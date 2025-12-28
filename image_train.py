"""
Train a diffusion model on Seismic Images (Numpy format).

Example:
    python image_train.py --data_dir datasets/seismic --batch_size 4 --image_size 128 --stream_from_dir
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _ensure_repo_imports() -> None:
    """
    Expect to run from the repository root; add the parent directory to sys.path if the
    guided_diffusion package is not already importable.
    """
    project_root = Path(__file__).resolve().parent
    module_missing = importlib.util.find_spec("guided_diffusion") is None
    if module_missing and str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        module_missing = importlib.util.find_spec("guided_diffusion") is None
    if module_missing:
        raise RuntimeError(
            "Unable to import guided_diffusion. Run this script from the repository root "
            f"or set PYTHONPATH to include {project_root}."
        )


_ensure_repo_imports()

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (  # noqa: E402
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    str2bool,
)
from guided_diffusion.train_util import TrainLoop  # noqa: E402


def _prepare_tensor(
    array: np.ndarray,
    image_size: int,
    file_path: Path,
    *,
    auto_normalize: bool,
    clamp_to_unit: bool,
    resize_on_mismatch: bool,
    log_actions: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    original_dtype = str(array.dtype)
    array = np.asarray(array)

    if array.ndim == 3:
        array = array[:, np.newaxis, :, :]
    elif array.ndim == 4:
        if array.shape[1] not in (1, 3):
            raise ValueError(
                f"Unsupported channel dimension {array.shape[1]} in {file_path}. "
                "Expected channels to be 1 or 3."
            )
    else:
        raise ValueError(
            f"Expected array with shape (N, H, W) or (N, C, H, W), got {array.shape} in {file_path}."
        )

    min_value = float(array.min())
    max_value = float(array.max())
    value_range_ok = -1.0 <= min_value <= 1.0 and -1.0 <= max_value <= 1.0

    array = array.astype(np.float32, copy=False)

    if not value_range_ok and log_actions:
        logger.log(
            f"Value range warning for {file_path}: min={min_value:.4f}, max={max_value:.4f} "
            "expected within [-1, 1]."
        )

    if auto_normalize and not value_range_ok:
        scale = max(abs(min_value), abs(max_value))
        if scale > 0:
            array = array / scale
            if log_actions:
                logger.log(
                    f"Auto-normalized {file_path} by dividing by {scale:.6f} to fit within [-1, 1]."
                )
        elif log_actions:
            logger.log(f"Skipping auto-normalization for {file_path} because values are constant.")
    elif clamp_to_unit and not value_range_ok:
        np.clip(array, -1.0, 1.0, out=array)
        if log_actions:
            logger.log(f"Clamped values in {file_path} to [-1, 1].")

    tensor = torch.from_numpy(array)
    resized = False
    _, channels, height, width = tensor.shape

    if (height, width) != (image_size, image_size):
        if not resize_on_mismatch:
            raise ValueError(
                f"{file_path} has spatial size {height}x{width}; expected {image_size}. "
                "Use --resize_on_mismatch to resize automatically."
            )
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        resized = True
        if log_actions:
            logger.log(f"Resized {file_path} from {height}x{width} to {image_size}x{image_size}.")

    tensor_min = float(tensor.min().item())
    tensor_max = float(tensor.max().item())

    info = {
        "file": str(file_path),
        "num_samples": tensor.shape[0],
        "channels": channels,
        "height": tensor.shape[2],
        "width": tensor.shape[3],
        "dtype": original_dtype,
        "min_value": tensor_min,
        "max_value": tensor_max,
        "resized": resized,
    }
    return tensor, info


def load_seismic_data(
    data_path: str,
    batch_size: int,
    image_size: int,
    *,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    stream_from_dir: bool = False,
    auto_normalize: bool = False,
    clamp_to_unit: bool = False,
    resize_on_mismatch: bool = False,
) -> Tuple[Iterable[Tuple[torch.Tensor, Dict]], Dict[str, object]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if path.is_dir():
        file_paths = sorted(p for p in path.glob("*.npy") if p.is_file())
        if not file_paths:
            raise ValueError(f"No .npy files found in directory: {data_path}")
    else:
        if path.suffix != ".npy":
            raise ValueError(f"Expected a .npy file, got: {data_path}")
        file_paths = [path]

    processed_tensors: List[torch.Tensor] = []
    per_file_info: List[Dict[str, object]] = []
    total_samples = 0
    global_min = float("inf")
    global_max = float("-inf")
    channels = None
    height = None
    width = None
    dtype_set = set()
    resized_any = False

    for file_path in file_paths:
        tensor, info = _prepare_tensor(
            np.load(file_path),
            image_size,
            file_path,
            auto_normalize=auto_normalize,
            clamp_to_unit=clamp_to_unit,
            resize_on_mismatch=resize_on_mismatch,
        )
        per_file_info.append(info)
        total_samples += info["num_samples"]
        global_min = min(global_min, info["min_value"])
        global_max = max(global_max, info["max_value"])
        resized_any = resized_any or info["resized"]
        dtype_set.add(info["dtype"])

        if channels is None:
            channels = info["channels"]
            height = info["height"]
            width = info["width"]
        elif info["channels"] != channels:
            raise ValueError(
                f"Channel mismatch across files: expected {channels}, found {info['channels']} in {file_path}."
            )

        if not stream_from_dir:
            processed_tensors.append(tensor)

    if channels is None or height is None or width is None:
        raise ValueError(f"No samples could be loaded from {data_path}.")
    if total_samples == 0:
        raise ValueError(f"Loaded tensors from {data_path} but found zero samples.")

    effective_drop_last = drop_last
    if drop_last and total_samples < batch_size:
        logger.log(
            f"drop_last=True would discard all {total_samples} samples with batch_size={batch_size}; "
            "overriding to drop_last=False."
        )
        effective_drop_last = False

    per_file_drop_last: List[bool] = []
    for info in per_file_info:
        if effective_drop_last and info["num_samples"] < batch_size:
            logger.log(
                f"drop_last disabled for {info['file']} because it only has {info['num_samples']} samples "
                f"(batch_size={batch_size})."
            )
            per_file_drop_last.append(False)
        else:
            per_file_drop_last.append(effective_drop_last)

    data_info = {
        "data_path": str(path),
        "files": [str(fp) for fp in file_paths],
        "num_files": len(file_paths),
        "num_samples": total_samples,
        "channels": channels,
        "height": height,
        "width": width,
        "dtype": ", ".join(sorted(dtype_set)),
        "min_value": global_min,
        "max_value": global_max,
        "resized": resized_any,
        "streaming": stream_from_dir,
    }

    if not stream_from_dir:
        if len(file_paths) > 1:
            logger.log(
                f"Concatenating {len(file_paths)} files ({total_samples} samples) into memory. "
                "Use --stream_from_dir to load one file at a time if memory constrained."
            )
        tensor_data = torch.cat(processed_tensors, dim=0)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=effective_drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        def _in_memory_iterator() -> Iterable[Tuple[torch.Tensor, Dict]]:
            while True:
                for batch in loader:
                    yield batch[0], {}

        return _in_memory_iterator(), data_info

    def _streaming_iterator() -> Iterable[Tuple[torch.Tensor, Dict]]:
        while True:
            for file_path, file_drop_last in zip(file_paths, per_file_drop_last):
                tensor, _ = _prepare_tensor(
                    np.load(file_path),
                    image_size,
                    file_path,
                    auto_normalize=auto_normalize,
                    clamp_to_unit=clamp_to_unit,
                    resize_on_mismatch=resize_on_mismatch,
                    log_actions=False,
                )
                dataset = TensorDataset(tensor)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=file_drop_last,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                for batch in loader:
                    yield batch[0], {}

    return _streaming_iterator(), data_info


def main() -> None:
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log(
        "Example usage: python image_train.py --data_dir datasets/seismic --batch_size 4 --image_size 128"
    )

    data, data_info = load_seismic_data(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        stream_from_dir=args.stream_from_dir,
        auto_normalize=args.auto_normalize,
        clamp_to_unit=args.clamp_to_unit,
        resize_on_mismatch=args.resize_on_mismatch,
    )

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {resume_path}")

    inferred_channels = data_info["channels"]
    in_channels = args.in_channels if args.in_channels is not None else inferred_channels
    out_channels = args.out_channels if args.out_channels is not None else in_channels
    args.in_channels = in_channels
    args.out_channels = out_channels

    logger.log(f"Data path(s): {', '.join(data_info['files'])}")
    logger.log(
        f"Loaded {data_info['num_samples']} samples from {data_info['num_files']} file(s); "
        f"sample shape={data_info['channels']}x{data_info['height']}x{data_info['width']}; "
        f"dtype={data_info['dtype']} value range=[{data_info['min_value']:.4f}, {data_info['max_value']:.4f}] "
        f"(resized={data_info['resized']}, streaming={data_info['streaming']})."
    )
    logger.log(
        "Model configuration: "
        f"image_size={args.image_size}, in_channels={in_channels}, out_channels={out_channels}, "
        f"num_channels={args.num_channels}, num_res_blocks={args.num_res_blocks}, "
        f"attention_resolutions={args.attention_resolutions}, dropout={args.dropout}"
    )

    logger.log("creating model and diffusion...")
    args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    args_dict["in_channels"] = in_channels
    args_dict["out_channels"] = out_channels
    model, diffusion = create_model_and_diffusion(**args_dict)

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    model_defaults = model_and_diffusion_defaults()
    model_defaults_without_channels = {
        k: v for k, v in model_defaults.items() if k not in ("in_channels", "out_channels")
    }
    defaults.update(model_defaults_without_channels)
    parser = argparse.ArgumentParser(description="Train a diffusion model on seismic image data.")
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--in_channels",
        type=int,
        default=None,
        help="Input channel count; defaults to the dataset's channel dimension.",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=None,
        help="Output channel count; defaults to match --in_channels or the dataset.",
    )
    parser.add_argument(
        "--drop_last",
        type=str2bool,
        default=True,
        help="Whether to drop the last incomplete batch. Disabled automatically when it would drop all data.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the PyTorch DataLoader.",
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help="Whether to pin dataloader memory.",
    )
    parser.add_argument(
        "--stream_from_dir",
        action="store_true",
        help="When data_dir is a directory, stream one .npy file at a time instead of concatenating in memory.",
    )
    parser.add_argument(
        "--auto_normalize",
        action="store_true",
        help="Automatically scale values into [-1, 1] when they fall outside that range.",
    )
    parser.add_argument(
        "--clamp_to_unit",
        action="store_true",
        help="Clamp values to [-1, 1] if they fall outside that range (ignored when --auto_normalize is set).",
    )
    parser.add_argument(
        "--resize_on_mismatch",
        action="store_true",
        help="Resize inputs whose spatial size does not match image_size instead of raising an error.",
    )
    return parser


if __name__ == "__main__":
    main()
