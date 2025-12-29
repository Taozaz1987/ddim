import inspect
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from datasets import load_seismic, load_seismic_from_mat
from datasets.utils import normalize
from guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    DDNMSampler,
    DDRMSampler,
    DPSSampler,
)
from guided_diffusion import dist_util
from guided_diffusion.ddim import R_DDIMSampler
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
)
from metrics import LPIPS, PSNR, SSIM, Metric
from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer


def prepare_model(algorithm, conf, device):
    logging_info("Prepare model...")
    create_model_kwargs = select_args(
        conf, inspect.signature(create_model).parameters.keys()
    )
    unet = create_model(**create_model_kwargs)
    SAMPLER_CLS = {
        "repaint": SpacedDiffusion,
        "ddim": DDIMSampler,
        "o_ddim": O_DDIMSampler,
        "resample": R_DDIMSampler,
        "ddnm": DDNMSampler,
        "ddrm": DDRMSampler,
        "dps": DPSSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler


def prepare_data(
    mask_type="half",
    dataset_starting_index=-1,
    dataset_ending_index=-1,
    image_size=None,
    data_path=None,
    mask_params=None,
    mat_path=None,
    mat_key=None,
    mat_stride=None,
):
    shape = (image_size, image_size) if image_size is not None else None
    if mat_path:
        datas = load_seismic_from_mat(
            mat_path=mat_path,
            mat_key=mat_key or "input",
            mask_type=mask_type,
            mask_params=mask_params,
            patch_size=image_size,
            stride=mat_stride,
        )
    else:
        datas = load_seismic(
            data_path=data_path or "qiepian/marmousi_patches.npy",
            mask_type=mask_type,
            shape=shape,
            mask_params=mask_params,
            include_unmasked=True,
        )

    dataset_starting_index = (
        0 if dataset_starting_index == -1 else dataset_starting_index
    )
    dataset_ending_index = (
        len(datas) if dataset_ending_index == -1 else dataset_ending_index
    )
    datas = datas[dataset_starting_index:dataset_ending_index]

    logging_info(f"Load {len(datas)} samples")
    return datas


def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True


def main():
    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    base_defaults = {
        "data_path": "qiepian/marmousi_patches.npy",
        "mask_drop_rate": 0.5,
        "mask_drop_indices": [],
        "mat_path": "",
        "mat_key": "input",
        "mat_stride": 64,
    }
    config = Config(
        default_config_file="configs/seismic.yaml",
        default_config_dict=base_defaults,
        use_argparse=True,
    )
    if not hasattr(config, "in_channels"):
        config["in_channels"] = model_defaults()["in_channels"]
        config.in_channels = config["in_channels"]
    if not hasattr(config, "out_channels"):
        config["out_channels"] = model_defaults()["out_channels"]
        config.out_channels = config["out_channels"]
    config.show()

    all_paths = get_all_paths(config.outdir)
    config.dump(all_paths["path_config"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    recorder = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=config,
        use_git=config.use_git,
    )
    set_random_seed(config.seed, deterministic=False, no_torch=False, no_tf=True)

    ###################################################################################
    # prepare data
    ###################################################################################
    mask_params = {}
    if hasattr(config, "mask_drop_rate") and config.mask_drop_rate is not None:
        mask_params["drop_rate"] = config.mask_drop_rate
    if hasattr(config, "mask_drop_indices") and config.mask_drop_indices not in (None, []):
        mask_params["drop_indices"] = [int(i) for i in config.mask_drop_indices]
    if len(mask_params) == 0:
        mask_params = None

    if config.input_image == "":  # if input image is not given, load dataset
        datas = prepare_data(
            config.mask_type,
            config.dataset_starting_index,
            config.dataset_ending_index,
            config.image_size,
            data_path=config.data_path if hasattr(config, "data_path") else None,
            mask_params=mask_params,
            mat_path=config.mat_path if getattr(config, "mat_path", "") else None,
            mat_key=getattr(config, "mat_key", None),
            mat_stride=getattr(config, "mat_stride", None),
        )
    else:
        # NOTE: the model should accepet this input image size
        target_shape = (
            (config.image_size, config.image_size)
            if config.image_size is not None
            else None
        )
        image = normalize(
            Image.open(config.input_image).convert(
                "L" if config.in_channels == 1 else "RGB"
            ),
            shape=target_shape,
            channels=config.in_channels,
            grayscale=config.in_channels == 1,
        )
        if config.mode != "super_resolution":
            mask_image = Image.open(config.mask).convert("1")
            if target_shape is not None:
                mask_image = mask_image.resize(target_shape)
            mask = (
                torch.from_numpy(np.array(mask_image, dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            mask = torch.from_numpy(np.array([0]))  # just a dummy value
        datas = [(image, mask, "sample0")]

    ###################################################################################
    # prepare model and device
    ###################################################################################
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    unet, sampler = prepare_model(config.algorithm, config, device)

    def model_fn(x, t, y=None, **kwargs):
        forward_params = inspect.signature(unet.forward).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in forward_params}
        if "y" in forward_params:
            filtered_kwargs["y"] = y
        return unet(x, t, **filtered_kwargs)
    
    cond_fn = None

    METRICS = {
        "lpips": Metric(LPIPS("alex", device)),
        "psnr": Metric(PSNR(), eval_type="max"),
        "ssim": Metric(SSIM(), eval_type="max"),
    }
    final_loss = []

    ###################################################################################
    # start sampling
    ###################################################################################
    logging_info("Start sampling")
    timer, num_image = Timer(), 0
    batch_size = config.n_samples

    for data in tqdm(datas):
        image, mask, image_name, *extra_items = data
        original_image = extra_items[0] if len(extra_items) > 0 else image
        # prepare save dir
        outpath = os.path.join(config.outdir, image_name)
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = max(len(os.listdir(outpath)) - 3, 0)

        # prepare batch data for processing
        batch_image = original_image.to(device)
        batch_mask = mask.to(device)
        if batch_mask.dim() < 4:
            batch_mask = torch.ones(
                (batch_image.shape[0], 1, batch_image.shape[2],
                 batch_image.shape[3]),
                device=device,
                dtype=batch_image.dtype,
            )
        if batch_image.shape[1] != config.in_channels:
            if batch_image.shape[1] == 1:
                batch_image = batch_image.repeat(1, config.in_channels, 1, 1)
            else:
                batch_image = batch_image[:, : config.in_channels]
        if batch_mask.dim() == 3:
            batch_mask = batch_mask.unsqueeze(0)
        if batch_mask.shape[1] != config.in_channels:
            batch_mask = batch_mask.repeat(1, config.in_channels, 1, 1)
        batch_image_masked = batch_image * batch_mask
        model_gt = batch_image_masked
        batch = {
            "image": batch_image,
            "mask": batch_mask,
            "masked_image": batch_image_masked,
        }
        model_kwargs = {
            "gt": model_gt.repeat(batch_size, 1, 1, 1),
            "gt_keep_mask": batch["mask"].repeat(batch_size, 1, 1, 1),
        }

        _, _, height, width = batch["image"].shape
        shape = (batch_size, config.in_channels, height, width)

        all_metric_paths = [
            os.path.join(outpath, i + ".last")
            for i in (list(METRICS.keys()) + ["final_loss"])
        ]
        if config.get("resume", False) and all_exist(all_metric_paths):
            for metric_name, metric in METRICS.items():
                metric.dataset_scores += torch.load(
                    os.path.join(outpath, metric_name + ".last")
                )
            logging_info("Results exists. Skip!")
        else:
            # sample images
            samples = []
            for n in range(config.n_iter):
                timer.start()
                result = sampler.p_sample_loop(
                    model_fn,
                    shape=shape,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    progress=True,
                    return_all=True,
                    conf=config,
                    sample_dir=outpath if config["debug"] else None,
                )
                timer.end()

                for metric in METRICS.values():
                    metric.update(result["sample"], batch["image"])

                if "loss" in result.keys() and result["loss"] is not None:
                    recorder.add_with_logging(
                        key=f"loss_{image_name}_{n}", value=result["loss"]
                    )
                    final_loss.append(result["loss"])
                else:
                    final_loss.append(None)

                inpainted = normalize_image(result["sample"])
                samples.append(inpainted.detach().cpu())

            samples = torch.cat(samples)

            # save images
            # save gt images
            save_grid(normalize_image(batch["image"]), os.path.join(outpath, f"gt.png"))
            save_grid(
                normalize_image(batch["masked_image"]),
                os.path.join(outpath, f"masked.png"),
            )
            # save generations
            for sample in samples:
                save_image(sample, os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1
            save_grid(
                samples,
                os.path.join(outpath, f"grid-{grid_count:04}.png"),
                nrow=batch_size,
            )
            # save metrics
            for metric_name, metric in METRICS.items():
                torch.save(metric.dataset_scores[-config.n_iter:], os.path.join(outpath, metric_name + ".last"))

            torch.save(
                final_loss[-config.n_iter:], os.path.join(outpath, "final_loss.last"))

            num_image += 1
            last_duration = timer.get_last_duration()
            logging_info(
                "It takes %.3lf seconds for image %s"
                % (float(last_duration), image_name)
            )

        # report batch scores
        for metric_name, metric in METRICS.items():
            recorder.add_with_logging(
                key=f"{metric_name}_score_{image_name}",
                value=metric.report_batch(),
            )

    # report over all results
    for metric_name, metric in METRICS.items():
        mean, colbest_mean = metric.report_all()
        recorder.add_with_logging(key=f"mean_{metric_name}", value=mean)
        recorder.add_with_logging(
            key=f"best_mean_{metric_name}", value=colbest_mean)
    if len(final_loss) > 0 and final_loss[0] is not None:
        recorder.add_with_logging(
            key="final_loss",
            value=np.mean(final_loss),
        )
    if num_image > 0:
        recorder.add_with_logging(
            key="mean time", value=timer.get_cumulative_duration() / num_image
        )

    logging_info(
        f"Your samples are ready and waiting for you here: \n{config.outdir} \n"
        f" \nEnjoy."
    )
    recorder.end_recording()


if __name__ == "__main__":
    main()
