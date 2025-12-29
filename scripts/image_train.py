import argparse
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.seismic import load_seismic
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.train_util import TrainLoop


class SeismicPatchDataset(Dataset):
    """
    Dataset wrapper around the seismic numpy patches for training.
    """

    def __init__(self, data_path: str, image_size: Optional[int]):
        shape = (image_size, image_size) if image_size is not None else None
        self.samples = load_seismic(
            data_path=data_path, mask_type="ones", shape=shape)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        image, _, _ = self.samples[idx]
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        return image, {}


def load_data(
    data_path: str, batch_size: int, image_size: Optional[int]
) -> Iterator:
    dataset = SeismicPatchDataset(data_path=data_path, image_size=image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1
    )
    while True:
        yield from loader


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("starting training...")
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
        max_steps=args.max_steps,
        target_loss=args.target_loss,
    ).run_loop(progress=args.progress_bar)


def create_argparser():
    defaults = dict(
        data_path="qiepian/marmousi_patches.npy",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        progress_bar=False,
        max_steps=0,
        target_loss=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults["in_channels"] = 1
    defaults["out_channels"] = 1

    parser = argparse.ArgumentParser(
        description="Train a diffusion model on seismic patches.")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
