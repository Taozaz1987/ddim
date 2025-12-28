import os

import numpy as np
import torch

from datasets.masks import mask_generators
from datasets.utils import normalize


def load_seismic(
    data_path="qiepian/marmousi_patches.npy",
    mask_type="ones",
    shape=None,
):
    """
    Load seismic patches saved in a numpy array and prepare them for sampling.

    Args:
        data_path: path to the .npy file containing patches shaped (N, H, W) or (N, H, W, 1).
        mask_type: mask type from datasets.masks.mask_generators, defaults to full ones.
        shape: optional spatial shape to resize patches and masks to.

    Returns:
        A list of tuples (image, mask, name) where image is a tensor of shape (1, 1, H, W).
    """
    full_path = data_path if os.path.isabs(data_path) else os.path.join(os.getcwd(), data_path)
    patches = np.load(full_path)

    if patches.ndim not in (3, 4):
        raise ValueError(f"Unexpected patch dimensions {patches.shape}, expected (N, H, W) or (N, H, W, 1).")
    if patches.ndim == 4 and patches.shape[-1] == 1:
        patches = patches[..., 0]

    if mask_type not in mask_generators:
        raise ValueError(f"Unsupported mask type '{mask_type}'.")
    mask_generator = mask_generators[mask_type]

    dataset = []
    for idx, patch in enumerate(patches):
        target_shape = shape if shape is not None else patch.shape[:2]
        image = normalize(patch, shape=target_shape, channels=1, grayscale=True)
        mask = mask_generator(target_shape)
        dataset.append((image, mask, f"seismic_{idx:05d}"))
    return dataset
