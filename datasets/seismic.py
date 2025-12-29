import os
import inspect
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import scipy.io as sio

from datasets.masks import mask_generators
from datasets.utils import normalize


def _call_mask_generator(mask_type: str, shape: Sequence[int], mask_params: Optional[dict]):
    if mask_type not in mask_generators:
        raise ValueError(f"Unsupported mask type '{mask_type}'.")

    mask_generator = mask_generators[mask_type]
    params = mask_params or {}
    sig = inspect.signature(mask_generator)
    kwargs = {
        k: v for k, v in params.items() if k in sig.parameters and k != "shape"
    }
    return mask_generator(shape, **kwargs)


def _get_spatial_shape(patch: np.ndarray) -> Tuple[int, int]:
    if patch.ndim == 3 and patch.shape[0] in (1, 3):
        return tuple(patch.shape[1:3])
    return tuple(patch.shape[:2])


def _normalize_section(section: np.ndarray) -> np.ndarray:
    section = section.astype(np.float32)
    max_val = np.max(np.abs(section))
    if max_val > 0:
        section = section / (max_val + 1e-8)
    return section


def _load_mat_section(mat_path: str, mat_key: str) -> np.ndarray:
    try:
        mat_data = sio.loadmat(mat_path)
        if mat_key not in mat_data:
            known_keys = [k for k in mat_data.keys() if not k.startswith("__")]
            raise KeyError(f"Key '{mat_key}' not found in {mat_path}. Available keys: {known_keys}")
        section = mat_data[mat_key]
    except Exception:
        section = np.load(mat_path)
    section = np.squeeze(section)
    if section.ndim != 2:
        raise ValueError(f"Expected 2D seismic section, got shape {section.shape}")
    return _normalize_section(section)


def _extract_patches(
    section: np.ndarray, patch_size: Optional[int], stride: Optional[int]
) -> List[np.ndarray]:
    if patch_size is None:
        return [section]
    stride = stride or patch_size
    h, w = section.shape
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Patch size {patch_size} is larger than seismic section {section.shape}"
        )
    patches = []
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patches.append(section[r: r + patch_size, c: c + patch_size])
    if not patches:
        patches.append(section)
    return patches


def _prepare_sample(
    patch: np.ndarray,
    target_shape: Tuple[int, int],
    mask_type: str,
    mask_params: Optional[dict],
    name: str,
    include_unmasked: bool = True,
):
    image = normalize(patch, shape=target_shape, channels=1, grayscale=True)
    mask = _call_mask_generator(mask_type, target_shape, mask_params)
    masked_image = image * mask
    if include_unmasked:
        return masked_image, mask, name, image
    return masked_image, mask, name


def load_seismic(
    data_path="qiepian/marmousi_patches.npy",
    mask_type="ones",
    shape=None,
    mask_params: Optional[dict] = None,
    include_unmasked: bool = True,
):
    """
    Load seismic patches saved in a numpy array and prepare them for sampling.

    Args:
        data_path: path to the .npy file containing patches shaped (N, H, W) or (N, H, W, 1).
        mask_type: mask type from datasets.masks.mask_generators.
        shape: optional spatial shape to resize patches and masks to.
        mask_params: optional parameters forwarded to the mask generator.
        include_unmasked: return the original unmasked tensor as the 4th tuple element.

    Returns:
        A list of tuples (masked_image, mask, name[, unmasked_image]) where image tensors are shaped (1, 1, H, W).
    """
    full_path = data_path if os.path.isabs(data_path) else os.path.join(os.getcwd(), data_path)
    patches = np.load(full_path)

    if patches.ndim not in (3, 4):
        raise ValueError(f"Unexpected patch dimensions {patches.shape}, expected (N, H, W) or (N, H, W, 1).")
    if patches.ndim == 4 and patches.shape[-1] == 1:
        patches = patches[..., 0]

    dataset = []
    for idx, patch in enumerate(patches):
        spatial_shape = _get_spatial_shape(patch)
        target_shape = shape if shape is not None else spatial_shape
        dataset.append(
            _prepare_sample(
                patch,
                target_shape=target_shape,
                mask_type=mask_type,
                mask_params=mask_params,
                include_unmasked=include_unmasked,
                name=f"seismic_{idx:05d}",
            )
        )
    return dataset


def load_seismic_from_mat(
    mat_path: str,
    mat_key: str = "input",
    mask_type: str = "trace_dropout",
    mask_params: Optional[dict] = None,
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
):
    """
    Load a seismic section from .mat or .npy file, normalize it, and slice into patches.

    Args:
        mat_path: path to the .mat/.npy file.
        mat_key: key used to retrieve data from the .mat file.
        mask_type: mask to apply to traces.
        mask_params: optional parameters forwarded to the mask generator.
        patch_size: size of square patches. If None, process the full section.
        stride: stride between patches; defaults to patch_size when provided.

    Returns:
        Dataset list of (masked_image, mask, name, unmasked_image).
    """
    section = _load_mat_section(mat_path, mat_key)
    patches = _extract_patches(section, patch_size=patch_size, stride=stride)

    dataset = []
    for idx, patch in enumerate(patches):
        target_shape = _get_spatial_shape(patch)
        dataset.append(
            _prepare_sample(
                patch,
                target_shape=target_shape,
                mask_type=mask_type,
                mask_params=mask_params,
                include_unmasked=True,
                name=f"seismic_mat_{idx:05d}",
            )
        )
    return dataset
