import torch
import numpy as np
import blobfile as bf
from PIL import Image


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    if arr.max() <= 1.0 and arr.min() >= -1.0:
        arr = (arr + 1.0) * 127.5
    elif arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _resize_array(arr: np.ndarray, shape, mode: str) -> np.ndarray:
    if shape is None or arr.shape[:2] == shape:
        return arr
    arr_for_pil = _to_uint8_image(arr)
    if arr_for_pil.ndim == 3 and arr_for_pil.shape[-1] == 1:
        arr_for_pil = arr_for_pil[..., 0]
    pil_image = Image.fromarray(arr_for_pil).convert(mode)
    pil_image = pil_image.resize(shape)
    return np.array(pil_image)


def normalize(image, shape=(256, 256), channels=3, grayscale=False):
    """
    Normalize an image or numpy array into [-1, 1] with shape (1, C, H, W).

    Args:
        image: image to be normalized, PIL.Image or numpy array.
        shape: the desired spatial shape of the image, or None to keep input.
        channels: desired number of channels in the output tensor.
        grayscale: force grayscale handling and channel count of 1.

    Returns: the normalized image tensor.

    """
    target_channels = 1 if grayscale or channels == 1 else channels

    if isinstance(image, np.ndarray):
        arr = image
        channel_first = arr.ndim == 3 and arr.shape[0] in (1, target_channels)
        if channel_first:
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = _resize_array(arr, shape, "L" if target_channels == 1 else "RGB")
    else:
        pil_image = image.convert("L" if target_channels == 1 else "RGB")
        if shape is not None:
            pil_image = pil_image.resize(shape)
        arr = np.array(pil_image)

    arr = arr.astype(np.float32)
    if arr.max() > 1.0 or arr.min() < -1.0:
        arr = arr / 255.0
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = arr * 2.0 - 1.0

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, target_channels) and arr.shape[1] != target_channels and arr.shape[2] != target_channels:
            arr = arr
        else:
            arr = arr.transpose(2, 0, 1)

    if target_channels == 1 and arr.shape[0] != 1:
        arr = arr[:1]
    arr = torch.from_numpy(arr).unsqueeze(0)
    return arr


# Copied from Repaint code
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
