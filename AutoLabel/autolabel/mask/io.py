"""Mask file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from autolabel.cv_io import imread, imwrite


def save_mask_png(path: Union[str, Path], mask: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = mask.astype(np.uint8)
    if arr.max() <= 1:
        arr = (arr > 0).astype(np.uint8) * 255
    if not imwrite(out, arr):
        raise OSError(f"Failed to write mask: {out}")
    return out


def load_mask_png(path: Union[str, Path]) -> np.ndarray:
    img = imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def save_mask_npy(path: Union[str, Path], mask: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), mask)
    return out


def resize_mask_nn(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
