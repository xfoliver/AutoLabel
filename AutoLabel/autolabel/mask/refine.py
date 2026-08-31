"""Binary mask morphology refine (pre-contour)."""

from __future__ import annotations

import cv2
import numpy as np


def _odd_ksize(ksize: int) -> int:
    k = int(ksize)
    if k <= 0:
        return 0
    if k % 2 == 0:
        k += 1
    return max(k, 1)


def refine_binary_mask(
    mask: np.ndarray,
    *,
    close_ksize: int = 3,
    open_ksize: int = 0,
) -> np.ndarray:
    """Optional close then open on a binary instance mask.

    ``ksize <= 0`` skips that step. Kernel is ``MORPH_ELLIPSE``.
    Returns uint8 {0,1} with the same HxW as input.
    """
    if mask is None:
        return mask
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.size == 0:
        return binary

    ck = _odd_ksize(close_ksize)
    if ck > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    ok = _odd_ksize(open_ksize)
    if ok > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return (binary > 0).astype(np.uint8)
