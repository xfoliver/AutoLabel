"""Original-image GrabCut refine for instance masks (edge hugging)."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from autolabel.mask.io import resize_mask_nn


def _odd_ksize(ksize: int) -> int:
    k = int(ksize)
    if k <= 0:
        return 0
    if k % 2 == 0:
        k += 1
    return max(k, 1)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _to_original_binary(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape[0] == height and binary.shape[1] == width:
        return binary
    return (resize_mask_nn(binary, width, height) > 0).astype(np.uint8)


def grabcut_refine_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    iters: int = 3,
    pad_px: int = 8,
    sure_fg_erode: int = 3,
    sure_bg_dilate: int = 8,
    min_area_ratio: float = 0.3,
) -> np.ndarray:
    """Refine a coarse instance mask against the original BGR image via GrabCut.

    Returns uint8 ``{0,1}`` with the same HxW as ``image_bgr``. Soft-fails to the
    input mask (resized to the image) on OpenCV errors, empty seeds, or when the
    refined area collapses below ``min_area_ratio`` of the seed area.
    """
    if image_bgr is None or mask is None:
        return (np.asarray(mask) > 0).astype(np.uint8) if mask is not None else mask

    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] < 3:
        h, w = img.shape[:2]
        return _to_original_binary(mask, w, h)

    oh, ow = img.shape[:2]
    seed = _to_original_binary(mask, ow, oh)
    seed_area = int(np.count_nonzero(seed))
    if seed_area <= 0 or oh < 2 or ow < 2:
        return seed

    bbox = _mask_bbox(seed)
    if bbox is None:
        return seed

    # ROI must include sure-background outside the seed (dilate radius + frame).
    pad = max(int(pad_px), _odd_ksize(sure_bg_dilate) + 2, 4)
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(ow, bbox[2] + pad)
    y1 = min(oh, bbox[3] + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return seed

    roi = np.ascontiguousarray(img[y0:y1, x0:x1, :3])
    seed_roi = seed[y0:y1, x0:x1]
    if int(np.count_nonzero(seed_roi)) <= 0:
        return seed

    # Standard mask init: sure BG outside dilated seed, PR_FGD on seed, sure FG eroded.
    gc = np.full(seed_roi.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc[seed_roi > 0] = cv2.GC_PR_FGD

    bg_k = _odd_ksize(sure_bg_dilate)
    if bg_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_k, bg_k))
        dilated = cv2.dilate(seed_roi, kernel)
        gc[dilated == 0] = cv2.GC_BGD
    else:
        gc[seed_roi == 0] = cv2.GC_BGD

    # Force a 1px sure-BG frame so GrabCut always has background samples in-ROI.
    gc[0, :] = cv2.GC_BGD
    gc[-1, :] = cv2.GC_BGD
    gc[:, 0] = cv2.GC_BGD
    gc[:, -1] = cv2.GC_BGD
    # Restore seed on the frame if the object touches the ROI border.
    gc[seed_roi > 0] = cv2.GC_PR_FGD

    fg_k = _odd_ksize(sure_fg_erode)
    if fg_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_k, fg_k))
        eroded = cv2.erode(seed_roi, kernel)
        if int(np.count_nonzero(eroded)) > 0:
            gc[eroded > 0] = cv2.GC_FGD

    if not np.any(gc == cv2.GC_FGD) and not np.any(gc == cv2.GC_PR_FGD):
        return seed

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            roi,
            gc,
            None,
            bgd_model,
            fgd_model,
            max(1, int(iters)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return seed

    refined_roi = np.where(
        (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)
    out = np.zeros((oh, ow), dtype=np.uint8)
    out[y0:y1, x0:x1] = refined_roi
    out_area = int(np.count_nonzero(out))
    if out_area <= 0:
        return seed
    if out_area < float(min_area_ratio) * float(seed_area):
        return seed
    return out
