"""Parse flat colored mask images into InstancePred list (MVP: one class)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autolabel.models import InstancePred


def _luma_bgr(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    return 0.114 * b.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.299 * r.astype(np.float32)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a_b, b_b).sum()
    return float(inter) / float(max(union, 1))


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter) / float(max(area_a + area_b - inter, 1))


def _masks_touch(a: np.ndarray, b: np.ndarray, *, gap_px: int) -> bool:
    """True if masks overlap, or come within ``gap_px`` after dilation."""
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    if np.logical_and(a_b, b_b).any():
        return True
    g = int(gap_px)
    if g <= 0:
        return False
    k = 2 * g + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    a_d = cv2.dilate(a_b.astype(np.uint8), kernel)
    return bool(np.logical_and(a_d.astype(bool), b_b).any())


def merge_overlapping_instances(
    instances: Sequence[InstancePred],
    *,
    iou_thresh: float = 0.15,
    bbox_iou_thresh: float = 0.20,
    min_relative_area: float = 0.12,
    gap_px: int = 5,
) -> List[InstancePred]:
    """Merge color-split fragments that cover the same object; drop tiny leftovers.

    - Pairwise union when mask IoU >= ``iou_thresh``, or bbox IoU >= ``bbox_iou_thresh``
      (covers overlapping color bands and later-paint replacement with mask IoU≈0).
    - Also union when masks touch / are within ``gap_px`` (adjacent color halves).
    - After merges, drop instances whose area is < ``min_relative_area`` * max area.
    """
    if not instances:
        return []

    items: List[InstancePred] = [
        InstancePred(
            instance_id=i.instance_id,
            class_id=i.class_id,
            class_name=i.class_name,
            mask=(i.mask > 0).astype(np.uint8),
            score=i.score,
        )
        for i in instances
    ]

    changed = True
    while changed and len(items) > 1:
        changed = False
        areas = [int(np.count_nonzero(it.mask)) for it in items]
        bboxes = [_mask_bbox(it.mask) for it in items]
        order = sorted(range(len(items)), key=lambda i: -areas[i])
        merged_into: Optional[Tuple[int, int]] = None
        for ii, i in enumerate(order):
            for j in order[ii + 1 :]:
                if areas[j] <= 0 or areas[i] <= 0:
                    continue
                miou = _mask_iou(items[i].mask, items[j].mask)
                biou = _bbox_iou(bboxes[i], bboxes[j])
                # Overlapping color bands, heavily overlapping boxes, or adjacent
                # halves of one object painted with different solid colors.
                if (
                    miou >= float(iou_thresh)
                    or biou >= float(bbox_iou_thresh)
                    or _masks_touch(items[i].mask, items[j].mask, gap_px=gap_px)
                ):
                    merged_into = (i, j)
                    break
            if merged_into is not None:
                break
        if merged_into is None:
            break
        keep, drop = merged_into
        # Prefer keeping the larger index as the host.
        if areas[keep] < areas[drop]:
            keep, drop = drop, keep
        items[keep].mask = np.logical_or(items[keep].mask.astype(bool), items[drop].mask.astype(bool)).astype(
            np.uint8
        )
        del items[drop]
        changed = True

    if not items:
        return []

    areas = [int(np.count_nonzero(it.mask)) for it in items]
    max_area = max(areas) if areas else 0
    rel = float(min_relative_area)
    kept = [
        it
        for it, area in zip(items, areas)
        if max_area <= 0 or area >= max(1, int(round(rel * max_area)))
    ]
    # Re-id stably by descending area.
    kept.sort(key=lambda it: -int(np.count_nonzero(it.mask)))
    out: List[InstancePred] = []
    for n, it in enumerate(kept, start=1):
        out.append(
            InstancePred(
                instance_id=n,
                class_id=it.class_id,
                class_name=it.class_name,
                mask=it.mask,
                score=it.score,
            )
        )
    return out


def colored_mask_to_instances(
    rgb_bgr: np.ndarray,
    *,
    class_name: str = "object",
    class_id: int = 1,
    min_area: int = 64,
    quantize: int = 16,
    bg_luma_thresh: float = 24.0,
    merge_iou: float = 0.15,
    merge_bbox_iou: float = 0.20,
    min_relative_area: float = 0.12,
    merge_gap_px: int = 5,
    merge_fragments: bool = True,
) -> List[InstancePred]:
    """Map each non-background quantized color to one instance.

    Expects OpenCV BGR uint8 image with near-black background and flat solid
    colors per instance. Anti-aliased edges are absorbed by quantization.

    When ``merge_fragments`` is True (default), overlapping / nested / adjacent
    color fragments of the same object are unioned and tiny leftovers dropped.
    """
    if rgb_bgr is None or rgb_bgr.size == 0:
        return []
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] < 3:
        raise ValueError("colored_mask_to_instances expects HxWx3 BGR image")

    img = rgb_bgr[:, :, :3]
    q = max(int(quantize), 1)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    luma = _luma_bgr(b, g, r)
    fg = luma >= float(bg_luma_thresh)

    qb = (b.astype(np.int32) // q) * q
    qg = (g.astype(np.int32) // q) * q
    qr = (r.astype(np.int32) // q) * q
    # Pack quantized BGR into int32 key
    keys = (qb << 16) | (qg << 8) | qr
    keys = np.where(fg, keys, -1)

    flat = keys.reshape(-1)
    uniq, counts = np.unique(flat, return_counts=True)
    instances: List[InstancePred] = []
    next_id = 1
    # Stable order: larger area first (deterministic for tests)
    order = np.argsort(-counts)
    for idx in order:
        key = int(uniq[idx])
        if key < 0:
            continue
        area = int(counts[idx])
        if area < int(min_area):
            continue
        mask = (keys == key).astype(np.uint8)
        instances.append(
            InstancePred(
                instance_id=next_id,
                class_id=int(class_id),
                class_name=str(class_name or "object"),
                mask=mask,
                score=None,
            )
        )
        next_id += 1

    if merge_fragments and instances:
        instances = merge_overlapping_instances(
            instances,
            iou_thresh=float(merge_iou),
            bbox_iou_thresh=float(merge_bbox_iou),
            min_relative_area=float(min_relative_area),
            gap_px=int(merge_gap_px),
        )
    return instances


def summarize_colors(
    rgb_bgr: np.ndarray,
    *,
    quantize: int = 16,
    bg_luma_thresh: float = 24.0,
    top_k: int = 20,
) -> List[Tuple[Tuple[int, int, int], int]]:
    """Debug helper: top quantized (B,G,R) colors and counts (excluding bg)."""
    if rgb_bgr is None or rgb_bgr.ndim != 3:
        return []
    q = max(int(quantize), 1)
    b, g, r = rgb_bgr[:, :, 0], rgb_bgr[:, :, 1], rgb_bgr[:, :, 2]
    luma = _luma_bgr(b, g, r)
    fg = luma >= float(bg_luma_thresh)
    qb = (b.astype(np.int32) // q) * q
    qg = (g.astype(np.int32) // q) * q
    qr = (r.astype(np.int32) // q) * q
    keys = (qb << 16) | (qg << 8) | qr
    keys = np.where(fg, keys, -1)
    uniq, counts = np.unique(keys.reshape(-1), return_counts=True)
    out: List[Tuple[Tuple[int, int, int], int]] = []
    for i in np.argsort(-counts):
        key = int(uniq[i])
        if key < 0:
            continue
        bb, gg, rr = (key >> 16) & 255, (key >> 8) & 255, key & 255
        out.append(((bb, gg, rr), int(counts[i])))
        if len(out) >= top_k:
            break
    return out
