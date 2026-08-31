"""Polygon helpers for Stage 2."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autolabel.geometry.transform import map_points_mask_to_original
from autolabel.models import TransformMeta

Points = List[List[float]]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace area of a polygon (absolute value)."""
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def mask_to_contours(
    mask: np.ndarray,
    *,
    min_area: float = 16.0,
) -> List[np.ndarray]:
    """Return external contours in mask coordinates (Nx1x2 int)."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        out.append(cnt)
    return out


def simplify_contour(contour: np.ndarray, epsilon_ratio: float = 0.001) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    eps = max(epsilon_ratio * peri, 0.5)
    approx = cv2.approxPolyDP(contour, eps, True)
    if approx is None or len(approx) < 3:
        return contour
    return approx


def contour_to_points(contour: np.ndarray) -> Points:
    pts = contour.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


def instance_mask_to_polygons(
    mask: np.ndarray,
    transform: TransformMeta,
    *,
    min_area: float = 64.0,
    min_area_space: str = "original",
    epsilon_ratio: float = 0.001,
    close_ksize: int = 3,
    open_ksize: int = 0,
    keep_largest: bool = True,
    clip: bool = True,
) -> List[Points]:
    """Mask (mask-space) → list of polygons in original image coordinates.

    ``min_area`` defaults to **original-image** pixel area when
    ``min_area_space="original"``. Mask-space prefilter is derived from the
    transform so tiny stretch fragments are dropped after mapping.

    Optional morphology (``close_ksize`` / ``open_ksize``) runs on the binary
    instance mask before contours; ``<=0`` disables that step.

    When ``keep_largest`` is True (default), only the largest surviving
    polygon is returned — one InstancePred → one LabelMe shape.
    """
    from autolabel.geometry.transform import original_min_area_to_mask
    from autolabel.mask.refine import refine_binary_mask

    space = (min_area_space or "original").lower()
    if space == "mask":
        mask_min = float(min_area)
        orig_min = 0.0
    else:
        orig_min = float(min_area)
        # Conservative mask prefilter: keep candidates that might survive mapping.
        mask_min = max(16.0, original_min_area_to_mask(orig_min, transform) * 0.5)

    refined = refine_binary_mask(mask, close_ksize=close_ksize, open_ksize=open_ksize)

    polygons: List[Points] = []
    for cnt in mask_to_contours(refined, min_area=mask_min):
        approx = simplify_contour(cnt, epsilon_ratio=epsilon_ratio)
        mask_pts = contour_to_points(approx)
        if len(mask_pts) < 3:
            continue
        ori_pts = map_points_mask_to_original(mask_pts, transform, clip=clip)
        if len(ori_pts) < 3:
            continue
        if orig_min > 0.0 and polygon_area(ori_pts) < orig_min:
            continue
        polygons.append(ori_pts)
    if keep_largest and len(polygons) > 1:
        polygons = [max(polygons, key=polygon_area)]
    return polygons


def bbox_from_points(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)
