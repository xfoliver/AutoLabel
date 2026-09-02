"""Geometry package exports."""

from autolabel.geometry.polygon import (
    instance_mask_to_polygons,
    mask_to_contours,
    polygon_area,
)
from autolabel.geometry.transform import (
    build_letterbox_meta,
    build_stretch_meta,
    letterbox_image_bgr,
    map_mask_to_original,
    map_original_to_mask,
    map_points_mask_to_original,
    mask_area_scale,
    original_min_area_to_mask,
    warp_mask_to_original,
)

__all__ = [
    "build_letterbox_meta",
    "build_stretch_meta",
    "letterbox_image_bgr",
    "map_mask_to_original",
    "map_original_to_mask",
    "map_points_mask_to_original",
    "mask_area_scale",
    "original_min_area_to_mask",
    "warp_mask_to_original",
    "instance_mask_to_polygons",
    "mask_to_contours",
    "polygon_area",
]
