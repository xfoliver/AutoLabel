"""Unit tests for original-space polygon area filtering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.geometry.polygon import (  # noqa: E402
    instance_mask_to_polygons,
    polygon_area,
)
from autolabel.geometry.transform import (  # noqa: E402
    build_stretch_meta,
    mask_area_scale,
    original_min_area_to_mask,
)
from autolabel.models import InstancePred, Stage1Result  # noqa: E402
from autolabel.stage2 import stage1_to_shapes  # noqa: E402


def test_polygon_area_unit_square():
    assert abs(polygon_area([[0, 0], [10, 0], [10, 5], [0, 5]]) - 50.0) < 1e-6


def test_mask_area_scale_stretch():
    # original 480x640, mask/input 1085x1450 → sx=1085/480, sy=1450/640
    meta = build_stretch_meta((480, 640), (1085, 1450), (1085, 1450))
    scale = mask_area_scale(meta)
    expected = (1085 / 480) * (1450 / 640)
    assert abs(scale - expected) < 1e-6
    assert original_min_area_to_mask(64.0, meta) > 300


def test_tiny_mask_blob_dropped_in_original_space():
    # Stretch 2x in both axes: mask 100x100 → original 50x50
    meta = build_stretch_meta((50, 50), (100, 100), (100, 100))
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 6x6 blob in mask → ~9 original px² after /4 area scale
    mask[10:16, 10:16] = 1
    polys = instance_mask_to_polygons(mask, meta, min_area=64.0, min_area_space="original")
    assert polys == []


def test_large_mask_blob_kept_in_original_space():
    meta = build_stretch_meta((50, 50), (100, 100), (100, 100))
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 40x40 in mask → 20x20 = 400 original px²
    mask[10:50, 10:50] = 1
    polys = instance_mask_to_polygons(mask, meta, min_area=64.0, min_area_space="original")
    assert len(polys) == 1
    assert polygon_area(polys[0]) >= 64.0


def test_stage1_to_shapes_uses_original_min_area():
    meta = build_stretch_meta((50, 50), (100, 100), (100, 100))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:16, 10:16] = 1
    result = Stage1Result(
        instances=[
            InstancePred(
                instance_id=1,
                class_id=1,
                class_name="手机",
                mask=mask,
                score=None,
            )
        ],
        transform=meta,
        raw={},
    )
    shapes = stage1_to_shapes(result, min_area=64.0)
    assert shapes == []
