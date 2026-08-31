"""Unit tests for colored mask → InstancePred."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.mask.color_instances import colored_mask_to_instances  # noqa: E402


def test_two_solid_colors_become_two_instances():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    # BGR: red block + blue block
    img[10:50, 10:50] = (0, 0, 255)
    img[55:95, 60:110] = (255, 0, 0)
    inst = colored_mask_to_instances(img, class_name="颗粒", min_area=16, quantize=16)
    assert len(inst) == 2
    assert all(i.class_name == "颗粒" for i in inst)
    assert all(i.score is None for i in inst)
    areas = sorted(int(i.mask.sum()) for i in inst)
    assert areas[0] == 40 * 40
    assert areas[1] == 40 * 50


def test_near_black_background_ignored():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:] = (8, 8, 8)  # below default luma thresh
    img[20:40, 20:40] = (0, 255, 0)
    inst = colored_mask_to_instances(img, min_area=16)
    assert len(inst) == 1
    assert int(inst[0].mask.sum()) == 20 * 20


def test_min_area_filters_noise():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[10:50, 10:50] = (0, 0, 255)
    img[0:2, 0:2] = (0, 255, 255)  # tiny noise
    inst = colored_mask_to_instances(img, min_area=64, quantize=16)
    assert len(inst) == 1


def test_rejects_non_bgr():
    with pytest.raises(ValueError):
        colored_mask_to_instances(np.zeros((10, 10), dtype=np.uint8))


def test_overlapping_color_bands_merge_into_one():
    """Same object painted with two overlapping colors should become 1 instance."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:70] = (0, 0, 255)  # red core
    img[25:75, 40:85] = (0, 255, 0)  # green overlapping band
    # Without merge → 2; with merge → 1
    raw = colored_mask_to_instances(img, min_area=16, quantize=16, merge_fragments=False)
    assert len(raw) == 2
    merged = colored_mask_to_instances(img, min_area=16, quantize=16, merge_fragments=True)
    assert len(merged) == 1
    assert int(merged[0].mask.sum()) >= max(int(i.mask.sum()) for i in raw)


def test_adjacent_color_halves_merge_into_one():
    """Same object painted as two abutting solid colors (no pixel overlap)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:50] = (0, 0, 255)  # left half
    img[20:80, 50:80] = (0, 255, 0)  # right half, shares edge col 50? actually 50 starts green
    # abutting at x=50: red ends at 49, green starts at 50 → gap 0 after dilation
    raw = colored_mask_to_instances(img, min_area=16, quantize=16, merge_fragments=False)
    assert len(raw) == 2
    merged = colored_mask_to_instances(
        img, min_area=16, quantize=16, merge_fragments=True, merge_bbox_iou=0.20, merge_gap_px=5
    )
    assert len(merged) == 1
    assert int(merged[0].mask.sum()) == 60 * 60


def test_separate_non_overlapping_colors_stay_separate():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    img[10:50, 10:50] = (0, 0, 255)
    img[55:95, 60:110] = (255, 0, 0)
    inst = colored_mask_to_instances(img, min_area=16, quantize=16, merge_fragments=True)
    assert len(inst) == 2


def test_tiny_relative_fragment_dropped():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:90, 10:90] = (0, 0, 255)  # large
    img[0:5, 0:5] = (0, 255, 0)  # tiny, no overlap
    # Pass absolute min_area so tiny color survives color split, then relative filter drops it.
    inst = colored_mask_to_instances(
        img,
        min_area=10,
        quantize=16,
        merge_fragments=True,
        min_relative_area=0.12,
    )
    assert len(inst) == 1
    assert int(inst[0].mask.sum()) == 80 * 80
