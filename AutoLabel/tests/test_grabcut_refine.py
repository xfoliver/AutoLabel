"""Unit tests for GrabCut original-image mask refine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.geometry.polygon import polygon_area  # noqa: E402
from autolabel.geometry.transform import build_stretch_meta  # noqa: E402
from autolabel.inference.openai_image_edits import OpenAIImageEditsBackend  # noqa: E402
from autolabel.mask.grabcut_refine import grabcut_refine_mask  # noqa: E402
from autolabel.mask.io import resize_mask_nn  # noqa: E402
from autolabel.models import InstancePred, Stage1Result  # noqa: E402
from autolabel.stage2 import stage1_to_shapes  # noqa: E402


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    return float(inter) / float(max(union, 1))


def _synthetic_rect_scene(h: int = 120, w: int = 160, margin: int = 8):
    """Bright rectangle on dark bg + intentionally fat seed mask."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    gt = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = 40, 30, 100, 90
    img[y0:y1, x0:x1] = (220, 40, 40)
    gt[y0:y1, x0:x1] = 1
    fat = np.zeros((h, w), dtype=np.uint8)
    fat[y0 - margin : y1 + margin, x0 - margin : x1 + margin] = 1
    return img, gt, fat


def test_grabcut_refine_improves_iou_on_fat_seed():
    img, gt, fat = _synthetic_rect_scene()
    refined = grabcut_refine_mask(
        img,
        fat,
        iters=3,
        pad_px=12,
        sure_fg_erode=7,
        sure_bg_dilate=11,
    )
    assert refined.shape == gt.shape
    assert int(refined.sum()) > 0
    assert _iou(refined, gt) > _iou(fat, gt)
    # Should shrink toward the true rect (less over-cover).
    assert int(refined.sum()) < int(fat.sum())


def test_grabcut_refine_soft_fail_empty_seed():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    empty = np.zeros((32, 32), dtype=np.uint8)
    out = grabcut_refine_mask(img, empty)
    assert out.shape == (32, 32)
    assert int(out.sum()) == 0


def test_grabcut_refine_resizes_mask_to_image():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[10:30, 10:40] = (200, 10, 10)
    small = np.zeros((20, 25), dtype=np.uint8)
    small[5:15, 5:20] = 1
    out = grabcut_refine_mask(img, small, iters=2, pad_px=4)
    assert out.shape == (40, 50)
    assert int(out.sum()) > 0


def test_stage1_to_shapes_grabcut_changes_polygon():
    img, gt, fat = _synthetic_rect_scene()
    h, w = img.shape[:2]
    # Mask space intentionally different size (stretch path).
    meta = build_stretch_meta((w, h), (w * 2, h * 2), (w * 2, h * 2))
    fat_mask = resize_mask_nn(fat, w * 2, h * 2)
    result = Stage1Result(
        instances=[
            InstancePred(
                instance_id=1,
                class_id=1,
                class_name="手机",
                mask=fat_mask,
                score=None,
            )
        ],
        transform=meta,
        raw={},
    )
    shapes_off = stage1_to_shapes(
        result,
        min_area=16.0,
        close_ksize=0,
        open_ksize=0,
        epsilon_ratio=0.001,
        grabcut=False,
        image_bgr=img,
    )
    shapes_on = stage1_to_shapes(
        result,
        min_area=16.0,
        close_ksize=0,
        open_ksize=0,
        epsilon_ratio=0.001,
        grabcut=True,
        image_bgr=img,
        grabcut_iters=3,
        grabcut_pad_px=12,
        grabcut_sure_fg_erode=7,
        grabcut_sure_bg_dilate=11,
    )
    assert len(shapes_off) == 1 and len(shapes_on) == 1
    area_off = polygon_area(shapes_off[0].points)
    area_on = polygon_area(shapes_on[0].points)
    # GrabCut should shrink the fat mask toward the true object.
    assert area_on < area_off


def test_openai_backend_stage2_options_includes_grabcut():
    be = OpenAIImageEditsBackend(
        "gpt_image_mask",
        {
            "base_url": "http://example",
            "api_key": "x",
            "grabcut": True,
            "grabcut_iters": 4,
            "mask_close_ksize": 0,
            "epsilon_ratio": 0.0005,
        },
    )
    opts = be.stage2_options()
    assert opts["grabcut"] is True
    assert opts["grabcut_iters"] == 4
    assert opts["close_ksize"] == 0
    assert opts["epsilon_ratio"] == 0.0005
    assert "image_bgr" not in opts  # injected by orchestrator, not backend
