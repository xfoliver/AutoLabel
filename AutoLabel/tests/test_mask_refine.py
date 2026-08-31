"""Unit tests for binary mask refine + epsilon / stage2 plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.geometry.polygon import instance_mask_to_polygons  # noqa: E402
from autolabel.geometry.transform import build_stretch_meta  # noqa: E402
from autolabel.inference.openai_image_edits import (  # noqa: E402
    OpenAIImageEditsBackend,
    build_mask_prompt,
)
from autolabel.label_spec import build_label_spec  # noqa: E402
from autolabel.mask.refine import refine_binary_mask  # noqa: E402
from autolabel.models import InstancePred, ShapeRecord, Stage1Result  # noqa: E402
from autolabel.stage2 import annotate_image_to_dict, stage1_to_shapes  # noqa: E402


def test_refine_binary_mask_close_fills_1px_gap():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:9] = 1
    mask[5:15, 10:15] = 1  # 1px gap at col 9
    assert mask[10, 9] == 0

    closed = refine_binary_mask(mask, close_ksize=3, open_ksize=0)
    assert closed[10, 9] == 1

    noop = refine_binary_mask(mask, close_ksize=0, open_ksize=0)
    assert noop[10, 9] == 0
    assert int(noop.sum()) == int(mask.sum())


def test_refine_binary_mask_open_removes_speck():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[8:22, 8:22] = 1
    mask[2, 2] = 1  # speck
    opened = refine_binary_mask(mask, close_ksize=0, open_ksize=3)
    assert opened[2, 2] == 0
    assert opened[15, 15] == 1


def test_epsilon_ratio_affects_point_count():
    meta = build_stretch_meta((64, 64), (64, 64), (64, 64))
    yy, xx = np.ogrid[:64, :64]
    # soft circle → many contour points before simplify
    mask = (((xx - 32) ** 2 + (yy - 32) ** 2) <= 20**2).astype(np.uint8)

    fine = instance_mask_to_polygons(
        mask, meta, min_area=16.0, epsilon_ratio=0.001, close_ksize=0, open_ksize=0
    )
    coarse = instance_mask_to_polygons(
        mask, meta, min_area=16.0, epsilon_ratio=0.02, close_ksize=0, open_ksize=0
    )
    assert len(fine) == 1 and len(coarse) == 1
    assert len(fine[0]) >= len(coarse[0])


def test_stage1_to_shapes_passes_morph_kwargs():
    meta = build_stretch_meta((40, 40), (40, 40), (40, 40))
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:18] = 1
    mask[10:30, 19:28] = 1  # 1px gap
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
    # Without close, findContours may yield 2 external blobs; with close → 1
    shapes_raw = stage1_to_shapes(result, min_area=16.0, close_ksize=0, open_ksize=0, epsilon_ratio=0.001)
    shapes_closed = stage1_to_shapes(result, min_area=16.0, close_ksize=3, open_ksize=0, epsilon_ratio=0.001)
    assert len(shapes_closed) == 1
    assert len(shapes_raw) >= len(shapes_closed)


def test_annotate_image_to_dict_reuses_shapes(tmp_path: Path):
    meta = build_stretch_meta((32, 32), (32, 32), (32, 32))
    img_path = tmp_path / "x.png"
    import cv2

    cv2.imwrite(str(img_path), np.zeros((32, 32, 3), dtype=np.uint8))
    result = Stage1Result(instances=[], transform=meta, raw={})
    fake = [
        ShapeRecord(
            label="手机",
            points=[[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]],
            shape_type="polygon",
        )
    ]
    data = annotate_image_to_dict(img_path, result, shapes=fake)
    assert len(data["shapes"]) == 1
    assert data["shapes"][0]["label"] == "手机"
    assert len(data["shapes"][0]["points"]) == 4


def test_openai_backend_stage2_options():
    be = OpenAIImageEditsBackend(
        "gpt_image_mask",
        {
            "base_url": "http://example",
            "api_key": "x",
            "epsilon_ratio": 0.0008,
            "mask_close_ksize": 5,
            "mask_open_ksize": 0,
            "min_area": 128,
            "quantize": 8,
            "grabcut": False,
        },
    )
    opts = be.stage2_options()
    assert opts["epsilon_ratio"] == 0.0008
    assert opts["close_ksize"] == 5
    assert opts["open_ksize"] == 0
    assert opts["min_area"] == 128.0
    assert opts["grabcut"] is False
    assert be.quantize == 8


def test_openai_backend_default_close_ksize_is_zero():
    be = OpenAIImageEditsBackend(
        "gpt_image_mask",
        {"base_url": "http://example", "api_key": "x"},
    )
    assert be.mask_close_ksize == 0
    assert be.grabcut is True
    assert be.stage2_options()["close_ksize"] == 0
    assert be.stage2_options()["grabcut"] is True


def test_build_mask_prompt_includes_silhouette_rules():
    spec = build_label_spec(label="手机", description="机身", exclude=["壳装饰"])
    p = build_mask_prompt(spec)
    assert "pixel-accurate" in p.lower() or "silhouette" in p.lower()
    assert "halo" in p.lower()
    assert "壳装饰" in p
