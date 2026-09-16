"""SAM3 local backend helpers — no real SAM3 / GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.inference.sam3_local import (  # noqa: E402
    Sam3LocalBackend,
    discovery_threshold,
    filter_by_score,
    nms_by_bbox,
    relative_min_area,
    resolve_model_prompt,
)
from autolabel.label_spec import build_label_spec  # noqa: E402
from autolabel.models import LabelSpec, TargetSpec  # noqa: E402


def test_nms_drops_duplicate_boxes():
    items = [
        {"score": 0.9, "bbox": [0, 0, 10, 10], "mask_path": "a"},
        {"score": 0.8, "bbox": [1, 1, 11, 11], "mask_path": "b"},
        {"score": 0.7, "bbox": [50, 50, 80, 80], "mask_path": "c"},
    ]
    kept = nms_by_bbox(items, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0]["mask_path"] == "a"
    assert kept[1]["mask_path"] == "c"


def test_resolve_model_prompt_prefers_field():
    spec = LabelSpec(
        class_names=["手机"],
        targets=[TargetSpec(label="手机", model_prompt="phone")],
    )
    label, prompt = resolve_model_prompt(spec)
    assert label == "手机"
    assert prompt == "phone"


def test_resolve_model_prompt_falls_back_to_label():
    spec = build_label_spec(label="car")
    label, prompt = resolve_model_prompt(spec)
    assert label == "car"
    assert prompt == "car"


def test_resolve_model_prompt_keeps_user_wording():
    spec = build_label_spec(label="alpha", model_prompt="alpha bead")
    label, prompt = resolve_model_prompt(spec)
    assert label == "alpha"
    assert prompt == "alpha bead"
    spec_plain = build_label_spec(label="圆形垫片")
    _, prompt_plain = resolve_model_prompt(spec_plain)
    assert prompt_plain == "圆形垫片"


def test_discovery_threshold_caps_user_slider():
    assert discovery_threshold(0.55) == 0.05
    assert discovery_threshold(0.03) == 0.03
    assert discovery_threshold(0) == 0.05


def test_filter_by_score_does_not_empty_low_class():
    items = [
        {"score": 0.12, "mask_path": "a"},
        {"score": 0.09, "mask_path": "b"},
    ]
    assert [x["mask_path"] for x in filter_by_score(items, 0.43)] == ["a", "b"]
    mixed = [
        {"score": 0.80, "mask_path": "hi"},
        {"score": 0.10, "mask_path": "lo"},
        {"score": None, "mask_path": "none"},
    ]
    kept = filter_by_score(mixed, 0.5)
    assert [x["mask_path"] for x in kept] == ["hi", "none"]


def test_relative_min_area_drops_fragments_when_clustered():
    beads = [40000, 45000, 48000, 50000, 52000, 55000]
    assert relative_min_area(beads + [800, 2000], floor=64) >= 8000
    assert relative_min_area([300], floor=64) == 64


def test_sam3_backend_identity_and_no_grabcut(tmp_path: Path, monkeypatch):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    image_path = tmp_path / "a.jpg"
    cv2.imwrite(str(image_path), img)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[5:20, 5:25] = 255
    mask_path = tmp_path / "a_sam3_mask_0.png"
    cv2.imwrite(str(mask_path), mask)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "instances": [
                    {
                        "score": 0.91,
                        "bbox": [5, 5, 25, 20],
                        "area": int(np.count_nonzero(mask)),
                        "mask_path": str(mask_path),
                    }
                ]
            }

    monkeypatch.setattr(
        "autolabel.inference.sam3_local.requests.post",
        lambda *a, **k: _Resp(),
    )
    backend = Sam3LocalBackend(
        "sam3_local",
        {"min_area": 16, "score_threshold": 0.55, "nms_iou_threshold": 0.7},
    )
    spec = build_label_spec(label="手机", model_prompt="phone")
    result = backend.predict(image_path, spec)
    assert result.transform.mask_to_input.mode == "identity"
    assert result.transform.input_to_original.mode == "identity"
    assert len(result.instances) == 1
    assert result.instances[0].class_name == "手机"
    assert result.instances[0].score == pytest.approx(0.91)
    opts = backend.stage2_options()
    assert opts["grabcut"] is False
    assert opts["close_ksize"] == 0


def test_sam3_backend_posts_discovery_threshold(tmp_path: Path, monkeypatch):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    image_path = tmp_path / "a.jpg"
    cv2.imwrite(str(image_path), img)
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"instances": []}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr("autolabel.inference.sam3_local.requests.post", fake_post)
    backend = Sam3LocalBackend(
        "sam3_local",
        {"score_threshold": 0.43, "min_area": 16, "base_url": "http://127.0.0.1:9892"},
    )
    backend.predict(image_path, build_label_spec(label="alpha", model_prompt="alpha bead"))
    assert captured["payload"]["prompt"] == "alpha bead"
    assert captured["payload"]["score_threshold"] == 0.05


def test_sam3_backend_keeps_any_shape_above_min_area(tmp_path: Path, monkeypatch):
    img = np.zeros((80, 200, 3), dtype=np.uint8)
    image_path = tmp_path / "a.jpg"
    cv2.imwrite(str(image_path), img)
    thin = np.zeros((80, 200), dtype=np.uint8)
    thin[40:43, 10:160] = 255
    thin_path = tmp_path / "thin.png"
    cv2.imwrite(str(thin_path), thin)
    blob = np.zeros((80, 200), dtype=np.uint8)
    blob[20:60, 40:160] = 255
    blob_path = tmp_path / "blob.png"
    cv2.imwrite(str(blob_path), blob)

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "instances": [
                    {"score": 0.12, "bbox": [10, 40, 160, 43], "area": 450, "mask_path": str(thin_path)},
                    {"score": 0.11, "bbox": [40, 20, 160, 60], "area": 4800, "mask_path": str(blob_path)},
                ]
            }

    monkeypatch.setattr("autolabel.inference.sam3_local.requests.post", lambda *a, **k: _Resp())
    backend = Sam3LocalBackend("sam3_local", {"min_area": 64, "score_threshold": 0.43})
    result = backend.predict(image_path, build_label_spec(label="target"))
    assert len(result.instances) == 2


def test_sam3_backend_filters_tiny_masks(tmp_path: Path, monkeypatch):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    image_path = tmp_path / "a.jpg"
    cv2.imwrite(str(image_path), img)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[0, 0] = 255
    mask_path = tmp_path / "tiny.png"
    cv2.imwrite(str(mask_path), mask)

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "instances": [
                    {"score": 0.99, "bbox": [0, 0, 1, 1], "area": 1, "mask_path": str(mask_path)}
                ]
            }

    monkeypatch.setattr("autolabel.inference.sam3_local.requests.post", lambda *a, **k: _Resp())
    backend = Sam3LocalBackend("sam3_local", {"min_area": 64})
    result = backend.predict(image_path, build_label_spec(label="obj"))
    assert result.instances == []


def test_create_backend_registers_sam3():
    from autolabel.inference.base import create_backend

    backend = create_backend("sam3_local", "sam3_local", {"host": "127.0.0.1", "port": 9892})
    assert backend.id == "sam3_local"
    assert backend.capabilities.instance_mask is True
    assert backend.capabilities.confidence is True


def test_sam3_refine_posts_to_refine(tmp_path: Path, monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"instances": []}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr("autolabel.inference.sam3_local.requests.post", fake_post)
    backend = Sam3LocalBackend("sam3_local", {"base_url": "http://127.0.0.1:9892"})
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), dtype=np.uint8))
    backend.refine(img, bbox=[1, 2, 8, 9], points=[[4, 5]])
    assert str(captured["url"]).endswith("/refine")
    assert captured["payload"]["bbox"] == [1.0, 2.0, 8.0, 9.0]
    assert captured["payload"]["points"] == [[4.0, 5.0]]
    assert captured["payload"]["save_raw_mask"] is True
