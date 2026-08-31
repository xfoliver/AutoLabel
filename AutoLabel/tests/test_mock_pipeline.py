"""Mock pipeline: adjacent instances stay separate; end-to-end stage1→json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.inference.mock import MockBackend  # noqa: E402
from autolabel.label_spec import build_label_spec  # noqa: E402
from autolabel.orchestrator import AutoLabelOrchestrator  # noqa: E402
from autolabel.stage2 import stage1_to_shapes  # noqa: E402


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dataset"
    d.mkdir()
    # create a few blank images
    for i in range(6):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cv2.circle(img, (80 + i * 10, 120), 30, (200, 200, 200), -1)
        cv2.imwrite(str(d / f"img_{i:02d}.jpg"), img)
    # one existing sidecar to test skip
    (d / "img_00.json").write_text('{"shapes": [], "imagePath": "img_00.jpg"}', encoding="utf-8")
    return d


def test_adjacent_instances_produce_two_shapes(sample_dir: Path):
    backend = MockBackend(
        "mock",
        {"adjacent": True, "n_instances": 2, "mode": "letterbox", "mask_scale": 0.25, "with_scores": True},
    )
    spec = build_label_spec(label="颗粒")
    img = next(p for p in sorted(sample_dir.glob("*.jpg")) if p.name != "img_00.jpg")
    s1 = backend.predict(img, spec)
    assert len(s1.instances) == 2
    shapes = stage1_to_shapes(s1)
    assert len(shapes) >= 2, f"expected >=2 polygons, got {len(shapes)}"


def test_orchestrator_preview_batch_commit_skip(sample_dir: Path):
    backend = MockBackend("mock", {"n_instances": 2, "adjacent": True, "mask_scale": 1.0, "mode": "identity"})
    orch = AutoLabelOrchestrator(backend=backend, log=lambda m: None)
    orch.make_manual_intent(label="颗粒", dataset_path=str(sample_dir), preview_count=3)
    plan = orch.build_plan()
    assert plan.skipped_existing == 1
    assert plan.pending_count == 5
    preview = orch.run_preview()
    assert len(preview) == 3
    orch.confirm_preview()
    report = orch.run_batch()
    assert report["processed"] == 5
    assert report["skipped_existing"] == 1
    # work jsons exist
    assert orch.state.run_dir is not None
    work_jsons = list((orch.state.run_dir / "json").glob("*.json"))
    assert len(work_jsons) == 5
    # commit should not overwrite img_00.json content blindly — img_00 not in mapping
    summary = orch.commit()
    assert summary["committed_count"] == 5
    # img_00 still the old stub
    data = json.loads((sample_dir / "img_00.json").read_text(encoding="utf-8"))
    assert data.get("shapes") == []
    # a committed sidecar has autolabel flag
    committed = sample_dir / "img_01.json"
    assert committed.exists()
    cdata = json.loads(committed.read_text(encoding="utf-8"))
    assert cdata["imageData"] is None
    assert cdata["shapes"][0]["flags"]["autolabel"] is True


def test_no_confirm_blocks_batch(sample_dir: Path):
    backend = MockBackend()
    orch = AutoLabelOrchestrator(backend=backend, log=lambda m: None)
    orch.make_manual_intent(label="x", dataset_path=str(sample_dir), preview_count=2)
    orch.build_plan()
    orch.run_preview()
    with pytest.raises(RuntimeError, match="试标未确认"):
        orch.run_batch()
