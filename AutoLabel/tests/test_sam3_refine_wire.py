"""SAM3 worker interactive-backbone wiring (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from sam3_worker.inference import _scores_list, _wire_interactive_backbone  # noqa: E402


def test_wire_shares_detector_backbone():
    backbone = object()
    tracker = SimpleNamespace(backbone=None)
    predictor = SimpleNamespace(model=tracker)
    model = SimpleNamespace(backbone=backbone, inst_interactive_predictor=predictor)
    assert _wire_interactive_backbone(model) is True
    assert tracker.backbone is backbone


def test_wire_noop_when_already_set():
    detector_bb = object()
    tracker_bb = object()
    tracker = SimpleNamespace(backbone=tracker_bb)
    predictor = SimpleNamespace(model=tracker)
    model = SimpleNamespace(backbone=detector_bb, inst_interactive_predictor=predictor)
    assert _wire_interactive_backbone(model) is True
    assert tracker.backbone is tracker_bb


def test_wire_fails_without_predictor():
    model = SimpleNamespace(backbone=object(), inst_interactive_predictor=None)
    assert _wire_interactive_backbone(model) is False


def test_scores_list_flattens_nested_tensor_like():
    import numpy as np

    nested = np.array([[[0.91], [0.4], [0.12]]])
    out = _scores_list(nested, 3)
    assert out[0] == pytest.approx(0.91)
    assert out[1] == pytest.approx(0.4)
    assert out[2] == pytest.approx(0.12)
    assert _scores_list(None, 2) == [None, None]
