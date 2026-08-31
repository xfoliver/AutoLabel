"""Overlay rendering: Chinese labels must not become '?'."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.models import ShapeRecord  # noqa: E402
from autolabel.viz import _resolve_font_path, draw_shapes_overlay  # noqa: E402


def test_chinese_label_overlay_not_question_marks(tmp_path: Path):
    if not _resolve_font_path():
        pytest.skip("no CJK font available on this machine")

    src = tmp_path / "src.png"
    img = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.imwrite(str(src), img)

    shapes = [
        ShapeRecord(
            label="手机",
            points=[[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]],
            shape_type="polygon",
        )
    ]
    out = draw_shapes_overlay(src, shapes, tmp_path / "overlay.jpg")
    assert out.exists()

    rendered = cv2.imread(str(out))
    assert rendered is not None
    # Crop near expected label origin; should contain non-background colored pixels
    # from the Chinese glyphs (not just the polygon fill region).
    crop = rendered[0:25, 15:90]
    # Question-mark glyphs from Hershey are sparse thin strokes; Chinese with outline
    # paints a denser blob. Require some bright-ish pixels from the label color/outline.
    assert int(np.count_nonzero(crop.max(axis=2) > 80)) > 30
