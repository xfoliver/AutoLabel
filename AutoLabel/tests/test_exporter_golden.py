"""Exporter schema vs golden ISEG sample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.exporters.labelmeai import build_labelme_dict  # noqa: E402
from autolabel.exporters.schema_v1 import DEFAULT_VERSION  # noqa: E402
from autolabel.models import ShapeRecord  # noqa: E402
from autolabel.validators.annotation_validator import validate_labelme_dict  # noqa: E402

GOLDEN = ROOT / "fixtures" / "golden" / "iseg_polygon.json"


def test_golden_file_exists_and_keys():
    assert GOLDEN.exists()
    with GOLDEN.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("imageData") is None
    assert "shapes" in data
    assert data["shapes"][0]["shape_type"] == "polygon"
    assert "imagePath" in data
    assert "imageWidth" in data
    assert "imageHeight" in data


def test_exporter_matches_golden_schema_shape():
    shapes = [
        ShapeRecord(
            label="凹坑",
            points=[[332.0, 433.0], [327.0, 439.0], [327.0, 458.0], [334.0, 467.0]],
            model_confidence=0.86,
            quality_risk=0.1,
            score_source="model",
            instance_source="model",
        )
    ]
    data = build_labelme_dict(
        image_path="Image_20250325144902743_604_22.bmp",
        image_width=894,
        image_height=1058,
        shapes=shapes,
        version=DEFAULT_VERSION,
    )
    assert data["version"] == "2025.9.1.0"
    assert data["imageData"] is None
    assert data["imagePath"] == "Image_20250325144902743_604_22.bmp"
    sh = data["shapes"][0]
    assert sh["group_id"] is None
    assert sh["description"] is None
    assert sh["mask"] is None
    assert sh["shape_type"] == "polygon"
    assert sh["flags"]["autolabel"] is True
    assert sh["flags"]["model_confidence"] == 0.86
    problems = validate_labelme_dict(data, image_wh=(894, 1058))
    assert problems == []
