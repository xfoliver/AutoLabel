"""LabelMeAI JSON exporter (Schema v1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from autolabel.exporters.schema_v1 import (
    DEFAULT_VERSION,
    FLAG_AUTOLABEL,
    FLAG_INSTANCE_SOURCE,
    FLAG_MODEL_CONFIDENCE,
    FLAG_QUALITY_RISK,
    FLAG_QUALITY_STATUS,
    FLAG_SCORE_SOURCE,
    ISEG_SHAPE_TYPE,
)
from autolabel.models import ShapeRecord


def shape_to_dict(shape: ShapeRecord) -> Dict[str, Any]:
    flags: Dict[str, Any] = {
        FLAG_AUTOLABEL: True,
        FLAG_QUALITY_RISK: float(shape.quality_risk),
        FLAG_SCORE_SOURCE: shape.score_source,
        FLAG_INSTANCE_SOURCE: shape.instance_source,
    }
    if shape.model_confidence is not None:
        flags[FLAG_MODEL_CONFIDENCE] = float(shape.model_confidence)
    if shape.quality_status:
        flags[FLAG_QUALITY_STATUS] = str(shape.quality_status)

    return {
        "label": shape.label,
        "points": [[float(p[0]), float(p[1])] for p in shape.points],
        "group_id": shape.group_id,
        "description": shape.description,
        "shape_type": shape.shape_type or ISEG_SHAPE_TYPE,
        "flags": flags,
        "mask": None,
    }


def build_labelme_dict(
    *,
    image_path: str,
    image_width: int,
    image_height: int,
    shapes: Sequence[ShapeRecord],
    version: str = DEFAULT_VERSION,
    top_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a LabelMeAI-compatible annotation dict (ISEG polygon)."""
    # imagePath: relative filename only
    name = Path(image_path).name
    return {
        "version": version,
        "flags": dict(top_flags or {}),
        "shapes": [shape_to_dict(s) for s in shapes],
        "imagePath": name,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }


def write_labelme_json(
    path: Union[str, Path],
    data: Dict[str, Any],
    *,
    indent: int = 2,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return out


def export_annotation(
    *,
    image_path: Union[str, Path],
    image_width: int,
    image_height: int,
    shapes: Sequence[ShapeRecord],
    output_json: Union[str, Path],
    version: str = DEFAULT_VERSION,
) -> Path:
    data = build_labelme_dict(
        image_path=str(image_path),
        image_width=image_width,
        image_height=image_height,
        shapes=shapes,
        version=version,
    )
    return write_labelme_json(output_json, data)
