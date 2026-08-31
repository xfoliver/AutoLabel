"""Locked AutoLabelExporterSchema v1 constants (ISEG MVP)."""

from __future__ import annotations

DEFAULT_VERSION = "2025.9.1.0"

TOP_LEVEL_KEYS = (
    "version",
    "flags",
    "shapes",
    "imagePath",
    "imageData",
    "imageHeight",
    "imageWidth",
)

SHAPE_KEYS = (
    "label",
    "points",
    "group_id",
    "description",
    "shape_type",
    "flags",
    "mask",
)

ISEG_SHAPE_TYPE = "polygon"

FLAG_AUTOLABEL = "autolabel"
FLAG_MODEL_CONFIDENCE = "model_confidence"
FLAG_QUALITY_RISK = "quality_risk"
FLAG_SCORE_SOURCE = "score_source"
FLAG_INSTANCE_SOURCE = "instance_source"
