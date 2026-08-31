"""Annotation validators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from autolabel.exporters.schema_v1 import ISEG_SHAPE_TYPE


def validate_labelme_dict(
    data: Dict[str, Any],
    *,
    require_polygon: bool = True,
    image_wh: Optional[Tuple[int, int]] = None,
) -> List[str]:
    problems: List[str] = []
    if not isinstance(data, dict):
        return ["NOT_OBJECT"]
    for key in ("shapes", "imagePath", "imageWidth", "imageHeight"):
        if key not in data:
            problems.append(f"MISSING_{key.upper()}")
    if "imageData" in data and data.get("imageData") is not None:
        problems.append("IMAGE_DATA_NOT_NULL")
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        problems.append("NO_SHAPES_FIELD")
        return problems
    for sh in shapes:
        if not isinstance(sh, dict):
            problems.append("BAD_SHAPE")
            continue
        st = sh.get("shape_type")
        if require_polygon and st != ISEG_SHAPE_TYPE:
            problems.append("BAD_SHAPE_TYPE")
        pts = sh.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            problems.append("BAD_POINTS")
        if not sh.get("label"):
            problems.append("MISSING_LABEL_NAME")
        if image_wh and isinstance(pts, list):
            w, h = image_wh
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    problems.append("BAD_POINT")
                    break
                x, y = float(p[0]), float(p[1])
                if x < -2 or y < -2 or x > w + 1 or y > h + 1:
                    problems.append("OOB_POINT")
                    break
    return problems
