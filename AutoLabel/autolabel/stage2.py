"""Stage-2: Stage1Result → ShapeRecords / LabelMe JSON dict."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from autolabel.cv_io import imread
from autolabel.exporters.labelmeai import build_labelme_dict, write_labelme_json
from autolabel.geometry.polygon import instance_mask_to_polygons
from autolabel.geometry.transform import build_stretch_meta
from autolabel.mask.grabcut_refine import grabcut_refine_mask
from autolabel.models import ShapeRecord, Stage1Result
from autolabel.scoring.risk import compute_quality_risk


def stage1_to_shapes(
    result: Stage1Result,
    *,
    min_area: float = 64.0,
    min_area_space: str = "original",
    epsilon_ratio: float = 0.001,
) -> List[ShapeRecord]:
    ow, oh = result.transform.original_size
    use_grabcut = bool(grabcut) and image_bgr is not None
    if use_grabcut:
        # After GrabCut the mask is in original image space — avoid double stretch.
        poly_transform = build_stretch_meta((ow, oh), (ow, oh), (ow, oh))
    else:
        poly_transform = result.transform

    shapes: List[ShapeRecord] = []
    for inst in result.instances:
        mask = inst.mask
        if use_grabcut:
            mask = grabcut_refine_mask(
                image_bgr,
                mask,
                iters=grabcut_iters,
                pad_px=grabcut_pad_px,
                sure_fg_erode=grabcut_sure_fg_erode,
                sure_bg_dilate=grabcut_sure_bg_dilate,
            )
        polys = instance_mask_to_polygons(
            mask,
            poly_transform,
            min_area=min_area,
            min_area_space=min_area_space,
            epsilon_ratio=epsilon_ratio,
            close_ksize=close_ksize,
            open_ksize=open_ksize,
            keep_largest=keep_largest,
        )
        score = inst.score
        score_source = "model" if score is not None else "none"
        for poly in polys:
            risk = compute_quality_risk(poly, image_width=ow, image_height=oh)
            shapes.append(
                ShapeRecord(
                    label=inst.class_name,
                    points=poly,
                    shape_type="polygon",
                    model_confidence=score,
                    quality_risk=risk,
                    score_source=score_source,
                    instance_source=instance_source,
                )
            )
    return shapes


def annotate_image_to_dict(
    image_path: Union[str, Path],
    result: Stage1Result,
    *,
    shapes: Optional[Sequence[ShapeRecord]] = None,
    **kwargs,
) -> dict:
    path = Path(image_path)
    ow, oh = result.transform.original_size
    # Prefer actual image size if readable
    img = imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        oh, ow = img.shape[:2]
    if shapes is None:
        if img is not None and "image_bgr" not in kwargs:
            kwargs = {**kwargs, "image_bgr": img}
        shapes = stage1_to_shapes(result, **kwargs)
    return build_labelme_dict(
        image_path=path.name,
        image_width=ow,
        image_height=oh,
        shapes=shapes,
    )


def annotate_image_to_json(
    image_path: Union[str, Path],
    result: Stage1Result,
    output_json: Union[str, Path],
    **kwargs,
) -> Path:
    data = annotate_image_to_dict(image_path, result, **kwargs)
    return write_labelme_json(output_json, data)
