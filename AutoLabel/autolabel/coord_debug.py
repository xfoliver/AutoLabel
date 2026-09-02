"""Coordinate-space debug dumps (EXIF / mask / polygon)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from autolabel.cv_io import imread, imwrite
from autolabel.image_orient import exif_orientation, open_normalized
from autolabel.models import ShapeRecord


def write_coord_debug(
    image_path: Path,
    *,
    out_dir: Path,
    shapes: Sequence[ShapeRecord],
    transform: Any = None,
    sam3_raw: Optional[Dict[str, Any]] = None,
    mask_path: Optional[Path] = None,
    overlay_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write 01–06 debug images + image_debug.json. Never overwrites the source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(image_path)
    raw = Image.open(p)
    raw_wh = [int(raw.size[0]), int(raw.size[1])]
    ori = exif_orientation(p, raw)
    raw_rgb = raw.convert("RGB")
    Image.fromarray(np.asarray(raw_rgb)).save(out_dir / "01_original_raw.jpg", quality=90)

    norm, meta = open_normalized(p)
    np.asarray(norm)
    Image.fromarray(np.asarray(norm)).save(out_dir / "02_exif_normalized.jpg", quality=90)
    nw, nh = int(norm.size[0]), int(norm.size[1])

    bgr = imread(p, cv2.IMREAD_COLOR)
    opencv_wh = [int(bgr.shape[1]), int(bgr.shape[0])] if bgr is not None else None

    mask_shape = None
    if mask_path and Path(mask_path).is_file():
        mask = imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask_shape = [int(mask.shape[0]), int(mask.shape[1])]
            imwrite(out_dir / "03_sam3_raw_mask.png", mask)
            if bgr is not None and mask.shape[:2] == bgr.shape[:2]:
                vis = bgr.copy()
                color = np.zeros_like(vis)
                color[:, :] = (0, 0, 255)
                m = mask > 0
                vis[m] = (vis[m] * 0.45 + color[m] * 0.55).astype(np.uint8)
                imwrite(out_dir / "04_sam3_mask_overlay.jpg", vis)

    if overlay_path and Path(overlay_path).is_file():
        ov = imread(overlay_path, cv2.IMREAD_COLOR)
        if ov is not None:
            imwrite(out_dir / "05_polygon_preview.jpg", ov)

    xs: List[float] = []
    ys: List[float] = []
    for sh in shapes:
        for pt in sh.points or []:
            if len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    xy_range = None
    if xs:
        xy_range = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
        }

    if bgr is not None:
        dbg = bgr.copy()
        h, w = dbg.shape[:2]
        cv2.putText(dbg, "(0,0)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dbg, f"(w={w},0)", (max(8, w - 140), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dbg, f"(0,h={h})", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            dbg,
            f"(w,h)=({w},{h})",
            (max(8, w - 180), h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if xs:
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
        imwrite(out_dir / "06_coordinate_debug.jpg", dbg)

    payload: Dict[str, Any] = {
        "path": str(p),
        "exif_orientation": ori,
        "pil_raw_size_wh": raw_wh,
        "after_exif_transpose_wh": [nw, nh],
        "opencv_normalized_wh": opencv_wh,
        "sam3_image_size_wh": (sam3_raw or {}).get("image_size"),
        "sam3_exif_orientation": (sam3_raw or {}).get("exif_orientation"),
        "sam3_pil_raw_size_wh": (sam3_raw or {}).get("pil_raw_size_wh"),
        "sam3_normalized_size_wh": (sam3_raw or {}).get("normalized_size_wh"),
        "mask_shape_hw": mask_shape,
        "transform": transform.to_dict() if transform is not None and hasattr(transform, "to_dict") else None,
        "polygon_xy_range": xy_range,
        "n_shapes": len(list(shapes)),
        "labelme_image_wh": [nw, nh],
    }
    (out_dir / "image_debug.json").write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
