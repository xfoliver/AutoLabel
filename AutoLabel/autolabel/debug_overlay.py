"""SAM3 debug overlays: raw mask / raw bbox / final polygon, kept separate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autolabel.cv_io import imread, imwrite
from autolabel.models import ShapeRecord, Stage1Result
from autolabel.viz import draw_shapes_overlay

_BBOX_COLOR = (0, 255, 255)
# Single cyan (BGR) so the pixel mask does not copy the class-colored polygon overlay.
_RAW_MASK_COLOR = (255, 220, 0)


def looks_like_axis_aligned_rectangle(points: Sequence[Sequence[float]], *, atol: float = 1.5) -> bool:
    """True when ``points`` are four corners of an axis-aligned rectangle."""
    if points is None or len(points) != 4:
        return False
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    uniq_x = _cluster_values(xs, atol)
    uniq_y = _cluster_values(ys, atol)
    if len(uniq_x) != 2 or len(uniq_y) != 2:
        return False
    corners = {(x, y) for x in uniq_x for y in uniq_y}
    got = {(_nearest(x, uniq_x), _nearest(y, uniq_y)) for x, y in zip(xs, ys)}
    return got == corners


def _cluster_values(values: Sequence[float], atol: float) -> List[float]:
    ordered = sorted(float(v) for v in values)
    groups: List[List[float]] = []
    for v in ordered:
        if not groups or abs(v - groups[-1][-1]) > atol:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [sum(g) / len(g) for g in groups]


def _nearest(value: float, options: Sequence[float]) -> float:
    return min(options, key=lambda o: abs(o - value))


def summarize_shapes(shapes: Sequence[ShapeRecord]) -> Dict[str, Any]:
    point_counts = [len(s.points or []) for s in shapes]
    rectangles = [looks_like_axis_aligned_rectangle(s.points or []) for s in shapes]
    return {
        "final_shapes": len(list(shapes)),
        "point_counts": point_counts,
        "rectangle_n": int(sum(1 for x in rectangles if x)),
        "four_point_n": int(sum(1 for n in point_counts if n == 4)),
        "shapes": [
            {
                "label": sh.label,
                "n_points": len(sh.points or []),
                "looks_rectangle": bool(rect),
                "instance_source": sh.instance_source,
                "quality_status": sh.quality_status,
                "model_confidence": sh.model_confidence,
            }
            for sh, rect in zip(shapes, rectangles)
        ],
    }


def resolve_preview_mask_path(
    *,
    raw: Optional[Dict[str, Any]] = None,
    debug_dir: Optional[Path] = None,
    colored_mask_path: Optional[Path] = None,
) -> Optional[Path]:
    """Pick the gallery's right-hand image: filtered union overlay, not one binary PNG.

    SAM3 writes one PNG per instance plus an unfiltered rainbow overlay of every
    keep-gate hit (scratches included). Preview should show the *exported*
    instances, class-colored, so it matches the left polygon overlay.
    """
    candidates: List[Path] = []
    if colored_mask_path:
        named = Path(colored_mask_path)
        if "rawmask" in named.name.lower() or "overlay" in named.name.lower():
            candidates.append(named)
    if debug_dir is not None:
        d = Path(debug_dir)
        candidates.extend(
            [
                d / "sam_raw_mask_overlay.jpg",
                d / "04_sam3_mask_overlay.jpg",
                d / "03_sam3_raw_mask.png",
            ]
        )
    if isinstance(raw, dict):
        overlay = raw.get("overlay_path")
        if overlay:
            candidates.append(Path(str(overlay)))
    if colored_mask_path:
        candidates.append(Path(colored_mask_path))
    if isinstance(raw, dict):
        for item in raw.get("instances") or []:
            if not isinstance(item, dict) or not item.get("mask_path"):
                continue
            p = Path(str(item["mask_path"]))
            if p.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}:
                candidates.append(p)
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def count_sam3_raw(result: Stage1Result) -> Dict[str, int]:
    """Count raw SAM3 boxes/masks vs exported instances. Does not count vision bboxes."""
    raw = result.raw if isinstance(result.raw, dict) else {}
    items = [x for x in (raw.get("instances") or []) if isinstance(x, dict)]
    raw_boxes = 0
    raw_masks = 0
    for item in items:
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            raw_boxes += 1
        if item.get("mask_path") or item.get("mask") is not None:
            raw_masks += 1
    inst_masks = 0
    for inst in result.instances or []:
        mask = getattr(inst, "mask", None)
        if mask is None:
            continue
        if int(np.count_nonzero(mask)) > 0:
            inst_masks += 1
    # Text-prompt SAM3 stores masks on instances even when raw.instances was stripped.
    if raw_masks == 0:
        raw_masks = inst_masks
    if raw_boxes == 0:
        raw_boxes = inst_masks
    return {
        "raw_boxes": int(raw_boxes),
        "raw_masks": int(raw_masks),
        "instance_masks": int(inst_masks),
    }


def _canvas(image_path: Path) -> Optional[np.ndarray]:
    img = imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img.copy()


def _put(canvas: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_xyxy(
    canvas: np.ndarray,
    bbox: Sequence[float],
    color: Tuple[int, int, int],
    *,
    label: str = "",
) -> None:
    if len(bbox) < 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        _put(canvas, label, (x1, max(16, y1 - 6)), color)


def _collect_masks(result: Stage1Result) -> List[np.ndarray]:
    """Filtered instance masks only (what AutoLabel actually exports)."""
    masks: List[np.ndarray] = []
    for inst in result.instances or []:
        mask = getattr(inst, "mask", None)
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim > 2:
            arr = arr[:, :, 0]
        if int(np.count_nonzero(arr)) > 0:
            masks.append((arr > 0).astype(np.uint8))
    return masks


def _collect_sam_bboxes(result: Stage1Result) -> List[Tuple[List[float], Optional[float]]]:
    out: List[Tuple[List[float], Optional[float]]] = []
    raw = result.raw if isinstance(result.raw, dict) else {}
    for item in raw.get("instances") or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        score = item.get("score")
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        out.append(([float(x) for x in bbox[:4]], score_f))
    if out:
        return out
    for inst in result.instances or []:
        mask = getattr(inst, "mask", None)
        if mask is None:
            continue
        ys, xs = np.where(np.asarray(mask) > 0)
        if xs.size == 0:
            continue
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        out.append((bbox, inst.score))
    return out


def write_sam_debug_overlays(
    image_path: Path,
    *,
    out_dir: Path,
    result: Stage1Result,
    shapes: Sequence[ShapeRecord],
) -> Dict[str, Any]:
    """Write the three debug images required to separate mask vs bbox vs export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    counts = count_sam3_raw(result)
    shape_info = summarize_shapes(shapes)
    bbox_fallback = bool(counts["raw_masks"] == 0 and shape_info["final_shapes"] > 0)

    mask_path = out_dir / "sam_raw_mask_overlay.jpg"
    bbox_path = out_dir / "sam_raw_bbox_overlay.jpg"
    poly_path = out_dir / "final_polygon_overlay.jpg"

    canvas = _canvas(image_path)
    if canvas is not None:
        # Dim the photo, then paint a single cyan on kept pixels. This must not
        # look like the class-colored polygon overlay on the left of the gallery.
        mask_vis = (canvas.astype(np.float32) * 0.42).astype(np.uint8)
        union = None
        for mask in _collect_masks(result):
            if mask.shape[:2] != mask_vis.shape[:2]:
                continue
            union = mask if union is None else np.maximum(union, mask)
        if union is not None:
            color = np.array(_RAW_MASK_COLOR, dtype=np.float32)
            m = union > 0
            mask_vis[m] = (mask_vis[m].astype(np.float32) * 0.22 + color * 0.78).astype(np.uint8)
        imwrite(mask_path, mask_vis)

        bbox_vis = canvas.copy()
        for i, (bbox, score) in enumerate(_collect_sam_bboxes(result), start=1):
            score_txt = f"{score:.2f}" if score is not None else "none"
            _draw_xyxy(bbox_vis, bbox, _BBOX_COLOR, label=f"SAM #{i} score={score_txt}")
        imwrite(bbox_path, bbox_vis)

    if canvas is not None:
        draw_shapes_overlay(image_path, shapes, poly_path)
    elif not poly_path.exists():
        # Keep a placeholder so the debug folder always has the three names.
        pass

    payload = {
        **counts,
        **shape_info,
        "bbox_fallback_used": bbox_fallback,
        "transform": result.transform.to_dict() if getattr(result, "transform", None) is not None else None,
        "files": {
            "sam_raw_mask_overlay": str(mask_path),
            "sam_raw_bbox_overlay": str(bbox_path),
            "final_polygon_overlay": str(poly_path),
        },
    }
    (out_dir / "debug_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
