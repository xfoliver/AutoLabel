"""Two-stage TransformMeta: mask → model input → original image."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from autolabel.models import AffineStep, TransformMeta

Point = Tuple[float, float]
Points = List[List[float]]


def _apply_forward_step(x: float, y: float, step: AffineStep) -> Point:
    mode = (step.mode or "identity").lower()
    if mode in ("identity", "none", ""):
        return x, y
    if mode == "stretch":
        return x * step.scale_x, y * step.scale_y
    if mode == "letterbox":
        scale = step.scale if step.scale is not None else step.scale_x
        return x * scale + step.pad_left, y * scale + step.pad_top
    if mode == "crop":
        crop = step.crop or {}
        return x + float(crop.get("x", 0.0)), y + float(crop.get("y", 0.0))
    return x * step.scale_x + step.pad_left, y * step.scale_y + step.pad_top


def _apply_inverse_step(x: float, y: float, step: AffineStep) -> Point:
    """Invert a FORWARD preprocess step (original→input or mask→input)."""
    mode = (step.mode or "identity").lower()
    if mode in ("identity", "none", ""):
        return x, y
    if mode == "stretch":
        sx = step.scale_x if step.scale_x else 1.0
        sy = step.scale_y if step.scale_y else 1.0
        return x / sx, y / sy
    if mode == "letterbox":
        # Forward: x_in = x_ori * scale + pad
        scale = step.scale if step.scale is not None else step.scale_x
        scale = scale if scale else 1.0
        return (x - step.pad_left) / scale, (y - step.pad_top) / scale
    if mode == "crop":
        crop = step.crop or {}
        return x - float(crop.get("x", 0.0)), y - float(crop.get("y", 0.0))
    sx = step.scale_x if step.scale_x else 1.0
    sy = step.scale_y if step.scale_y else 1.0
    return (x - step.pad_left) / sx, (y - step.pad_top) / sy


def map_mask_to_original(x: float, y: float, meta: TransformMeta) -> Point:
    """Map point from mask space to original image space.

    `mask_to_input` is applied forward (mask→input).
    `input_to_original` stores the FORWARD preprocess (original→input) and is inverted.
    """
    xi, yi = _apply_forward_step(x, y, meta.mask_to_input)
    return _apply_inverse_step(xi, yi, meta.input_to_original)


def map_points_mask_to_original(
    points: Iterable[Sequence[float]],
    meta: TransformMeta,
    *,
    clip: bool = True,
) -> Points:
    ow, oh = meta.original_size
    out: Points = []
    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        xo, yo = map_mask_to_original(x, y, meta)
        if clip:
            xo = min(max(xo, 0.0), float(max(ow - 1, 0)))
            yo = min(max(yo, 0.0), float(max(oh - 1, 0)))
        out.append([float(xo), float(yo)])
    return out


def build_letterbox_meta(
    original_wh: Tuple[int, int],
    model_input_wh: Tuple[int, int],
    mask_wh: Tuple[int, int] | None = None,
) -> TransformMeta:
    """Letterbox original into model input; optional lower-res mask grid."""
    ow, oh = int(original_wh[0]), int(original_wh[1])
    iw, ih = int(model_input_wh[0]), int(model_input_wh[1])
    mw, mh = (iw, ih) if mask_wh is None else (int(mask_wh[0]), int(mask_wh[1]))

    scale = min(iw / max(ow, 1), ih / max(oh, 1))
    new_w, new_h = ow * scale, oh * scale
    pad_left = (iw - new_w) / 2.0
    pad_top = (ih - new_h) / 2.0
    pad_right = iw - new_w - pad_left
    pad_bottom = ih - new_h - pad_top

    if (mw, mh) == (iw, ih):
        m2i = AffineStep.identity()
    else:
        m2i = AffineStep.stretch(iw / max(mw, 1), ih / max(mh, 1))

    return TransformMeta(
        original_size=(ow, oh),
        model_input_size=(iw, ih),
        mask_size=(mw, mh),
        mask_to_input=m2i,
        input_to_original=AffineStep.letterbox(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        ),
    )


def _forward_xy_scales(step: AffineStep) -> tuple[float, float]:
    mode = (step.mode or "identity").lower()
    if mode in ("identity", "none", ""):
        return 1.0, 1.0
    if mode == "stretch":
        return float(step.scale_x or 1.0), float(step.scale_y or 1.0)
    if mode == "letterbox":
        s = float(step.scale if step.scale is not None else step.scale_x or 1.0)
        return s, s
    return float(step.scale_x or 1.0), float(step.scale_y or 1.0)


def mask_area_scale(meta: TransformMeta) -> float:
    """Approximate A_mask / A_original from TransformMeta forward scales.

    ``mask_to_input`` maps mask→input (A_input = A_mask * sx_m * sy_m).
    ``input_to_original`` stores FORWARD ori→input (A_input = A_ori * sx_i * sy_i).
    Therefore A_mask / A_ori = (sx_i * sy_i) / (sx_m * sy_m).
    """
    sx_m, sy_m = _forward_xy_scales(meta.mask_to_input)
    sx_i, sy_i = _forward_xy_scales(meta.input_to_original)
    denom = max(sx_m * sy_m, 1e-12)
    return max((sx_i * sy_i) / denom, 1e-12)


def original_min_area_to_mask(min_area_original: float, meta: TransformMeta) -> float:
    """Convert an original-image area threshold into mask-space area."""
    return float(min_area_original) * mask_area_scale(meta)


def build_stretch_meta(
    original_wh: Tuple[int, int],
    model_input_wh: Tuple[int, int],
    mask_wh: Tuple[int, int] | None = None,
) -> TransformMeta:
    """Stretch original into model input.

    `input_to_original` stores the FORWARD preprocess scales (iw/ow, ih/oh),
    matching letterbox: map_mask_to_original inverts this step.
    """
    ow, oh = int(original_wh[0]), int(original_wh[1])
    iw, ih = int(model_input_wh[0]), int(model_input_wh[1])
    mw, mh = (iw, ih) if mask_wh is None else (int(mask_wh[0]), int(mask_wh[1]))
    if (mw, mh) == (iw, ih):
        m2i = AffineStep.identity()
    else:
        m2i = AffineStep.stretch(iw / max(mw, 1), ih / max(mh, 1))
    # Forward preprocess: x_in = x_ori * (iw/ow)  → store iw/ow; inverse divides.
    return TransformMeta(
        original_size=(ow, oh),
        model_input_size=(iw, ih),
        mask_size=(mw, mh),
        mask_to_input=m2i,
        input_to_original=AffineStep.stretch(iw / max(ow, 1), ih / max(oh, 1)),
    )
