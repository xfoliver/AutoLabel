"""Overlay visualization for preview UI / reports."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from autolabel.cv_io import imread, imwrite
from autolabel.models import ShapeRecord


_COLORS = [
    (108, 92, 231),
    (0, 184, 148),
    (9, 132, 227),
    (253, 203, 110),
    (225, 112, 85),
]

def color_for_label(label: str, fallback_index: int = 0) -> Tuple[int, int, int]:
    """Stable BGR color for a class. Same label → same color (not rainbow-per-instance)."""
    key = (label or "").strip()
    if not key:
        return _COLORS[int(fallback_index) % len(_COLORS)]
    digest = hashlib.md5(key.encode("utf-8")).digest()
    # Keep channels away from 0 so fills stay visible on dark metal.
    return (80 + digest[0] % 160, 80 + digest[1] % 160, 80 + digest[2] % 160)


_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
)


@lru_cache(maxsize=1)
def _resolve_font_path() -> Optional[str]:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            return str(p)
    return None


def _needs_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _draw_label(
    canvas: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw label text; use Pillow+CJK font when needed, else OpenCV putText."""
    label = (text or "").strip()
    if not label:
        return
    x, y = org
    if not _needs_cjk(label):
        cv2.putText(
            canvas,
            label,
            (x, max(15, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
        return

    font_path = _resolve_font_path()
    if not font_path:
        # Last resort: still call putText (will show '?'), but avoid crash.
        cv2.putText(
            canvas,
            label,
            (x, max(15, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
        return

    from PIL import Image, ImageDraw, ImageFont

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, 16)
    except OSError:
        font = ImageFont.load_default()
    # Pillow uses top-left; keep label slightly above the first polygon vertex.
    ty = max(0, y - 18)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    # Soft dark outline for contrast on busy backgrounds.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, ty + dy), label, font=font, fill=(0, 0, 0))
    draw.text((x, ty), label, font=font, fill=color_rgb)
    canvas[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def draw_shapes_overlay(
    image_path: Union[str, Path],
    shapes: Sequence[ShapeRecord],
    output_path: Union[str, Path],
) -> Path:
    img = imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    canvas = img.copy()
    for i, sh in enumerate(shapes):
        color = color_for_label(sh.label, i)
        pts = np.array(sh.points, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 2)
            x, y = int(pts[0, 0, 0]), int(pts[0, 0, 1])
            _draw_label(canvas, sh.label, (x, max(15, y - 5)), color)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite(out, canvas):
        raise OSError(f"Failed to write overlay: {out}")
    return out
