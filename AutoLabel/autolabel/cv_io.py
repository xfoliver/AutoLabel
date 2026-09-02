"""Unicode-safe OpenCV image I/O (Windows Chinese paths).

Color reads go through EXIF-normalized PIL so SAM3 / Preview / LabelMe
share one coordinate space. Mask / unchanged reads stay raw imdecode
(those files are written by us without Orientation tags).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def _is_color_flag(flags: int) -> bool:
    # IMREAD_COLOR == 1. IGNORE_ORIENTATION is OR'd by some callers.
    color = int(getattr(cv2, "IMREAD_COLOR", 1))
    gray = int(getattr(cv2, "IMREAD_GRAYSCALE", 0))
    unchanged = int(getattr(cv2, "IMREAD_UNCHANGED", -1))
    if flags == unchanged or flags == gray:
        return False
    if flags == color:
        return True
    # Treat default / color-like flags as color (including COLOR|IGNORE_ORIENTATION).
    return flags >= 0 and flags != gray


def _imdecode(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image; returns None if missing / undecodable (same as cv2.imread).

    For color photos, pixels are EXIF-transposed (display orientation).
    """
    p = Path(path)
    if _is_color_flag(flags):
        try:
            from autolabel.image_orient import normalized_rgb_array

            rgb, _meta = normalized_rgb_array(p)
            if rgb is None or getattr(rgb, "size", 0) == 0:
                return _imdecode(p, flags)
            if rgb.ndim == 2:
                return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return _imdecode(p, flags)
    return _imdecode(p, flags)


def imwrite(path: PathLike, image: np.ndarray, params: Optional[list] = None) -> bool:
    """Write an image; returns False on encode/write failure (same as cv2.imwrite)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower() or ".png"
    if not ext.startswith("."):
        ext = "." + ext
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    try:
        buf.tofile(str(out))
    except OSError:
        return False
    return True
