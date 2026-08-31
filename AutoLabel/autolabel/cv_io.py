"""Unicode-safe OpenCV image I/O (Windows Chinese paths).

``cv2.imread`` / ``cv2.imwrite`` fail on non-ASCII paths under Windows.
Use ``np.fromfile`` + ``imdecode`` / ``imencode`` + ``tofile`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image; returns None if missing / undecodable (same as cv2.imread)."""
    p = Path(path)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


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
