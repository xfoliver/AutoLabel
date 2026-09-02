"""Unified image load: EXIF-normalized RGB/BGR pixels.

Windows viewers / Qt / LabelMe apply JPEG Orientation. OpenCV IMREAD_COLOR
usually does too. Pillow ``Image.open().convert("RGB")`` does **not**.

AutoLabel's coordinate space is the EXIF-transposed image:
    open → ImageOps.exif_transpose → RGB
so SAM3 / polygon / Preview / LabelMe share one (width, height).

Never overwrite the user's original file; this module only returns arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image, ImageOps

PathLike = Union[str, Path]


def exif_orientation(path: PathLike, image: Optional[Image.Image] = None) -> Optional[int]:
    im = image
    close = False
    if im is None:
        im = Image.open(path)
        close = True
    try:
        exif = im.getexif()
        if not exif:
            return None
        value = exif.get(274) or exif.get(0x0112)
        return int(value) if value is not None else None
    except Exception:
        return None
    finally:
        if close:
            im.close()


def open_normalized(path: PathLike) -> Tuple[Image.Image, Dict[str, Any]]:
    """Open a still image, apply EXIF Orientation, return RGB PIL + debug meta.

    Does not write back to ``path``.
    """
    p = Path(path)
    raw = Image.open(p)
    raw_size = (int(raw.size[0]), int(raw.size[1]))
    ori = exif_orientation(p, raw)
    normalized = ImageOps.exif_transpose(raw)
    if normalized is None:
        normalized = raw
    if normalized.mode != "RGB":
        normalized = normalized.convert("RGB")
    elif normalized is raw:
        normalized = raw.convert("RGB")
    meta: Dict[str, Any] = {
        "path": str(p),
        "exif_orientation": ori,
        "pil_raw_size_wh": [raw_size[0], raw_size[1]],
        "normalized_size_wh": [int(normalized.size[0]), int(normalized.size[1])],
        "applied_exif_transpose": bool(ori not in (None, 0, 1) and tuple(normalized.size) != raw_size),
    }
    return normalized, meta


def normalized_rgb_array(path: PathLike) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np

    image, meta = open_normalized(path)
    arr = np.asarray(image)
    meta["normalized_shape_hw"] = [int(arr.shape[0]), int(arr.shape[1])]
    return arr, meta
