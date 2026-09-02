"""EXIF orientation: SAM3 / OpenCV / Preview must share one coordinate space."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.cv_io import imread  # noqa: E402
from autolabel.image_orient import open_normalized  # noqa: E402
from autolabel.inference.sam3_local import _load_mask  # noqa: E402
from autolabel.models import ShapeRecord  # noqa: E402
from autolabel.viz import draw_shapes_overlay  # noqa: E402
from sam3_worker.inference import open_image_normalized  # noqa: E402


def _jpeg_with_orientation(tmp_path: Path, *, ori: int, raw_size=(40, 80), name="t.jpg") -> Path:
    """Raw pixels W×H = raw_size. Paint a unique blob in raw pixel space."""
    w, h = raw_size
    img = Image.new("RGB", (w, h), (10, 10, 10))
    px = img.load()
    # bright block near top-left of RAW pixels (x=2..10, y=2..18)
    for y in range(2, 18):
        for x in range(2, 10):
            px[x, y] = (255, 0, 0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw_bytes = buf.getvalue()
    # Inject EXIF Orientation via Pillow
    tagged = Image.open(io.BytesIO(raw_bytes))
    exif = tagged.getexif()
    exif[274] = ori
    dest = tmp_path / name
    tagged.save(dest, format="JPEG", quality=95, exif=exif)
    return dest


def test_open_normalized_orientation_6(tmp_path: Path):
    path = _jpeg_with_orientation(tmp_path, ori=6, raw_size=(40, 80))
    raw = Image.open(path)
    assert raw.size == (40, 80)
    assert raw.getexif().get(274) == 6
    after = ImageOps.exif_transpose(raw)
    assert after.size == (80, 40)  # 90° CW

    pil, meta = open_normalized(path)
    assert tuple(pil.size) == (80, 40)
    assert meta["exif_orientation"] == 6
    assert meta["applied_exif_transpose"] is True

    worker_im, worker_meta = open_image_normalized(path)
    assert tuple(worker_im.size) == tuple(pil.size)
    assert worker_meta["exif_orientation"] == 6

    bgr = imread(path, cv2.IMREAD_COLOR)
    assert bgr is not None
    assert (bgr.shape[1], bgr.shape[0]) == pil.size


def test_open_normalized_orientation_1_unchanged(tmp_path: Path):
    path = _jpeg_with_orientation(tmp_path, ori=1, raw_size=(40, 80))
    pil, meta = open_normalized(path)
    assert tuple(pil.size) == (40, 80)
    assert meta["applied_exif_transpose"] is False
    bgr = imread(path)
    assert (bgr.shape[1], bgr.shape[0]) == (40, 80)


def test_load_mask_rejects_transposed_shape(tmp_path: Path):
    mask = np.zeros((80, 40), dtype=np.uint8)
    mask[0, 0] = 255
    p = tmp_path / "m.png"
    cv2.imwrite(str(p), mask)
    with pytest.raises(RuntimeError, match="transposed"):
        _load_mask(str(p), (40, 80))  # image H,W swapped vs mask


def test_load_mask_same_shape_ok(tmp_path: Path):
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    p = tmp_path / "m.png"
    cv2.imwrite(str(p), mask)
    out = _load_mask(str(p), (40, 80))
    assert out.shape == (40, 80)
    assert int(out[7, 7]) == 1


def test_overlay_uses_normalized_pixels(tmp_path: Path):
    path = _jpeg_with_orientation(tmp_path, ori=6, raw_size=(40, 80), name="src.jpg")
    # After transpose, display size is 80x40. Polygon in that space.
    shapes = [
        ShapeRecord(
            label="x",
            points=[[2.0, 2.0], [20.0, 2.0], [20.0, 20.0], [2.0, 20.0]],
        )
    ]
    out = draw_shapes_overlay(path, shapes, tmp_path / "ov.jpg")
    rendered = cv2.imread(str(out))
    assert rendered is not None
    assert rendered.shape[1] == 80
    assert rendered.shape[0] == 40
