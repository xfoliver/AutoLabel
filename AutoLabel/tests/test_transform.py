"""Unit tests for TransformMeta two-stage mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from autolabel.geometry.transform import (  # noqa: E402
    build_letterbox_meta,
    build_stretch_meta,
    map_mask_to_original,
)
from autolabel.models import AffineStep, TransformMeta  # noqa: E402


def test_identity_roundtrip():
    meta = TransformMeta.identity_for(1920, 1080)
    x, y = map_mask_to_original(100.0, 200.0, meta)
    assert abs(x - 100.0) < 1e-6
    assert abs(y - 200.0) < 1e-6


def test_letterbox_quarter_mask_center():
    # original 1920x1080 → letterbox into 640x640 → mask 160x160 (1/4)
    meta = build_letterbox_meta((1920, 1080), (640, 640), (160, 160))
    assert meta.mask_to_input.mode == "stretch"
    assert abs(meta.mask_to_input.scale_x - 4.0) < 1e-6

    # Center of original should map near center of mask after forward preprocess.
    # Pick a known original point, push through forward, then invert via map.
    # Forward: scale = min(640/1920, 640/1080) = 640/1920 = 1/3
    scale = 640 / 1920
    assert abs(meta.input_to_original.scale - scale) < 1e-6
    # original center (960, 540) → input
    xin = 960 * scale + meta.input_to_original.pad_left
    yin = 540 * scale + meta.input_to_original.pad_top
    # mask coords = input / 4
    mx, my = xin / 4.0, yin / 4.0
    xo, yo = map_mask_to_original(mx, my, meta)
    assert abs(xo - 960) < 1.0
    assert abs(yo - 540) < 1.0


def test_stretch_corners():
    meta = build_stretch_meta((100, 50), (50, 25), (50, 25))
    xo, yo = map_mask_to_original(50, 25, meta)
    # mask==input; stretch input→original by (2,2)
    # Wait: input_to_original stretch stores scale_x = ow/iw = 2
    # Inverse of stretch(scale_x=2): x_ori = x_in / ?
    # Our AffineStep.stretch for input_to_original uses scale_x = ow/iw meaning
    # forward original→input would be divide...
    # Convention: input_to_original.stretch means x_ori = x_in * scale_x
    # So inverse of preprocess: if preprocess is stretch-to-input, forward ori→in is x_in = x_ori * (iw/ow)
    # We stored scale_x = ow/iw which is the INVERSE scale for going in→ori as multiply.
    #
    # map does: inverse_step on input_to_original.
    # For stretch inverse: x / scale_x. That would be wrong if scale_x=ow/iw.
    #
    # Fix expectation to match implementation: we need consistency with build_stretch_meta.
    # build_stretch_meta sets input_to_original = stretch(ow/iw, oh/ih)
    # map: inverse stretch => x_ori = x_in / (ow/iw) = x_in * (iw/ow)  -- WRONG direction!
    #
    # Actually for stretch preprocess: x_in = x_ori * (iw/ow)
    # Inverse: x_ori = x_in * (ow/iw)
    # So input_to_original should be applied FORWARD when going input→original for stretch,
    # OR we store the preprocess scales differently.
    #
    # Looking at letterbox: we store FORWARD preprocess (scale, pad) and invert correctly.
    # For stretch consistency, input_to_original.stretch should store the FORWARD preprocess
    # scales (iw/ow), and inverse divides.
    pass  # detailed assert in test_stretch_meta_consistent


def test_stretch_meta_consistent():
    """Stretch: original 200x100 → input 100x50; mask=input."""
    # Rebuild with preprocess-forward convention: store iw/ow in step, invert by divide.
    # Current build_stretch_meta stores ow/iw — verify and document behavior.
    meta = build_stretch_meta((200, 100), (100, 50), (100, 50))
    # If implementation stores ow/iw=2 and inverse divides, mapping input(50,25) → (25,12.5) which is wrong.
    # Correct mapping of input center (50,25) should be original center (100,50).
    xo, yo = map_mask_to_original(50, 25, meta)
    # Adjust implementation if needed — for now assert the mathematically intended result
    # and fix transform if failing.
    assert abs(xo - 100) < 1.0, f"got {(xo, yo)}"
    assert abs(yo - 50) < 1.0, f"got {(xo, yo)}"


def test_from_dict_roundtrip_letterbox():
    meta = build_letterbox_meta((1920, 1080), (640, 640), (160, 160))
    restored = TransformMeta.from_dict(meta.to_dict())
    x1, y1 = map_mask_to_original(10, 20, meta)
    x2, y2 = map_mask_to_original(10, 20, restored)
    assert abs(x1 - x2) < 1e-6
    assert abs(y1 - y2) < 1e-6
