"""SAM3 worker numpy conversion — no real torch / GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from sam3_worker.inference import _scores_list, _to_numpy  # noqa: E402


class _FakeF32:
    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([0.41, 0.87], dtype=np.float32)


class _FakeBF16:
    """Mimic CUDA bfloat16: ``.numpy()`` raises until ``.float()``."""

    def detach(self):
        return self

    def float(self):
        return _FakeF32()

    def cpu(self):
        return self

    def numpy(self):
        raise TypeError("Got unsupported ScalarType BFloat16")


def test_to_numpy_casts_bfloat16_via_float():
    arr = _to_numpy(_FakeBF16())
    assert arr.dtype == np.float32
    assert arr.tolist() == pytest.approx([0.41, 0.87], rel=1e-5)


def test_scores_list_reads_bfloat16():
    out = _scores_list(_FakeBF16(), 2)
    assert out[0] == pytest.approx(0.41, rel=1e-5)
    assert out[1] == pytest.approx(0.87, rel=1e-5)
    assert _scores_list(_FakeBF16(), 0) == []
