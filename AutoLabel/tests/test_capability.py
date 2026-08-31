"""Capability check tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.capability import CapabilityError, check_backend_for_spec  # noqa: E402
from autolabel.inference.mock import MockBackend, NoInstanceMaskBackend  # noqa: E402
from autolabel.label_spec import build_label_spec  # noqa: E402


def test_mock_passes_iseg():
    backend = MockBackend()
    spec = build_label_spec(label="王不留行")
    report = check_backend_for_spec(backend, spec)
    assert report.ok


def test_no_instance_rejected():
    backend = NoInstanceMaskBackend()
    spec = build_label_spec(label="x")
    report = check_backend_for_spec(backend, spec)
    assert not report.ok
    with pytest.raises(CapabilityError):
        report.raise_if_failed()


def test_fallback_warning_only_when_allowed():
    backend = NoInstanceMaskBackend()
    spec = build_label_spec(label="x")
    report = check_backend_for_spec(backend, spec, allow_class_mask_fallback=True)
    assert report.ok
    assert report.warnings
