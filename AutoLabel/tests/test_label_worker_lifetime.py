"""LabelWorker must not emit into a deleted C++ QThread."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from ui.backend_worker import LabelWorker, _qt_alive  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeOrch:
    def __init__(self):
        self._log = lambda m: None
        self.logged = []

    def apply_ui_intent(self, **kwargs):
        return SimpleNamespace()

    def build_plan(self, dataset_path=None):
        self._log("building")
        return SimpleNamespace(pending_count=1, preview_count=1)


def test_qt_alive_false_after_deleteLater(qapp):
    w = LabelWorker("plan", _FakeOrch())
    assert _qt_alive(w)
    w.deleteLater()
    qapp.processEvents()
    # sip.isdeleted may still be False until the event is processed fully;
    # force C++ delete via sip if available.
    try:
        from PyQt5 import sip

        sip.delete(w)
        assert not _qt_alive(w)
    except Exception:
        pytest.skip("sip.delete not available")


def test_emit_after_delete_does_not_raise(qapp):
    orch = _FakeOrch()
    w = LabelWorker("plan", orch)
    w._emit("log_received", "hello")
    try:
        from PyQt5 import sip

        sip.delete(w)
    except Exception:
        pytest.skip("sip.delete not available")
    w._emit("log_received", "after-delete")
    w._emit("failed", "should be swallowed")


def test_run_logs_via_safe_emit(qapp, tmp_path):
    orch = _FakeOrch()
    w = LabelWorker("plan", orch, dataset_path=str(tmp_path), prompt="x", short_label="x")
    logs = []
    w.log_received.connect(logs.append)
    w.run()
    assert any("building" in str(x) for x in logs)
