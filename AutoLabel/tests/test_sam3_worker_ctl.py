"""SAM3 worker control helpers (mocked HTTP)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.sam3_worker_ctl import probe_health, worker_url, _worker_env  # noqa: E402


def test_worker_url_from_options():
    assert worker_url({"host": "127.0.0.1", "port": 9892}) == "http://127.0.0.1:9892"
    assert worker_url({"base_url": "http://127.0.0.1:9892/"}) == "http://127.0.0.1:9892"


def test_probe_health_ok(monkeypatch):
    class _Resp:
        status_code = 200
        content = b'{"status":"ok","model_loaded":true}'

        def json(self):
            return {"status": "ok", "model_loaded": True, "device": "cuda"}

    monkeypatch.setattr("autolabel.sam3_worker_ctl.requests.get", lambda *a, **k: _Resp())
    ok, data = probe_health("http://127.0.0.1:9892")
    assert ok is True
    assert data["model_loaded"] is True


def test_probe_health_loading(monkeypatch):
    class _Resp:
        status_code = 200
        content = b'{"status":"loading","model_loaded":false}'

        def json(self):
            return {"status": "loading", "model_loaded": False}

    monkeypatch.setattr("autolabel.sam3_worker_ctl.requests.get", lambda *a, **k: _Resp())
    ok, data = probe_health("http://127.0.0.1:9892")
    assert ok is True
    assert data["model_loaded"] is False


def test_probe_health_down(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("refused")

    monkeypatch.setattr("autolabel.sam3_worker_ctl.requests.get", _boom)
    ok, data = probe_health("http://127.0.0.1:9892")
    assert ok is False
    assert data["status"] == "down"


def test_worker_env_drops_parent_pythonpath(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "wrong"))
    py = Path(r"C:\dlcv\envs\sam3\python.exe")
    env = _worker_env(py)
    assert "wrong" not in (env.get("PYTHONPATH") or "")
    assert str(ROOT) in env["PYTHONPATH"]
    assert str(py.parent) in env["PATH"]
    assert env.get("PYTHONUNBUFFERED") == "1"
