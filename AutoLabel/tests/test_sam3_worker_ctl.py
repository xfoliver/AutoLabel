"""SAM3 worker control helpers (mocked HTTP)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.sam3_worker_ctl import (  # noqa: E402
    forget_owned_pid,
    is_autolabel_sam3_worker,
    owned_worker_pids,
    parse_netstat_listening_pids,
    probe_health,
    probe_refine_supported,
    remember_owned_pid,
    stop_owned_worker,
    worker_url,
    _worker_env,
)


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


def test_parse_netstat_listening_pids():
    text = (
        "  TCP    127.0.0.1:9892         0.0.0.0:0              LISTENING       4242\n"
        "  TCP    127.0.0.1:19892        0.0.0.0:0              LISTENING       7\n"
        "  TCP    127.0.0.1:9892         127.0.0.1:50000        ESTABLISHED     9\n"
    )
    assert parse_netstat_listening_pids(text, 9892) == [4242]


def test_probe_refine_supported_404(monkeypatch):
    class _Resp:
        status_code = 404

    monkeypatch.setattr("autolabel.sam3_worker_ctl.requests.post", lambda *a, **k: _Resp())
    assert probe_refine_supported("http://127.0.0.1:9892") is False


def test_probe_refine_supported_from_health():
    assert probe_refine_supported(
        "http://127.0.0.1:9892",
        {"refine_mode": "interactive", "api_version": "0.4", "endpoints": ["/health", "/segment", "/refine"]},
    )


def test_probe_refine_supported_rejects_unwired_0_3():
    # 0.3 advertised interactive refine but tracker.backbone was still None.
    assert probe_refine_supported(
        "http://127.0.0.1:9892",
        {"refine_mode": "interactive", "api_version": "0.3", "endpoints": ["/refine"]},
    ) is False


def test_probe_refine_supported_rejects_broken_mode():
    assert probe_refine_supported(
        "http://127.0.0.1:9892",
        {"refine_mode": "broken", "api_version": "0.4", "endpoints": ["/refine"]},
    ) is False


def test_probe_refine_supported_rejects_geometric_mode():
    assert probe_refine_supported(
        "http://127.0.0.1:9892",
        {"refine_mode": "geometric", "api_version": "0.2", "endpoints": ["/refine"]},
    ) is False


def test_probe_refine_supported_rejects_legacy_health():
    assert probe_refine_supported("http://127.0.0.1:9892", {"status": "ok", "model_loaded": True}) is False


def test_probe_refine_supported_route_exists(monkeypatch):
    class _Resp:
        status_code = 400

    monkeypatch.setattr("autolabel.sam3_worker_ctl.requests.post", lambda *a, **k: _Resp())
    assert probe_refine_supported(
        "http://127.0.0.1:9892",
        {"api_version": "0.4", "endpoints": ["/refine"]},
    ) is True


def test_ensure_worker_restarts_stale_refine(monkeypatch):
    from autolabel.sam3_worker_ctl import ensure_worker

    calls = {"health": 0, "stopped": 0, "started": 0}

    def fake_health(url, timeout=2.0):
        calls["health"] += 1
        if calls["stopped"]:
            if calls["health"] < 4:
                return False, {"status": "down"}
            return True, {"status": "ok", "model_loaded": True, "device": "cuda", "endpoints": ["/refine"]}
        return True, {"status": "ok", "model_loaded": True, "device": "cuda"}

    monkeypatch.setattr("autolabel.sam3_worker_ctl.probe_health", fake_health)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.probe_refine_supported", lambda *a, **k: bool(calls["stopped"]))

    def fake_stop(opt, log=None):
        calls["stopped"] += 1

    class _Proc:
        def poll(self):
            return None

    def fake_start(opt, log=None):
        calls["started"] += 1
        return _Proc()

    monkeypatch.setattr("autolabel.sam3_worker_ctl.stop_worker", fake_stop)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.start_worker", fake_start)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.time.sleep", lambda *_: None)
    logs: list[str] = []
    data = ensure_worker({"port": 9892}, log=logs.append, timeout=30)
    assert calls["stopped"] == 1
    assert calls["started"] == 1
    assert data.get("model_loaded") is True
    assert any("过旧" in m or "重启" in m for m in logs)


def test_ensure_worker_restarts_on_version_mismatch(monkeypatch):
    from autolabel.sam3_worker_ctl import ensure_worker

    calls = {"health": 0, "stopped": 0, "started": 0}

    def fake_health(url, timeout=2.0):
        calls["health"] += 1
        if calls["stopped"]:
            return True, {
                "status": "ok",
                "model_loaded": True,
                "device": "cuda",
                "sam3_version": "sam3.1",
                "api_version": "0.4",
                "endpoints": ["/refine"],
                "refine_mode": "interactive",
            }
        return True, {
            "status": "ok",
            "model_loaded": True,
            "device": "cuda",
            "sam3_version": "sam3",
            "api_version": "0.4",
            "endpoints": ["/refine"],
            "refine_mode": "interactive",
        }

    monkeypatch.setattr("autolabel.sam3_worker_ctl.probe_health", fake_health)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.probe_refine_supported", lambda *a, **k: True)

    def fake_stop(opt, log=None):
        calls["stopped"] += 1

    class _Proc:
        def poll(self):
            return None

    def fake_start(opt, log=None):
        calls["started"] += 1
        return _Proc()

    monkeypatch.setattr("autolabel.sam3_worker_ctl.stop_worker", fake_stop)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.start_worker", fake_start)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.time.sleep", lambda *_: None)
    logs: list[str] = []
    data = ensure_worker({"port": 9892, "sam3_version": "sam3.1"}, log=logs.append, timeout=30)
    assert calls["stopped"] == 1
    assert calls["started"] == 1
    assert data.get("sam3_version") == "sam3.1"
    assert any("sam3.1" in m for m in logs)


def test_worker_env_drops_parent_pythonpath(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "wrong"))
    py = Path(r"C:\dlcv\envs\sam3\python.exe")
    env = _worker_env(py)
    assert "wrong" not in (env.get("PYTHONPATH") or "")
    assert str(ROOT) in env["PYTHONPATH"]
    assert str(py.parent) in env["PATH"]
    assert env.get("PYTHONUNBUFFERED") == "1"


def test_worker_env_injects_local_checkpoint(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "models" / "sam3.1_multiplex.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"weights")
    (tmp_path / "models" / "sam3.pt").write_bytes(b"old")
    bpe = tmp_path / "sam3_src" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    bpe.parent.mkdir(parents=True)
    bpe.write_bytes(b"gz")
    (tmp_path / "sam3_src" / "sam3" / "model_builder.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    monkeypatch.delenv("SAM3_BPE", raising=False)
    monkeypatch.delenv("SAM3_ROOT", raising=False)
    py = tmp_path / "runtime" / "sam3" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    env = _worker_env(py)
    assert env["SAM3_CHECKPOINT"] == str(ckpt.resolve())
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("SAM3_BPE") == str(bpe.resolve())
    assert env.get("SAM3_VERSION") == "sam3.1"


def test_worker_env_injects_sam31_checkpoint(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "models" / "sam3.1_multiplex.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"weights31")
    (tmp_path / "models" / "sam3.pt").write_bytes(b"old")
    bpe = tmp_path / "sam3_src" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    bpe.parent.mkdir(parents=True)
    bpe.write_bytes(b"gz")
    (tmp_path / "sam3_src" / "sam3" / "model_builder.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    monkeypatch.delenv("SAM3_BPE", raising=False)
    monkeypatch.delenv("SAM3_ROOT", raising=False)
    py = tmp_path / "runtime" / "sam3" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    env = _worker_env(py, sam3_version="sam3.1")
    assert env["SAM3_CHECKPOINT"] == str(ckpt.resolve())
    assert env.get("SAM3_VERSION") == "sam3.1"


def test_worker_options_picks_sam31_from_engine():
    from autolabel.sam3_worker_ctl import worker_options

    cfg = {
        "annotation_engine": "sam3",
        "backends": [
            {"id": "sam3_local", "type": "sam3_local", "options": {"port": 9892, "sam3_version": "sam3"}},
            {"id": "sam3_1_local", "type": "sam3_local", "options": {"port": 9892, "sam3_version": "sam3.1"}},
        ],
    }
    opt = worker_options(cfg)
    assert opt.get("sam3_version") == "sam3.1"


def test_is_autolabel_sam3_worker_cmdline():
    assert is_autolabel_sam3_worker(
        r"C:\dlcv\envs\sam3\python.exe -u C:\Users\Admin\kimi_gui_automation\AutoLabel\sam3_worker\server.py --host 127.0.0.1 --port 9892"
    )
    assert is_autolabel_sam3_worker("python -u AutoLabel/sam3_worker/server.py --port 9892")
    assert not is_autolabel_sam3_worker("C:\\dlcv\\envs\\sam3\\python.exe -m http.server 9892")
    assert not is_autolabel_sam3_worker("")


def test_stop_owned_worker_skips_foreign_pid(monkeypatch):
    from autolabel import sam3_worker_ctl as ctl

    ctl._OWNED_WORKER_PIDS.clear()
    remember_owned_pid(111)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.pids_listening_on_port", lambda port: [222])
    monkeypatch.setattr("autolabel.sam3_worker_ctl.process_command_line", lambda pid: "python -m http.server 9892")
    killed: list[int] = []
    monkeypatch.setattr("autolabel.sam3_worker_ctl.kill_pid", lambda pid: killed.append(pid) or True)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.probe_health", lambda *a, **k: (False, {}))
    monkeypatch.setattr("autolabel.sam3_worker_ctl.time.sleep", lambda *_: None)
    stop_owned_worker({"port": 9892})
    assert killed == [111]
    assert 111 not in owned_worker_pids()
    forget_owned_pid(111)


def test_hidden_popen_kwargs_hides_console():
    from autolabel.sam3_worker_ctl import CREATE_NO_WINDOW, _hidden_popen_kwargs
    import os

    kwargs = _hidden_popen_kwargs()
    if os.name == "nt":
        assert kwargs.get("creationflags") == CREATE_NO_WINDOW
        info = kwargs.get("startupinfo")
        assert info is not None
        assert info.wShowWindow == 0
    else:
        assert kwargs == {}


def test_stop_owned_worker_atexit_skips_when_already_cleared(monkeypatch):
    from autolabel import sam3_worker_ctl as ctl

    ctl._OWNED_WORKER_PIDS.clear()
    ctl._STOPPING_OWNED_WORKER = False
    called = {"netstat": 0}

    def boom(_port):
        called["netstat"] += 1
        return [1]

    monkeypatch.setattr("autolabel.sam3_worker_ctl.pids_listening_on_port", boom)
    stop_owned_worker({"port": 9892}, from_atexit=True)
    assert called["netstat"] == 0


def test_stop_owned_worker_kills_orphan_sam3_cmdline(monkeypatch):
    from autolabel import sam3_worker_ctl as ctl

    ctl._OWNED_WORKER_PIDS.clear()
    monkeypatch.setattr("autolabel.sam3_worker_ctl.pids_listening_on_port", lambda port: [48008])
    monkeypatch.setattr(
        "autolabel.sam3_worker_ctl.process_command_line",
        lambda pid: r"C:\dlcv\envs\sam3\python.exe -u C:\repo\AutoLabel\sam3_worker\server.py --port 9892",
    )
    killed: list[int] = []
    monkeypatch.setattr("autolabel.sam3_worker_ctl.kill_pid", lambda pid: killed.append(pid) or True)
    monkeypatch.setattr("autolabel.sam3_worker_ctl.probe_health", lambda *a, **k: (False, {}))
    monkeypatch.setattr("autolabel.sam3_worker_ctl.time.sleep", lambda *_: None)
    stop_owned_worker({"port": 9892})
    assert killed == [48008]
