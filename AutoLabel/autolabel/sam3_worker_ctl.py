"""Start / probe the resident SAM3 HTTP worker from AutoLabel's main process.

AutoLabel must never import sam3/torch. This module only uses subprocess + HTTP.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

AUTO_LABEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = r"C:\dlcv\envs\sam3\python.exe"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9892
WORKER_LOG = AUTO_LABEL_ROOT / "runs" / "sam3" / "worker.log"

# CREATE_NO_WINDOW keeps a GUI parent from flashing a console; we still capture
# stdout/stderr to a log file. DETACHED_PROCESS is avoided — it often kills
# conda python before it can bind the port.
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def worker_options(backends_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = backends_config or {}
    active = cfg.get("active_backend")
    for entry in cfg.get("backends") or []:
        if isinstance(entry, dict) and entry.get("id") == active:
            return dict(entry.get("options") or {})
    for entry in cfg.get("backends") or []:
        if isinstance(entry, dict) and (entry.get("id") == "sam3_local" or entry.get("type") == "sam3_local"):
            return dict(entry.get("options") or {})
    return {}


def worker_url(options: Optional[Dict[str, Any]] = None) -> str:
    opt = options or {}
    base = str(opt.get("base_url") or "").strip().rstrip("/")
    if base:
        return base
    host = str(opt.get("host") or DEFAULT_HOST).strip() or DEFAULT_HOST
    port = int(opt.get("port") or DEFAULT_PORT)
    return f"http://{host}:{port}"


def probe_health(url: str, *, timeout: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
    try:
        resp = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        if resp.status_code != 200:
            return False, {"status": "error", "error": f"HTTP {resp.status_code}"}
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            return False, {"status": "error", "error": "invalid health payload"}
        loaded = bool(data.get("model_loaded"))
        status = str(data.get("status") or "")
        ok = loaded or status in {"ok", "loading", "error"}
        return ok, data
    except Exception as e:
        return False, {"status": "down", "error": str(e)}


def _conda_env_root(python_path: Path) -> Path:
    """C:\\dlcv\\envs\\sam3\\python.exe → C:\\dlcv\\envs\\sam3."""
    return python_path.resolve().parent


def _worker_env(python_path: Path, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Give the child a conda-like PATH and drop the parent's PYTHONPATH."""
    env = os.environ.copy()
    root = _conda_env_root(python_path)
    prepend = [
        str(root),
        str(root / "Scripts"),
        str(root / "Library" / "bin"),
        str(root / "Library" / "mingw-w64" / "bin"),
        str(root / "Library" / "usr" / "bin"),
        str(root / "bin"),
    ]
    old_path = env.get("PATH") or env.get("Path") or ""
    env["PATH"] = os.pathsep.join(prepend + [old_path])
    env["CONDA_PREFIX"] = str(root)
    env["CONDA_DEFAULT_ENV"] = root.name
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    # Parent AutoLabel PYTHONPATH would leak PyQt / wrong torch into sam3.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    sam3_repo = AUTO_LABEL_ROOT.parent / "sam3"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(AUTO_LABEL_ROOT), str(sam3_repo)] if sam3_repo.is_dir() else [str(AUTO_LABEL_ROOT)]
    )
    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env


def _tail_log(path: Path, n: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def start_worker(
    options: Optional[Dict[str, Any]] = None,
    *,
    log=None,
) -> subprocess.Popen:
    opt = options or {}
    python_path = Path(str(opt.get("python_path") or DEFAULT_PYTHON))
    server = AUTO_LABEL_ROOT / "sam3_worker" / "server.py"
    host = str(opt.get("host") or DEFAULT_HOST)
    port = str(int(opt.get("port") or DEFAULT_PORT))
    if not python_path.is_file():
        raise FileNotFoundError(
            f"SAM3 Python 不存在: {python_path}\n"
            "请安装 sam3 Conda 环境，或在 backends.json 的 sam3_local.options.python_path 中指定。"
        )
    if not server.is_file():
        raise FileNotFoundError(f"SAM3 worker 脚本不存在: {server}")

    WORKER_LOG.parent.mkdir(parents=True, exist_ok=True)
    extra_env: Dict[str, str] = {
        "SAM3_WORKER_HOST": host,
        "SAM3_WORKER_PORT": port,
        "SAM3_WORKER_RUNS": str(AUTO_LABEL_ROOT / "runs" / "sam3"),
    }
    proxy = str(opt.get("http_proxy") or "").strip()
    if proxy:
        extra_env["HTTP_PROXY"] = proxy
        extra_env["HTTPS_PROXY"] = proxy
        extra_env["http_proxy"] = proxy
        extra_env["https_proxy"] = proxy

    env = _worker_env(python_path, extra_env)
    cmd = [str(python_path), "-u", str(server), "--host", host, "--port", port]
    if log:
        log(f"启动 SAM3 Worker: {python_path} {server}")
        log(f"Worker 日志: {WORKER_LOG}")

    log_fh = open(WORKER_LOG, "ab", buffering=0)
    kwargs: Dict[str, Any] = {
        "cwd": str(AUTO_LABEL_ROOT),
        "env": env,
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = False
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    # Keep the handle alive on the Popen object so the log is not closed early.
    proc._autolabel_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def ensure_worker(
    options: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 180.0,
    log=None,
    autostart: bool = True,
) -> Dict[str, Any]:
    """Return health payload. Start the worker if /health is down.

    Raises RuntimeError with a clear message if the worker cannot become ready.
    """
    opt = options or {}
    url = worker_url(opt)
    ok, data = probe_health(url, timeout=2.0)
    if ok and data.get("model_loaded"):
        if log:
            log(f"SAM3 Worker 已就绪 ({url}, device={data.get('device') or '?'})")
        return data
    if ok and data.get("status") == "error" and data.get("error"):
        raise RuntimeError(f"SAM3 Worker 模型加载失败: {data.get('error')}")
    if ok and not autostart:
        return data
    if not autostart:
        raise RuntimeError(f"SAM3 Worker 未运行: {url}/health")

    proc = None
    if not ok:
        if log:
            log(f"SAM3 Worker 未就绪，正在启动… ({url})")
        try:
            proc = start_worker(opt, log=log)
        except Exception as e:
            raise RuntimeError(f"无法启动 SAM3 Worker: {e}") from e

        # Fail fast if the process dies before binding.
        for _ in range(20):
            time.sleep(0.25)
            if proc.poll() is not None:
                tail = _tail_log(WORKER_LOG)
                raise RuntimeError(
                    f"SAM3 Worker 进程已退出 (code={proc.returncode})。\n"
                    f"日志 {WORKER_LOG}:\n{tail or '(空)'}"
                )
            ok, data = probe_health(url, timeout=1.0)
            if ok:
                break
        if not ok:
            if proc.poll() is not None:
                tail = _tail_log(WORKER_LOG)
                raise RuntimeError(
                    f"SAM3 Worker 进程已退出 (code={proc.returncode})。\n"
                    f"日志 {WORKER_LOG}:\n{tail or '(空)'}"
                )
            if log:
                log("端口尚未打开，继续等待模型加载…")

    deadline = time.time() + max(30.0, float(timeout))
    last = data
    last_status = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _tail_log(WORKER_LOG)
            raise RuntimeError(
                f"SAM3 Worker 在加载过程中退出 (code={proc.returncode})。\n"
                f"日志 {WORKER_LOG}:\n{tail or '(空)'}"
            )
        time.sleep(1.5)
        ok, last = probe_health(url, timeout=3.0)
        if ok and last.get("model_loaded"):
            if log:
                log(f"SAM3 Worker 模型已加载 ({last.get('device') or ''})")
            return last
        if ok and last.get("status") == "error" and last.get("error"):
            raise RuntimeError(f"SAM3 Worker 模型加载失败: {last.get('error')}")
        status = str((last or {}).get("status") or "")
        if log and status and status != last_status:
            last_status = status
            if status == "loading":
                log("SAM3 模型加载中（首次可能下载权重，请等待）…")
            else:
                log(f"SAM3 Worker 状态: {status}")
    err = last.get("error") if isinstance(last, dict) else last
    tail = _tail_log(WORKER_LOG)
    extra = f"\n日志 {WORKER_LOG}:\n{tail}" if tail else f"\n日志文件: {WORKER_LOG}"
    raise RuntimeError(
        f"SAM3 Worker 在 {timeout:.0f}s 内未就绪 ({url}/health)。"
        f" 最后状态: {err or last}。"
        "可用 sam3 Conda Python 手动运行 AutoLabel/sam3_worker/server.py。"
        f"{extra}"
    )
