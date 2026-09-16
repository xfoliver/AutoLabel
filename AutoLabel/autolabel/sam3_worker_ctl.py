"""Start / probe the resident SAM3 HTTP worker from AutoLabel's main process.

AutoLabel must never import sam3/torch. This module only uses subprocess + HTTP.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from autolabel.paths import (
    app_root,
    bpe_vocab_path,
    checkpoint_path,
    resolve_python_path,
    runs_dir,
    sam3_src_dir,
    worker_log_path,
    worker_script,
)

AUTO_LABEL_ROOT = app_root()
DEFAULT_PYTHON = r"C:\dlcv\envs\sam3\python.exe"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9892
WORKER_LOG = worker_log_path()

# PIDs started by this AutoLabel process.
_OWNED_WORKER_PIDS: List[int] = []
# Windows Job Object: GUI process death also kills assigned workers (task manager).
_KILL_ON_CLOSE_JOB = None
_ATEXIT_REGISTERED = False

# CREATE_NO_WINDOW keeps a GUI parent from flashing a console; we still capture
# stdout/stderr to a log file. DETACHED_PROCESS is avoided — it often kills
# conda python before it can bind the port.
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
_STOPPING_OWNED_WORKER = False


def _hidden_popen_kwargs() -> Dict[str, Any]:
    """Hide console windows when a windowed GUI spawns netstat / powershell / taskkill."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 1)
    startupinfo.wShowWindow = 0  # SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": CREATE_NO_WINDOW,
    }


def _run_hidden(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    extra = _hidden_popen_kwargs()
    extra.update(kwargs)
    return subprocess.run(args, **extra)


def worker_options(backends_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = backends_config or {}
    from autolabel.engines import ENGINE_SAM31, normalize_engine, sam3_version_for_engine

    engine = normalize_engine(cfg.get("annotation_engine") or cfg.get("active_backend"), cfg)
    version = sam3_version_for_engine(engine) or ENGINE_SAM31
    fallback: Dict[str, Any] = {}
    for entry in cfg.get("backends") or []:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("id") or "")
        etype = str(entry.get("type") or "")
        if eid == "sam3_1_local":
            opt = dict(entry.get("options") or {})
            opt["sam3_version"] = version
            return opt
        if not fallback and (eid == "sam3_local" or etype == "sam3_local"):
            fallback = dict(entry.get("options") or {})
    if fallback:
        fallback["sam3_version"] = version
        return fallback
    return {"sam3_version": version}


def worker_url(options: Optional[Dict[str, Any]] = None) -> str:
    opt = options or {}
    base = str(opt.get("base_url") or "").strip().rstrip("/")
    if base:
        return base
    host = str(opt.get("host") or DEFAULT_HOST).strip() or DEFAULT_HOST
    port = int(opt.get("port") or DEFAULT_PORT)
    return f"http://{host}:{port}"


def parse_netstat_listening_pids(text: str, port: int) -> List[int]:
    """Parse ``netstat -ano`` / ``ss`` style output for LISTENING PIDs on ``port``."""
    pids: List[int] = []
    needle = f":{int(port)}"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        joined = " ".join(parts).upper()
        if "LISTEN" not in joined:
            continue
        local = parts[1] if len(parts) > 1 else ""
        if needle not in local and not local.endswith(needle):
            continue
        host, _, port_txt = local.rpartition(":")
        if port_txt.rstrip("]") != str(int(port)):
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def pids_listening_on_port(port: int) -> List[int]:
    port = int(port)
    if os.name == "nt":
        try:
            completed = _run_hidden(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except Exception:
            return []
        return parse_netstat_listening_pids((completed.stdout or "") + (completed.stderr or ""), port)
    try:
        completed = _run_hidden(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    pids: List[int] = []
    for line in (completed.stdout or "").splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def process_command_line(pid: int) -> str:
    """Best-effort command line for ``pid`` (Windows CIM / Linux /proc)."""
    pid = int(pid)
    if pid <= 0:
        return ""
    if os.name == "nt":
        try:
            completed = _run_hidden(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                check=False,
            )
        except Exception:
            return ""
        return (completed.stdout or "").strip()
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def is_autolabel_sam3_worker(cmdline: str) -> bool:
    """True when the process is AutoLabel's ``sam3_worker/server.py``."""
    text = (cmdline or "").replace("\\", "/").lower()
    return "sam3_worker" in text and "server.py" in text


def _ensure_kill_on_close_job():
    """Create a Windows job so assigned children die with this process."""
    global _KILL_ON_CLOSE_JOB
    if os.name != "nt":
        return None
    if _KILL_ON_CLOSE_JOB:
        return _KILL_ON_CLOSE_JOB
    try:
        import ctypes
        from ctypes import wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            handle,
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(handle)
            return None
        _KILL_ON_CLOSE_JOB = handle
        return handle
    except Exception:
        return None


def assign_pid_to_kill_on_close_job(pid: int) -> bool:
    """Tie ``pid`` to this process; parent exit then kills the worker tree."""
    pid = int(pid)
    if pid <= 0 or os.name != "nt":
        return False
    job = _ensure_kill_on_close_job()
    if not job:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not handle:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(job, handle))
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def _register_atexit_stop() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(lambda: stop_owned_worker(from_atexit=True))
    _ATEXIT_REGISTERED = True


def kill_pid(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            completed = _run_hidden(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            return completed.returncode == 0
        os.kill(pid, 9)
        return True
    except Exception:
        return False


def remember_owned_pid(pid: int) -> None:
    pid = int(pid)
    if pid > 0 and pid not in _OWNED_WORKER_PIDS:
        _OWNED_WORKER_PIDS.append(pid)


def owned_worker_pids() -> List[int]:
    return list(_OWNED_WORKER_PIDS)


def forget_owned_pid(pid: int) -> None:
    pid = int(pid)
    while pid in _OWNED_WORKER_PIDS:
        _OWNED_WORKER_PIDS.remove(pid)


def _wait_port_free(port: int, url: str, *, seconds: float = 15.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        ok, _ = probe_health(url, timeout=1.0)
        if not ok and not pids_listening_on_port(port):
            return
        time.sleep(0.4)


def stop_worker(options: Optional[Dict[str, Any]] = None, *, log=None) -> None:
    """Kill whatever is listening on the SAM3 worker port (typically 9892)."""
    opt = options or {}
    port = int(opt.get("port") or DEFAULT_PORT)
    url = worker_url(opt)
    pids = pids_listening_on_port(port)
    if log:
        if pids:
            log(f"停止旧 SAM3 Worker: port={port} pid={pids}")
        else:
            log(f"未找到占用 {port} 的 SAM3 Worker 进程")
    for pid in pids:
        kill_pid(pid)
        forget_owned_pid(pid)
    _wait_port_free(port, url)


def stop_owned_worker(
    options: Optional[Dict[str, Any]] = None,
    *,
    log=None,
    from_atexit: bool = False,
) -> None:
    """Stop AutoLabel's SAM3 Worker so GPU memory is released with the GUI.

    Kills PIDs this process started, plus any leftover ``sam3_worker/server.py``
    still listening on the worker port (orphans after a crash). Other programs
    on that port are left alone.
    """
    global _STOPPING_OWNED_WORKER
    if _STOPPING_OWNED_WORKER:
        return
    owned = owned_worker_pids()
    if from_atexit and not owned:
        # closeEvent already ran the netstat / powershell / taskkill cleanup.
        return
    opt = options or {}
    port = int(opt.get("port") or DEFAULT_PORT)
    url = worker_url(opt)
    _STOPPING_OWNED_WORKER = True
    try:
        listening = pids_listening_on_port(port)
        targets: List[int] = []
        for pid in owned:
            if pid > 0 and pid not in targets:
                targets.append(pid)
        for pid in listening:
            if pid in targets:
                continue
            if is_autolabel_sam3_worker(process_command_line(pid)):
                targets.append(pid)
        if not targets:
            if log:
                log("没有需要停止的 SAM3 Worker")
            return
        if log:
            log(f"停止 SAM3 Worker（释放显存）: pid={targets}")
        for pid in targets:
            kill_pid(pid)
            forget_owned_pid(pid)
        _wait_port_free(port, url)
    finally:
        _STOPPING_OWNED_WORKER = False


def probe_refine_supported(url: str, health: Optional[Dict[str, Any]] = None, *, timeout: float = 2.0) -> bool:
    """True when the live worker can cut a mask from a box (interactive refine)."""
    info = health if isinstance(health, dict) else {}
    mode = str(info.get("refine_mode") or "").strip().lower()
    version = str(info.get("api_version") or "").strip()
    parts = (0, 0)
    if version:
        try:
            parts = tuple(int(x) for x in version.split(".")[:2])
        except ValueError:
            parts = (0, 0)
    # 0.3 advertised interactive refine but the tracker backbone was still None.
    if parts < (0, 4):
        return False
    if mode == "interactive":
        return True
    if mode in {"geometric", "broken"}:
        return False
    endpoints = info.get("endpoints")
    if isinstance(endpoints, list) and "/refine" in endpoints:
        return True
    if not version:
        return False
    try:
        resp = requests.post(f"{url.rstrip('/')}/refine", json={}, timeout=timeout)
    except Exception:
        return False
    if resp.status_code == 404:
        return False
    return True


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


def _worker_env(
    python_path: Path,
    extra: Optional[Dict[str, str]] = None,
    *,
    sam3_version: Optional[str] = None,
) -> Dict[str, str]:
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
    label_root = app_root()
    pythonpath = [str(label_root)]
    sam3_repo = sam3_src_dir()
    if sam3_repo is not None:
        pythonpath.append(str(sam3_repo))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    from autolabel.engines import ENGINE_SAM31, sam3_version_for_engine

    version = sam3_version_for_engine(sam3_version or env.get("SAM3_VERSION"))
    env["SAM3_VERSION"] = version
    ckpt = checkpoint_path(version)
    if ckpt is not None and ckpt.is_file():
        env["SAM3_CHECKPOINT"] = str(ckpt)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env.pop("SAM3_CHECKPOINT", None)
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        if version == ENGINE_SAM31:
            env["SAM3_HF_VERSION"] = ENGINE_SAM31
    bpe = bpe_vocab_path()
    if bpe is not None:
        env["SAM3_BPE"] = str(bpe)
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
    python_path = resolve_python_path(opt.get("python_path"))
    server = worker_script()
    host = str(opt.get("host") or DEFAULT_HOST)
    port = str(int(opt.get("port") or DEFAULT_PORT))
    if not python_path.is_file():
        raise FileNotFoundError(
            f"SAM3.1 Python 不存在: {python_path}\n"
            "便携版请确认 runtime/sam3/python.exe；开发机请安装 sam3 Conda，"
            "或在 backends.json 的 sam3_1_local.options.python_path 中指定。"
        )
    if not server.is_file():
        raise FileNotFoundError(f"SAM3 worker 脚本不存在: {server}")

    log_path = worker_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    extra_env: Dict[str, str] = {
        "SAM3_WORKER_HOST": host,
        "SAM3_WORKER_PORT": port,
        "SAM3_WORKER_RUNS": str(runs_dir()),
    }
    proxy = str(opt.get("http_proxy") or "").strip()
    if proxy:
        extra_env["HTTP_PROXY"] = proxy
        extra_env["HTTPS_PROXY"] = proxy
        extra_env["http_proxy"] = proxy
        extra_env["https_proxy"] = proxy

    from autolabel.engines import sam3_version_for_engine
    from autolabel.paths import checkpoint_filename, checkpoint_search_paths

    version = sam3_version_for_engine(opt.get("sam3_version"))
    extra_env["SAM3_VERSION"] = version
    env = _worker_env(python_path, extra_env, sam3_version=version)
    cmd = [str(python_path), "-u", str(server), "--host", host, "--port", port]
    ckpt = env.get("SAM3_CHECKPOINT")
    if not ckpt:
        name = checkpoint_filename(version)
        searched = "\n".join(f"  - {p}" for p in checkpoint_search_paths(version))
        raise FileNotFoundError(
            f"未找到本地 {version} 权重（{name}）。\n"
            f"已搜索:\n{searched or '  (无)'}\n"
            "请把权重放到以上任一 models/ 目录后重试。"
            "Worker 不会自动从 Hugging Face 下载 3GB，否则界面会长时间停在试标 0%。"
        )
    if log:
        log(f"启动 SAM3 Worker ({version}): {python_path} {server}")
        log(f"Worker 日志: {log_path}")
        log(f"本地权重: {ckpt}")

    log_fh = open(log_path, "ab", buffering=0)
    kwargs: Dict[str, Any] = {
        "cwd": str(app_root()),
        "env": env,
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        hidden = _hidden_popen_kwargs()
        kwargs["startupinfo"] = hidden.get("startupinfo")
        kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = False
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    # Keep the handle alive on the Popen object so the log is not closed early.
    proc._autolabel_log_fh = log_fh  # type: ignore[attr-defined]
    if proc.pid:
        remember_owned_pid(int(proc.pid))
        assign_pid_to_kill_on_close_job(int(proc.pid))
    _register_atexit_stop()
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
    from autolabel.engines import sam3_version_for_engine

    want_version = sam3_version_for_engine(opt.get("sam3_version"))
    ok, data = probe_health(url, timeout=2.0)
    stale = False
    if ok and data.get("model_loaded"):
        live_version = str(data.get("sam3_version") or "").strip()
        if live_version and live_version != want_version:
            stale = True
            if log:
                log(f"SAM3 Worker 当前是 {live_version}，需要 {want_version}，正在重启…")
        elif probe_refine_supported(url, data):
            if log:
                label = "SAM 3.1" if want_version == "sam3.1" else "SAM3"
                log(f"{label} Worker 已就绪 ({url}, device={data.get('device') or '?'})")
            return data
        else:
            stale = True
            if log:
                log("SAM3 Worker 过旧（refine 不是交互式 box 切 mask），正在重启…")
    if ok and data.get("status") == "error" and data.get("error"):
        raise RuntimeError(f"SAM3 Worker 模型加载失败: {data.get('error')}")
    if ok and not autostart and not stale:
        return data
    if not autostart:
        raise RuntimeError(f"SAM3 Worker 未运行: {url}/health")

    proc = None
    restarted_for_refine = stale
    if (not ok) or stale:
        if stale:
            stop_worker(opt, log=log)
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
                tail = _tail_log(worker_log_path())
                raise RuntimeError(
                    f"SAM3 Worker 进程已退出 (code={proc.returncode})。\n"
                    f"日志 {worker_log_path()}:\n{tail or '(空)'}"
                )
            ok, data = probe_health(url, timeout=1.0)
            if ok:
                break
        if not ok:
            if proc.poll() is not None:
                tail = _tail_log(worker_log_path())
                raise RuntimeError(
                    f"SAM3 Worker 进程已退出 (code={proc.returncode})。\n"
                    f"日志 {worker_log_path()}:\n{tail or '(空)'}"
                )
            if log:
                log("端口尚未打开，继续等待模型加载…")

    deadline = time.time() + max(30.0, float(timeout))
    last = data
    last_status = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _tail_log(worker_log_path())
            raise RuntimeError(
                f"SAM3 Worker 在加载过程中退出 (code={proc.returncode})。\n"
                f"日志 {worker_log_path()}:\n{tail or '(空)'}"
            )
        time.sleep(1.5)
        ok, last = probe_health(url, timeout=3.0)
        if ok and last.get("model_loaded"):
            live_version = str(last.get("sam3_version") or "").strip()
            version_ok = (not live_version) or live_version == want_version
            if version_ok and (restarted_for_refine or probe_refine_supported(url, last)):
                if log:
                    log(f"SAM3 Worker 模型已加载 ({last.get('device') or ''} {live_version or want_version})")
                return last
            if live_version and live_version != want_version:
                restarted_for_refine = True
                if log:
                    log(f"SAM3 Worker 版本是 {live_version}，需要 {want_version}，正在重启…")
                stop_worker(opt, log=log)
                proc = start_worker(opt, log=log)
                continue
            restarted_for_refine = True
            if log:
                log("SAM3 Worker 过旧（refine 不是交互式 box 切 mask），正在重启…")
            stop_worker(opt, log=log)
            proc = start_worker(opt, log=log)
            continue
        if ok and last.get("status") == "error" and last.get("error"):
            raise RuntimeError(f"SAM3 Worker 模型加载失败: {last.get('error')}")
        status = str((last or {}).get("status") or "")
        if log and status and status != last_status:
            last_status = status
            if status == "loading":
                log("SAM3 模型加载中，请等待…")
            else:
                log(f"SAM3 Worker 状态: {status}")
    err = last.get("error") if isinstance(last, dict) else last
    log_path = worker_log_path()
    tail = _tail_log(log_path)
    extra = f"\n日志 {log_path}:\n{tail}" if tail else f"\n日志文件: {log_path}"
    raise RuntimeError(
        f"SAM3 Worker 在 {timeout:.0f}s 内未就绪 ({url}/health)。"
        f" 最后状态: {err or last}。"
        "可用 sam3 Python 手动运行 sam3_worker/server.py。"
        f"{extra}"
    )
