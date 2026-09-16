"""Runtime path helpers for development and the portable (frozen) layout.

Frozen onedir: ``sys.executable`` sits in ``AutoLabel_Portable/``.
Development: this file lives in ``AutoLabel/autolabel/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

DEV_CONDA_PYTHON = r"C:\dlcv\envs\sam3\python.exe"
PORTABLE_PYTHON_REL = Path("runtime") / "sam3" / "python.exe"
PORTABLE_SAM3_SRC_REL = Path("sam3_src")
PORTABLE_CHECKPOINT_REL = Path("models") / "sam3.pt"
PORTABLE_CHECKPOINT_SAM31_REL = Path("models") / "sam3.1_multiplex.pt"
PORTABLE_WORKER_REL = Path("sam3_worker") / "server.py"
SAM3_VERSION_SAM3 = "sam3"
SAM3_VERSION_SAM31 = "sam3.1"
SAM3_VERSIONS = (SAM3_VERSION_SAM3, SAM3_VERSION_SAM31)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Portable folder (next to AutoLabel.exe) or the AutoLabel source tree."""
    override = (os.environ.get("AUTOLABEL_ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    """Read-only resource. Prefer PyInstaller ``_MEIPASS``, else ``app_root``."""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        packed = Path(getattr(sys, "_MEIPASS")).joinpath(*parts)
        if packed.exists():
            return packed
    return app_root().joinpath(*parts)


def user_data_dir() -> Path:
    """Writable data. Frozen builds use ``%APPDATA%\\AutoLabel``."""
    override = (os.environ.get("AUTOLABEL_USER_DATA") or "").strip()
    if override:
        dest = Path(override).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    if is_frozen():
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        dest = Path(base) / "AutoLabel"
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    dest = app_root()
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def config_dir() -> Path:
    if is_frozen():
        dest = user_data_dir() / "config"
    else:
        dest = app_root() / "config"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def bundled_config_dir() -> Path:
    """Templates shipped next to the exe (or AutoLabel/config in dev)."""
    bundled = app_root() / "config"
    if bundled.is_dir():
        return bundled
    return resource_path("config")


def settings_path() -> Path:
    return config_dir() / "settings.json"


def backends_path() -> Path:
    return config_dir() / "backends.json"


def ensure_user_config() -> None:
    """Copy ``*.example.json`` into the writable config dir if missing."""
    dest = config_dir()
    src_dir = bundled_config_dir()
    mapping = {
        "settings.json": "settings.example.json",
        "backends.json": "backends.example.json",
    }
    for dest_name, example_name in mapping.items():
        target = dest / dest_name
        if target.exists():
            continue
        example = src_dir / example_name
        if not example.is_file():
            example = app_root() / "config" / example_name
        if example.is_file():
            target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")


def runs_dir() -> Path:
    dest = user_data_dir() / "runs" / "sam3"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def worker_log_path() -> Path:
    return runs_dir() / "worker.log"


def worker_script() -> Path:
    env = (os.environ.get("SAM3_WORKER_SCRIPT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    portable = app_root() / PORTABLE_WORKER_REL
    if portable.is_file():
        return portable
    return app_root() / "sam3_worker" / "server.py"


def sam3_src_dir() -> Optional[Path]:
    """Official SAM3 checkout (repo root that contains the ``sam3`` package)."""
    env = (os.environ.get("SAM3_ROOT") or "").strip()
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(app_root() / PORTABLE_SAM3_SRC_REL)
    candidates.append(app_root() / "sam3")
    if not is_frozen():
        candidates.append(app_root().parent / "sam3")
    for raw in candidates:
        path = raw
        if not path.is_dir():
            continue
        if (path / "sam3" / "model_builder.py").is_file() or (path / "model_builder.py").is_file():
            return path.resolve()
    return None


def bpe_vocab_path() -> Optional[Path]:
    env = (os.environ.get("SAM3_BPE") or "").strip()
    if env:
        path = Path(env).expanduser()
        return path.resolve() if path.is_file() else None
    src = sam3_src_dir()
    if src is None:
        return None
    for cand in (
        src / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        src / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ):
        if cand.is_file():
            return cand.resolve()
    return None


def normalize_sam3_version(raw: Optional[str] = None) -> str:
    """AutoLabel only ships SAM 3.1; retired SAM3 names still resolve to 3.1."""
    del raw
    return SAM3_VERSION_SAM31


def checkpoint_rel(version: Optional[str] = None) -> Path:
    if normalize_sam3_version(version) == SAM3_VERSION_SAM31:
        return PORTABLE_CHECKPOINT_SAM31_REL
    return PORTABLE_CHECKPOINT_REL


def checkpoint_filename(version: Optional[str] = None) -> str:
    return checkpoint_rel(version).name


def checkpoint_search_paths(version: Optional[str] = None) -> List[Path]:
    """Places we look for ``sam3.pt`` / ``sam3.1_multiplex.pt``.

    Dev layout keeps the 3GB weights next to the repo (``kimi_gui_automation/models``),
    not inside ``AutoLabel/models``. Portable layout uses ``<app>/models``.
    """
    ver = normalize_sam3_version(version or os.environ.get("SAM3_VERSION"))
    name = checkpoint_filename(ver)
    roots: List[Path] = [app_root()]
    if not is_frozen():
        parent = app_root().parent
        roots.append(parent)
        roots.append(parent / "dist" / "AutoLabel_Portable")
    src = sam3_src_dir()
    if src is not None:
        roots.append(src.parent)
        roots.append(src.parent / "AutoLabel")
        roots.append(src.parent / "dist" / "AutoLabel_Portable")
    out: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        cand = (Path(root) / "models" / name).expanduser()
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def checkpoint_path(version: Optional[str] = None) -> Optional[Path]:
    """Resolve a local SAM3 / SAM 3.1 weight file.

    Env ``SAM3_CHECKPOINT`` still wins (used by the worker process after the
    parent injects the chosen file). Otherwise search known ``models/`` folders.
    """
    env = (os.environ.get("SAM3_CHECKPOINT") or "").strip()
    if env:
        path = Path(env).expanduser()
        return path.resolve() if path.exists() else path
    ver = normalize_sam3_version(version or os.environ.get("SAM3_VERSION"))
    for cand in checkpoint_search_paths(ver):
        if cand.is_file():
            return cand.resolve()
    return None


def resolve_python_path(configured: Optional[str] = None) -> Path:
    """First existing interpreter: env, config, portable runtime, then conda."""
    env = (os.environ.get("SAM3_PYTHON") or "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    text = str(configured or "").strip()
    if text:
        raw = Path(text).expanduser()
        if not raw.is_absolute():
            raw = app_root() / raw
        candidates.append(raw)
    candidates.append(app_root() / PORTABLE_PYTHON_REL)
    candidates.append(Path(DEV_CONDA_PYTHON))

    seen: set[str] = set()
    existing: list[Path] = []
    missing: list[Path] = []
    for cand in candidates:
        try:
            key = str(cand)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            existing.append(cand)
        else:
            missing.append(cand)
    if existing:
        return existing[0].resolve()
    if is_frozen():
        return (app_root() / PORTABLE_PYTHON_REL)
    if text:
        raw = Path(text).expanduser()
        if not raw.is_absolute():
            raw = app_root() / raw
        return raw
    return Path(DEV_CONDA_PYTHON)
