"""Shared process bootstrap: dotenv + Windows UTF-8 stdout/stderr."""

from __future__ import annotations

import sys
from pathlib import Path


def load_dotenv_from_root(root: Path | None = None) -> None:
    """Load project-root .env if present; silently skip if python-dotenv missing."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    base = root or Path(__file__).resolve().parent.parent
    dotenv_path = base / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)


def ensure_utf8_stdio() -> None:
    """Keep Chinese logs readable on Windows Git Bash / CMD / PowerShell."""
    if sys.platform != "win32":
        return
    if getattr(sys.stdout, "encoding", None) in ("utf-8", "UTF-8"):
        return
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass


def bootstrap(root: Path | None = None) -> None:
    """Apply dotenv + UTF-8 stdio. Safe to call multiple times."""
    load_dotenv_from_root(root)
    ensure_utf8_stdio()
