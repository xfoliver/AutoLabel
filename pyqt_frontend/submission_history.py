"""训练需求提交历史：JSON 持久化（无 Qt 依赖）。

存储：``record/submission_history.json``（与 chat_history 同目录）。
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 200
HISTORY_VERSION = 1
STALE_RUNNING_HOURS = 12

_ALLOWED_UPDATE_KEYS = frozenset(
    {
        "status",
        "exit_code",
        "elapsed_s",
        "run_dir",
        "task_type_hint",
        "training_completed",
        "error_summary",
        "prompt",
        "submitted_at",
    }
)


def default_history_path() -> Path:
    """仓库根目录下的默认历史文件路径。"""
    return Path(__file__).resolve().parent.parent / "record" / "submission_history.json"


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _new_id(existing: Optional[Iterable[str]] = None) -> str:
    existing_set = set(existing or ())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(8):
        suffix = secrets.token_hex(2)
        candidate = f"sub_{stamp}_{suffix}"
        if candidate not in existing_set:
            return candidate
    return f"sub_{stamp}_{uuid.uuid4().hex[:8]}"


def _normalize_entry(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    prompt = str(raw.get("prompt") or "").strip()
    entry_id = str(raw.get("id") or "").strip()
    if not prompt or not entry_id:
        return None
    status = str(raw.get("status") or "done").strip() or "done"
    if status not in {"running", "done", "error"}:
        status = "done"
    exit_code = raw.get("exit_code", None)
    if exit_code is not None:
        try:
            exit_code = int(exit_code)
        except (TypeError, ValueError):
            exit_code = None
    elapsed_s = raw.get("elapsed_s", None)
    if elapsed_s is not None:
        try:
            elapsed_s = float(elapsed_s)
        except (TypeError, ValueError):
            elapsed_s = None
    return {
        "id": entry_id,
        "submitted_at": str(raw.get("submitted_at") or ""),
        "prompt": prompt,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": elapsed_s,
        "run_dir": str(raw.get("run_dir") or ""),
        "task_type_hint": str(raw.get("task_type_hint") or ""),
        "training_completed": bool(raw.get("training_completed", False)),
        "error_summary": str(raw.get("error_summary") or ""),
    }


def _mark_stale_running(entries: List[Dict[str, Any]], hours: float = STALE_RUNNING_HOURS) -> bool:
    """将超时仍为 running 的条目标为 error。返回是否有修改。"""
    if hours <= 0:
        return False
    changed = False
    cutoff = datetime.now() - timedelta(hours=hours)
    for entry in entries:
        if entry.get("status") != "running":
            continue
        submitted = str(entry.get("submitted_at") or "")
        try:
            ts = datetime.fromisoformat(submitted)
        except ValueError:
            ts = None
        if ts is None or ts < cutoff:
            entry["status"] = "error"
            if not entry.get("error_summary"):
                entry["error_summary"] = "进程异常中断或超时未更新"
            changed = True
    return changed


def load_history(
    path: Optional[Path] = None,
    *,
    mark_stale: bool = True,
) -> List[Dict[str, Any]]:
    """加载提交历史；文件缺失/损坏时返回空列表。"""
    history_path = Path(path) if path is not None else default_history_path()
    if not history_path.exists():
        return []
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取提交历史失败 %s: %s", history_path, exc)
        return []

    if isinstance(raw, list):
        entries_raw = raw
    elif isinstance(raw, dict):
        entries_raw = raw.get("entries") or []
    else:
        return []

    entries: List[Dict[str, Any]] = []
    for item in entries_raw:
        normalized = _normalize_entry(item)
        if normalized is not None:
            entries.append(normalized)

    if mark_stale and _mark_stale_running(entries):
        try:
            save_history(entries, path=history_path)
        except Exception as exc:
            logger.warning("写回陈旧 running 状态失败: %s", exc)
    return entries


def save_history(
    entries: List[Dict[str, Any]],
    path: Optional[Path] = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """保存历史，自动裁剪到 max_entries（保留最新）。"""
    history_path = Path(path) if path is not None else default_history_path()
    cleaned: List[Dict[str, Any]] = []
    for item in entries:
        normalized = _normalize_entry(item)
        if normalized is not None:
            cleaned.append(normalized)
    if max_entries > 0:
        cleaned = cleaned[:max_entries]
    payload = {
        "version": HISTORY_VERSION,
        "max_entries": max_entries,
        "entries": cleaned,
    }
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("写入提交历史失败 %s: %s", history_path, exc)


def add_submission(
    prompt: str,
    *,
    task_type_hint: str = "",
    path: Optional[Path] = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> Dict[str, Any]:
    """追加一条 running 状态的提交（插到列表头部）。"""
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt 不能为空")
    history_path = Path(path) if path is not None else default_history_path()
    entries = load_history(history_path, mark_stale=False)
    entry = {
        "id": _new_id(e.get("id", "") for e in entries),
        "submitted_at": _now_iso(),
        "prompt": text,
        "status": "running",
        "exit_code": None,
        "elapsed_s": None,
        "run_dir": "",
        "task_type_hint": (task_type_hint or "").strip(),
        "training_completed": False,
        "error_summary": "",
    }
    entries.insert(0, entry)
    save_history(entries, path=history_path, max_entries=max_entries)
    return dict(entry)


def update_submission(
    entry_id: str,
    path: Optional[Path] = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    **fields: Any,
) -> bool:
    """按 id 更新字段。成功返回 True。"""
    if not entry_id:
        return False
    history_path = Path(path) if path is not None else default_history_path()
    entries = load_history(history_path, mark_stale=False)
    found = False
    for entry in entries:
        if entry.get("id") != entry_id:
            continue
        for key, value in fields.items():
            if key not in _ALLOWED_UPDATE_KEYS:
                continue
            if key == "status" and value not in {"running", "done", "error"}:
                continue
            if key == "exit_code" and value is not None:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            if key == "elapsed_s" and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
            if key == "training_completed":
                value = bool(value)
            if key in {"prompt", "run_dir", "task_type_hint", "error_summary", "submitted_at"}:
                value = str(value or "")
            entry[key] = value
        found = True
        break
    if not found:
        return False
    save_history(entries, path=history_path, max_entries=max_entries)
    return True


def delete_submission(
    entry_id: str,
    path: Optional[Path] = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> bool:
    """删除一条历史。"""
    if not entry_id:
        return False
    history_path = Path(path) if path is not None else default_history_path()
    entries = load_history(history_path, mark_stale=False)
    new_entries = [e for e in entries if e.get("id") != entry_id]
    if len(new_entries) == len(entries):
        return False
    save_history(new_entries, path=history_path, max_entries=max_entries)
    return True


def clear_history(path: Optional[Path] = None) -> None:
    """清空全部历史。"""
    history_path = Path(path) if path is not None else default_history_path()
    save_history([], path=history_path)


def filter_submissions(
    entries: List[Dict[str, Any]],
    *,
    query: str = "",
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按关键字与状态过滤（不修改原列表）。"""
    q = (query or "").strip().lower()
    st = (status or "").strip().lower()
    if st in {"", "all", "全部"}:
        st = ""
    result: List[Dict[str, Any]] = []
    for entry in entries:
        if st and str(entry.get("status") or "").lower() != st:
            continue
        if q:
            hay = " ".join(
                [
                    str(entry.get("prompt") or ""),
                    str(entry.get("run_dir") or ""),
                    str(entry.get("task_type_hint") or ""),
                    str(entry.get("error_summary") or ""),
                    str(entry.get("id") or ""),
                ]
            ).lower()
            if q not in hay:
                continue
        result.append(entry)
    return result
