"""Run reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from autolabel.dataset import write_json
from autolabel.models import ReviewItem


def write_review_queue(run_dir: Path, items: List[ReviewItem]) -> Path:
    data = {"items": [it.to_dict() for it in items]}
    return write_json(Path(run_dir) / "report" / "review_queue.json", data)


def write_run_report(run_dir: Path, report: Dict[str, Any]) -> Path:
    return write_json(Path(run_dir) / "report" / "summary.json", report)


def format_summary_text(report: Dict[str, Any]) -> str:
    lines = [
        f"run_id: {report.get('run_id')}",
        f"processed: {report.get('processed', 0)}",
        f"empty: {report.get('empty', 0)}",
        f"buckets: {report.get('buckets', {})}",
        f"work_dir: {report.get('run_dir')}",
    ]
    return "\n".join(lines)
