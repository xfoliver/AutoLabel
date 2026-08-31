"""Commit work JSON → dataset sidecar."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from autolabel.dataset import has_sidecar, sidecar_json_path, write_json


def commit_sidecars(
    mapping: Dict[Path, Path],
    *,
    overwrite_policy: str = "skip_existing",
) -> Tuple[List[str], List[str], List[str]]:
    """Copy work json paths → image sidecar.

    mapping: image_path -> work_json_path
    Returns (committed, skipped, errors)
    """
    committed: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []
    for image_path, work_json in mapping.items():
        image_path = Path(image_path)
        work_json = Path(work_json)
        dest = sidecar_json_path(image_path)
        try:
            if overwrite_policy == "skip_existing" and dest.exists():
                skipped.append(str(dest))
                continue
            if not work_json.exists():
                errors.append(f"missing work json: {work_json}")
                continue
            if overwrite_policy == "overwrite_backup" and dest.exists():
                bak = dest.with_suffix(dest.suffix + ".bak")
                shutil.copy2(dest, bak)
            shutil.copy2(work_json, dest)
            committed.append(str(dest))
        except Exception as e:
            errors.append(f"{dest}: {e}")
    return committed, skipped, errors


def load_commit_manifest(run_dir: Path) -> Dict[str, str]:
    """image_path_str -> work_json_str from report/commit_manifest.json"""
    import json

    path = Path(run_dir) / "report" / "commit_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in (data or {}).items()}


def save_commit_manifest(run_dir: Path, mapping: Dict[Path, Path]) -> Path:
    data = {str(k): str(v) for k, v in mapping.items()}
    return write_json(Path(run_dir) / "report" / "commit_manifest.json", data)
