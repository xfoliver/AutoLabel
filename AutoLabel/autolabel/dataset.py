"""Dataset image listing and run directory helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Prefer autotrain4 dataset_fs; fall back to local set
try:
    from autotrain4.dataset_fs import IMAGE_EXTENSIONS, find_dataset_dir, is_image_file
except Exception:  # pragma: no cover
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

    def is_image_file(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def find_dataset_dir(path, *, require_images: bool = True):
        return Path(path)


def list_images(dataset_dir: Path, *, recursive: bool = True) -> List[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        return []
    try:
        root = Path(find_dataset_dir(root, require_images=False))
    except Exception:
        pass
    paths: List[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and is_image_file(p):
                # skip work dirs
                parts_lower = {x.lower() for x in p.parts}
                if "work" in parts_lower or "__pycache__" in parts_lower:
                    continue
                paths.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and is_image_file(p):
                paths.append(p)
    paths.sort(key=lambda x: str(x).lower())
    return paths


def sidecar_json_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def has_sidecar(image_path: Path) -> bool:
    return sidecar_json_path(image_path).exists()


def filter_pending(
    images: Sequence[Path],
    *,
    overwrite_policy: str = "skip_existing",
) -> Tuple[List[Path], List[Path]]:
    """Return (pending, skipped)."""
    pending: List[Path] = []
    skipped: List[Path] = []
    for img in images:
        if overwrite_policy == "skip_existing" and has_sidecar(img):
            skipped.append(img)
        else:
            pending.append(img)
    return pending, skipped


def generate_run_id(prefix: str = "autolabel") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d_%H%M%S')}"


def create_run_dir(project_or_dataset: Path, run_id: Optional[str] = None) -> Path:
    """Create work/autolabel/<run_id>/ under project root (parent of 训练集 if needed)."""
    root = Path(project_or_dataset)
    # If inside a train split, lift to parent for work/
    name = root.name.lower()
    if name in {"训练集", "1-训练集", "训练", "train", "dataset"} and root.parent.exists():
        project = root.parent
    else:
        project = root
    rid = run_id or generate_run_id()
    run_dir = project / "work" / "autolabel" / rid
    for sub in ("meta", "preview", "stage1", "json", "report", "overlays"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
