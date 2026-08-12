"""
数据集文件系统单源：图片扩展名、训练集根探测、类别探测。

合并原 resolver._find_dataset_dir / data_quality.find_dataset_dir 两套逻辑，
供质检、解析器、（阶段 D 后）auto_train2 DVA 共用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# 图片扩展名（小写，带点）
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}

# 数据集子目录探测优先级（项目根下常见训练集文件夹名）
DATASET_CANDIDATES: List[str] = [
    "训练集",
    "1-训练集",
    "训练",
    "train",
    "dataset",
]

# 扫描时跳过的目录名（小写）
SKIP_DIR_NAMES: Set[str] = {
    "work",
    "checkpoints",
    "__pycache__",
    ".git",
    "node_modules",
    "baseline models",
    "models",
    "model",
    "输出",
    "模型输出",
    "模型",
    "annotations",
    "labels",
    "images",
    "image",
    "ok",
    "ng",
}

# 分类子目录探测时额外跳过
CLS_SKIP_DIR_NAMES: Set[str] = SKIP_DIR_NAMES | {
    "train",
    "val",
    "test",
    "valid",
    "validation",
    "训练集",
    "验证集",
    "测试集",
}

# 类别文件夹名黑名单（检测/分类探测）
CATEGORY_SKIP_NAMES: Set[str] = {
    "annotations",
    "labels",
    "image",
    "images",
}
# 注意：分类任务里 ok/ng 本身可能是合法类别名，不要全局跳过。


def is_image_file(path: Union[str, Path]) -> bool:
    p = Path(path)
    try:
        return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    except OSError:
        return False


def _has_direct_images(directory: Path) -> bool:
    try:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def _has_class_subdirs(directory: Path) -> bool:
    """是否存在「子文件夹内有图」的分类结构。"""
    try:
        for p in directory.iterdir():
            if not p.is_dir():
                continue
            if p.name.lower() in CLS_SKIP_DIR_NAMES or p.name in CLS_SKIP_DIR_NAMES:
                continue
            try:
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def count_images(
    directory: Union[str, Path],
    *,
    max_files: int = 0,
    recursive: bool = True,
) -> int:
    """统计目录下图片数；``max_files>0`` 时提前返回（存在性探测）。"""
    root = Path(directory)
    n = 0
    try:
        iterator = root.rglob("*") if recursive else root.iterdir()
        for p in iterator:
            try:
                if not p.is_file():
                    continue
                if recursive and any(part.lower() in SKIP_DIR_NAMES for part in p.parts):
                    continue
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    n += 1
                    if max_files and n >= max_files:
                        return n
            except OSError:
                continue
    except OSError:
        return n
    return n


def find_nested_train_subdir(folder: Union[str, Path]) -> Optional[Path]:
    """仅在 ``folder`` 下一层按 ``DATASET_CANDIDATES`` 找训练集子目录（可不含图检查）。

    兼容原 ``resolver._find_dataset_dir``：有同名子目录即返回。
    """
    root = Path(folder)
    try:
        if not root.is_dir():
            return None
    except OSError:
        return None
    for candidate in DATASET_CANDIDATES:
        cand_path = root / candidate
        try:
            if cand_path.is_dir():
                return cand_path
        except OSError:
            continue
    return None


def find_dataset_dir(path: Union[str, Path], *, require_images: bool = True) -> Path:
    """
    解析实际训练集目录。

    优先进入 DATASET_CANDIDATES（训练集/train/…），避免停在项目根
    （根下往往还有 work/模型，且分类的类文件夹在「训练集」内）。

    Parameters
    ----------
    require_images:
        True（默认，质检/解析）时，候选目录需能看到至少 1 张图才采纳。
        False 时只要候选子目录存在即返回（更接近旧 resolver 行为）。
    """
    root = Path(path).expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass

    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    def _ok(cand: Path) -> bool:
        if not require_images:
            return cand.is_dir()
        return cand.is_dir() and count_images(cand, max_files=1) > 0

    # 1) 显式训练子目录优先
    for name in DATASET_CANDIDATES:
        cand = root / name
        try:
            if _ok(cand):
                return cand
        except OSError:
            continue

    # 2) 根目录自身已是数据目录
    if not require_images:
        return root
    if _has_direct_images(root) or _has_class_subdirs(root):
        return root

    # 3) 再往下一层：如 5-ISEG/1-训练集、4-DETR/训练
    try:
        children = sorted(root.iterdir())
    except OSError:
        return root

    for child in children:
        try:
            if not child.is_dir():
                continue
            if child.name.lower() in SKIP_DIR_NAMES or child.name in SKIP_DIR_NAMES:
                continue
        except OSError:
            continue
        for name in DATASET_CANDIDATES:
            cand = child / name
            try:
                if _ok(cand):
                    return cand
            except OSError:
                continue
        if _has_direct_images(child) or _has_class_subdirs(child):
            return child
        if count_images(child, max_files=1) > 0:
            return child

    return root


def find_train_split_root(dataset_dir: Union[str, Path]) -> Path:
    """若目录下全是 train/val/test 划分，返回 train 子目录；否则返回自身。

    对应原 auto_train2 ``_find_train_root``。
    """
    root = Path(dataset_dir)
    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return root
    if not children:
        return root
    split_names = {"train", "val", "valid", "validation", "test"}
    if all(p.name.lower() in split_names for p in children) and len(children) <= 4:
        train_dir = root / "train"
        if train_dir.exists():
            return train_dir
    return root


def detect_categories(
    dataset_dir: Union[str, Path],
    *,
    with_counts: bool = False,
    use_train_split: bool = False,
) -> Union[Tuple[List[str], int], Tuple[Dict[str, int], Dict[str, str]]]:
    """探测数据集类别。

    默认（resolver 风格）::
        with_counts=False -> (categories: List[str], total_images: int)

    训练 DVA 风格::
        with_counts=True -> (category_dist: Dict[str,int], category_map: Dict[str,str])

    use_train_split=True 时先走 ``find_train_split_root``。
    """
    root = Path(dataset_dir)
    if use_train_split:
        root = find_train_split_root(root)

    categories: List[str] = []
    total = 0
    category_dist: Dict[str, int] = {}
    category_map: Dict[str, str] = {}

    if not root.exists():
        if with_counts:
            return {}, {}
        return [], 0

    try:
        subdirs = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        if with_counts:
            return {}, {}
        return [], 0

    for subdir in subdirs:
        name_l = subdir.name.lower()
        if name_l in CATEGORY_SKIP_NAMES or subdir.name in CATEGORY_SKIP_NAMES:
            continue
        try:
            # 默认只计一层（与 resolver/2.0 一致）；避免 annotations 深树拖慢
            count = sum(1 for f in subdir.iterdir() if is_image_file(f))
        except OSError:
            count = 0
        if count > 0:
            categories.append(subdir.name)
            total += count
            category_dist[subdir.name] = count
            category_map[subdir.name] = subdir.name

    if not categories:
        try:
            root_count = sum(1 for f in root.iterdir() if is_image_file(f))
        except OSError:
            root_count = 0
        if root_count > 0:
            total = root_count

    if with_counts:
        return category_dist, category_map
    return categories, total
