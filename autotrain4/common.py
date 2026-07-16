"""
AutoTrain 4.0 共享常量与工具函数。

包含任务类型映射、模板配置文件、文件夹关键词、数据集子目录探测优先级，
以及配置/模型路径查找等通用工具。
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# 任务类型映射
# ---------------------------------------------------------------------------
# 用户可能说的别名 -> 内部规范任务类型
TASK_ALIASES: Dict[str, str] = {
    # OCR
    "ocr": "OCR",
    "OCR": "OCR",
    # 分类
    "cls": "分类",
    "CLS": "分类",
    "分类": "分类",
    "classification": "分类",
    # 目标检测
    "det": "目标检测",
    "DET": "目标检测",
    "目标检测": "目标检测",
    "detection": "目标检测",
    # DETR（底层任务类型仍用 目标检测，但模板用 DETR 专用）
    "detr": "DETR",
    "DETR": "DETR",
    # 实例分割
    "iseg": "实例分割",
    "ISEG": "实例分割",
    "实例分割": "实例分割",
    "instance": "实例分割",
    # 语义分割
    "sseg": "语义分割",
    "SSEG": "语义分割",
    "语义分割": "语义分割",
    "semantic": "语义分割",
    # 旋转框检测
    "rotate": "旋转框检测",
    "ROTATE": "旋转框检测",
    "旋转框检测": "旋转框检测",
    "rotation": "旋转框检测",
}

# 规范任务类型 -> 模板配置文件
TASK_TEMPLATES: Dict[str, str] = {
    "OCR": "config/ocr_train_automation_config.json",
    "分类": "config/cls_train_automation_config.json",
    "目标检测": "config/det_train_automation_config.json",
    "DETR": "config/detr_train_automation_config.json",
    "实例分割": "config/iseg_train_automation_config.json",
    "语义分割": "config/sseg_train_automation_config.json",
    "旋转框检测": "config/rotate_train_automation_config.json",
}

# 规范任务类型 -> baseline 模型文件名前缀
BASELINE_PREFIXES: Dict[str, str] = {
    "OCR": "OCR",
    "分类": "CLS",
    "目标检测": "DET",
    "DETR": "DETR",
    "实例分割": "ISEG",
    "语义分割": "SSEG",
    "旋转框检测": "ROTATE",
}

# 用于匹配文件夹名的关键字（不区分大小写）
TASK_FOLDER_KEYWORDS: Dict[str, List[str]] = {
    "OCR": ["ocr"],
    "分类": ["cls", "分类", "classification"],
    "目标检测": ["det", "目标检测", "detection"],
    "DETR": ["detr"],
    "实例分割": ["iseg", "实例分割", "instance"],
    "语义分割": ["sseg", "语义分割", "semantic"],
    "旋转框检测": ["rotate", "旋转框检测", "rotation"],
}

# 数据集子目录探测优先级
DATASET_CANDIDATES = ["训练集", "1-训练集", "训练", "train", "dataset"]

# 默认 WebBridge 配置（Playwright，推荐）
DEFAULT_WEBBRIDGE = {
    "enabled": True,
    "adapter": "playwright",
    "frontend_url": "http://localhost:8843",
    "timeout": 120,
    "frontend_ready_timeout": 60,
    "frontend_ready_interval": 1,
    "screenshot_dir": "./webbridge_screenshots",
    "headless": False,
    "browser": "chromium",
    "slow_mo": 300,
    "step_delay": 1.0,
    "wait_loading_done": False,
    "post_click_delay": 2.0,
    "keep_browser_open": True,
    "record_video_dir": None,
    "record_video_size": {"width": 1280, "height": 720},
}

# 图片扩展名，用于数据集类别自动检测
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


# ---------------------------------------------------------------------------
# 通用路径工具
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """返回项目根目录（本文件位于 autotrain4/ 下）。"""
    return Path(__file__).parent.parent.resolve()


def config_dir() -> Path:
    """返回配置目录。"""
    return project_root() / "config"


def baseline_dir() -> Path:
    """返回 baseline 模型目录。"""
    return project_root() / "Baseline Models"


def resolve_path(raw: Union[str, Path]) -> Path:
    """解析路径：支持环境变量、相对路径转绝对路径。"""
    expanded = Path(os.path.expandvars(str(raw)))
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_root() / expanded).resolve()


def find_latest_config(task_type: str = "") -> Optional[Path]:
    """查找最新的 auto_latest_*.json 配置文件。"""
    cfg_dir = config_dir()
    if not cfg_dir.exists():
        return None
    candidates = list(cfg_dir.glob("auto_latest_*.json"))
    if not candidates:
        return None
    if task_type:
        safe = re.sub(r"[^\w\-]+", "_", task_type)
        matched = [p for p in candidates if safe in p.name]
        if matched:
            return max(matched, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_dvp(directory: Union[str, Path]) -> Optional[Path]:
    """查找目录下最新的 .dvp 文件。"""
    path = resolve_path(directory)
    if not path.exists():
        return None
    dvps = list(path.glob("*.dvp"))
    return max(dvps, key=lambda p: p.stat().st_mtime) if dvps else None


def find_baseline_model(task_type: str) -> Optional[Path]:
    """按任务类型在 Baseline Models 目录下找最新的 baseline .dvp。"""
    base = baseline_dir()
    if not base.exists():
        return None
    prefix = BASELINE_PREFIXES.get(task_type)
    if not prefix:
        return None
    candidates = [
        p for p in base.glob("*.dvp")
        if p.name.upper().startswith(prefix.upper() + "-")
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
