"""
AutoTrain 4.0 共享常量。

包含任务类型映射、模板配置文件、文件夹关键词、数据集子目录探测优先级等。
"""

from typing import Dict, List

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
    "record_frontend": True,
    "record_dir": "./record",
    "record_size": {"width": 1280, "height": 720},
}

# 图片扩展名，用于数据集类别自动检测
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
