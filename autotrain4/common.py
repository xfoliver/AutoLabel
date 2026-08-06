"""
AutoTrain 4.0 共享常量与工具函数。

包含任务类型映射、模板配置文件、文件夹关键词、数据集子目录探测优先级，
以及配置/模型路径查找等通用工具。
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

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


def normalize_scan_dirs(
    *values: Union[str, Path, None, List[Union[str, Path]], tuple],
    default: str = "AI训练平台测试",
) -> List[Path]:
    """规范化一个或多个数据集扫描根目录。

    支持：
    - 单个路径字符串 / Path
    - 多个路径（可变参数或 list/tuple）
    - 同一字符串内用 ``;`` / ``|`` / 换行分隔的多根
    - 重复路径去重（按 resolve 后的绝对路径），保持首次出现顺序

    未提供任何有效输入时，回退到 ``default``（相对项目根解析）。
    """
    tokens: List[str] = []

    def _consume(item: Union[str, Path, None, List, tuple]) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                _consume(sub)
            return
        text = str(item).strip()
        if not text:
            return
        # 同一参数内允许多根：Z:\\data;Y:\\data 或 Z:\\data|Y:\\data
        parts = re.split(r"[;|\n\r]+", text)
        for part in parts:
            part = part.strip()
            if part:
                tokens.append(part)

    for value in values:
        _consume(value)

    if not tokens:
        tokens = [default]

    resolved: List[Path] = []
    seen = set()
    for token in tokens:
        path = resolve_path(token)
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


# 从自然语言中识别文件系统路径时去掉的尾部标点
_PATH_TRAILING_JUNK = " \t\r\n\"'“”‘’，,。．.；;：:、)）]】>》"

# 路径前后更像“模型输出/保存位置”的提示词
_OUTPUT_PATH_MARKERS = (
    "模型输出",
    "输出到",
    "输出至",
    "输出目录",
    "保存到",
    "保存至",
    "导出到",
    "导出至",
    "output",
    "save to",
    "export to",
)

# 路径前后更像“训练集/数据位置”的提示词
_DATASET_PATH_MARKERS = (
    "训练集",
    "训练数据",
    "数据集",
    "数据目录",
    "数据路径",
    "样本",
    "dataset",
    "train set",
    "training set",
    "data dir",
    "data path",
)


def _clean_extracted_path(raw: str) -> str:
    """去掉提取路径两端的引号与中英文标点。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    # 循环剥离尾部标点，避免 "Z:\\data。" 这类情况
    while text and text[-1] in _PATH_TRAILING_JUNK:
        text = text[:-1].rstrip()
    while text and text[0] in "\"'“”‘’":
        text = text[1:].lstrip()
    return text.strip()


def extract_filesystem_paths(text: str) -> List[str]:
    """从文本中提取 Windows 绝对路径 / UNC / 常见 POSIX 绝对路径。

    用于提示词里直接给出 ``Z:\\项目\\训练集`` 时的路径直达，不依赖 scan_dir。

    无引号路径在空白与中文标点处截断；引号内路径可含空格。
    """
    if not text:
        return []

    # 路径段：不含空白、引号、Windows 非法字符、以及常见中文/英文断句标点
    _seg = r'[^\s\\/:*?"<>|\'"，。；：、（）【】《》]+'

    patterns = [
        # 引号包裹（可含空格）
        r'["\']([A-Za-z]:\\[^"\']+)["\']',
        r'["\'](\\\\[^"\']+)["\']',
        r'["\'](/[^"\']+)["\']',
        # 无引号 Windows 盘符路径
        rf'(?<![A-Za-z0-9])([A-Za-z]:\\(?:{_seg}\\)*{_seg})',
        # 无引号 UNC
        rf'(?<![A-Za-z0-9])(\\\\{_seg}\\(?:{_seg}\\)*{_seg})',
    ]

    found: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            cleaned = _clean_extracted_path(match.group(1))
            if not cleaned:
                continue
            # 至少要有一个路径分隔符，避免误伤 "C:" 之类
            if "\\" not in cleaned and not cleaned.startswith("/"):
                continue
            # 基准模型文件不是数据集目录
            if cleaned.lower().endswith(".dvp"):
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(cleaned)
    return found


def looks_like_filesystem_path(raw: str) -> bool:
    """判断字符串是否像文件系统路径（绝对路径或明显的相对路径片段）。"""
    text = _clean_extracted_path(raw)
    if not text:
        return False
    if text.lower().endswith(".dvp"):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return True
    if text.startswith("\\\\"):
        return True
    if text.startswith("/") and not text.startswith("//"):
        # 简单排除 URL
        if re.match(r"^//", text) or "://" in text:
            return False
        return True
    if "\\" in text or "/" in text:
        return True
    return False


def coerce_existing_dir(
    raw: Union[str, Path, None],
    *search_roots: Union[str, Path, None],
) -> Optional[Path]:
    """把用户/LLM 给的路径字符串解析为已存在的目录。

    尝试顺序：原样绝对路径 → 各 search_root 拼接 → 项目根相对路径。
    """
    if raw is None:
        return None
    text = _clean_extracted_path(str(raw))
    if not text:
        return None

    expanded = Path(os.path.expandvars(text)).expanduser()
    candidates: List[Path] = []
    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        for root in search_roots:
            if root is None:
                continue
            candidates.append(Path(root) / expanded)
        candidates.append(project_root() / expanded)
        candidates.append(expanded)

    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved.exists() and resolved.is_dir():
            return resolved
    return None


def _path_context_window(prompt: str, path: str, radius: int = 24) -> str:
    """取路径在提示词中前后一段文本，用于判断是训练集还是输出目录。"""
    idx = prompt.find(path)
    if idx < 0:
        # 容错：去掉末尾标点再找一次
        cleaned = _clean_extracted_path(path)
        idx = prompt.find(cleaned)
        if idx < 0:
            return ""
        path = cleaned
    start = max(0, idx - radius)
    end = min(len(prompt), idx + len(path) + radius)
    return prompt[start:end]


def select_dataset_path_from_prompt(prompt: str) -> Optional[str]:
    """从提示词提取的多条路径中，选出最像训练集的那条。"""
    paths = extract_filesystem_paths(prompt)
    if not paths:
        return None

    scored: List[tuple] = []
    for path in paths:
        ctx = _path_context_window(prompt, path).lower()
        name = Path(_clean_extracted_path(path)).name.lower()
        score = 0
        if any(m.lower() in ctx for m in _DATASET_PATH_MARKERS):
            score += 5
        if any(m.lower() in name for m in ("训练", "训练集", "train", "dataset", "data")):
            score += 3
        if any(m.lower() in ctx for m in _OUTPUT_PATH_MARKERS):
            score -= 5
        if any(m.lower() in name for m in ("模型", "model", "output", "export")):
            score -= 3
        # 已存在的目录略优先
        try:
            if Path(os.path.expandvars(path)).expanduser().exists():
                score += 1
        except OSError:
            pass
        scored.append((score, path))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path = scored[0]
    # 若最高分明显是输出目录且没有正分，则放弃
    if best_score < 0:
        return None
    return _clean_extracted_path(best_path)


def select_output_path_from_prompt(prompt: str) -> Optional[str]:
    """从提示词中识别模型输出目录（若有）。"""
    paths = extract_filesystem_paths(prompt)
    if not paths:
        return None
    dataset = select_dataset_path_from_prompt(prompt)
    best: Optional[str] = None
    best_score = 0
    for path in paths:
        if dataset and _clean_extracted_path(path).lower() == _clean_extracted_path(dataset).lower():
            continue
        ctx = _path_context_window(prompt, path).lower()
        name = Path(_clean_extracted_path(path)).name.lower()
        score = 0
        if any(m.lower() in ctx for m in _OUTPUT_PATH_MARKERS):
            score += 5
        if any(m.lower() in name for m in ("模型", "model", "output", "export")):
            score += 3
        if score > best_score:
            best_score = score
            best = _clean_extracted_path(path)
    return best if best_score > 0 else None


def resolve_output_dir(raw: Union[str, Path, None]) -> Optional[Path]:
    """解析用户指定的模型输出目录；目录可不存在，会在使用时创建。

    与数据集路径不同：输出目录允许预先不存在。
    """
    if raw is None:
        return None
    text = _clean_extracted_path(str(raw))
    if not text:
        return None
    expanded = Path(os.path.expandvars(text)).expanduser()
    if expanded.is_absolute():
        try:
            return expanded.resolve()
        except OSError:
            return expanded
    # 相对路径：相对项目根
    try:
        return (project_root() / expanded).resolve()
    except OSError:
        return project_root() / expanded


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
