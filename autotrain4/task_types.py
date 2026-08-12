"""
任务类型单源真相。

规范名（中文/OCR/DETR）用于配置、NLU、质检、auto_latest_*.json。
UI 短码（CLS/DET）与下拉 value（cls/det）只做展示/控件映射，禁止第二份手写别名表。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# 核心表
# ---------------------------------------------------------------------------

# 用户/LLM 可能说的别名 -> 内部规范任务类型
TASK_ALIASES: Dict[str, str] = {
    # OCR
    "ocr": "OCR",
    "OCR": "OCR",
    # 分类
    "cls": "分类",
    "CLS": "分类",
    "分类": "分类",
    "classification": "分类",
    "classify": "分类",
    "classifier": "分类",
    # 目标检测
    "det": "目标检测",
    "DET": "目标检测",
    "目标检测": "目标检测",
    "detection": "目标检测",
    "检测": "目标检测",
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
    "旋转框": "旋转框检测",
    "旋转检测": "旋转框检测",
}

# 规范任务类型 -> 模板配置文件（相对项目根）
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

# 规范名 -> UI 展示短码（指标栏 / 建议引擎）
TASK_UI_CODES: Dict[str, str] = {
    "OCR": "OCR",
    "分类": "CLS",
    "目标检测": "DET",
    "DETR": "DETR",
    "实例分割": "ISEG",
    "语义分割": "SSEG",
    "旋转框检测": "ROTATE",
}

# 规范名 -> 质检面板下拉 value（小写）
TASK_DROPDOWN_VALUES: Dict[str, str] = {
    "OCR": "ocr",
    "分类": "cls",
    "目标检测": "det",
    "DETR": "detr",
    "实例分割": "iseg",
    "语义分割": "sseg",
    "旋转框检测": "rotate",
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

# 建议引擎额外口语（不进 normalize 主表也可匹配展示）
_SUGGESTION_EXTRA: Dict[str, List[str]] = {
    "OCR": ["文字识别", "文本识别"],
    "分类": [],
    "目标检测": [],
    "DETR": [],
    "实例分割": [],
    "语义分割": [],
    "旋转框检测": ["旋转"],
}

CANONICAL_TASK_TYPES: Set[str] = set(TASK_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# 规范化 / 映射
# ---------------------------------------------------------------------------


def normalize_task_type(raw: Optional[str], *, unknown: str = "") -> str:
    """将用户/LLM/UI 输入规范为内部任务名。

    - 识别别名、UI 短码、下拉 value、规范中文名
    - 无法识别时：若 ``unknown`` 为 ``\"keep\"`` 则返回 strip 原文，否则返回 ``unknown``（默认空串）
    """
    if raw is None:
        return "" if unknown != "keep" else ""
    key = str(raw).strip()
    if not key:
        return "" if unknown != "keep" else ""

    if key in TASK_ALIASES:
        return TASK_ALIASES[key]

    low = key.lower()
    for alias, canon in TASK_ALIASES.items():
        if alias.lower() == low:
            return canon

    if key in CANONICAL_TASK_TYPES:
        return key
    for canon in CANONICAL_TASK_TYPES:
        if canon.lower() == low:
            return canon

    # UI 短码
    for canon, code in TASK_UI_CODES.items():
        if code.lower() == low:
            return canon

    if unknown == "keep":
        return key
    return unknown if unknown != "keep" else key


def to_ui_code(task_type: Optional[str]) -> str:
    """规范任务名 -> CLS/DET/…；无法识别返回空串。"""
    canon = normalize_task_type(task_type)
    if not canon:
        return ""
    return TASK_UI_CODES.get(canon, "")


def from_ui_code(code: Optional[str]) -> str:
    """CLS/DET/… 或 cls/det -> 规范任务名。"""
    return normalize_task_type(code)


def to_dropdown_value(task_type: Optional[str]) -> Optional[str]:
    """规范任务名/任意别名 -> 质检下拉 value（ocr/cls/…）。"""
    canon = normalize_task_type(task_type)
    if not canon:
        return None
    return TASK_DROPDOWN_VALUES.get(canon)


def guess_task_type_from_text(text: Optional[str]) -> Optional[str]:
    """从路径或自由文本子串猜测规范任务类型（较长关键字优先，减少误伤）。"""
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    # 直接整段规范化
    direct = normalize_task_type(raw)
    if direct:
        return direct

    low = raw.lower().replace("\\", "/")
    # 候选：(keyword_len, canon) 取最长命中
    hits: List[tuple] = []
    for alias, canon in TASK_ALIASES.items():
        a = alias.lower()
        if len(a) < 2:
            continue
        if a in low:
            hits.append((len(a), canon))
    for canon, kws in TASK_FOLDER_KEYWORDS.items():
        for kw in kws:
            k = kw.lower()
            if len(k) < 2:
                continue
            if k in low:
                hits.append((len(k), canon))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[0][1]


def ui_suggestion_aliases() -> Dict[str, List[str]]:
    """建议引擎用：UI 短码 -> 匹配用别名列表（小写/原文混合，调用方 lower 比较）。"""
    out: Dict[str, List[str]] = {}
    for canon, code in TASK_UI_CODES.items():
        tokens: List[str] = [code, code.lower()]
        # 所有指向该规范名的别名
        for alias, c in TASK_ALIASES.items():
            if c == canon:
                tokens.append(alias)
                tokens.append(alias.lower())
        tokens.extend(_SUGGESTION_EXTRA.get(canon, []))
        # 规范名本身
        tokens.append(canon)
        # 去重保序
        seen = set()
        uniq: List[str] = []
        for t in tokens:
            t = str(t).strip()
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(t if t != t.lower() and t.isupper() is False else t)
            # 统一存小写便于 in lower 匹配；同时保留中文原样
            if t.lower() not in seen:
                pass
        # 规范化：中文保留，英文小写
        cleaned: List[str] = []
        seen2 = set()
        for t in tokens:
            t = str(t).strip()
            if not t:
                continue
            item = t if any("一" <= ch <= "鿿" for ch in t) else t.lower()
            if item in seen2:
                continue
            seen2.add(item)
            cleaned.append(item)
        out[code] = cleaned
    return out


def all_alias_keys_lower() -> Set[str]:
    """供 NL 停用词等使用。"""
    keys = {a.lower() for a in TASK_ALIASES.keys()}
    keys |= {t.lower() for t in TASK_ALIASES.values()}
    keys |= {c.lower() for c in TASK_UI_CODES.values()}
    return keys


def task_dropdown_options() -> List[tuple]:
    """质检面板等：(显示标签, value) 列表。"""
    return [
        ("自动推断", ""),
        ("OCR (ocr)", "ocr"),
        ("分类 (cls)", "cls"),
        ("目标检测 (det)", "det"),
        ("DETR (detr)", "detr"),
        ("实例分割 (iseg)", "iseg"),
        ("语义分割 (sseg)", "sseg"),
        ("旋转框检测 (rotate)", "rotate"),
    ]
