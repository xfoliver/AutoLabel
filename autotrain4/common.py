"""
AutoTrain 4.0 兼容门面 + 路径 / NL 定位工具。

**权威实现（新代码请直接 import）**
  - ``autotrain4.task_types`` — 任务别名 / UI 码 / baseline 前缀
  - ``autotrain4.dataset_fs`` — 数据集根 / 类别 / 图片扩展名
  - ``autotrain4.constants`` — 默认 frontend/backend/inference URL

**本模块仍拥有的实现**（路径解析、扫描根、NL 线索、latest config/dvp/baseline）
  - ``project_root`` / ``resolve_path`` / ``normalize_scan_dirs`` / …
  - ``extract_filesystem_paths`` / ``extract_location_clues`` / …
  - ``find_latest_config`` / ``find_latest_dvp`` / ``find_baseline_model``
  - ``DEFAULT_WEBBRIDGE``（``frontend_url`` 对齐 constants）

对 task_types / dataset_fs / constants 的符号仅做 **re-export**，不维护第二份表。
今年不强制全仓改 import；新代码优先直连权威模块。
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

from .constants import DEFAULT_BACKEND_URL, DEFAULT_FRONTEND_URL, DEFAULT_INFERENCE_URL
from .dataset_fs import (
    DATASET_CANDIDATES,
    IMAGE_EXTENSIONS,
    detect_categories,
    find_dataset_dir,
    find_nested_train_subdir,
    find_train_split_root,
    is_image_file,
)
from .task_types import (
    BASELINE_PREFIXES,
    TASK_ALIASES,
    TASK_FOLDER_KEYWORDS,
    TASK_TEMPLATES,
    TASK_UI_CODES,
    all_alias_keys_lower,
    from_ui_code,
    normalize_task_type,
    to_ui_code,
)

logger = logging.getLogger(__name__)

# 默认 WebBridge 配置（Playwright，推荐）；frontend_url 与 constants 单源对齐
DEFAULT_WEBBRIDGE = {
    "enabled": True,
    "adapter": "playwright",
    "frontend_url": DEFAULT_FRONTEND_URL,
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

# NL 定位：深搜时跳过的目录名（小写比较）
NL_SKIP_DIR_NAMES = {
    "work",
    "checkpoints",
    "__pycache__",
    ".git",
    "node_modules",
    "baseline models",
    "$recycle.bin",
    "system volume information",
    "models",
    "model",
    "输出",
    "模型输出",
}
# NL 线索提取时忽略的停用词（小写 / 原文）
_NL_STOPWORDS = {
    "训练",
    "训练集",
    "训练数据",
    "数据集",
    "数据",
    "目录",
    "文件夹",
    "路径",
    "模型",
    "文件",
    "默认",
    "参数",
    "帮我",
    "请",
    "一个",
    "一下",
    "使用",
    "用",
    "做",
    "跑",
    "启动",
    "开始",
    "里面",
    "里边",
    "中的",
    "中",
    "里的",
    "里",
    "下的",
    "下",
    "上的",
    "上",
    "那个",
    "这个",
    "我的",
    "以及",
    "和",
    "与",
    "及",
    "的",
    "了",
    "吧",
    "呢",
    "啊",
    "是",
    "在",
    "到",
    "把",
    "将",
    "从",
    "为",
    "并",
    "且",
    "或",
    "等",
    "项目",
    "工程",
    "平台",
    "测试",
    "正式",
    "nas",
    "disk",
    "drive",
    "folder",
    "directory",
    "path",
    # 注意：data / zzh 等短目录名不要放进停用词，否则无法作 path_fragment
    "dataset",
    "train",
    "training",
    "model",
    "default",
    "params",
    "parameter",
    "please",
    "help",
    "me",
    "a",
    "an",
    "the",
    "my",
    "this",
    "that",
    "with",
    "from",
    "under",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
}
# 任务别名也视为停用词，避免 iseg/cls 被当成路径片段
_NL_STOPWORDS.update(all_alias_keys_lower())

# 有界深搜默认限制
NL_DEFAULT_MAX_DEPTH = 4
NL_DEFAULT_MAX_DIRS_VISITED = 800
NL_DEFAULT_MAX_CANDIDATES = 40
NL_DEFAULT_TOP_K = 12
NL_SINGLE_CANDIDATE_SCORE = 3.0
NL_SCORE_GAP_FALLBACK = 1.5


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


# ---------------------------------------------------------------------------
# 自然语言位置线索（模糊定位训练集，非完整绝对路径）
# ---------------------------------------------------------------------------


@dataclass
class LocationClues:
    """从提示词提取的位置线索，用于有界锚点深搜。"""

    drives: List[str] = field(default_factory=list)  # 如 ["Z:\\", "D:\\"]
    path_fragments: List[str] = field(default_factory=list)  # 如 ["zzh", "data"]
    name_keywords: List[str] = field(default_factory=list)  # 如 ["易格", "易格技术"]
    task_type: str = ""

    def has_searchable_signal(self) -> bool:
        """是否具备可触发 NL 深搜的信号（禁止仅凭盘符整盘扫）。"""
        return bool(self.path_fragments or self.name_keywords)

    def to_dict(self) -> Dict[str, object]:
        return {
            "drives": list(self.drives),
            "path_fragments": list(self.path_fragments),
            "name_keywords": list(self.name_keywords),
            "task_type": self.task_type,
        }


def normalize_name_for_match(name: str) -> str:
    """文件夹名模糊匹配用的归一化：小写、去常见连接符与空白。"""
    text = str(name or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[\s_\-—–·.]+", "", text)
    return text


def fuzzy_name_match(name: str, keyword: str) -> bool:
    """判断目录名是否模糊包含关键词（子串 / 归一化子串）。"""
    raw_name = str(name or "").strip()
    raw_kw = str(keyword or "").strip()
    if not raw_name or not raw_kw:
        return False
    if raw_kw.lower() in raw_name.lower():
        return True
    n = normalize_name_for_match(raw_name)
    k = normalize_name_for_match(raw_kw)
    if not n or not k:
        return False
    if k in n or n in k:
        return True
    # 短英文 token：要求更严，避免 z 命中 zzh 以外的误伤已由长度控制
    if len(k) >= 2 and k in n:
        return True
    return False


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _is_stopword_token(token: str) -> bool:
    t = str(token or "").strip()
    if not t:
        return True
    if t.lower() in _NL_STOPWORDS:
        return True
    # 纯标点
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    return False


def _chinese_stopwords_sorted() -> List[str]:
    return sorted(
        (s for s in _NL_STOPWORDS if re.fullmatch(r"[一-鿿]+", s)),
        key=len,
        reverse=True,
    )


def _strip_stopword_affixes(token: str) -> str:
    """反复剥掉中文 token 首尾停用词/助词，得到更干净的项目名。"""
    t = str(token or "").strip()
    if not t:
        return ""
    # 混有拉丁时先丢掉盘符口语残留
    t = re.sub(r"(?i)[a-z]\s*盘", "", t).strip()
    stops = _chinese_stopwords_sorted()
    particles = ("的", "了", "呢", "吧", "啊", "吗", "么")
    changed = True
    while t and changed:
        changed = False
        for s in stops:
            if t.startswith(s) and len(t) > len(s):
                t = t[len(s) :].strip()
                changed = True
                break
            if t.endswith(s) and len(t) > len(s):
                t = t[: -len(s)].strip()
                changed = True
                break
        if not changed:
            for p in particles:
                if t.startswith(p) and len(t) > len(p):
                    t = t[len(p) :].strip()
                    changed = True
                    break
                if t.endswith(p) and len(t) > len(p):
                    t = t[: -len(p)].strip()
                    changed = True
                    break
    return t


def _is_noisy_keyword(token: str) -> bool:
    """过滤由停用词拼接而成的中文噪声，如「帮我训练」「数据在」「文件夹中的易格技术」整段。"""
    t = str(token or "").strip()
    if not t:
        return True
    if _is_stopword_token(t):
        return True
    # 盘符口语残留或「z盘zzh」这类不应进 name_keywords
    if re.search(r"(?i)[a-z]\s*盘", t):
        return True
    # 过短英文已在别处处理；中文至少 2 字且不能整段由停用词覆盖
    if re.fullmatch(r"[一-鿿]+", t):
        rest = t
        for s in _chinese_stopwords_sorted():
            if s in rest:
                rest = rest.replace(s, "")
        rest = rest.strip()
        if not rest:
            return True
        # 剥前后缀后若与原文相同且以功能结构开头，仍可能是噪声短语
        cleaned = _strip_stopword_affixes(t)
        if not cleaned or _is_stopword_token(cleaned):
            return True
        # 以强功能词/路径虚词开头的整段（未剥离干净前）视为噪声，
        # 真正的项目名会通过剥离或「中的XX」规则另路进入
        for prefix in (
            "帮我",
            "请帮",
            "数据在",
            "训练集是",
            "模型是",
            "文件夹",
            "目录",
            "里面",
            "里边",
        ):
            if t.startswith(prefix) and t != cleaned:
                # 若 cleaned 已是有效短名，由调用方用 cleaned；整段 t 丢弃
                return True
            if t.startswith(prefix) and len(t) <= len(prefix) + 1:
                return True
    return False


def _finalize_name_keywords(keywords: Sequence[str]) -> List[str]:
    """清洗、剥停用词前后缀，并去掉被更短关键词主导的噪声超集。"""
    cleaned: List[str] = []
    for raw in keywords:
        tok = str(raw or "").strip()
        if not tok:
            continue
        peeled = _strip_stopword_affixes(tok) or tok
        # 剥完后若变成拉丁片段，不进 name_keywords
        if re.fullmatch(r"[A-Za-z0-9_\-]+", peeled):
            continue
        if _is_noisy_keyword(peeled) or _is_stopword_token(peeled):
            continue
        if len(peeled) < 2:
            continue
        cleaned.append(peeled)

    cleaned = _dedupe_preserve(cleaned)
    # 若短词是长词的实质核心（长词去掉停用词后等于短词），丢掉长词
    stops = _chinese_stopwords_sorted()
    kept: List[str] = []
    for kw in sorted(cleaned, key=len):  # 短的优先保留
        dominated = False
        for short in kept:
            if short == kw:
                dominated = True
                break
            if short in kw:
                rest = kw
                for s in stops:
                    rest = rest.replace(s, "")
                rest = re.sub(r"[的了呢吧啊吗么]", "", rest)
                if normalize_name_for_match(rest) == normalize_name_for_match(short):
                    dominated = True
                    break
                # 长词以短词结尾/开头且多余部分全是虚词
                if kw.endswith(short) or kw.startswith(short):
                    extra = kw.replace(short, "", 1)
                    if not extra or _is_noisy_keyword(extra) or _is_stopword_token(extra):
                        dominated = True
                        break
                    extra2 = _strip_stopword_affixes(extra)
                    if not extra2:
                        dominated = True
                        break
        if not dominated:
            kept.append(kw)
    return _dedupe_preserve(kept)


def extract_drive_letters(text: str) -> List[str]:
    """从自然语言提取盘符根路径，如 z盘 / Z: → ``Z:\\``。"""
    if not text:
        return []
    found: List[str] = []
    seen: Set[str] = set()

    def _add(letter: str) -> None:
        L = letter.upper()
        if not re.fullmatch(r"[A-Z]", L):
            return
        root = f"{L}:\\"
        if root.lower() in seen:
            return
        seen.add(root.lower())
        found.append(root)

    # 中文：z盘 / Z 盘
    for m in re.finditer(r"(?i)(?<![a-z0-9])([a-z])\s*盘", text):
        _add(m.group(1))
    # 英文盘符：Z: 或 Z:\ 或 Z:/
    for m in re.finditer(r"(?i)(?<![a-z0-9])([a-z]):(?=[\\/\s]|$|[^a-z0-9])", text):
        _add(m.group(1))

    return found


def _strip_known_absolute_paths(text: str) -> str:
    """去掉已识别的绝对路径，避免再被拆成 fragments。"""
    cleaned = text
    for path in extract_filesystem_paths(text):
        cleaned = cleaned.replace(path, " ")
    # 再清一遍无引号盘符路径粗匹配
    cleaned = re.sub(r"[A-Za-z]:\\[^\s\"']+", " ", cleaned)
    cleaned = re.sub(r"[A-Za-z]:/[^\s\"']+", " ", cleaned)
    return cleaned


def _iter_hint_tokens(hint: str) -> List[str]:
    if not hint:
        return []
    parts = re.split(r"[\s,，、/;|\\]+", str(hint).strip())
    return [p for p in parts if p and not _is_stopword_token(p)]


def extract_location_clues(
    prompt: str = "",
    dataset_hint: str = "",
    task_type: str = "",
    extra_keywords: Optional[Sequence[str]] = None,
    *,
    drives: Optional[Sequence[str]] = None,
    path_fragments: Optional[Sequence[str]] = None,
    name_keywords: Optional[Sequence[str]] = None,
) -> LocationClues:
    """Assemble location clues from LLM fields only."""
    _ = prompt

    drive_list: List[str] = []
    for d in drives or []:
        text_d = str(d or "").strip()
        if not text_d:
            continue
        m = re.match('^\\s*([A-Za-z])\\s*:?\\s*[\\\\/]?\\s*$', text_d)
        if m:
            drive_list.append(m.group(1).upper() + ":\\")
        elif re.match('^[A-Za-z]:[\\\\/]', text_d):
            drive_list.append(text_d[:2].upper() + "\\")
        else:
            if not text_d.endswith(("\\", "/")):
                text_d = text_d + "\\"
            drive_list.append(text_d)

    fragments: List[str] = [
        str(x).strip() for x in (path_fragments or []) if str(x).strip()
    ]
    keywords: List[str] = [
        str(x).strip() for x in (name_keywords or []) if str(x).strip()
    ]

    hint = str(dataset_hint or "").strip()
    if hint and not looks_like_filesystem_path(hint):
        for part in re.split('[\\s,，、/;|\\\\]+', hint):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch('[A-Za-z][A-Za-z0-9_\\-]{1,63}', part):
                fragments.append(part)
            else:
                keywords.append(part)

    if extra_keywords:
        for kw in extra_keywords:
            text_k = str(kw or "").strip()
            if text_k:
                keywords.append(text_k)

    return LocationClues(
        drives=_dedupe_preserve(drive_list),
        path_fragments=_dedupe_preserve(fragments),
        name_keywords=_dedupe_preserve(keywords),
        task_type=str(task_type or "").strip(),
    )



def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _path_exists_dir(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def _list_child_dirs(root: Path) -> List[Path]:
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.warning("无法读取目录 %s: %s", root, exc)
        return []
    out: List[Path] = []
    for entry in entries:
        try:
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                continue
            if not entry.is_dir():
                continue
            if entry.name.lower() in NL_SKIP_DIR_NAMES:
                continue
        except OSError:
            continue
        out.append(entry)
    return out


def resolve_search_anchors(
    clues: LocationClues,
    scan_dirs: Optional[Sequence[Union[str, Path]]] = None,
    *,
    drive_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> List[Path]:
    """根据位置线索解析有界深搜的锚点目录（必须已存在）。

    ``drive_roots`` 仅供测试注入：按与 ``clues.drives`` 相同顺序映射盘符根，
    避免单测依赖真实 ``Z:\\``。
    """
    if clues is None or not clues.has_searchable_signal():
        return []

    roots = normalize_scan_dirs(list(scan_dirs) if scan_dirs is not None else None)
    anchors: List[Path] = []
    seen: Set[str] = set()

    def _add_anchor(path: Path) -> None:
        if not _path_exists_dir(path):
            return
        resolved = _safe_resolve(path)
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        anchors.append(resolved)

    # 盘符根列表
    drive_paths: List[Path] = []
    if drive_roots is not None:
        for item in drive_roots:
            if item is None:
                continue
            drive_paths.append(Path(item))
    else:
        for d in clues.drives:
            drive_paths.append(Path(d))

    fragments = list(clues.path_fragments)
    keywords = list(clues.name_keywords)

    # 1) drive + 每个 fragment → drive\fragment
    for drive in drive_paths:
        for frag in fragments:
            _add_anchor(Path(drive) / frag)

    # 2) 禁止：仅 drive 根且无 fragment/keyword 时不加入（has_searchable_signal 已保证有其一）
    #    若有 keyword 但无 fragment：在 drive 下一层按名称模糊匹配
    if keywords:
        for drive in drive_paths:
            if not _path_exists_dir(Path(drive)):
                continue
            for child in _list_child_dirs(Path(drive)):
                if any(fuzzy_name_match(child.name, kw) for kw in keywords):
                    _add_anchor(child)
                # fragment 也在 drive 下一层再匹配一次（大小写不同）
            for frag in fragments:
                direct = Path(drive) / frag
                if _path_exists_dir(direct):
                    _add_anchor(direct)
                else:
                    for child in _list_child_dirs(Path(drive)):
                        if fuzzy_name_match(child.name, frag):
                            _add_anchor(child)

    # 3) scan_dirs 下一层：fragment / keyword 模糊匹配
    for root in roots:
        if not _path_exists_dir(root):
            continue
        for child in _list_child_dirs(root):
            name = child.name
            if fragments and any(fuzzy_name_match(name, f) for f in fragments):
                _add_anchor(child)
                continue
            if keywords and any(fuzzy_name_match(name, kw) for kw in keywords):
                _add_anchor(child)

        # scan_dir 自身若名命中也可作为锚点（少见）
        if keywords and any(fuzzy_name_match(root.name, kw) for kw in keywords):
            _add_anchor(root)
        if fragments and any(fuzzy_name_match(root.name, f) for f in fragments):
            _add_anchor(root)

    # 4) 仅有 keyword、无 drive：已在 scan_dirs 一层处理；若无 anchor 且只有一个 fragment
    #    尝试 project_root / fragment
    if not anchors and fragments:
        for frag in fragments:
            _add_anchor(project_root() / frag)

    if anchors:
        logger.info(
            "NL 定位锚点: %s (clues=%s)",
            [str(a) for a in anchors],
            clues.to_dict(),
        )
    else:
        logger.info("NL 定位未解析到有效锚点: clues=%s", clues.to_dict())

    return anchors


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
