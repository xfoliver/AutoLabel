"""
自然语言意图解析器。

把用户输入（如"训练一个cls模型来辨别郁金香和菊花"）解析成结构化意图，
供数据集解析和配置生成使用。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import TASK_ALIASES
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TrainIntent:
    """用户训练意图。"""

    task_type: str
    categories: List[str] = field(default_factory=list)
    dataset_hint: str = ""
    mode: str = "train"  # train / dry_run / evaluate
    raw_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "categories": self.categories,
            "dataset_hint": self.dataset_hint,
            "mode": self.mode,
            "raw_prompt": self.raw_prompt,
        }


class IntentParser:
    """训练意图解析器。"""

    # 常见的任务类型中文表达
    TASK_KEYWORDS: Dict[str, List[str]] = {
        "OCR": ["ocr", "文字识别"],
        "分类": ["cls", "分类", "classif", "识别", "辨别", "区分"],
        "目标检测": ["det", "目标检测", "检测"],
        "DETR": ["detr"],
        "实例分割": ["iseg", "实例分割"],
        "语义分割": ["sseg", "语义分割", "分割"],
        "旋转框检测": ["rotate", "旋转框检测", "旋转框"],
    }

    # 类别提取的停用词
    STOP_WORDS = {
        "一个",
        "模型",
        "来",
        "训练",
        "做",
        "进行",
        "任务",
        "项目",
        "数据",
        "集",
        "和",
        "与",
        "或",
        "的",
        "了",
        "在",
        "用",
        "使用",
        "基于",
        "通过",
        "对",
        "针对",
        "类别",
        "种类",
        "类型",
        "识别",
        "辨别",
        "区分",
        "分辨",
        "检测",
        "分",
        "出",
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def parse(self, prompt: str) -> TrainIntent:
        """解析用户输入为训练意图。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("输入不能为空")

        # 1. 任务类型
        task_type = self._extract_task_type(prompt)

        # 2. 类别（先规则，再 LLM 兜底）
        # OCR 任务的"文字"是识别对象，不是分类类别，通常不需要类别匹配
        if task_type == "OCR":
            categories = []
        else:
            categories = self._extract_categories(prompt)
            if not categories and self.llm.available:
                logger.info("规则未提取到类别，尝试 LLM 提取")
                categories = self._extract_categories_with_llm(prompt)

        # 3. 数据集提示
        dataset_hint = self._extract_dataset_hint(prompt)

        # 4. 模式
        mode = "dry_run" if "dry" in prompt.lower() or "只生成" in prompt else "train"

        return TrainIntent(
            task_type=task_type,
            categories=categories,
            dataset_hint=dataset_hint,
            mode=mode,
            raw_prompt=prompt,
        )

    def _extract_task_type(self, prompt: str) -> str:
        """从用户输入中提取规范任务类型。"""
        lower = prompt.lower()

        # 先匹配显式别名（如 cls、detr、OCR）
        # 边界只限制英文数字，中文前后可直接相邻
        for alias, canonical in TASK_ALIASES.items():
            pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
            if re.search(pattern, lower):
                return canonical

        # 再按关键词匹配中文表达
        # 优先检测更具体的词，避免"分割"被"语义分割"和"实例分割"同时命中
        priority_order = ["DETR", "旋转框检测", "实例分割", "语义分割", "目标检测", "分类", "OCR"]
        for canonical in priority_order:
            for kw in self.TASK_KEYWORDS.get(canonical, []):
                if kw.lower() in lower:
                    return canonical

        raise ValueError(
            f"无法从输入中识别任务类型: {prompt}。"
            f"支持的类型: {', '.join(sorted(set(TASK_ALIASES.values())))}"
        )

    def _extract_categories(self, prompt: str) -> List[str]:
        """
        规则提取类别名。
        支持：
        - 引号包裹：「郁金香」、"tulip"、'cat'
        - 顿号/逗号分隔：郁金香、菊花
        - "A 和 B" / "A 与 B" / "辨别 A 和 B"
        """
        categories: List[str] = []

        # 1. 引号包裹
        quoted = re.findall(r"[「\"']([^「\"']+?)[」\"']", prompt)
        categories.extend([self._clean_cat(c) for c in quoted if self._clean_cat(c)])

        # 2. "辨别/识别/区分/分辨 A 和 B" 或 "A 和 B"
        # 尝试找到动词后的对象，优先匹配更长的动词
        m = re.search(r"(?:辨别|识别|区分|分辨|辨别出|识别出|分辨出|分出|分类)\s*([\u4e00-\u9fa5a-zA-Z0-9\s、,和与]+)", prompt)
        if m:
            categories.extend(self._split_categories(m.group(1)))

        # 3. 如果还没找到，尝试提取末尾的并列名词
        if not categories:
            # 匹配 "来/用于/做 X 和 Y" 中的 X 和 Y
            m = re.search(r"(?:来|用于|做)\s*([\u4e00-\u9fa5a-zA-Z0-9\s、,和与]+?)(?:模型|任务|项目)?$", prompt)
            if m:
                categories.extend(self._split_categories(m.group(1)))

        # 去重并保持顺序
        seen = set()
        result: List[str] = []
        for c in categories:
            key = c.lower()
            if key and key not in seen:
                seen.add(key)
                result.append(c)
        return result

    def _extract_categories_with_llm(self, prompt: str) -> List[str]:
        """LLM 兜底提取类别。"""
        intent = self.llm.extract_intent(prompt)
        if not intent:
            return []
        cats = intent.get("categories", [])
        return [self._clean_cat(str(c)) for c in cats if self._clean_cat(str(c))]

    def _extract_dataset_hint(self, prompt: str) -> str:
        """提取用户可能提到的数据集文件夹名。"""
        # 匹配 "文件夹 XXX" / "数据集 XXX" / "用 XXX 数据"
        patterns = [
            r"(?:文件夹|目录|dataset|folder)\s*[是为:]?\s*([\w\-]+)",
            r"(?:数据集|数据)\s*[是为:]?\s*([\w\-]+)",
            r"用\s+([\w\-]+)\s*(?:数据|数据集|文件夹)",
        ]
        for pat in patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    def _split_categories(self, text: str) -> List[str]:
        """把 'A、B和C' 或 'A, B and C' 拆成类别列表。"""
        # 统一分隔符
        normalized = re.sub(r"[,，、]\s*", "|", text)
        # 中文连词不需要空格
        normalized = re.sub(r"(?:和|与|and|or)", "|", normalized)
        parts = [self._clean_cat(p) for p in normalized.split("|")]
        return [p for p in parts if p]

    def _clean_cat(self, text: str) -> str:
        """清理类别名。"""
        text = text.strip()
        # 去掉口语化后缀：螺丝什么的、水果之类、A和B等等
        text = re.sub(r"(什么的|什么|之类|等等|等)$", "", text).strip()
        # 去掉常见后缀
        text = re.sub(r"[图张]?$", "", text).strip()
        # 过滤任务描述性短语（如 "电子元器件上的凹坑"、"图片中的文字"）
        if self._is_task_description(text):
            return ""
        if text.lower() in {"and", "or", "和", "与", ""}:
            return ""
        # 去掉前导的停用词（如 "模型识别菊花" -> "菊花"）
        changed = True
        while changed:
            changed = False
            for sw in self.STOP_WORDS:
                if text.startswith(sw):
                    text = text[len(sw):].strip()
                    changed = True
                    break
        # 去掉末尾的停用词
        for sw in self.STOP_WORDS:
            if text.endswith(sw):
                text = text[:-len(sw)].strip()
        if text in self.STOP_WORDS or not text:
            return ""
        return text

    @staticmethod
    def _is_task_description(text: str) -> bool:
        """判断文本是任务描述而非具体类别名。"""
        # 包含 "X上的Y"、"X中的Y"、"X里的Y" 等空间/位置关系描述
        if re.search(r"(上的|中的|里的|表面|内部|外部|区域|部位)", text):
            return True
        # 包含图片/图像/文字等媒体泛化词
        if re.search(r"(图片|图像|照片|文字)", text):
            return True
        # 过长且包含抽象泛化词，才认为是描述
        if len(text) > 6 and re.search(r"(对象|目标|东西|行为|场景)", text):
            return True
        # 超长短语（类别名通常较短）
        if len(text) > 8:
            return True
        return False
