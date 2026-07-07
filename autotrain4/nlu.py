"""
自然语言意图解析器（LLM 直接驱动，无规则兜底）。

把用户输入（如"训练一个cls模型来辨别郁金香和菊花"）解析成结构化意图，
供数据集解析和配置生成使用。
"""

import logging
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
    mode: str = "train"  # train / dry_run / evaluate / test
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
    """训练意图解析器（第一层 LLM）。"""

    # 测试/评估意图关键词（仅用于轻量预判断，不决定最终 mode）
    TEST_KEYWORDS = [
        "测试", "试一下", "试下", "评估", "评测", "验证", "跑一遍",
        "test", "testing", "evaluate", "eval", "infer", "inference",
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def parse(self, prompt: str) -> TrainIntent:
        """解析用户输入为训练意图，完全由 LLM 负责。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("输入不能为空")

        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法解析自然语言意图（已取消规则兜底）")

        logger.info("第一层 LLM：解析用户意图")
        parsed = self.llm.extract_intent(prompt)
        if not parsed:
            raise RuntimeError("LLM 意图解析失败，无法继续（已取消规则兜底）")

        task_type = self._canonicalize_task_type(parsed.get("task_type", ""))
        mode = self._canonicalize_mode(parsed.get("mode", "train"), prompt)
        categories = self._canonicalize_categories(parsed.get("categories", []))
        dataset_hint = str(parsed.get("dataset_hint", "")).strip()

        if not task_type:
            # 测试类指令可能只含"刚刚训练的模型"而不含任务类型，允许为空，由下游推断
            if mode != "test":
                raise ValueError(f"LLM 未返回有效任务类型: {parsed}")

        # dry_run 覆盖
        if mode == "train" and ("dry" in prompt.lower() or "只生成" in prompt):
            mode = "dry_run"

        return TrainIntent(
            task_type=task_type,
            categories=categories,
            dataset_hint=dataset_hint,
            mode=mode,
            raw_prompt=prompt,
        )

    def _canonicalize_task_type(self, raw: str) -> str:
        """把 LLM 返回的任务类型规范化为内部标准名。"""
        if not raw:
            return ""
        lower = raw.strip().lower()
        # 先匹配显式别名
        for alias, canonical in TASK_ALIASES.items():
            if alias.lower() == lower:
                return canonical
        # 再按规范名称匹配
        for canonical in set(TASK_ALIASES.values()):
            if canonical.lower() == lower:
                return canonical
        return ""

    def _canonicalize_mode(self, raw: str, prompt: str) -> str:
        """规范 mode，优先信任 LLM 返回值，再用关键词复核。"""
        if raw:
            lower = str(raw).strip().lower()
            if lower in {"test", "testing", "evaluate", "eval", "infer", "inference"}:
                return "test"
            if lower in {"dry_run", "dry-run", "dryrun"}:
                return "dry_run"
            if lower in {"train", "training"}:
                return "train"

        lower = prompt.lower()
        for kw in self.TEST_KEYWORDS:
            if kw.lower() in lower:
                return "test"
        return "train"

    def _canonicalize_categories(self, raw: List[Any]) -> List[str]:
        """清理并去重类别列表。"""
        seen = set()
        result: List[str] = []
        for c in raw:
            cat = str(c).strip()
            if not cat:
                continue
            key = cat.lower()
            if key not in seen:
                seen.add(key)
                result.append(cat)
        return result
