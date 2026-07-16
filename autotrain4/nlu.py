"""
自然语言意图解析器（LLM 直接驱动，无规则兜底）。

把用户输入（如"训练一个cls模型来辨别郁金香和菊花"）解析成结构化意图，
供数据集解析和配置生成使用。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import TASK_ALIASES
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# 数据增强 UI 选项（用于归一化 LLM 输出，允许“随即”/“随机”混用）
DATA_AUGMENTATION_OPTIONS = [
    "调整图像大小",
    "随机比例缩放",
    "随机翻转",
    "随机固定角度旋转",
    "随机裁剪",
    "随机调整亮度",
    "随机调整对比度",
    "随机JPG编码",
]


@dataclass
class TrainIntent:
    """用户训练意图。"""

    task_type: str
    categories: List[str] = field(default_factory=list)
    dataset_hint: str = ""
    mode: str = "train"  # train / dry_run / evaluate / test
    raw_prompt: str = ""
    needs_training: bool = True
    data_analysis: bool = True
    mentions_augmentation: bool = False
    data_augmentation: List[str] = field(default_factory=list)
    baseline_model: Optional[str] = None
    wants_baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "categories": self.categories,
            "dataset_hint": self.dataset_hint,
            "mode": self.mode,
            "raw_prompt": self.raw_prompt,
            "needs_training": self.needs_training,
            "data_analysis": self.data_analysis,
            "mentions_augmentation": self.mentions_augmentation,
            "data_augmentation": self.data_augmentation,
            "baseline_model": self.baseline_model,
            "wants_baseline": self.wants_baseline,
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

    def parse(self, prompt: str) -> List[TrainIntent]:
        """解析用户输入为一个或多个训练意图，完全由 LLM 负责。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("输入不能为空")

        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法解析自然语言意图（已取消规则兜底）")

        logger.info("第一层 LLM：解析用户意图")
        parsed_list = self.llm.extract_intent(prompt)
        if parsed_list is None:
            raise RuntimeError("LLM 意图解析失败，无法继续（已取消规则兜底）")

        top_level_needs_training = True
        if isinstance(parsed_list, dict):
            top_level_needs_training = bool(parsed_list.get("needs_training", True))
            parsed_list = parsed_list.get("tasks", [parsed_list])

        if not isinstance(parsed_list, list):
            raise ValueError(f"LLM 返回的任务格式不正确: {parsed_list}")

        # LLM 返回空列表，说明没有训练任务，进入通用对话
        if not parsed_list:
            logger.info("LLM 返回空任务列表，进入通用对话模式")
            return [
                TrainIntent(
                    task_type="",
                    categories=[],
                    dataset_hint="",
                    mode="chat",
                    raw_prompt=prompt,
                    needs_training=False,
                    data_analysis=False,
                    mentions_augmentation=False,
                    data_augmentation=[],
                    baseline_model=None,
                    wants_baseline=False,
                )
            ]

        # 用户只是闲聊/提问，没有训练任务
        if not top_level_needs_training:
            logger.info("用户输入与训练无关，进入通用对话模式")
            return [
                TrainIntent(
                    task_type="",
                    categories=[],
                    dataset_hint="",
                    mode="chat",
                    raw_prompt=prompt,
                    needs_training=False,
                    data_analysis=False,
                    mentions_augmentation=False,
                    data_augmentation=[],
                    baseline_model=None,
                    wants_baseline=False,
                )
            ]

        intents: List[TrainIntent] = []
        for idx, parsed in enumerate(parsed_list):
            if not isinstance(parsed, dict):
                raise ValueError(f"第 {idx + 1} 个任务格式不正确: {parsed}")

            task_type = self._canonicalize_task_type(parsed.get("task_type", ""))
            mode = self._canonicalize_mode(parsed.get("mode", "train"), prompt)
            categories = self._canonicalize_categories(parsed.get("categories", []))
            dataset_hint = str(parsed.get("dataset_hint", "")).strip()
            baseline_model = self._extract_baseline_model(parsed.get("baseline_model"), prompt)
            wants_baseline = self._wants_baseline_comparison(
                parsed.get("wants_baseline"), baseline_model, prompt
            )
            needs_training = bool(parsed.get("needs_training", top_level_needs_training))
            data_analysis = bool(parsed.get("data_analysis", True))
            mentions_augmentation = bool(parsed.get("mentions_augmentation", False))
            data_augmentation = self._canonicalize_data_augmentation(
                parsed.get("data_augmentation", [])
            )

            # 只有当用户明确提到数据增强时才保留增强选项，否则清空，避免前端误点增强菜单。
            if data_augmentation and not mentions_augmentation:
                logger.warning("LLM 返回了数据增强选项但用户未明确提及，已忽略: %s", data_augmentation)
                data_augmentation = []

            if not task_type:
                if mode != "test" and needs_training:
                    raise ValueError(f"第 {idx + 1} 个任务未返回有效任务类型: {parsed}")

            # dry_run 覆盖：仅当整个请求明确提到 dry/只生成 时才生效
            if mode == "train" and ("dry" in prompt.lower() or "只生成" in prompt):
                mode = "dry_run"

            intents.append(
                TrainIntent(
                    task_type=task_type,
                    categories=categories,
                    dataset_hint=dataset_hint,
                    mode=mode,
                    raw_prompt=prompt,
                    needs_training=needs_training,
                    data_analysis=data_analysis,
                    mentions_augmentation=mentions_augmentation,
                    data_augmentation=data_augmentation,
                    baseline_model=baseline_model,
                    wants_baseline=wants_baseline,
                )
            )

        if not intents:
            raise ValueError("LLM 未返回任何有效任务")

        return intents

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
            # 即使 LLM 返回 train，如果用户明确提到测试/评估关键词，也进入测试模式
            if lower in {"train", "training"}:
                prompt_lower = prompt.lower()
                if any(kw.lower() in prompt_lower for kw in self.TEST_KEYWORDS):
                    return "test"
                return "train"

        lower = prompt.lower()
        for kw in self.TEST_KEYWORDS:
            if kw.lower() in lower:
                return "test"
        return "train"

    BASELINE_KEYWORDS = ["baseline", "基准", "对比", "比较", "compare"]

    def _extract_baseline_model(self, raw: Any, prompt: str) -> Optional[str]:
        """从 LLM 输出或用户原始输入中提取基准模型 .dvp 文件路径。"""
        if raw:
            s = str(raw).strip().strip('"\'')
            if s.lower().endswith(".dvp"):
                return s

        # 兜底：用正则从 prompt 中匹配 .dvp 文件路径
        # 支持带空格、斜杠、下划线、连字符、点号的相对/绝对路径
        matches = re.findall(r"[\w\s/\\._-]+?\.dvp", prompt, re.IGNORECASE)
        for m in matches:
            s = m.strip().strip('"\'，,。')
            if s and s.lower().endswith(".dvp"):
                return s
        return None

    def _wants_baseline_comparison(
        self,
        raw: Any,
        baseline_model: Optional[str],
        prompt: str,
    ) -> bool:
        """判断用户是否希望与 baseline 模型做对比。"""
        if raw is not None:
            return bool(raw)
        if baseline_model:
            return True
        lower = prompt.lower()
        return any(kw.lower() in lower for kw in self.BASELINE_KEYWORDS)

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

    @staticmethod
    def _canonicalize_data_augmentation(raw: Any) -> List[str]:
        """把 LLM 返回的增强选项归一化为标准 UI 文本。"""
        if not raw:
            return []
        if isinstance(raw, str):
            raw = [raw]

        result: List[str] = []
        for item in raw:
            text = str(item).strip().replace("随即", "随机")
            if not text:
                continue
            # 找最匹配的标准选项（忽略大小写和全半角）
            for option in DATA_AUGMENTATION_OPTIONS:
                if option.lower() in text.lower() or text.lower() in option.lower():
                    if option not in result:
                        result.append(option)
                    break
        return result
