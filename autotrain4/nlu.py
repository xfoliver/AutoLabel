"""
自然语言意图解析器（LLM 全权理解用户输入）。

本地只做：
- 字段规范化（任务别名、增强选项名、mode 枚举）
- 路径存在性校验（拒绝 LLM 臆造的不存在路径）
- 绝不使用「测试/定位/文件夹」等关键词去改写用户意图
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    TASK_ALIASES,
    _clean_extracted_path,
    extract_filesystem_paths,
    looks_like_filesystem_path,
)
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# 数据增强 UI 选项（仅用于把 LLM 输出归一成前端可点的标准文案）
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

_VALID_MODES = frozenset({"train", "dry_run", "test", "locate", "chat"})


@dataclass
class TrainIntent:
    """用户意图（训练 / 测试 / 定位数据集 / 对话）。"""

    task_type: str
    categories: List[str] = field(default_factory=list)
    dataset_hint: str = ""
    dataset_path: str = ""
    output_dir: str = ""
    mode: str = "train"  # train / dry_run / test / locate / chat
    raw_prompt: str = ""
    needs_training: bool = True
    data_analysis: bool = True
    mentions_augmentation: bool = False
    data_augmentation: List[str] = field(default_factory=list)
    baseline_model: Optional[str] = None
    wants_baseline: bool = False
    # LLM 给出的位置线索（供有界扫盘；非关键词规则）
    location_drives: List[str] = field(default_factory=list)
    location_path_fragments: List[str] = field(default_factory=list)
    location_name_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "categories": self.categories,
            "dataset_hint": self.dataset_hint,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "mode": self.mode,
            "raw_prompt": self.raw_prompt,
            "needs_training": self.needs_training,
            "data_analysis": self.data_analysis,
            "mentions_augmentation": self.mentions_augmentation,
            "data_augmentation": self.data_augmentation,
            "baseline_model": self.baseline_model,
            "wants_baseline": self.wants_baseline,
            "location_drives": self.location_drives,
            "location_path_fragments": self.location_path_fragments,
            "location_name_keywords": self.location_name_keywords,
        }

    def to_task_dict(self) -> Dict[str, Any]:
        """返回供单任务下游使用的意图，避免原始多任务提示词串扰。"""
        data = self.to_dict()
        data.pop("raw_prompt", None)
        return data


class IntentParser:
    """意图解析器：完全信任 LLM 结构化输出。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    @staticmethod
    def _chat_intent(prompt: str) -> List[TrainIntent]:
        return [
            TrainIntent(
                task_type="",
                mode="chat",
                raw_prompt=prompt,
                needs_training=False,
                data_analysis=False,
            )
        ]

    def parse(self, prompt: str) -> List[TrainIntent]:
        """解析用户输入为一个或多个意图（LLM 负责理解，本地只规范字段）。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("输入不能为空")

        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法解析自然语言意图（已取消规则兜底）")

        logger.info("第一层 LLM：解析用户意图")
        extracted = self.llm.extract_intent(prompt)
        if extracted is None:
            raise RuntimeError("LLM 意图解析失败，无法继续（已取消规则兜底）")

        top_level_needs_training = True
        intent_action = ""
        if isinstance(extracted, dict):
            intent_action = str(extracted.get("intent_action") or "").strip().lower()
            if "needs_training" in extracted:
                top_level_needs_training = bool(extracted.get("needs_training"))
            elif intent_action in {"test", "locate", "chat"}:
                top_level_needs_training = False
            elif intent_action in {"train", "dry_run"}:
                top_level_needs_training = True
            parsed_list = extracted.get("tasks", [])
            if parsed_list is None:
                parsed_list = []
            if "tasks" not in extracted and any(
                k in extracted
                for k in ("task_type", "mode", "dataset_hint", "dataset_path")
            ):
                parsed_list = [extracted]
        elif isinstance(extracted, list):
            parsed_list = extracted
        else:
            raise ValueError(f"LLM 返回的任务格式不正确: {extracted}")

        if not isinstance(parsed_list, list):
            raise ValueError(f"LLM 返回的任务格式不正确: {parsed_list}")

        if not parsed_list:
            if intent_action == "locate":
                logger.info("空 tasks 且 intent_action=locate，构造 locate 意图")
                return [
                    TrainIntent(
                        task_type="",
                        mode="locate",
                        raw_prompt=prompt,
                        needs_training=False,
                        data_analysis=False,
                    )
                ]
            logger.info("LLM 返回空任务列表，进入通用对话模式")
            return self._chat_intent(prompt)

        if not top_level_needs_training and intent_action == "chat":
            logger.info("用户输入为通用对话（intent_action=chat）")
            return self._chat_intent(prompt)

        # 仅当用户提示词里出现绝对路径字面量时作补充（不理解语义，只认路径形状）
        prompt_paths = extract_filesystem_paths(prompt)

        intents: List[TrainIntent] = []
        for idx, parsed in enumerate(parsed_list):
            if not isinstance(parsed, dict):
                raise ValueError(f"第 {idx + 1} 个任务格式不正确: {parsed}")

            task_type = self._canonicalize_task_type(parsed.get("task_type", ""))
            mode = self._canonicalize_mode(parsed.get("mode") or intent_action or "")
            categories = self._canonicalize_categories(parsed.get("categories", []))
            dataset_hint = str(parsed.get("dataset_hint", "") or "").strip()
            dataset_path = self._resolve_dataset_path(parsed, dataset_hint, prompt_paths)
            if dataset_path and (not dataset_hint or looks_like_filesystem_path(dataset_hint)):
                try:
                    dataset_hint = Path(dataset_path).name
                except Exception:
                    pass
            output_dir = self._resolve_output_dir(parsed)
            baseline_model = self._extract_baseline_model(parsed.get("baseline_model"))
            wants_baseline = bool(parsed.get("wants_baseline", False)) or bool(baseline_model)

            location_drives, location_frags, location_kws = self._parse_location_fields(
                parsed, dataset_hint
            )

            if "needs_training" in parsed:
                needs_training = bool(parsed.get("needs_training"))
            else:
                needs_training = mode in {"train", "dry_run"}

            # 模式与 needs_training 对齐（仅枚举一致性，不读用户原句）
            if mode in {"locate", "chat", "test"}:
                needs_training = False
            elif needs_training and mode not in {"train", "dry_run"}:
                mode = "train"

            data_analysis = bool(
                parsed.get(
                    "data_analysis",
                    True if mode in {"train", "dry_run"} else False,
                )
            )
            mentions_augmentation = bool(parsed.get("mentions_augmentation", False))
            data_augmentation = self._canonicalize_data_augmentation(
                parsed.get("data_augmentation", [])
            )
            if data_augmentation and not mentions_augmentation:
                logger.warning(
                    "LLM 返回了数据增强选项但 mentions_augmentation=false，已忽略: %s",
                    data_augmentation,
                )
                data_augmentation = []

            if not task_type and mode == "train" and needs_training:
                raise ValueError(f"第 {idx + 1} 个任务未返回有效任务类型: {parsed}")

            intents.append(
                TrainIntent(
                    task_type=task_type,
                    categories=categories,
                    dataset_hint=dataset_hint,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    mode=mode,
                    raw_prompt=prompt,
                    needs_training=needs_training,
                    data_analysis=data_analysis,
                    mentions_augmentation=mentions_augmentation,
                    data_augmentation=data_augmentation,
                    baseline_model=baseline_model,
                    wants_baseline=wants_baseline,
                    location_drives=location_drives,
                    location_path_fragments=location_frags,
                    location_name_keywords=location_kws,
                )
            )

        if not intents:
            raise ValueError("LLM 未返回任何有效任务")

        if (
            not top_level_needs_training
            and intent_action in {"", "chat"}
            and all(i.mode == "chat" for i in intents)
            and not any(
                i.task_type or i.dataset_hint or i.dataset_path for i in intents
            )
        ):
            logger.info("无实质任务字段，进入通用对话")
            return self._chat_intent(prompt)

        return intents

    def _resolve_dataset_path(
        self,
        parsed: Dict[str, Any],
        dataset_hint: str,
        prompt_paths: List[str],
    ) -> str:
        """接受 LLM/提示词中的路径；不存在的臆造路径降级为空（hint 仍保留）。"""
        raw = str(
            parsed.get("dataset_path")
            or parsed.get("dataset_dir")
            or parsed.get("data_path")
            or ""
        ).strip()
        if not raw and dataset_hint and looks_like_filesystem_path(dataset_hint):
            raw = dataset_hint
        if not raw and prompt_paths:
            for p in prompt_paths:
                accepted = self._accept_path(p, prompt_paths)
                if accepted:
                    return accepted
            return _clean_extracted_path(prompt_paths[0])

        if not raw:
            return ""
        accepted = self._accept_path(raw, prompt_paths)
        if accepted:
            logger.info("解析到显式数据集路径: %s", accepted)
            return accepted
        logger.info("忽略不可信/不存在的 dataset_path: %s", raw)
        return ""

    @staticmethod
    def _accept_path(raw_path: str, prompt_paths: List[str]) -> str:
        text = _clean_extracted_path(raw_path)
        if not text:
            return ""
        try:
            p = Path(os.path.expandvars(text)).expanduser()
            if p.exists() and p.is_dir():
                return text
        except OSError:
            pass
        lowered = {_clean_extracted_path(x).lower() for x in prompt_paths}
        if text.lower() in lowered:
            return text
        return ""

    def _resolve_output_dir(self, parsed: Dict[str, Any]) -> str:
        raw = str(
            parsed.get("output_dir")
            or parsed.get("model_output_dir")
            or parsed.get("export_dir")
            or ""
        ).strip()
        if raw:
            logger.info("解析到模型输出目录: %s", raw)
        return raw

    @staticmethod
    def _parse_location_fields(
        parsed: Dict[str, Any],
        dataset_hint: str,
    ) -> tuple:
        """读取 LLM 的位置线索字段。"""

        def _as_str_list(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, str):
                parts = re.split(r"[\s,，、/;|]+", val.strip())
                return [p.strip() for p in parts if p.strip()]
            if isinstance(val, list):
                out: List[str] = []
                for item in val:
                    s = str(item or "").strip()
                    if s:
                        out.append(s)
                return out
            return []

        drives = _as_str_list(parsed.get("location_drives") or parsed.get("drives"))
        norm_drives: List[str] = []
        for d in drives:
            m = re.match(r"^\s*([A-Za-z])\s*:?\s*[\\/]?\s*$", d)
            if m:
                norm_drives.append(f"{m.group(1).upper()}:\\")
            elif re.match(r"^[A-Za-z]:[\\/]", d):
                norm_drives.append(d[:2].upper() + "\\")
            else:
                norm_drives.append(d if d.endswith(("\\", "/")) else d + "\\")
        seen = set()
        drives_out: List[str] = []
        for d in norm_drives:
            k = d.lower()
            if k not in seen:
                seen.add(k)
                drives_out.append(d)

        frags = _as_str_list(
            parsed.get("location_path_fragments") or parsed.get("path_fragments")
        )
        kws = _as_str_list(
            parsed.get("location_name_keywords") or parsed.get("name_keywords")
        )
        if dataset_hint and not looks_like_filesystem_path(dataset_hint):
            for part in re.split(r"[\s,，、/;|]+", dataset_hint):
                part = part.strip()
                if part and part not in kws and part not in frags:
                    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{1,63}", part):
                        frags.append(part)
                    else:
                        kws.append(part)

        return drives_out, frags, kws

    def _canonicalize_task_type(self, raw: str) -> str:
        """把 LLM 返回的任务类型规范化为内部标准名。"""
        if not raw:
            return ""
        lower = raw.strip().lower()
        for alias, canonical in TASK_ALIASES.items():
            if alias.lower() == lower:
                return canonical
        for canonical in set(TASK_ALIASES.values()):
            if canonical.lower() == lower:
                return canonical
        return ""

    def _canonicalize_mode(self, raw: str) -> str:
        """只规范化 LLM 返回的 mode，不根据 prompt 关键词覆盖。"""
        if not raw:
            return "train"
        lower = str(raw).strip().lower()
        if lower in {"locate", "find", "search", "inspect", "browse"}:
            return "locate"
        if lower in {"chat", "talk", "general"}:
            return "chat"
        if lower in {"test", "testing", "evaluate", "eval", "infer", "inference"}:
            return "test"
        if lower in {"dry_run", "dry-run", "dryrun", "dry"}:
            return "dry_run"
        if lower in {"train", "training"}:
            return "train"
        if lower in _VALID_MODES:
            return lower
        return "train"

    def _extract_baseline_model(self, raw: Any) -> Optional[str]:
        """只接受 LLM 给出的 .dvp 路径。"""
        if not raw:
            return None
        s = str(raw).strip().strip('"\'')
        if s.lower().endswith(".dvp"):
            return s
        return None

    def _canonicalize_categories(self, raw: Any) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            raw = [raw]
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
            for option in DATA_AUGMENTATION_OPTIONS:
                if option.lower() in text.lower() or text.lower() in option.lower():
                    if option not in result:
                        result.append(option)
                    break
        return result
