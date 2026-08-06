"""
自然语言意图解析器（LLM 直接驱动，无规则兜底）。

把用户输入（如"训练一个cls模型来辨别郁金香和菊花"）解析成结构化意图，
供数据集解析和配置生成使用。
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
    select_dataset_path_from_prompt,
    select_output_path_from_prompt,
)
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
        }

    def to_task_dict(self) -> Dict[str, Any]:
        """返回供单任务下游使用的意图，避免原始多任务提示词串扰。"""
        data = self.to_dict()
        data.pop("raw_prompt", None)
        return data


class IntentParser:
    """意图解析器（第一层 LLM）。"""

    # 测试/评估意图关键词（仅用于轻量预判断，不单独决定最终 mode）
    TEST_KEYWORDS = [
        "测试", "试一下", "试下", "评估", "评测", "验证", "跑一遍",
        "test", "testing", "evaluate", "eval", "infer", "inference",
    ]
    # 仅定位/查找数据集（无开训）
    LOCATE_KEYWORDS = [
        "定位", "查找", "找到", "找一下", "找下", "搜一下", "搜索",
        "在哪", "在哪儿", "哪里", "哪儿", "路径是", "目录是",
        "locate", "find dataset", "where is", "which folder",
    ]
    # 明确开训/启动训练流程
    TRAIN_KEYWORDS = [
        "训练", "开训", "训一个", "训个", "跑训练", "开始训练", "启动训练",
        "fine-tune", "finetune", "fine tune", "train a", "train an", "training",
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    @staticmethod
    def _chat_intent(prompt: str) -> List[TrainIntent]:
        return [
            TrainIntent(
                task_type="",
                categories=[],
                dataset_hint="",
                dataset_path="",
                output_dir="",
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

    def parse(self, prompt: str) -> List[TrainIntent]:
        """解析用户输入为一个或多个意图，完全由 LLM 负责（本地仅做规范与纠偏）。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("输入不能为空")

        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法解析自然语言意图（已取消规则兜底）")

        logger.info("第一层 LLM：解析用户意图")
        extracted = self.llm.extract_intent(prompt)
        if extracted is None:
            raise RuntimeError("LLM 意图解析失败，无法继续（已取消规则兜底）")

        # extract_intent 应返回完整 dict；兼容误返回 list / 单 task 对象
        top_level_needs_training = True
        intent_action = ""
        if isinstance(extracted, dict):
            intent_action = str(extracted.get("intent_action") or "").strip().lower()
            if "needs_training" in extracted:
                top_level_needs_training = bool(extracted.get("needs_training"))
            elif intent_action in {"test", "locate", "chat"}:
                top_level_needs_training = False
            elif intent_action == "train":
                top_level_needs_training = True
            parsed_list = extracted.get("tasks", [])
            if parsed_list is None:
                parsed_list = []
            # 旧单对象：无 tasks 键时整个 dict 当 task（extract_intent 已尽量包好）
            if "tasks" not in extracted and any(
                k in extracted for k in ("task_type", "mode", "dataset_hint", "dataset_path")
            ):
                parsed_list = [extracted]
        elif isinstance(extracted, list):
            parsed_list = extracted
        else:
            raise ValueError(f"LLM 返回的任务格式不正确: {extracted}")

        if not isinstance(parsed_list, list):
            raise ValueError(f"LLM 返回的任务格式不正确: {parsed_list}")

        # 空任务 → 通用对话（除非顶层标明 locate 却漏了 task，下面还会用线索补）
        if not parsed_list:
            if intent_action == "locate" or self._prompt_looks_like_locate(prompt):
                logger.info("空 tasks 但像定位请求，构造 locate 意图")
                return [
                    TrainIntent(
                        task_type=self._reconcile_task_type_with_prompt("", prompt),
                        categories=[],
                        dataset_hint="",
                        dataset_path=select_dataset_path_from_prompt(prompt) or "",
                        output_dir="",
                        mode="locate",
                        raw_prompt=prompt,
                        needs_training=False,
                        data_analysis=False,
                        mentions_augmentation=False,
                        data_augmentation=[],
                        baseline_model=None,
                        wants_baseline=False,
                    )
                ]
            logger.info("LLM 返回空任务列表，进入通用对话模式")
            return self._chat_intent(prompt)

        # 顶层 needs_training=false 且全部是 chat/无实质 task → 对话
        # 但 locate/test 必须保留 tasks，不能再被旧逻辑一律打成 chat
        if not top_level_needs_training and intent_action == "chat":
            logger.info("用户输入为通用对话（intent_action=chat）")
            return self._chat_intent(prompt)

        # 整句级路径兜底：即使用户写了绝对路径而 LLM 漏提，也能直达
        prompt_dataset_path = select_dataset_path_from_prompt(prompt) or ""
        prompt_output_dir = select_output_path_from_prompt(prompt) or ""

        intents: List[TrainIntent] = []
        for idx, parsed in enumerate(parsed_list):
            if not isinstance(parsed, dict):
                raise ValueError(f"第 {idx + 1} 个任务格式不正确: {parsed}")

            task_type = self._canonicalize_task_type(parsed.get("task_type", ""))
            # LLM 偶发把 iseg 误判成语义分割等；用提示词别名复核
            task_type = self._reconcile_task_type_with_prompt(task_type, prompt)
            mode = self._canonicalize_mode(
                parsed.get("mode") or intent_action or "train",
                prompt,
            )
            categories = self._canonicalize_categories(parsed.get("categories", []))
            dataset_hint = str(parsed.get("dataset_hint", "") or "").strip()
            dataset_path, dataset_hint = self._resolve_dataset_fields(
                parsed,
                dataset_hint=dataset_hint,
                prompt=prompt,
                prompt_dataset_path=prompt_dataset_path,
            )
            output_dir = self._resolve_output_dir(
                parsed,
                prompt=prompt,
                prompt_output_dir=prompt_output_dir,
            )
            baseline_model = self._extract_baseline_model(parsed.get("baseline_model"), prompt)
            wants_baseline = self._wants_baseline_comparison(
                parsed.get("wants_baseline"), baseline_model, prompt
            )

            if "needs_training" in parsed:
                needs_training = bool(parsed.get("needs_training"))
            elif mode in {"locate", "chat", "test"}:
                needs_training = False
            elif mode in {"train", "dry_run"}:
                needs_training = True
            else:
                needs_training = bool(top_level_needs_training)

            # 模式与 needs_training 对齐
            if mode == "locate":
                needs_training = False
            elif mode == "chat":
                needs_training = False
            elif mode == "test":
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

            # 只有当用户明确提到数据增强时才保留增强选项，否则清空，避免前端误点增强菜单。
            if data_augmentation and not mentions_augmentation:
                logger.warning("LLM 返回了数据增强选项但用户未明确提及，已忽略: %s", data_augmentation)
                data_augmentation = []

            if not task_type:
                # locate 允许暂时无 task_type；train 仍要求
                if mode == "train" and needs_training:
                    raise ValueError(f"第 {idx + 1} 个任务未返回有效任务类型: {parsed}")
                if mode == "test" and needs_training:
                    raise ValueError(f"第 {idx + 1} 个任务未返回有效任务类型: {parsed}")

            # dry_run 覆盖：仅当整个请求明确提到 dry/只生成 时才生效
            if mode == "train" and ("dry" in prompt.lower() or "只生成" in prompt):
                mode = "dry_run"

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
                )
            )

        if not intents:
            raise ValueError("LLM 未返回任何有效任务")

        # 顶层误标 needs_training=false 且 action=chat，但 tasks 实际是 locate/test：保留 tasks
        if (
            not top_level_needs_training
            and intent_action in {"", "chat"}
            and all(i.mode == "chat" for i in intents)
            and not any(i.task_type or i.dataset_hint or i.dataset_path for i in intents)
        ):
            logger.info("无实质任务字段，进入通用对话")
            return self._chat_intent(prompt)

        return intents

    def _prompt_looks_like_locate(self, prompt: str) -> bool:
        """提示词是否像「只定位数据集」。"""
        text = prompt or ""
        if not text:
            return False
        lower = text.lower()
        has_locate = any(kw.lower() in lower or kw in text for kw in self.LOCATE_KEYWORDS)
        has_train = self._prompt_has_train_verb(text)
        return has_locate and not has_train

    def _prompt_has_train_verb(self, prompt: str) -> bool:
        """是否出现真正的开训动词（「训练集/训练数据」等名词不算）。"""
        text = prompt or ""
        if not text:
            return False
        # 去掉「训练集/训练数据…」后再匹配开训动词，避免「定位 OCR 训练集」误判为 train
        scrubbed = re.sub(
            r"训练集|训练数据|训练样本|训练图片|训练图像|"
            r"training\s*set|train\s*set|train\s*data|training\s*data",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        lower = scrubbed.lower()
        for kw in self.TRAIN_KEYWORDS:
            kl = kw.lower()
            if kl in lower or kw in scrubbed:
                return True
        if re.search(r"(?i)\btrain(?:ing)?\b", scrubbed):
            if not re.search(
                r"(?i)\btrain(?:ing)?\s*(set|data|folder|dir|directory)\b",
                text,
            ):
                return True
        return False

    def _resolve_dataset_fields(
        self,
        parsed: Dict[str, Any],
        dataset_hint: str,
        prompt: str,
        prompt_dataset_path: str,
    ) -> tuple:
        """合并 LLM 字段与提示词路径，得到 (dataset_path, dataset_hint)。

        仅当路径真实存在，或路径字面量来自用户提示词时，才写入 dataset_path。
        LLM 脑补的不存在路径降级为 dataset_hint，交给 NL 锚点深搜。
        """
        raw_path = str(
            parsed.get("dataset_path")
            or parsed.get("dataset_dir")
            or parsed.get("data_path")
            or ""
        ).strip()

        # dataset_hint 若本身就是路径，提升为 dataset_path 候选
        if not raw_path and dataset_hint and looks_like_filesystem_path(dataset_hint):
            raw_path = dataset_hint

        if not raw_path and prompt_dataset_path:
            raw_path = prompt_dataset_path

        hint_out = dataset_hint
        dataset_path = ""

        if raw_path:
            accepted = self._accept_dataset_path(raw_path, prompt, prompt_dataset_path)
            if accepted:
                dataset_path = accepted
                if not hint_out or looks_like_filesystem_path(hint_out):
                    try:
                        hint_out = Path(dataset_path).name
                    except Exception:
                        hint_out = dataset_hint
                logger.info("解析到显式数据集路径: %s", dataset_path)
            else:
                # 降级：用路径最后一段或原 hint 做模糊关键字
                try:
                    demoted = Path(_clean_extracted_path(raw_path)).name
                except Exception:
                    demoted = raw_path
                if demoted and not looks_like_filesystem_path(demoted):
                    if not hint_out:
                        hint_out = demoted
                    elif demoted not in hint_out and hint_out not in demoted:
                        hint_out = f"{hint_out} {demoted}".strip()
                logger.info(
                    "忽略不存在的 LLM/推断路径，保留为 hint 供 NL 定位: path=%s hint=%s",
                    raw_path,
                    hint_out,
                )

        return dataset_path, hint_out

    @staticmethod
    def _accept_dataset_path(
        raw_path: str,
        prompt: str,
        prompt_dataset_path: str = "",
    ) -> str:
        """返回可信任的 dataset_path；不可信则返回空串。"""
        text = _clean_extracted_path(raw_path)
        if not text:
            return ""

        # 1) 目录真实存在 → 接受
        try:
            p = Path(os.path.expandvars(text)).expanduser()
            if p.exists() and p.is_dir():
                return text
        except OSError:
            pass

        # 2) 路径字面量来自提示词正则提取 → 接受（后续 resolve 再报不存在）
        prompt_paths = set()
        if prompt_dataset_path:
            prompt_paths.add(_clean_extracted_path(prompt_dataset_path).lower())
        for item in extract_filesystem_paths(prompt or ""):
            prompt_paths.add(_clean_extracted_path(item).lower())
        if text.lower() in prompt_paths:
            return text

        # 3) 提示词中连续出现该路径子串（用户手打）
        if text in (prompt or "") or text.replace("/", "\\") in (prompt or ""):
            return text

        return ""

    def _resolve_output_dir(
        self,
        parsed: Dict[str, Any],
        prompt: str,
        prompt_output_dir: str,
    ) -> str:
        """合并 LLM 与提示词中的模型输出目录。"""
        raw = str(
            parsed.get("output_dir")
            or parsed.get("model_output_dir")
            or parsed.get("export_dir")
            or ""
        ).strip()
        if not raw and prompt_output_dir:
            raw = prompt_output_dir
        if raw:
            logger.info("解析到模型输出目录: %s", raw)
        return raw

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

    def _reconcile_task_type_with_prompt(self, task_type: str, prompt: str) -> str:
        """当提示词含明确任务别名时，纠正 LLM 的错误 task_type。

        例如用户写 iseg，但模型返回「语义分割」。多个别名同时出现时不覆盖。
        """
        text = str(prompt or "")
        if not text:
            return task_type

        # 按别名长度降序，避免 det 误伤 detr
        aliases = sorted(TASK_ALIASES.items(), key=lambda kv: len(str(kv[0])), reverse=True)
        hits: List[str] = []
        seen = set()
        lower = text.lower()
        for alias, canonical in aliases:
            a = str(alias)
            if not a:
                continue
            # 英文/数字别名用词边界；中文直接子串
            if re.fullmatch(r"[A-Za-z0-9_\-]+", a):
                if not re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", text):
                    continue
            else:
                if a not in text and a.lower() not in lower:
                    continue
            if canonical not in seen:
                seen.add(canonical)
                hits.append(canonical)

        if len(hits) == 1 and hits[0] != task_type:
            if task_type:
                logger.info(
                    "任务类型按提示词别名纠正: %s -> %s",
                    task_type,
                    hits[0],
                )
            return hits[0]
        if not task_type and len(hits) == 1:
            return hits[0]
        return task_type

    def _canonicalize_mode(self, raw: str, prompt: str) -> str:
        """规范 mode：理解用户动作，而不是默认 train。

        优先级：
        1. 提示词同时像「定位」且无开训动词 → locate（纠正 LLM 误标 train）
        2. LLM 返回的 mode / intent_action
        3. 提示词测试关键词 → test
        4. 默认 train（仅当 LLM 已给出 train 或未给出可识别动作时）
        """
        prompt_text = prompt or ""
        prompt_lower = prompt_text.lower()
        looks_locate = self._prompt_looks_like_locate(prompt_text)
        looks_test = any(kw.lower() in prompt_lower for kw in self.TEST_KEYWORDS)
        looks_train = self._prompt_has_train_verb(prompt_text)

        # 强信号：只定位不训练，覆盖 LLM 误返回的 train
        if looks_locate and not looks_train:
            return "locate"

        if raw:
            lower = str(raw).strip().lower()
            if lower in {"locate", "find", "search", "inspect", "browse"}:
                return "locate"
            if lower in {"chat", "talk", "general"}:
                return "chat"
            if lower in {"test", "testing", "evaluate", "eval", "infer", "inference"}:
                return "test"
            if lower in {"dry_run", "dry-run", "dryrun"}:
                return "dry_run"
            if lower in {"train", "training"}:
                # LLM 说 train，但用户其实在测试
                if looks_test and not looks_train:
                    return "test"
                if looks_locate and not looks_train:
                    return "locate"
                return "train"

        if looks_test and not looks_train:
            return "test"
        if looks_locate:
            return "locate"
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
