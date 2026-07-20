"""
AutoTrain 4.0 智能建议引擎（Smart Suggestion Engine）。

采用 LLM + 本地规则混合方案：
- 本地语义分类器立即识别意图（任务类型、阶段、缺失信息）
- 多生成器并行产出候选建议
- LLM 异步补全语义层面的下一句话预测
- 混合排序、去重、截断，最终返回 Top-K

目标延迟：本地建议 0ms，LLM 建议 100~300ms。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple


def _has_env_key() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@dataclass
class IntentProfile:
    """当前输入的语义画像。"""

    task_types: List[str] = field(default_factory=list)
    stage: str = "start"  # start / config / chain / question / unknown
    mentioned_augmentation: bool = False
    mentioned_baseline: bool = False
    mentioned_categories: bool = False
    mentioned_dataset: bool = False
    is_multi_task: bool = False
    raw_text: str = ""


@dataclass
class SuggestionCandidate:
    text: str
    score: float = 0.0
    source: str = "local"  # local / llm / pattern


class SmartSuggestionEngine:
    """
    智能建议引擎。

    对外接口保持简单：
        suggest(text) -> List[str]
        take_suggestion(current_text, suggestion) -> str
        on_llm_result(callback)
    """

    TASK_TYPES = ["OCR", "CLS", "DET", "DETR", "ISEG", "SSEG", "ROTATE"]
    TASK_ALIASES: Dict[str, List[str]] = {
        "OCR": ["ocr", "文字识别", "文本识别"],
        "CLS": ["cls", "分类", "classify", "classifier"],
        "DET": ["det", "目标检测", "检测"],
        "DETR": ["detr"],
        "ISEG": ["iseg", "实例分割"],
        "SSEG": ["sseg", "语义分割"],
        "ROTATE": ["rotate", "旋转框", "旋转检测"],
    }

    AUGMENTATIONS: List[Tuple[str, List[str]]] = [
        ("启用随机固定角度旋转", ["旋转", "随机旋转"]),
        ("启用随机比例缩放", ["缩放", "随机缩放"]),
        ("启用随机翻转", ["翻转"]),
        ("启用随机裁剪", ["裁剪"]),
        ("启用随机调整亮度", ["亮度"]),
        ("启用随机调整对比度", ["对比度"]),
    ]

    def __init__(self, llm_client: Optional[object] = None) -> None:
        self.llm_client = llm_client
        self._llm_enabled = llm_client is not None and _has_env_key()
        self._cache: Dict[str, List[str]] = {}
        self._cache_lock = threading.Lock()
        self._pending_text: Optional[str] = None
        self._pending_intent: Optional[IntentProfile] = None
        self._pending_time: float = 0.0
        self._debounce_ms: int = 180
        self._llm_thread: Optional[threading.Thread] = None
        self._llm_callback: Optional[Callable[[str, List[str]], None]] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def suggest(self, text: str) -> List[str]:
        """根据当前输入返回建议列表（本地结果立即返回，LLM 异步增强）。"""
        raw_text = text
        text = text.strip()

        # 1. 语义分类
        intent = self._classify_intent(raw_text)

        # 2. 立即生成本地候选
        local_candidates = self._generate_all_candidates(intent, raw_text)

        # 3. 触发 LLM 异步补全
        if text:
            self._request_llm_suggestions(raw_text, intent)

        # 4. 排序去重返回 Top-5
        return self._rank_and_dedupe(local_candidates)[:5]

    def take_suggestion(self, current_text: str, suggestion: str) -> str:
        """把建议追加到当前输入文本中。"""
        current_text = current_text.strip()
        if not current_text:
            return suggestion

        if suggestion.startswith(("。", "，", "；", "、", " ")):
            return current_text + suggestion

        if current_text[-1] not in "，。；！？、":
            return current_text + "，" + suggestion

        return current_text + suggestion

    def on_llm_result(self, callback: Callable[[str, List[str]], None]) -> None:
        """注册 LLM 建议返回时的回调（text, suggestions）。"""
        self._llm_callback = callback

    # ------------------------------------------------------------------
    # 语义分类
    # ------------------------------------------------------------------
    def _classify_intent(self, text: str) -> IntentProfile:
        lower = text.lower()
        intent = IntentProfile(raw_text=text)

        # 识别任务类型
        found_tasks: List[str] = []
        for task, aliases in self.TASK_ALIASES.items():
            if any(alias in lower for alias in aliases):
                found_tasks.append(task)
        intent.task_types = found_tasks

        # 判断是否多任务
        intent.is_multi_task = len(found_tasks) > 1 or "和" in text or "然后" in text or "再" in text

        # 判断阶段
        if re.search(r"哪里|在哪|怎么|什么|如何|吗|？|\?", text):
            intent.stage = "question"
        elif any(alias in lower for _, aliases in self.AUGMENTATIONS for alias in aliases) or "数据增强" in text:
            intent.mentioned_augmentation = True
            intent.stage = "config"
        elif "类别" in text or "辨别" in text or "区分" in text:
            intent.mentioned_categories = True
            intent.stage = "config"
        elif "baseline" in lower or "基准" in text or "对比" in text:
            intent.mentioned_baseline = True
            intent.stage = "config"
        elif "数据集" in text or "文件夹" in text:
            intent.mentioned_dataset = True
        elif intent.is_multi_task or "然后" in text or "再" in text:
            intent.stage = "chain"
        elif found_tasks:
            intent.stage = "start"

        return intent

    # ------------------------------------------------------------------
    # 候选生成器
    # ------------------------------------------------------------------
    def _generate_all_candidates(
        self, intent: IntentProfile, text: str
    ) -> List[SuggestionCandidate]:
        candidates: List[SuggestionCandidate] = []
        lower = text.lower()

        # 1. 空输入：默认推荐
        if not text.strip():
            for task in self.TASK_TYPES[:5]:
                candidates.append(SuggestionCandidate(f"训练 {task} 模型", 1.0, "local"))
            return candidates

        # 2. 问题类输入：不训练，推荐常见训练表述
        if intent.stage == "question":
            candidates.append(SuggestionCandidate("训练 OCR 模型", 0.8, "local"))
            candidates.append(SuggestionCandidate("训练 CLS 模型", 0.8, "local"))
            candidates.append(SuggestionCandidate("训练 DET 模型", 0.75, "local"))
            return candidates

        # 3. 已识别任务类型：推荐该任务的配置或下一步
        if intent.task_types:
            primary = intent.task_types[0]
            candidates.extend(self._task_config_candidates(primary, intent, lower))

        # 4. 多任务：推荐链式下一步
        if intent.is_multi_task or "然后" in text or "再" in text:
            candidates.extend(self._chain_candidates(intent, lower))

        # 5. 未识别任务类型：推荐可能的任务
        if not intent.task_types:
            candidates.extend(self._task_guess_candidates(lower))

        # 6. 数据增强上下文推荐
        if "增强" in text or "数据" in text or "预处理" in text:
            candidates.extend(self._augmentation_candidates(lower))

        # 7. 通用补全模式
        candidates.extend(self._pattern_candidates(text, intent))

        return candidates

    def _task_config_candidates(
        self, task: str, intent: IntentProfile, lower: str
    ) -> List[SuggestionCandidate]:
        cands: List[SuggestionCandidate] = []

        if task == "CLS" and not intent.mentioned_categories:
            cands.append(SuggestionCandidate("辨别 A 和 B", 0.95, "local"))
            cands.append(SuggestionCandidate("区分类别", 0.85, "local"))

        if task == "OCR" and not intent.mentioned_augmentation:
            cands.append(SuggestionCandidate("启用随机比例缩放", 0.9, "local"))

        if not intent.mentioned_augmentation:
            cands.append(SuggestionCandidate("不用数据增强", 0.82, "local"))
            cands.append(SuggestionCandidate("启用随机固定角度旋转", 0.78, "local"))
        else:
            cands.append(SuggestionCandidate("再启用随机翻转", 0.75, "local"))

        if not intent.mentioned_baseline:
            cands.append(SuggestionCandidate("与 baseline 对比", 0.7, "local"))

        if not intent.mentioned_dataset:
            cands.append(SuggestionCandidate("使用训练集", 0.65, "local"))

        return cands

    def _chain_candidates(self, intent: IntentProfile, lower: str) -> List[SuggestionCandidate]:
        cands: List[SuggestionCandidate] = []
        existing = set(intent.task_types)
        for task in self.TASK_TYPES:
            if task not in existing:
                cands.append(SuggestionCandidate(f"然后训练 {task} 模型", 0.85, "local"))
        # 多任务收尾
        cands.append(SuggestionCandidate("都不启用数据增强", 0.7, "local"))
        return cands

    def _task_guess_candidates(self, lower: str) -> List[SuggestionCandidate]:
        cands: List[SuggestionCandidate] = []
        if "文字" in lower or "字符" in lower or "识别" in lower:
            cands.append(SuggestionCandidate("训练 OCR 模型", 0.9, "local"))
        if "框" in lower or "角度" in lower or "旋转" in lower:
            cands.append(SuggestionCandidate("训练 ROTATE 模型", 0.85, "local"))
        if "分割" in lower or "mask" in lower or "像素" in lower:
            cands.append(SuggestionCandidate("训练 ISEG 模型", 0.85, "local"))
            cands.append(SuggestionCandidate("训练 SSEG 模型", 0.8, "local"))
        if "检测" in lower or "目标" in lower or "bbox" in lower:
            cands.append(SuggestionCandidate("训练 DET 模型", 0.9, "local"))
            cands.append(SuggestionCandidate("训练 DETR 模型", 0.8, "local"))
        if not cands:
            for task in self.TASK_TYPES[:5]:
                cands.append(SuggestionCandidate(f"训练 {task} 模型", 0.6, "local"))
        return cands

    def _augmentation_candidates(self, lower: str) -> List[SuggestionCandidate]:
        cands: List[SuggestionCandidate] = []
        already: Set[str] = set()
        for phrase, aliases in self.AUGMENTATIONS:
            if any(alias in lower for alias in aliases):
                already.add(phrase)
        for phrase, _ in self.AUGMENTATIONS:
            if phrase not in already:
                cands.append(SuggestionCandidate(phrase, 0.8, "local"))
        return cands

    def _pattern_candidates(self, text: str, intent: IntentProfile) -> List[SuggestionCandidate]:
        cands: List[SuggestionCandidate] = []
        lower = text.lower()

        # 如果结尾像是还没说完，补全
        if text.rstrip().endswith(("和", "与", "跟")):
            cands.append(SuggestionCandidate("DET 模型", 0.85, "pattern"))
            cands.append(SuggestionCandidate("CLS 模型", 0.8, "pattern"))

        if text.rstrip().endswith(("启用", "打开", "使用")):
            cands.append(SuggestionCandidate("随机固定角度旋转", 0.85, "pattern"))
            cands.append(SuggestionCandidate("随机比例缩放", 0.82, "pattern"))

        if text.rstrip().endswith(("辨别", "区分", "分类")):
            cands.append(SuggestionCandidate("A 和 B", 0.85, "pattern"))

        if not intent.mentioned_baseline and "baseline" not in lower and "基准" not in lower:
            cands.append(SuggestionCandidate("，与 baseline 对比", 0.6, "pattern"))

        return cands

    # ------------------------------------------------------------------
    # 排序与去重
    # ------------------------------------------------------------------
    def _rank_and_dedupe(self, candidates: List[SuggestionCandidate]) -> List[str]:
        seen: Set[str] = set()
        unique: List[SuggestionCandidate] = []
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            key = c.text.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        return [c.text for c in unique[:5]]

    # ------------------------------------------------------------------
    # LLM 异步补全
    # ------------------------------------------------------------------
    def _request_llm_suggestions(self, text: str, intent: IntentProfile) -> None:
        if not self._llm_enabled:
            return

        with self._cache_lock:
            if text in self._cache:
                # 命中缓存，直接回调
                if self._llm_callback:
                    try:
                        self._llm_callback(text, list(self._cache[text]))
                    except Exception:
                        pass
                return

        with self._cache_lock:
            self._pending_text = text
            self._pending_intent = intent
            self._pending_time = time.time()

        if self._llm_thread is None or not self._llm_thread.is_alive():
            self._llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
            self._llm_thread.start()

    def _llm_worker(self) -> None:
        while True:
            with self._cache_lock:
                text = self._pending_text
                intent = self._pending_intent
                pending_time = self._pending_time

            if text is None:
                return

            elapsed_ms = (time.time() - pending_time) * 1000
            if elapsed_ms < self._debounce_ms:
                time.sleep(0.02)
                continue

            with self._cache_lock:
                self._pending_text = None
                self._pending_intent = None

            suggestions = self._fetch_llm_suggestions(text, intent)
            if suggestions:
                with self._cache_lock:
                    self._cache[text] = suggestions
                if self._llm_callback:
                    try:
                        self._llm_callback(text, suggestions)
                    except Exception:
                        pass

    def _fetch_llm_suggestions(
        self, text: str, intent: Optional[IntentProfile]
    ) -> List[str]:
        if self.llm_client is None:
            return []
        try:
            system_msg = (
                "你是 AutoTrain 4.0 的智能输入建议助手。用户正在用自然语言描述深度学习训练需求。\n"
                "请根据当前输入的完整语义，预测用户下一步最可能继续输入的 3-5 个简短片段。\n"
                "平台支持的任务类型：OCR、CLS（分类）、DET（目标检测）、DETR、ISEG（实例分割）、SSEG（语义分割）、ROTATE（旋转框检测）。\n"
                "可配置项：数据增强（随机旋转、随机缩放、随机翻转、随机裁剪、随机亮度、随机对比度）、baseline 对比、多任务链式训练。\n"
                "要求：\n"
                "1. 基于语义预测，不要简单重复关键词\n"
                "2. 片段应该自然延续当前输入，可接在末尾\n"
                "3. 若输入已提及某个任务，推荐相关配置或下一个任务\n"
                "4. 若输入是问题，推荐转化为训练需求的表述\n"
                "5. 每个片段不超过 15 个字符\n"
                "6. 只返回 JSON 数组，不要解释"
            )
            user_msg = f"当前输入：{text}\n请给出 3-5 个后续建议片段。"
            raw = self.llm_client.chat(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.25,
            )
            if not raw:
                return []

            arr = self._extract_json_array(raw)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:5]
        except Exception:
            pass
        return []

    @staticmethod
    def _extract_json_array(raw: str):
        """从 LLM 输出中提取 JSON 数组。"""
        # 优先找方括号包裹的内容
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # 可能没有方括号，尝试整个字符串解析
        try:
            return json.loads(raw)
        except Exception:
            return []
