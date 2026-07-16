"""
第三层：DeepSeek 需求分析与判断层。

在意图解析、数据集匹配之后，调用 DeepSeek API 对用户需求做最终分析：
    1. 判断需求是否明确、可行。
    2. 判断选中的数据集是否真正满足用户意图。
    3. 输出是否继续执行训练的最终决策和必要修正建议。
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .llm_client import LLMClient
from .nlu import TrainIntent
from .resolver import DatasetMatch

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """DeepSeek 判断结果。"""

    proceed: bool
    reasoning: str
    adjustments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proceed": self.proceed,
            "reasoning": self.reasoning,
            "adjustments": self.adjustments,
        }


class RequirementJudge:
    """需求判断器（第三层 DeepSeek）。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # 默认使用 DeepSeek 专用客户端
        self.llm = llm_client or LLMClient.deepseek()

    def judge(
        self,
        intent: TrainIntent,
        match: DatasetMatch,
        task_context: Optional[str] = None,
    ) -> JudgeResult:
        """调用 DeepSeek 做最终需求分析与判断。"""
        if not self.llm.available:
            raise RuntimeError("DeepSeek 未配置，无法执行第三层判断")

        logger.info("第三层 DeepSeek：分析需求并做出最终判断")
        system_msg = (
            "你是 DLCV 训练平台的资深需求评审助手。请根据用户原始输入、解析出的意图以及最终匹配的数据集，"
            "判断当前方案是否可以继续执行训练。\n"
            "返回 JSON 格式：\n"
            "{\n"
            '  "proceed": true,\n'
            '  "reasoning": "对需求的理解、数据集匹配度、潜在风险的简要说明",\n'
            '  "adjustments": {"key": "value"}\n'
            "}\n"
            "注意：\n"
            "1. proceed 为 true 表示可以继续训练；为 false 时必须说明原因。\n"
            "2. adjustments 中可以给出对 project_name、description、hyperparameters 等的修正建议，没有则留空对象。\n"
            "3. 只返回 JSON，不要附加说明。"
        )
        context = task_context if task_context is not None else intent.raw_prompt
        user_msg = (
            f"用户原始输入：{context}\n"
            f"解析意图：{json.dumps(intent.to_dict(), ensure_ascii=False)}\n"
            f"数据集匹配：{json.dumps(match.to_dict(), ensure_ascii=False)}"
        )

        content = self.llm.chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )
        if not content:
            raise RuntimeError("DeepSeek 判断调用失败，无法继续")

        parsed = self.llm._safe_parse_json(content)
        if not parsed:
            raise RuntimeError(f"DeepSeek 返回内容无法解析为 JSON: {content[:200]}")

        proceed = bool(parsed.get("proceed", False))
        reasoning = str(parsed.get("reasoning", ""))
        adjustments = parsed.get("adjustments", {}) or {}

        if not proceed:
            raise RuntimeError(f"DeepSeek 判断不可继续: {reasoning}")

        logger.info("DeepSeek 判断结果: proceed=%s, reasoning=%s", proceed, reasoning)
        return JudgeResult(proceed=proceed, reasoning=reasoning, adjustments=adjustments)
