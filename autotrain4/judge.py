"""
第三层：需求分析与判断层。

在意图解析、数据集匹配之后，对用户需求做最终分析：
    1. 判断需求是否明确、可行。
    2. 判断选中的数据集是否真正满足用户意图。
    3. 输出是否继续执行**当前动作**（训练 / 仅定位等）的最终决策和必要修正建议。
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
    """需求判断结果。"""

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
    """需求判断器（第三层）。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # 默认使用 DeepSeek 专用客户端
        self.llm = llm_client or LLMClient.deepseek()

    def judge(
        self,
        intent: TrainIntent,
        match: DatasetMatch,
        task_context: Optional[str] = None,
    ) -> JudgeResult:
        """调用 LLM 做最终需求分析与判断。"""
        if not self.llm.available:
            raise RuntimeError("DeepSeek 未配置，无法执行第三层判断")

        mode = (intent.mode or "train").strip().lower()
        logger.info("第三层：分析需求并做出最终判断（mode=%s）", mode)

        if mode == "locate":
            action_desc = (
                "当前用户动作是 **locate（仅定位数据集）**：只需确认路径/文件夹是否匹配，"
                "**禁止**建议训练、建项目或启动 WebBridge。"
            )
            proceed_desc = (
                "proceed 为 true 表示定位结果可用，可以向用户回报路径；"
                "为 false 时说明数据集不匹配或无法确认。"
            )
        elif mode == "test":
            action_desc = "当前用户动作是 **test（测试/评估已有模型）**，不是开新训练。"
            proceed_desc = "proceed 为 true 表示可以继续测试；为 false 时必须说明原因。"
        else:
            action_desc = (
                "当前用户动作是 **train（训练）**。"
                "若用户原始输入其实是「定位/查找数据集」且没有开训意愿，"
                "必须 proceed=false，并在 reasoning 中说明应改为 locate。"
            )
            proceed_desc = "proceed 为 true 表示可以继续训练；为 false 时必须说明原因。"

        system_msg = (
            "你是 DLCV 训练平台的资深需求评审助手。"
            "请根据用户原始输入、解析意图和匹配到的数据集做判断。\n"
            f"{action_desc}\n"
            "返回 JSON：\n"
            "{\n"
            '  "proceed": true,\n'
            '  "reasoning": "对用户真实意图、数据集匹配度、风险的简要说明",\n'
            '  "adjustments": {"project_name": "可选", "description": "可选", '
            '"hyperparameters": {"max_epochs": 12}}\n'
            "}\n"
            "注意：\n"
            f"1. {proceed_desc}\n"
            "2. adjustments 只允许 project_name、description、hyperparameters；"
            "locate 模式下 adjustments 应为空对象；不要返回数据增强，"
            "数据增强必须严格遵守解析意图。\n"
            "3. 必须对照用户原始输入理解意图，不要被「解析意图」里误标的 train 带偏。\n"
            "4. 只返回 JSON，不要附加说明。"
        )
        # 始终带上原始提示词，避免被改写后的 task_context 抹掉「定位」语义
        raw = intent.raw_prompt or ""
        context = task_context if task_context is not None else raw
        user_msg = (
            f"用户原始输入：{raw}\n"
            f"当前任务描述：{context}\n"
            f"解析意图：{json.dumps(intent.to_task_dict(), ensure_ascii=False)}\n"
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
        if mode == "locate":
            adjustments = {}

        if not proceed:
            raise RuntimeError(f"DeepSeek 判断不可继续: {reasoning}")

        logger.info("判断结果: proceed=%s, reasoning=%s", proceed, reasoning)
        return JudgeResult(proceed=proceed, reasoning=reasoning, adjustments=adjustments)
