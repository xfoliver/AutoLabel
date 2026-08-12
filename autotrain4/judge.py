"""第三层：需求分析与判断层。

在意图解析、数据集匹配之后，对**当前动作**给出超参/项目名等 adjustments。

设计约定（阶段 D）：
- **mode 以 NLU 为准**，Judge 不再二次裁决 train/locate/test。
- locate / chat 路径本来就不调用 Judge；本模块只服务 train（及 dry_run）配置生成前。
- adjustments 仅允许 project_name / description / hyperparameters。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .llm import LLMClient
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
    """需求判断器（第三层）——只产出 adjustments，不改 mode。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # 与 NLU/Resolver 一致：默认读 AUTOTRAIN_LLM_* / 环境配置，不再写死 DeepSeek
        self.llm = llm_client or LLMClient()

    def judge(
        self,
        intent: TrainIntent,
        match: DatasetMatch,
        task_context: Optional[str] = None,
    ) -> JudgeResult:
        """调用 LLM 做需求分析，返回是否继续与 adjustments。"""
        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法执行第三层判断")

        mode = (intent.mode or "train").strip().lower()
        logger.info("第三层：分析需求并给出 adjustments（mode=%s，以 NLU 为准）", mode)

        # mode 已由 NLU 决定；此处仅描述当前动作，避免 LLM 再改 mode
        if mode == "dry_run":
            action_desc = (
                "当前用户动作是 **dry_run（仅生成配置）**。"
                "请评估数据集是否匹配，并给出合理的 project_name / description / hyperparameters 建议。"
            )
        else:
            action_desc = (
                "当前用户动作是 **train（训练）**（mode 已由上游 NLU 确定，不要建议改为 locate/test）。"
                "请评估选中的数据集是否满足意图，并给出必要的配置修正。"
            )

        system_msg = (
            "你是 DLCV 训练平台的资深需求评审助手。"
            "请根据用户原始输入、解析意图和匹配到的数据集做判断。\n"
            f"{action_desc}\n"
            "返回 JSON：\n"
            "{\n"
            '  "proceed": true,\n'
            '  "reasoning": "对数据集匹配度、风险与建议调整的简要说明",\n'
            '  "adjustments": {"project_name": "可选", "description": "可选", '
            '"hyperparameters": {"max_epochs": 12}}\n'
            "}\n"
            "注意：\n"
            "1. proceed 为 true 表示可以继续当前动作；为 false 时必须说明原因"
            "（例如数据集明显不匹配）。\n"
            "2. adjustments 只允许 project_name、description、hyperparameters；"
            "不要返回数据增强，数据增强必须严格遵守解析意图。\n"
            "3. **不要**尝试修改或否定 mode（train/locate/test/chat）；"
            "mode 由上游 NLU 决定，你只负责 adjustments。\n"
            "4. 只返回 JSON，不要附加说明。"
        )
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
            raise RuntimeError("LLM 判断调用失败，无法继续")

        parsed = self.llm._safe_parse_json(content)
        if not parsed:
            raise RuntimeError(f"LLM 返回内容无法解析为 JSON: {content[:200]}")

        proceed = bool(parsed.get("proceed", False))
        reasoning = str(parsed.get("reasoning", ""))
        adjustments = parsed.get("adjustments", {}) or {}
        if not isinstance(adjustments, dict):
            adjustments = {}
        # 白名单：防止 LLM 塞入 mode / data_augmentation 等
        allowed = {"project_name", "description", "hyperparameters"}
        adjustments = {k: v for k, v in adjustments.items() if k in allowed}

        if not proceed:
            raise RuntimeError(f"LLM 判断不可继续: {reasoning}")

        logger.info("判断结果: proceed=%s, reasoning=%s", proceed, reasoning)
        return JudgeResult(proceed=proceed, reasoning=reasoning, adjustments=adjustments)
