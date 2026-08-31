"""NL intent parsing via autotrain4.llm — fail hard; no local keyword guessing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from autolabel.errors import IntentError
from autolabel.labels import require_nonempty_label
from autolabel.models import LabelIntent, TargetSpec

logger = logging.getLogger(__name__)


INTENT_SYSTEM = """你是工业视觉自动标注意图解析器。用户用自然语言描述要标注的数据集与目标。
只输出一个 JSON 对象，不要 markdown。字段：
{
  "action": "auto_label",
  "annotation_task": "实例分割",
  "object_granularity": "instance",
  "targets": [{"label": "短类别名", "description": "语义描述", "include": [], "exclude": ["不要的东西"]}],
  "dataset_path": "若用户给了明确路径则填，否则 null",
  "dataset_clues": {"drive": "Z:", "path_fragments": ["片段"], "dataset_name": null},
  "references": {"positive": [], "negative": []},
  "overwrite_policy": "skip_existing",
  "preview_count": 5,
  "backend_preference": "auto",
  "review_policy": "low_confidence_only"
}
关键规则（强制）：
- targets[].label 必须是 1～8 个字的短名词，例如「手机」「螺栓」「裂纹」。
- 禁止把整句指令放进 label（错误例：「只识别图片中的手」「把该目录中的手机」）。
- 详细规则、排除项、掩码画法说明一律放进 description / exclude。
MVP 只支持实例分割；若用户说检测/分类等，仍输出 annotation_task=实例分割 并在 description 里保留原意提示。
"""


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        from autotrain4.llm.tasks import safe_parse_json  # type: ignore

        return safe_parse_json(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _validate_intent_targets(intent: LabelIntent) -> LabelIntent:
    if not intent.targets:
        raise IntentError("意图缺少 targets；LLM 必须返回至少一个带短 label 的目标")
    validated = []
    for t in intent.targets:
        validated.append(
            TargetSpec(
                label=require_nonempty_label(t.label or ""),
                description=(t.description or "").strip(),
                include=list(t.include or []),
                exclude=list(t.exclude or []),
            )
        )
    intent.targets = validated
    return intent


def parse_intent_manual(
    *,
    label: str = "",
    description: str = "",
    dataset_path: str = "",
    exclude: Optional[list] = None,
    preview_count: int = 5,
) -> LabelIntent:
    """Build intent from explicit caller fields. Empty label → IntentError."""
    intent = LabelIntent(
        targets=[
            TargetSpec(
                label=require_nonempty_label(label),
                description=(description or "").strip(),
                exclude=list(exclude or []),
            )
        ],
        dataset_path=dataset_path or None,
        preview_count=preview_count,
    )
    return _validate_intent_targets(intent)


def parse_intent_from_text(
    prompt: str,
    *,
    dataset_path: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
) -> LabelIntent:
    """Parse NL with LLM. On unavailable LLM / bad JSON / empty label → IntentError.

    Does not guess class nouns locally from keywords or regex.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise IntentError("自然语言意图为空；请输入要标注的目标描述")

    client = llm_client
    if client is None:
        try:
            from autotrain4.llm import LLMClient

            if settings:
                client = LLMClient.from_settings(settings)
            else:
                client = LLMClient.from_environ()
        except Exception as e:
            logger.warning("LLM client unavailable: %s", e)
            client = None

    if client is None or not getattr(client, "available", False):
        raise IntentError(
            "LLM 不可用，无法解析自然语言意图。"
            "请配置 LLM，或改用手工模式填写短类别名（如「手机」）"
        )

    try:
        resp = client.chat(
            [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        content = resp if isinstance(resp, str) else getattr(resp, "content", None) or str(resp)
    except Exception as e:
        raise IntentError(f"LLM 意图解析调用失败: {e}") from e

    data = _safe_parse_json(content)
    if data is None:
        raise IntentError(
            "LLM 未返回可解析的意图 JSON。"
            "请重试，或改用手工模式填写短类别名（如「手机」）"
        )

    intent = LabelIntent.from_dict(data)
    if dataset_path:
        intent.dataset_path = dataset_path
    # Force MVP task
    intent.annotation_task = "实例分割"
    intent.object_granularity = "instance"
    return _validate_intent_targets(intent)
