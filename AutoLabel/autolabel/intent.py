"""NL intent parsing — LLM optional; simple mode skips the LLM."""

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
  "targets": [{"label": "短类别名（显示用）", "model_prompt": "short English visual concept for SAM3", "description": "语义描述", "include": [], "exclude": ["不要的东西"]}],
  "dataset_path": "若用户给了明确路径则填，否则 null",
  "dataset_clues": {"drive": "Z:", "path_fragments": ["片段"], "dataset_name": null},
  "references": {"positive": [], "negative": []},
  "overwrite_policy": "skip_existing",
  "preview_count": 5,
  "backend_preference": "auto",
  "review_policy": "low_confidence_only"
}
关键规则（强制）：
- targets[].label 必须是 1～8 个字的短名词，例如「手机」「汽车」「人」。用于 LabelMe 显示。
- targets[].model_prompt 必须是简短视觉概念（优先英文），例如 phone / car / person / red car。
- 禁止把整句指令放进 label 或 model_prompt（错误例：「只识别图片中的目标」「把该目录中的物体」）。
- 详细规则、排除项一律放进 description / exclude。
- 不要把业务描述整段发给视觉模型。
MVP 只支持实例分割；若用户说检测/分类等，仍输出 annotation_task=实例分割 并在 description 里保留原意提示。
"""


_SIMPLE_PROMPT_RE = re.compile(r"^[\w\-./一-鿿 ]{1,32}$", re.UNICODE)


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
        raise IntentError("意图缺少 targets；必须提供至少一个带短 label 的目标")
    validated = []
    for t in intent.targets:
        validated.append(
            TargetSpec(
                label=require_nonempty_label(t.label or ""),
                description=(t.description or "").strip(),
                include=list(t.include or []),
                exclude=list(t.exclude or []),
                model_prompt=str(getattr(t, "model_prompt", "") or "").strip(),
            )
        )
    intent.targets = validated
    return intent


def looks_like_simple_prompt(text: str) -> bool:
    """True when the user typed a short class noun, not a sentence."""
    s = (text or "").strip()
    if not s or "\n" in s:
        return False
    if len(s) > 32:
        return False
    instructional = ("标注", "标出", "识别", "把该", "所有", "目录", "图片中", "please")
    lower = s.lower()
    if any(tok in s for tok in instructional):
        return False
    if any(tok in lower for tok in ("please", "annotate", "detect all", "label all")):
        return False
    if " " in s and len(s.split()) > 4:
        return False
    return bool(_SIMPLE_PROMPT_RE.match(s))


def parse_intent_manual(
    *,
    label: str = "",
    description: str = "",
    dataset_path: str = "",
    exclude: Optional[list] = None,
    preview_count: int = 5,
    model_prompt: str = "",
) -> LabelIntent:
    """Build intent from explicit caller fields. Empty label → IntentError."""
    clean_label = require_nonempty_label(label)
    prompt = (model_prompt or "").strip() or clean_label
    intent = LabelIntent(
        targets=[
            TargetSpec(
                label=clean_label,
                description=(description or "").strip(),
                exclude=list(exclude or []),
                model_prompt=prompt,
            )
        ],
        dataset_path=dataset_path or None,
        preview_count=preview_count,
    )
    return _validate_intent_targets(intent)


def parse_intent_simple(
    prompt: str,
    *,
    dataset_path: Optional[str] = None,
    preview_count: int = 5,
) -> LabelIntent:
    """Bypass LLM: treat the whole string as label + model_prompt."""
    text = require_nonempty_label(prompt)
    return parse_intent_manual(
        label=text,
        description="",
        dataset_path=dataset_path or "",
        preview_count=preview_count,
        model_prompt=text,
    )


def resolve_intent_mode(settings: Optional[Dict[str, Any]] = None) -> str:
    raw = str((settings or {}).get("intent_mode") or "auto").strip().lower()
    if raw in {"llm", "simple", "auto"}:
        return raw
    return "auto"


def parse_intent_from_text(
    prompt: str,
    *,
    dataset_path: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    intent_mode: Optional[str] = None,
) -> LabelIntent:
    """Parse NL.

    intent_mode:
      - simple: never call LLM
      - llm: always call LLM (fail if unavailable)
      - auto: short noun → simple; otherwise LLM, falling back to simple if LLM is down
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise IntentError("自然语言意图为空；请输入要标注的目标描述")

    mode = (intent_mode or resolve_intent_mode(settings)).strip().lower()
    if mode == "simple":
        return parse_intent_simple(prompt, dataset_path=dataset_path)
    if mode == "auto" and looks_like_simple_prompt(prompt) and llm_client is None:
        return parse_intent_simple(prompt, dataset_path=dataset_path)

    client = llm_client
    if client is None and mode != "simple":
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
        if mode == "llm":
            raise IntentError(
                "LLM 不可用，无法解析自然语言意图。"
                "请配置 LLM，或改用 simple 模式直接填写短类别名（如 phone）"
            )
        logger.info("LLM unavailable; using simple intent for %r", prompt[:40])
        return parse_intent_simple(prompt, dataset_path=dataset_path)

    try:
        resp = client.chat(
            [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        content = resp if isinstance(resp, str) else getattr(resp, "content", None) or str(resp)
    except Exception as e:
        if mode != "llm":
            logger.warning("LLM call failed, falling back to simple: %s", e)
            return parse_intent_simple(prompt, dataset_path=dataset_path)
        raise IntentError(f"LLM 意图解析调用失败: {e}") from e

    data = _safe_parse_json(content)
    if data is None:
        if mode != "llm":
            return parse_intent_simple(prompt, dataset_path=dataset_path)
        raise IntentError(
            "LLM 未返回可解析的意图 JSON。"
            "请重试，或改用 simple 模式填写短类别名（如 phone）"
        )

    intent = LabelIntent.from_dict(data)
    if dataset_path:
        intent.dataset_path = dataset_path
    intent.annotation_task = "实例分割"
    intent.object_granularity = "instance"
    return _validate_intent_targets(intent)
