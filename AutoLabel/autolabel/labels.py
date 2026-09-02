"""Label validation only — no keyword / heuristic rewriting.

Short class nouns are the LLM's (or explicit manual caller's) responsibility.
Local code only strips whitespace and rejects empty labels.
"""

from __future__ import annotations

from typing import Tuple

from autolabel.errors import IntentError
from autolabel.models import TargetSpec


def require_nonempty_label(label: str) -> str:
    """Return stripped label or raise IntentError if empty."""
    s = (label or "").strip()
    if not s:
        raise IntentError(
            "targets[].label 不能为空；请让 LLM 输出短类别名（如「零件」），"
            "或在手工模式下填写短类别名"
        )
    return s


def sanitize_class_label(label: str, description: str = "") -> str:
    """Compatibility wrapper: nonempty validation only (description ignored)."""
    del description  # kept for call-site compatibility; never used for guessing
    return require_nonempty_label(label)


def sanitize_target_spec(target: TargetSpec) -> TargetSpec:
    """Pass-through with nonempty label validation."""
    return TargetSpec(
        label=require_nonempty_label(target.label or ""),
        description=(target.description or "").strip(),
        include=list(target.include or []),
        exclude=list(target.exclude or []),
        model_prompt=str(getattr(target, "model_prompt", "") or "").strip(),
    )


def sanitize_label_and_description(label: str, description: str = "") -> Tuple[str, str]:
    """Convenience for builders: validate label, keep description as provided."""
    return require_nonempty_label(label), (description or "").strip()
