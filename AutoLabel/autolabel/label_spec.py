"""LabelSpec helpers."""

from __future__ import annotations

from autolabel.labels import require_nonempty_label
from autolabel.models import LabelIntent, LabelSpec, TargetSpec


def build_label_spec(
    *,
    label: str = "",
    description: str = "",
    exclude: list | None = None,
    include: list | None = None,
    annotation_task: str = "实例分割",
    model_prompt: str = "",
) -> LabelSpec:
    clean_label = require_nonempty_label(label)
    clean_desc = (description or "").strip()
    target = TargetSpec(
        label=clean_label,
        description=clean_desc,
        include=list(include or []),
        exclude=list(exclude or []),
        model_prompt=(model_prompt or "").strip(),
    )
    return LabelSpec(
        annotation_task=annotation_task,
        object_granularity="instance",
        class_names=[clean_label],
        targets=[target],
    )


def intent_to_label_spec(intent: LabelIntent) -> LabelSpec:
    return LabelSpec.from_intent(intent)
