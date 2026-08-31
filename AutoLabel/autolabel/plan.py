"""Plan preview builder."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from autolabel.capability import check_backend_for_intent
from autolabel.dataset import filter_pending, list_images
from autolabel.inference.base import InferenceBackend
from autolabel.label_spec import intent_to_label_spec
from autolabel.models import LabelIntent, PlanPreview


def build_plan(
    intent: LabelIntent,
    backend: InferenceBackend,
    *,
    dataset_path: Optional[str] = None,
    allow_class_mask_fallback: bool = False,
) -> PlanPreview:
    path = Path(dataset_path or intent.dataset_path or "").expanduser()
    if not str(path):
        raise ValueError("dataset_path is required for plan preview")
    if not path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {path}")

    images = list_images(path)
    pending, skipped = filter_pending(images, overwrite_policy=intent.overwrite_policy)

    cap = check_backend_for_intent(
        backend,
        intent,
        allow_class_mask_fallback=allow_class_mask_fallback,
    )
    warnings = list(cap.warnings)
    if not cap.ok:
        warnings.extend(cap.errors)

    targets = intent.targets or []
    summary_parts = []
    for t in targets:
        bit = t.label
        if t.description:
            bit += f"（{t.description}）"
        if t.exclude:
            bit += f" 排除:{','.join(t.exclude)}"
        summary_parts.append(bit)
    targets_summary = "；".join(summary_parts) or "(未指定目标)"

    spec = intent_to_label_spec(intent)
    plan = PlanPreview(
        dataset_path=str(path.resolve()),
        image_count=len(images),
        pending_count=len(pending),
        skipped_existing=len(skipped),
        annotation_task=intent.annotation_task,
        object_granularity=intent.object_granularity,
        targets_summary=targets_summary,
        backend_id=backend.id,
        backend_capabilities=backend.capabilities.to_dict(),
        overwrite_policy=intent.overwrite_policy,
        preview_count=min(int(intent.preview_count or 5), max(len(pending), 0)),
        review_policy=intent.review_policy,
        warnings=warnings,
        label_spec=spec,
        intent=intent,
    )
    if not cap.ok:
        # Attach failure by raising after plan built for UI display — orchestrator raises
        plan.warnings = warnings
    return plan


def format_plan_text(plan: PlanPreview) -> str:
    lines = [
        f"数据集: {plan.dataset_path}",
        f"图片总数: {plan.image_count}  待处理: {plan.pending_count}  已有JSON跳过: {plan.skipped_existing}",
        f"任务: {plan.annotation_task} / 粒度: {plan.object_granularity}",
        f"目标: {plan.targets_summary}",
        f"Backend: {plan.backend_id}  caps={plan.backend_capabilities}",
        f"覆盖策略: {plan.overwrite_policy}  试标张数: {plan.preview_count}",
        f"复核策略: {plan.review_policy}",
        f"输出: LabelMeAI 旁路 JSON（先写 work/，确认后 Commit）",
    ]
    if plan.warnings:
        lines.append("警告:")
        for w in plan.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
