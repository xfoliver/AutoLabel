"""Capability checks before burning GPU/API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from autolabel.inference.base import BackendCapabilities, InferenceBackend
from autolabel.models import LabelIntent, LabelSpec


class CapabilityError(RuntimeError):
    """Raised when backend cannot satisfy the requested annotation task."""


@dataclass
class CapabilityReport:
    ok: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise CapabilityError("; ".join(self.errors) or "capability check failed")


def check_backend_for_intent(
    backend: InferenceBackend,
    intent: LabelIntent,
    *,
    allow_class_mask_fallback: bool = False,
) -> CapabilityReport:
    return check_backend_for_spec(
        backend,
        LabelSpec.from_intent(intent),
        allow_class_mask_fallback=allow_class_mask_fallback,
    )


def check_backend_for_spec(
    backend: InferenceBackend,
    spec: LabelSpec,
    *,
    allow_class_mask_fallback: bool = False,
) -> CapabilityReport:
    caps = backend.capabilities
    errors: List[str] = []
    warnings: List[str] = []

    if not caps.image_input:
        errors.append(f"backend {backend.id} does not accept image input")

    need_instance = (
        spec.object_granularity == "instance"
        or spec.annotation_task in ("实例分割", "ISEG", "iseg")
    )
    if need_instance:
        if not caps.instance_mask:
            if allow_class_mask_fallback and caps.class_mask:
                warnings.append(
                    f"backend {backend.id} lacks instance_mask; "
                    "class_mask fallback may merge adjacent same-class objects"
                )
            else:
                errors.append(
                    f"ISEG requires capabilities.instance_mask=true; "
                    f"backend {backend.id} does not provide instance masks "
                    f"(can-chat / bbox-only VLM is not enough)"
                )

    if spec.reference_images.positive or spec.reference_images.negative:
        if not caps.reference_images:
            warnings.append(
                f"backend {backend.id} ignores reference images "
                "(capabilities.reference_images=false); refs are for human confirm only"
            )

    if not caps.confidence:
        warnings.append(
            "backend provides no model_confidence; results will enter sample/review, "
            "not high-confidence auto-accept"
        )

    return CapabilityReport(ok=not errors, errors=errors, warnings=warnings)
