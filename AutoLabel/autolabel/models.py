"""Core dataclasses for AutoLabel MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SizeWH = Tuple[int, int]  # (width, height)


@dataclass
class TargetSpec:
    label: str
    description: str = ""
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    model_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetSpec":
        return cls(
            label=str(data.get("label") or "").strip() or "object",
            description=str(data.get("description") or ""),
            include=[str(x) for x in (data.get("include") or [])],
            exclude=[str(x) for x in (data.get("exclude") or [])],
            model_prompt=str(data.get("model_prompt") or "").strip(),
        )


@dataclass
class DatasetClues:
    drive: Optional[str] = None
    path_fragments: List[str] = field(default_factory=list)
    dataset_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DatasetClues":
        data = data or {}
        return cls(
            drive=data.get("drive"),
            path_fragments=[str(x) for x in (data.get("path_fragments") or [])],
            dataset_name=data.get("dataset_name"),
        )


@dataclass
class References:
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "References":
        data = data or {}
        return cls(
            positive=[str(x) for x in (data.get("positive") or [])],
            negative=[str(x) for x in (data.get("negative") or [])],
        )


@dataclass
class LabelIntent:
    action: str = "auto_label"
    annotation_task: str = "实例分割"
    object_granularity: str = "instance"
    targets: List[TargetSpec] = field(default_factory=list)
    dataset_path: Optional[str] = None
    dataset_clues: DatasetClues = field(default_factory=DatasetClues)
    references: References = field(default_factory=References)
    instance_policy: str = "separate_each_object"
    review_policy: str = "low_confidence_only"
    overwrite_policy: str = "skip_existing"
    output_format: str = "labelmeai"
    backend_preference: str = "auto"
    preview_count: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "annotation_task": self.annotation_task,
            "object_granularity": self.object_granularity,
            "targets": [t.to_dict() for t in self.targets],
            "dataset_path": self.dataset_path,
            "dataset_clues": self.dataset_clues.to_dict(),
            "references": self.references.to_dict(),
            "instance_policy": self.instance_policy,
            "review_policy": self.review_policy,
            "overwrite_policy": self.overwrite_policy,
            "output_format": self.output_format,
            "backend_preference": self.backend_preference,
            "preview_count": self.preview_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelIntent":
        targets_raw = data.get("targets") or []
        targets = [TargetSpec.from_dict(t) if isinstance(t, dict) else TargetSpec(label=str(t)) for t in targets_raw]
        # Keep empty list so callers (intent validation / LabelSpec.from_intent) can fail explicitly.
        return cls(
            action=str(data.get("action") or "auto_label"),
            annotation_task=str(data.get("annotation_task") or "实例分割"),
            object_granularity=str(data.get("object_granularity") or "instance"),
            targets=targets,
            dataset_path=data.get("dataset_path"),
            dataset_clues=DatasetClues.from_dict(data.get("dataset_clues")),
            references=References.from_dict(data.get("references")),
            instance_policy=str(data.get("instance_policy") or "separate_each_object"),
            review_policy=str(data.get("review_policy") or "low_confidence_only"),
            overwrite_policy=str(data.get("overwrite_policy") or "skip_existing"),
            output_format=str(data.get("output_format") or "labelmeai"),
            backend_preference=str(data.get("backend_preference") or "auto"),
            preview_count=int(data.get("preview_count") or 5),
        )


@dataclass
class LabelSpec:
    annotation_task: str = "实例分割"
    object_granularity: str = "instance"
    class_names: List[str] = field(default_factory=lambda: ["object"])
    targets: List[TargetSpec] = field(default_factory=list)
    reference_images: References = field(default_factory=References)
    model_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_task": self.annotation_task,
            "object_granularity": self.object_granularity,
            "class_names": list(self.class_names),
            "targets": [t.to_dict() for t in self.targets],
            "reference_images": self.reference_images.to_dict(),
            "model_ref": self.model_ref,
        }

    @classmethod
    def from_intent(cls, intent: LabelIntent) -> "LabelSpec":
        from autolabel.errors import IntentError
        from autolabel.labels import require_nonempty_label

        raw = list(intent.targets or [])
        if not raw:
            raise IntentError("意图缺少 targets；无法构建 LabelSpec")
        targets = [
            TargetSpec(
                label=require_nonempty_label(t.label or ""),
                description=(t.description or "").strip(),
                include=list(t.include or []),
                exclude=list(t.exclude or []),
                model_prompt=str(getattr(t, "model_prompt", "") or "").strip(),
            )
            for t in raw
        ]
        names = [t.label for t in targets]
        return cls(
            annotation_task=intent.annotation_task,
            object_granularity=intent.object_granularity,
            class_names=names,
            targets=targets,
            reference_images=intent.references,
            model_ref=None,
        )


@dataclass
class AffineStep:
    """One stage of mask→input or input→original transform."""

    mode: str = "identity"  # identity | stretch | letterbox | crop
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale: Optional[float] = None
    pad_left: float = 0.0
    pad_top: float = 0.0
    pad_right: float = 0.0
    pad_bottom: float = 0.0
    crop: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"mode": self.mode}
        if self.mode == "stretch":
            d["scale_x"] = self.scale_x
            d["scale_y"] = self.scale_y
        elif self.mode == "letterbox":
            d["scale"] = self.scale if self.scale is not None else self.scale_x
            d["pad_left"] = self.pad_left
            d["pad_top"] = self.pad_top
            d["pad_right"] = self.pad_right
            d["pad_bottom"] = self.pad_bottom
        elif self.mode == "crop" and self.crop:
            d["crop"] = dict(self.crop)
        elif self.mode == "identity":
            pass
        else:
            d["scale_x"] = self.scale_x
            d["scale_y"] = self.scale_y
            d["pad_left"] = self.pad_left
            d["pad_top"] = self.pad_top
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AffineStep":
        if not data:
            return cls(mode="identity")
        scale = data.get("scale")
        sx = float(data.get("scale_x", scale if scale is not None else 1.0))
        sy = float(data.get("scale_y", scale if scale is not None else 1.0))
        return cls(
            mode=str(data.get("mode") or "identity"),
            scale_x=sx,
            scale_y=sy,
            scale=float(scale) if scale is not None else None,
            pad_left=float(data.get("pad_left") or 0.0),
            pad_top=float(data.get("pad_top") or 0.0),
            pad_right=float(data.get("pad_right") or 0.0),
            pad_bottom=float(data.get("pad_bottom") or 0.0),
            crop=data.get("crop"),
        )

    @classmethod
    def identity(cls) -> "AffineStep":
        return cls(mode="identity")

    @classmethod
    def stretch(cls, scale_x: float, scale_y: float) -> "AffineStep":
        return cls(mode="stretch", scale_x=scale_x, scale_y=scale_y)

    @classmethod
    def letterbox(
        cls,
        scale: float,
        pad_left: float,
        pad_top: float,
        pad_right: float = 0.0,
        pad_bottom: float = 0.0,
    ) -> "AffineStep":
        return cls(
            mode="letterbox",
            scale=scale,
            scale_x=scale,
            scale_y=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        )


@dataclass
class TransformMeta:
    """Two-stage transform: mask space → model input → original image."""

    original_size: SizeWH  # (W, H)
    model_input_size: SizeWH
    mask_size: SizeWH
    mask_to_input: AffineStep = field(default_factory=AffineStep.identity)
    input_to_original: AffineStep = field(default_factory=AffineStep.identity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_size": [int(self.original_size[0]), int(self.original_size[1])],
            "model_input_size": [int(self.model_input_size[0]), int(self.model_input_size[1])],
            "mask_size": [int(self.mask_size[0]), int(self.mask_size[1])],
            "mask_to_input": self.mask_to_input.to_dict(),
            "input_to_original": self.input_to_original.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformMeta":
        def _wh(key: str, default: Sequence[int]) -> SizeWH:
            raw = data.get(key) or default
            return (int(raw[0]), int(raw[1]))

        # Flat legacy compatibility: resize_mode / scale / pad_* at top level
        if "input_to_original" not in data and data.get("resize_mode"):
            mode = str(data.get("resize_mode"))
            if mode == "letterbox":
                step = AffineStep.letterbox(
                    scale=float(data.get("scale") or 1.0),
                    pad_left=float(data.get("pad_left") or 0.0),
                    pad_top=float(data.get("pad_top") or 0.0),
                    pad_right=float(data.get("pad_right") or 0.0),
                    pad_bottom=float(data.get("pad_bottom") or 0.0),
                )
            elif mode == "stretch":
                ow, oh = _wh("original_size", (1, 1))
                iw, ih = _wh("model_input_size", data.get("mask_size") or (1, 1))
                # Forward preprocess scales (input/original); inverted by map_mask_to_original
                step = AffineStep.stretch(iw / max(ow, 1), ih / max(oh, 1))
            else:
                step = AffineStep.identity()
            mask_size = _wh("mask_size", data.get("model_input_size") or (1, 1))
            input_size = _wh("model_input_size", mask_size)
            m2i = AffineStep.identity()
            if mask_size != input_size:
                m2i = AffineStep.stretch(
                    input_size[0] / max(mask_size[0], 1),
                    input_size[1] / max(mask_size[1], 1),
                )
            return cls(
                original_size=_wh("original_size", (1, 1)),
                model_input_size=input_size,
                mask_size=mask_size,
                mask_to_input=m2i,
                input_to_original=step,
            )

        return cls(
            original_size=_wh("original_size", (1, 1)),
            model_input_size=_wh("model_input_size", (1, 1)),
            mask_size=_wh("mask_size", (1, 1)),
            mask_to_input=AffineStep.from_dict(data.get("mask_to_input")),
            input_to_original=AffineStep.from_dict(data.get("input_to_original")),
        )

    @classmethod
    def identity_for(cls, width: int, height: int) -> "TransformMeta":
        size = (int(width), int(height))
        return cls(
            original_size=size,
            model_input_size=size,
            mask_size=size,
            mask_to_input=AffineStep.identity(),
            input_to_original=AffineStep.identity(),
        )


@dataclass
class InstancePred:
    instance_id: int
    class_id: int
    class_name: str
    mask: np.ndarray  # bool/uint8 HxW
    score: Optional[float] = None

    def to_meta_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "area": int(np.count_nonzero(self.mask)),
        }


@dataclass
class Stage1Result:
    instances: List[InstancePred]
    transform: TransformMeta
    class_mask: Optional[np.ndarray] = None
    colored_mask: Optional[np.ndarray] = None  # gpt-image RGB mask (BGR uint8), if any
    viz_rgb_path: Optional[str] = None
    raw: Any = None


@dataclass
class ShapeRecord:
    label: str
    points: List[List[float]]
    shape_type: str = "polygon"
    model_confidence: Optional[float] = None
    quality_risk: float = 0.0
    score_source: str = "none"  # model | none
    instance_source: str = "model"  # model | connected_components
    group_id: Any = None
    description: Any = None


@dataclass
class PlanPreview:
    dataset_path: str
    image_count: int
    pending_count: int
    skipped_existing: int
    annotation_task: str
    object_granularity: str
    targets_summary: str
    backend_id: str
    backend_capabilities: Dict[str, Any]
    overwrite_policy: str
    preview_count: int
    review_policy: str
    warnings: List[str] = field(default_factory=list)
    label_spec: Optional[LabelSpec] = None
    intent: Optional[LabelIntent] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "image_count": self.image_count,
            "pending_count": self.pending_count,
            "skipped_existing": self.skipped_existing,
            "annotation_task": self.annotation_task,
            "object_granularity": self.object_granularity,
            "targets_summary": self.targets_summary,
            "backend_id": self.backend_id,
            "backend_capabilities": self.backend_capabilities,
            "overwrite_policy": self.overwrite_policy,
            "preview_count": self.preview_count,
            "review_policy": self.review_policy,
            "warnings": list(self.warnings),
        }


@dataclass
class ReviewItem:
    image_path: str
    json_path: str
    instance_count: int
    model_confidence: Optional[float]
    quality_risk: float
    bucket: str  # suggest_accept | review | sample | empty | problem
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageAnnotationResult:
    image_path: Path
    shapes: List[ShapeRecord]
    transform: TransformMeta
    work_json_path: Optional[Path] = None
    preview_overlay_path: Optional[Path] = None
    preview_mask_path: Optional[Path] = None
    review: Optional[ReviewItem] = None
    error: Optional[str] = None
    skipped: bool = False
