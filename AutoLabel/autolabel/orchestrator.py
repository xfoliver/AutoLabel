"""AutoLabel orchestrator: plan → preview → batch → commit."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2

from autolabel.capability import CapabilityError, check_backend_for_intent
from autolabel.commit import commit_sidecars, save_commit_manifest
from autolabel.dataset import (
    create_run_dir,
    filter_pending,
    list_images,
    write_json,
)
from autolabel.diagnostics.report import format_summary_text, write_review_queue, write_run_report
from autolabel.errors import ImageModerationBlocked
from autolabel.exporters.labelmeai import write_labelme_json
from autolabel.inference.base import InferenceBackend, build_backend_from_config, load_backends_config
from autolabel.intent import parse_intent_from_text, parse_intent_manual
from autolabel.label_spec import intent_to_label_spec
from autolabel.models import (
    ImageAnnotationResult,
    LabelIntent,
    PlanPreview,
    ReviewItem,
    ShapeRecord,
    TransformMeta,
)
from autolabel.plan import build_plan, format_plan_text
from autolabel.preview import sample_preview_images
from autolabel.scoring.risk import summarize_image_bucket
from autolabel.cv_io import imread
from autolabel.stage2 import annotate_image_to_dict, stage1_to_shapes
from autolabel.viz import draw_shapes_overlay

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]


@dataclass
class OrchestratorState:
    intent: Optional[LabelIntent] = None
    plan: Optional[PlanPreview] = None
    backend: Optional[InferenceBackend] = None
    run_dir: Optional[Path] = None
    preview_results: List[ImageAnnotationResult] = field(default_factory=list)
    batch_results: List[ImageAnnotationResult] = field(default_factory=list)
    mapping: Dict[Path, Path] = field(default_factory=dict)
    preview_confirmed: bool = False
    report: Dict[str, Any] = field(default_factory=dict)


class AutoLabelOrchestrator:
    def __init__(
        self,
        *,
        backends_config: Optional[Dict[str, Any]] = None,
        backend: Optional[InferenceBackend] = None,
        backend_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        log: Optional[LogFn] = None,
    ):
        self.settings = settings or {}
        self._log = log or (lambda m: logger.info("%s", m))
        self.backends_config = backends_config or load_backends_config()
        self.backend = backend or build_backend_from_config(self.backends_config, backend_id)
        self.state = OrchestratorState(backend=self.backend)

    def log(self, msg: str) -> None:
        self._log(msg)

    # ------------------------------------------------------------------ plan
    def set_intent(self, intent: LabelIntent) -> None:
        self.state.intent = intent

    def parse_prompt(self, prompt: str, dataset_path: Optional[str] = None) -> LabelIntent:
        intent = parse_intent_from_text(
            prompt,
            dataset_path=dataset_path,
            settings=self.settings,
        )
        if dataset_path:
            intent.dataset_path = dataset_path
        self.state.intent = intent
        return intent

    def make_manual_intent(self, **kwargs) -> LabelIntent:
        intent = parse_intent_manual(**kwargs)
        self.state.intent = intent
        return intent

    def build_plan(self, *, dataset_path: Optional[str] = None) -> PlanPreview:
        if self.state.intent is None:
            raise RuntimeError("intent not set")
        plan = build_plan(
            self.state.intent,
            self.backend,
            dataset_path=dataset_path or self.state.intent.dataset_path,
        )
        self.state.plan = plan
        # Hard fail capability
        cap = check_backend_for_intent(self.backend, self.state.intent)
        if not cap.ok:
            raise CapabilityError("; ".join(cap.errors))
        self.log(format_plan_text(plan))
        return plan

    # ---------------------------------------------------------------- preview
    def run_preview(
        self,
        *,
        progress: Optional[ProgressFn] = None,
        seed: Optional[int] = None,
    ) -> List[ImageAnnotationResult]:
        plan = self.state.plan or self.build_plan()
        intent = self.state.intent
        assert intent is not None
        images = list_images(Path(plan.dataset_path))
        pending, _ = filter_pending(images, overwrite_policy=intent.overwrite_policy)
        sample = sample_preview_images(pending, plan.preview_count or 5, seed=seed)
        if not sample:
            raise RuntimeError("没有待标注图片（可能均已存在 JSON）")

        run_dir = self.state.run_dir or create_run_dir(Path(plan.dataset_path))
        self.state.run_dir = run_dir
        write_json(run_dir / "meta" / "intent.json", intent.to_dict())
        write_json(run_dir / "meta" / "plan.json", plan.to_dict())

        spec = intent_to_label_spec(intent)
        results: List[ImageAnnotationResult] = []
        total = len(sample)
        for i, img in enumerate(sample, start=1):
            if progress:
                progress(i, total, str(img))
            self.log(f"[preview] {i}/{total} {img.name}")
            result = self._annotate_one(img, spec, run_dir, kind="preview")
            results.append(result)
        self.state.preview_results = results
        self.state.preview_confirmed = False
        return results

    def confirm_preview(self) -> None:
        if not self.state.preview_results:
            raise RuntimeError("无试标结果可确认")
        self.state.preview_confirmed = True
        self.log("试标已确认，可开始全量标注")

    # ------------------------------------------------------------------ batch
    def run_batch(
        self,
        *,
        progress: Optional[ProgressFn] = None,
        require_preview_confirm: bool = True,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        if require_preview_confirm and not self.state.preview_confirmed:
            raise RuntimeError("试标未确认，不能进入全量（工业安全闸）")
        plan = self.state.plan or self.build_plan()
        intent = self.state.intent
        assert intent is not None
        run_dir = self.state.run_dir or create_run_dir(Path(plan.dataset_path))
        self.state.run_dir = run_dir

        images = list_images(Path(plan.dataset_path))
        pending, skipped = filter_pending(images, overwrite_policy=intent.overwrite_policy)
        spec = intent_to_label_spec(intent)

        results: List[ImageAnnotationResult] = []
        mapping: Dict[Path, Path] = {}
        buckets: Dict[str, int] = {}
        empty = 0
        moderation_blocked = 0
        total = len(pending)
        for i, img in enumerate(pending, start=1):
            if stop_flag and stop_flag():
                self.log("收到停止信号，中断全量标注")
                break
            if progress:
                progress(i, total, str(img))
            self.log(f"[batch] {i}/{total} {img.name}")
            result = self._annotate_one(img, spec, run_dir, kind="batch")
            results.append(result)
            if result.skipped:
                moderation_blocked += 1
            elif result.work_json_path:
                mapping[img] = result.work_json_path
            bucket = result.review.bucket if result.review else "unknown"
            buckets[bucket] = buckets.get(bucket, 0) + 1
            if not result.shapes and not result.skipped:
                empty += 1

        self.state.batch_results = results
        self.state.mapping = mapping
        save_commit_manifest(run_dir, mapping)

        review_items = [r.review for r in results if r.review is not None]
        write_review_queue(run_dir, review_items)

        report = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "processed": len(results),
            "skipped_existing": len(skipped),
            "empty": empty,
            "moderation_blocked": moderation_blocked,
            "buckets": buckets,
            "pending_total": total,
            "committed": False,
        }
        write_run_report(run_dir, report)
        self.state.report = report
        self.log(format_summary_text(report))
        return report

    def commit(self, *, overwrite_policy: Optional[str] = None) -> Dict[str, Any]:
        intent = self.state.intent
        policy = overwrite_policy or (intent.overwrite_policy if intent else "skip_existing")
        if not self.state.mapping:
            raise RuntimeError("没有可 Commit 的标注（请先全量）")
        committed, skipped, errors = commit_sidecars(self.state.mapping, overwrite_policy=policy)
        summary = {
            "committed": committed,
            "skipped": skipped,
            "errors": errors,
            "committed_count": len(committed),
            "skipped_count": len(skipped),
            "error_count": len(errors),
        }
        if self.state.run_dir:
            write_json(self.state.run_dir / "report" / "commit_result.json", summary)
            if self.state.report:
                self.state.report["committed"] = True
                self.state.report["commit_result"] = summary
                write_run_report(self.state.run_dir, self.state.report)
        self.log(
            f"Commit 完成: 写入 {len(committed)}, 跳过 {len(skipped)}, 错误 {len(errors)}"
        )
        return summary

    # --------------------------------------------------------------- internal
    def _annotate_one(
        self,
        image_path: Path,
        spec,
        run_dir: Path,
        *,
        kind: str,
    ) -> ImageAnnotationResult:
        try:
            s1 = self.backend.predict(image_path, spec, references=spec.reference_images)
        except ImageModerationBlocked as e:
            return self._moderation_skip_result(image_path, run_dir, kind=kind, error=e)

        # Persist transform + simple mask meta
        stem = image_path.stem
        meta_path = run_dir / "stage1" / f"{stem}_transform.json"
        write_json(meta_path, s1.transform.to_dict())

        stage2_opts = {}
        if self.backend is not None and hasattr(self.backend, "stage2_options"):
            stage2_opts = dict(self.backend.stage2_options() or {})
        img = imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            stage2_opts.setdefault("image_bgr", img)

        shapes = stage1_to_shapes(s1, **stage2_opts)
        data = annotate_image_to_dict(image_path, s1, shapes=shapes)
        json_subdir = "preview" if kind == "preview" else "json"
        out_json = run_dir / json_subdir / f"{stem}.json"
        write_labelme_json(out_json, data)

        overlay = None
        try:
            overlay = draw_shapes_overlay(
                image_path,
                shapes,
                run_dir / "overlays" / f"{stem}_overlay.jpg",
            )
        except Exception as e:
            self.log(f"overlay failed for {image_path.name}: {e}")

        bucket = summarize_image_bucket(shapes)
        confs = [s.model_confidence for s in shapes if s.model_confidence is not None]
        risks = [s.quality_risk for s in shapes]
        review = ReviewItem(
            image_path=str(image_path),
            json_path=str(out_json),
            instance_count=len(shapes),
            model_confidence=(sum(confs) / len(confs)) if confs else None,
            quality_risk=(sum(risks) / len(risks)) if risks else 0.0,
            bucket=bucket,
        )
        return ImageAnnotationResult(
            image_path=image_path,
            shapes=shapes,
            transform=s1.transform,
            work_json_path=out_json,
            preview_overlay_path=overlay,
            review=review,
        )

    def _moderation_skip_result(
        self,
        image_path: Path,
        run_dir: Path,
        *,
        kind: str,
        error: ImageModerationBlocked,
    ) -> ImageAnnotationResult:
        """Record a moderation-blocked image and continue the batch/preview."""
        stem = image_path.stem
        msg = str(error)
        self.log(f"[skip:moderation] {image_path.name}: {msg[:240]}")

        width, height = 1, 1
        try:
            from autolabel.cv_io import imread
            import cv2

            img = imread(image_path, cv2.IMREAD_COLOR)
            if img is not None:
                height, width = img.shape[:2]
        except Exception:
            pass
        transform = TransformMeta.identity_for(width, height)
        write_json(
            run_dir / "stage1" / f"{stem}_transform.json",
            {
                **transform.to_dict(),
                "error": "moderation_blocked",
                "error_message": msg[:800],
            },
        )

        json_subdir = "preview" if kind == "preview" else "json"
        out_json = run_dir / json_subdir / f"{stem}.json"
        write_labelme_json(
            out_json,
            {
                "version": "5.0.1",
                "flags": {"autolabel": True, "moderation_blocked": True},
                "shapes": [],
                "imagePath": image_path.name,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width,
            },
        )
        review = ReviewItem(
            image_path=str(image_path),
            json_path=str(out_json),
            instance_count=0,
            model_confidence=None,
            quality_risk=1.0,
            bucket="moderation_blocked",
            notes=msg[:500],
        )
        return ImageAnnotationResult(
            image_path=image_path,
            shapes=[],
            transform=transform,
            work_json_path=None,  # do not commit empty moderation-blocked sidecars
            preview_overlay_path=None,
            review=review,
            error=msg,
            skipped=True,
        )
