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
from autolabel.errors import ImageModerationBlocked, IntentError
from autolabel.exporters.labelmeai import write_labelme_json
from autolabel.inference.base import InferenceBackend, build_backend_from_config, load_backends_config
from autolabel.intent import apply_display_label, looks_like_simple_prompt, parse_intent_from_text, parse_intent_manual, resolve_intent_mode
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
from autolabel.cv_io import imread, imwrite
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
        self._requested_backend_id = backend_id or self.backends_config.get("annotation_engine")
        self._backend_injected = backend is not None
        self.backend = backend or build_backend_from_config(
            self.backends_config,
            self._requested_backend_id,
            settings=self.settings,
            log=self._log,
        )
        self._bind_backend_log()
        self.state = OrchestratorState(backend=self.backend)

    def log(self, msg: str) -> None:
        self._log(msg)

    def _bind_backend_log(self) -> None:
        if self.backend is not None and hasattr(self.backend, "bind_log"):
            self.backend.bind_log(self._log)

    def _resolve_engine_for_intent(self) -> None:
        """Keep the selected engine after intent is known. No auto-routing."""
        self._bind_backend_log()
        from autolabel.engines import engine_label

        bid = str(getattr(self.backend, "id", "") or "")
        self.log(f"[Engine] {engine_label(bid)} id={bid}")

    # ------------------------------------------------------------------ plan
    def set_intent(self, intent: LabelIntent) -> None:
        self.state.intent = intent
        self._resolve_engine_for_intent()

    def parse_prompt(self, prompt: str, dataset_path: Optional[str] = None) -> LabelIntent:
        intent = parse_intent_from_text(
            prompt,
            dataset_path=dataset_path,
            settings=self.settings,
        )
        if dataset_path:
            intent.dataset_path = dataset_path
        self.state.intent = intent
        self._resolve_engine_for_intent()
        return intent

    def apply_ui_intent(
        self,
        *,
        prompt: str = "",
        dataset_path: str = "",
        short_label: str = "",
        preview_count: int = 5,
    ) -> LabelIntent:
        """Parse GUI fields. Must run off the Qt GUI thread (LLM / network)."""
        prompt = (prompt or "").strip()
        short_label = (short_label or "").strip()
        if not prompt and not short_label:
            raise IntentError("请填写标注目标")
        visual = short_label or prompt
        if short_label and (not prompt or prompt == short_label) and looks_like_simple_prompt(visual):
            intent = self.make_manual_intent(
                label=short_label,
                description="",
                dataset_path=dataset_path or "",
                exclude=[],
                preview_count=int(preview_count or 5),
                model_prompt=short_label,
            )
        elif prompt:
            mode = resolve_intent_mode(self.settings)
            if mode == "llm":
                self.log("正在用 LLM 解析自然语言意图，请稍候…")
            elif mode == "simple":
                self.log(f"simple 模式：直接使用「{prompt}」作为目标")
            else:
                self.log("正在解析意图…")
            intent = self.parse_prompt(prompt, dataset_path=dataset_path or None)
            intent.preview_count = int(preview_count or 5)
            if short_label and intent.targets:
                apply_display_label(intent, short_label)
        else:
            intent = self.make_manual_intent(
                label=short_label,
                description="",
                dataset_path=dataset_path or "",
                exclude=[],
                preview_count=int(preview_count or 5),
                model_prompt=short_label,
            )
        t0 = intent.targets[0]
        visual = (getattr(t0, "model_prompt", "") or t0.label).strip()
        self.log(f"意图就绪：显示名={t0.label}  /  视觉提示={visual}")
        return intent

    def make_manual_intent(self, **kwargs) -> LabelIntent:
        intent = parse_intent_manual(**kwargs)
        self.state.intent = intent
        self._resolve_engine_for_intent()
        return intent

    def _needs_sam3_worker(self) -> bool:
        backend = self.backend
        if backend is None:
            return False
        bid = str(getattr(backend, "id", "") or "")
        return bid in {"sam3_local", "sam3", "sam3_1_local", "sam3.1"} or type(backend).__name__ == "Sam3LocalBackend"

    def _ensure_sam3_worker(self) -> None:
        if not self._needs_sam3_worker():
            return
        from autolabel.sam3_worker_ctl import ensure_worker, worker_options

        opt = dict(getattr(self.backend, "options", None) or worker_options(self.backends_config))
        ensure_worker(opt, log=self.log, timeout=float(opt.get("startup_timeout") or 600))

    def build_plan(self, *, dataset_path: Optional[str] = None) -> PlanPreview:
        if self.state.intent is None:
            raise RuntimeError("intent not set")
        self._ensure_sam3_worker()
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
        stop_flag: Optional[Callable[[], bool]] = None,
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
            if stop_flag and stop_flag():
                self.log("收到停止信号，中断试标")
                break
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
            if hasattr(self.backend, "debug_dir"):
                self.backend.debug_dir = run_dir / "stage1"
            s1 = self.backend.predict(image_path, spec, references=spec.reference_images)
        except ImageModerationBlocked as e:
            return self._moderation_skip_result(image_path, run_dir, kind=kind, error=e)

        # Persist transform + simple mask meta
        stem = image_path.stem
        meta_path = run_dir / "stage1" / f"{stem}_transform.json"
        write_json(meta_path, s1.transform.to_dict())
        raw_early = getattr(s1, "raw", None)
        if isinstance(raw_early, dict) and raw_early.get("candidates") is not None:
            write_json(
                run_dir / "stage1" / f"{stem}_candidates.json",
                {
                    "target": raw_early.get("target"),
                    "engine": raw_early.get("engine"),
                    "candidates": raw_early.get("candidates"),
                    "quality": raw_early.get("quality"),
                },
            )

        mask_path = None
        if getattr(s1, "colored_mask", None) is not None:
            mask_path = run_dir / "stage1" / f"{stem}_rawmask.png"
            try:
                if not imwrite(mask_path, s1.colored_mask):
                    self.log(f"raw mask write failed for {image_path.name}")
                    mask_path = None
            except Exception as e:
                self.log(f"raw mask write failed for {image_path.name}: {e}")
                mask_path = None
        raw = getattr(s1, "raw", None)
        if isinstance(raw, dict):
            first_mask = None
            for item in raw.get("instances") or []:
                if not isinstance(item, dict) or not item.get("mask_path"):
                    continue
                candidate = Path(str(item["mask_path"]))
                if candidate.is_file() and candidate.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}:
                    first_mask = candidate
                    break
            if first_mask is not None and mask_path is None:
                mask_path = first_mask

        stage2_opts = {}
        if self.backend is not None and hasattr(self.backend, "stage2_options"):
            stage2_opts = dict(self.backend.stage2_options() or {})
        img = imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            stage2_opts.setdefault("image_bgr", img)

        shapes = stage1_to_shapes(s1, **stage2_opts)
        self.log(f"[Export] annotations={len(shapes)}")
        try:
            from autolabel.debug_overlay import (
                count_sam3_raw,
                resolve_preview_mask_path,
                summarize_shapes,
                write_sam_debug_overlays,
            )

            counts = count_sam3_raw(s1)
            shape_info = summarize_shapes(shapes)
            self.log(
                f"[SAM3] raw_boxes={counts['raw_boxes']} "
                f"raw_masks={counts['raw_masks']} "
                f"final_shapes={shape_info['final_shapes']}"
            )
            if counts["raw_masks"] == 0 and shape_info["final_shapes"] > 0:
                self.log(
                    "[SAM3] WARNING: raw_masks=0 but final_shapes>0 — "
                    "bbox fallback is forbidden; inspect debug overlays"
                )
            if shape_info["rectangle_n"]:
                self.log(
                    f"[Export] rectangle_shapes={shape_info['rectangle_n']}/"
                    f"{shape_info['final_shapes']} point_counts={shape_info['point_counts']}"
                )
            write_sam_debug_overlays(
                image_path,
                out_dir=run_dir / "stage1" / "debug" / stem,
                result=s1,
                shapes=shapes,
            )
            preview_mask = resolve_preview_mask_path(
                raw=raw if isinstance(raw, dict) else None,
                debug_dir=run_dir / "stage1" / "debug" / stem,
                colored_mask_path=mask_path,
            )
            if preview_mask is not None:
                mask_path = preview_mask
        except Exception as e:
            self.log(f"SAM3 debug overlays failed for {image_path.name}: {e}")

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

        try:
            from autolabel.coord_debug import write_coord_debug

            dbg = write_coord_debug(
                image_path,
                out_dir=run_dir / "stage1" / "debug" / stem,
                shapes=shapes,
                transform=s1.transform,
                sam3_raw=raw if isinstance(raw, dict) else None,
                mask_path=mask_path,
                overlay_path=overlay,
            )
            self.log(
                f"[IMAGE DEBUG] {image_path.name} EXIF={dbg.get('exif_orientation')} "
                f"PIL_raw={dbg.get('pil_raw_size_wh')} "
                f"normalized={dbg.get('after_exif_transpose_wh')} "
                f"OpenCV={dbg.get('opencv_normalized_wh')} "
                f"SAM3={dbg.get('sam3_image_size_wh')} "
                f"mask_hw={dbg.get('mask_shape_hw')} "
                f"poly={dbg.get('polygon_xy_range')}"
            )
        except Exception as e:
            self.log(f"coord debug failed for {image_path.name}: {e}")

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
            preview_mask_path=mask_path,
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
