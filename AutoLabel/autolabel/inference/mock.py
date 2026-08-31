"""Mock instance-segmentation backend for development and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from autolabel.cv_io import imread
from autolabel.geometry.transform import build_letterbox_meta, build_stretch_meta
from autolabel.inference.base import BackendCapabilities, InferenceBackend, register_backend
from autolabel.models import InstancePred, LabelSpec, References, Stage1Result, TransformMeta


def _read_image_size(image_path: Path) -> Tuple[int, int, np.ndarray]:
    img = imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        # Synthetic blank if missing (tests may pass non-image paths carefully)
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        return w, h, img
    h, w = img.shape[:2]
    return w, h, img


def _make_blob_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2 <= 1.0
    return mask.astype(np.uint8)


class MockBackend(InferenceBackend):
    """Synthesizes 1–N instance masks with full TransformMeta.

    Options:
      - mode: letterbox | stretch | identity
      - model_input: [W,H] default [640,640]
      - mask_scale: float, e.g. 0.25 → mask is 1/4 of model input
      - n_instances: int (default 2)
      - adjacent: bool — place two same-class blobs nearly touching (for anti-merge tests)
      - with_scores: bool
      - seed: int
    """

    def __init__(self, backend_id: str = "mock", options: Optional[Dict[str, Any]] = None):
        self.id = backend_id
        self.options = options or {}
        self._caps = BackendCapabilities(
            image_input=True,
            text_guided_segmentation=True,
            instance_mask=True,
            class_mask=False,
            confidence=bool(self.options.get("with_scores", True)),
            reference_images=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def predict(
        self,
        image_path: str | Path,
        label_spec: LabelSpec,
        references: Optional[References] = None,
    ) -> Stage1Result:
        path = Path(image_path)
        ow, oh, _ = _read_image_size(path)
        mode = str(self.options.get("mode") or "letterbox").lower()
        mi = self.options.get("model_input") or [640, 640]
        iw, ih = int(mi[0]), int(mi[1])
        mask_scale = float(self.options.get("mask_scale") or 1.0)
        mw = max(1, int(round(iw * mask_scale)))
        mh = max(1, int(round(ih * mask_scale)))

        if mode == "identity":
            transform = TransformMeta.identity_for(ow, oh)
            mw, mh = ow, oh
            iw, ih = ow, oh
        elif mode == "stretch":
            transform = build_stretch_meta((ow, oh), (iw, ih), (mw, mh))
        else:
            transform = build_letterbox_meta((ow, oh), (iw, ih), (mw, mh))

        n = int(self.options.get("n_instances") or 2)
        adjacent = bool(self.options.get("adjacent", True))
        seed = int(self.options.get("seed") or 0)
        rng = np.random.default_rng(seed + (abs(hash(path.name)) % 10000))

        class_name = (label_spec.class_names[0] if label_spec.class_names else "object")
        instances: List[InstancePred] = []

        if adjacent and n >= 2:
            # Two blobs side-by-side in mask space (same class) — must stay 2 instances
            cx1, cx2 = mw * 0.35, mw * 0.55
            cy = mh * 0.5
            rx = mw * 0.12
            ry = mh * 0.18
            masks = [
                _make_blob_mask(mh, mw, cx1, cy, rx, ry),
                _make_blob_mask(mh, mw, cx2, cy, rx, ry),
            ]
            for i, m in enumerate(masks, start=1):
                score = float(0.9 - 0.05 * (i - 1)) if self._caps.confidence else None
                instances.append(
                    InstancePred(
                        instance_id=i,
                        class_id=1,
                        class_name=class_name,
                        mask=m,
                        score=score,
                    )
                )
            for i in range(3, n + 1):
                cx = float(rng.uniform(0.2, 0.8) * mw)
                cy = float(rng.uniform(0.2, 0.8) * mh)
                rx = float(rng.uniform(0.05, 0.1) * mw)
                ry = float(rng.uniform(0.05, 0.1) * mh)
                m = _make_blob_mask(mh, mw, cx, cy, rx, ry)
                instances.append(
                    InstancePred(
                        instance_id=i,
                        class_id=1,
                        class_name=class_name,
                        mask=m,
                        score=float(rng.uniform(0.4, 0.95)) if self._caps.confidence else None,
                    )
                )
        else:
            for i in range(1, n + 1):
                cx = float(rng.uniform(0.25, 0.75) * mw)
                cy = float(rng.uniform(0.25, 0.75) * mh)
                rx = float(rng.uniform(0.08, 0.15) * mw)
                ry = float(rng.uniform(0.08, 0.15) * mh)
                m = _make_blob_mask(mh, mw, cx, cy, rx, ry)
                instances.append(
                    InstancePred(
                        instance_id=i,
                        class_id=1,
                        class_name=class_name,
                        mask=m,
                        score=float(rng.uniform(0.35, 0.95)) if self._caps.confidence else None,
                    )
                )

        return Stage1Result(instances=instances, transform=transform, raw={"backend": self.id})


@register_backend("mock")
def _factory(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return MockBackend(backend_id, options)


class NoInstanceMaskBackend(InferenceBackend):
    """Stub that only claims class_mask — used to test Capability Check rejection."""

    def __init__(self, backend_id: str = "no_instance", options: Optional[Dict[str, Any]] = None):
        self.id = backend_id
        self.options = options or {}
        self._caps = BackendCapabilities(
            image_input=True,
            text_guided_segmentation=True,
            instance_mask=False,
            class_mask=True,
            confidence=False,
            reference_images=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def predict(self, image_path, label_spec, references=None) -> Stage1Result:
        raise RuntimeError("NoInstanceMaskBackend cannot produce instance masks")


@register_backend("no_instance")
def _factory_no(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return NoInstanceMaskBackend(backend_id, options)
