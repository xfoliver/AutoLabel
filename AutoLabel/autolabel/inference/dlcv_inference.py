"""DLCV local inference adapter (9890). Optional — requires capable .dvp / API masks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from autolabel.cv_io import imread
from autolabel.inference.base import BackendCapabilities, InferenceBackend, register_backend
from autolabel.models import InstancePred, LabelSpec, References, Stage1Result, TransformMeta

logger = logging.getLogger(__name__)


class DlcvInferenceBackend(InferenceBackend):
    """Wraps autotrain4.InferenceClient.

    NOTE: Many DLCV models return boxes/polygons rather than instance masks.
    This adapter:
      - Prefer response fields that already look like masks
      - Else rasterize polygon/points onto an identity-sized mask and set TransformMeta.identity
      - Declares instance_mask=True only if options.force_instance_mask or capabilities override

    For MVP, prefer Mock / HTTP backends unless you verified 9890 returns usable instances.
    """

    def __init__(self, backend_id: str = "dlcv", options: Optional[Dict[str, Any]] = None):
        self.id = backend_id
        self.options = options or {}
        self.base_url = str(self.options.get("base_url") or "http://127.0.0.1:9890")
        self.model_path = str(self.options.get("model_path") or "")
        self.timeout = int(self.options.get("timeout") or 60)
        caps = self.options.get("capabilities") or {}
        # Default optimistic if user selected dlcv for ISEG; Capability Check still gates on this flag
        self._caps = BackendCapabilities(
            image_input=True,
            text_guided_segmentation=bool(caps.get("text_guided_segmentation", False)),
            instance_mask=bool(caps.get("instance_mask", True)),
            class_mask=bool(caps.get("class_mask", False)),
            confidence=bool(caps.get("confidence", True)),
            reference_images=bool(caps.get("reference_images", False)),
        )
        self._client = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from autotrain4.inference_client import InferenceClient
        except ImportError as e:
            raise RuntimeError("autotrain4.inference_client not available") from e
        self._client = InferenceClient(base_url=self.base_url, timeout=self.timeout, progress=False)
        return self._client

    def predict(
        self,
        image_path: str | Path,
        label_spec: LabelSpec,
        references: Optional[References] = None,
    ) -> Stage1Result:
        path = Path(image_path)
        img = imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        client = self._get_client()
        if self.model_path:
            try:
                client.load_model(self.model_path)
            except Exception as e:
                logger.warning("load_model failed (may already be loaded): %s", e)
        raw = client.infer(str(path), model_path=self.model_path or None)

        transform = TransformMeta.identity_for(w, h)
        instances: List[InstancePred] = []
        class_name = label_spec.class_names[0] if label_spec.class_names else "object"

        # Heuristic parse — adapt as real API shapes become known
        items = []
        if isinstance(raw, dict):
            for key in ("instances", "results", "data", "shapes", "objects"):
                if isinstance(raw.get(key), list):
                    items = raw[key]
                    break
            if not items and isinstance(raw.get("result"), list):
                items = raw["result"]
        elif isinstance(raw, list):
            items = raw

        for i, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            score = item.get("score") or item.get("confidence")
            label = str(item.get("label") or item.get("class_name") or class_name)
            mask = None
            if item.get("mask") is not None and hasattr(item["mask"], "shape"):
                mask = np.asarray(item["mask"]).astype(np.uint8)
            elif item.get("points") and item.get("shape_type") in (None, "polygon"):
                pts = np.array(item["points"], dtype=np.float32).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 1)
            elif item.get("bbox") or item.get("box"):
                box = item.get("bbox") or item.get("box")
                x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
            if mask is None:
                continue
            instances.append(
                InstancePred(
                    instance_id=i,
                    class_id=1,
                    class_name=label,
                    mask=(mask > 0).astype(np.uint8),
                    score=float(score) if score is not None else None,
                )
            )

        return Stage1Result(instances=instances, transform=transform, raw=raw)


@register_backend("dlcv")
def _factory(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return DlcvInferenceBackend(backend_id, options)
