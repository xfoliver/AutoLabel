"""Generic HTTP vision segmentation API adapter (swap vendors via config)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from autolabel.cv_io import imread
from autolabel.inference.base import BackendCapabilities, InferenceBackend, register_backend
from autolabel.models import AffineStep, InstancePred, LabelSpec, References, Stage1Result, TransformMeta


def _decode_mask_b64(data: str, shape_hw: Optional[tuple] = None) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    # Try PNG/bytes via imdecode
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 3:
            img = img[:, :, 0]
        return (img > 0).astype(np.uint8)
    if shape_hw is not None:
        return arr.reshape(shape_hw).astype(np.uint8)
    raise ValueError("Cannot decode mask payload")


class HttpApiBackend(InferenceBackend):
    """Call a configurable HTTP endpoint that returns instance masks.

    Expected response JSON (flexible):
    {
      "instances": [
        {"mask_png_base64": "...", "class_name": "...", "score": 0.9, "instance_id": 1}
      ],
      "transform": { ... TransformMeta ... }
    }

    If transform omitted, identity_for(original) is used (caller should prefer APIs that return it).
    """

    def __init__(self, backend_id: str = "http_api", options: Optional[Dict[str, Any]] = None):
        self.id = backend_id
        self.options = options or {}
        self.base_url = str(self.options.get("base_url") or "").rstrip("/")
        self.endpoint = str(self.options.get("endpoint") or "/v1/segment")
        self.model = str(self.options.get("model") or "")
        env_key = str(self.options.get("api_key_env") or "AUTOLABEL_VISION_KEY")
        self.api_key = str(self.options.get("api_key") or os.environ.get(env_key) or "")
        self.timeout = float(self.options.get("timeout") or 120)
        caps = self.options.get("capabilities") or {}
        self._caps = BackendCapabilities(
            image_input=True,
            text_guided_segmentation=bool(caps.get("text_guided_segmentation", True)),
            instance_mask=bool(caps.get("instance_mask", True)),
            class_mask=bool(caps.get("class_mask", False)),
            confidence=bool(caps.get("confidence", True)),
            reference_images=bool(caps.get("reference_images", False)),
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
        if not self.base_url:
            raise RuntimeError(f"Backend {self.id}: base_url is required")
        path = Path(image_path)
        img = imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("Failed to encode image")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload: Dict[str, Any] = {
            "model": self.model,
            "image_base64": b64,
            "image_path": path.name,
            "label_spec": label_spec.to_dict(),
        }
        if references is not None:
            payload["references"] = references.to_dict()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}{self.endpoint}"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        transform_raw = data.get("transform")
        if transform_raw:
            transform = TransformMeta.from_dict(transform_raw)
        else:
            transform = TransformMeta.identity_for(w, h)

        instances: List[InstancePred] = []
        for i, item in enumerate(data.get("instances") or [], start=1):
            if not isinstance(item, dict):
                continue
            mask_b64 = item.get("mask_png_base64") or item.get("mask_base64")
            if not mask_b64:
                continue
            mask = _decode_mask_b64(mask_b64)
            score = item.get("score")
            instances.append(
                InstancePred(
                    instance_id=int(item.get("instance_id") or i),
                    class_id=int(item.get("class_id") or 1),
                    class_name=str(item.get("class_name") or (label_spec.class_names[0] if label_spec.class_names else "object")),
                    mask=(mask > 0).astype(np.uint8),
                    score=float(score) if score is not None else None,
                )
            )

        return Stage1Result(instances=instances, transform=transform, raw=data)


@register_backend("http_api")
def _factory(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return HttpApiBackend(backend_id, options)
