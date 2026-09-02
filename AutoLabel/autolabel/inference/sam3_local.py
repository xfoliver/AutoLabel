"""SAM3 local backend: HTTP client only.

Forbidden in this module: import sam3 / torch / triton.
Talks to AutoLabel/sam3_worker over 127.0.0.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from autolabel.cv_io import imread
from autolabel.inference.base import BackendCapabilities, InferenceBackend, register_backend
from autolabel.models import InstancePred, LabelSpec, References, Stage1Result, TransformMeta


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a[:4]]
    bx0, by0, bx1, by1 = [float(x) for x in b[:4]]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return float(inter) / float(max(area_a + area_b - inter, 1.0))


def nms_by_bbox(
    items: List[Dict[str, Any]],
    *,
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """Greedy NMS on worker instances using bbox IoU (highest score first)."""
    if iou_threshold <= 0 or len(items) <= 1:
        return items
    ranked = sorted(
        items,
        key=lambda x: float(x.get("score") if x.get("score") is not None else 0.0),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    for inst in ranked:
        bbox = inst.get("bbox") or [0, 0, 0, 0]
        drop = False
        for other in kept:
            if _bbox_iou(bbox, other.get("bbox") or [0, 0, 0, 0]) >= iou_threshold:
                drop = True
                break
        if not drop:
            kept.append(inst)
    return kept


def resolve_model_prompt(label_spec: LabelSpec) -> Tuple[str, str]:
    """Return (display_label, sam3_text_prompt)."""
    label = (label_spec.class_names[0] if label_spec.class_names else None) or "object"
    prompt = ""
    if label_spec.targets:
        t0 = label_spec.targets[0]
        label = (t0.label or label).strip() or label
        prompt = str(getattr(t0, "model_prompt", "") or "").strip()
        if not prompt:
            # include[] is a good place for an English visual synonym
            for item in t0.include or []:
                text = str(item).strip()
                if text:
                    prompt = text
                    break
        if not prompt and (t0.description or "").strip():
            desc = t0.description.strip()
            if len(desc) <= 32 and "\n" not in desc:
                prompt = desc
    if not prompt:
        prompt = label
    return label, prompt


def _load_mask(mask_path: str, fallback_hw: Tuple[int, int]) -> np.ndarray:
    img = imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"SAM3 mask missing: {mask_path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    binary = (img > 0).astype(np.uint8)
    h, w = fallback_hw
    mh, mw = binary.shape[:2]
    if (mh, mw) == (h, w):
        return binary
    # Swapped (H,W) vs (W,H) is EXIF / axis confusion. NEVER stretch — that
    # shears a 90°-oriented mask onto the display image (phone → chair).
    if (mh, mw) == (w, h):
        raise RuntimeError(
            f"SAM3 mask shape {(mh, mw)} is transposed vs image {(h, w)}. "
            "Worker and AutoLabel are not using the same EXIF orientation. "
            "Restart SAM3 Worker after updating AutoLabel/sam3_worker."
        )
    if mh * mw == 0:
        return np.zeros((h, w), dtype=np.uint8)
    # Same orientation, slightly different size only (letterbox leftovers).
    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return (binary > 0).astype(np.uint8)


class Sam3LocalBackend(InferenceBackend):
    """HTTP client for the resident SAM3 worker."""

    def __init__(self, backend_id: str = "sam3_local", options: Optional[Dict[str, Any]] = None):
        self.id = backend_id
        self.options = options or {}
        host = str(self.options.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        port = int(self.options.get("port") or 9892)
        base = str(self.options.get("base_url") or "").strip().rstrip("/")
        self.base_url = base or f"http://{host}:{port}"
        self.host = host
        self.port = port
        self.python_path = str(
            self.options.get("python_path") or r"C:\dlcv\envs\sam3\python.exe"
        )
        self.timeout = float(self.options.get("timeout") or 180)
        self.score_threshold = float(self.options.get("score_threshold") or 0.55)
        self.min_area = int(self.options.get("min_area") or 64)
        self.epsilon_ratio = float(self.options.get("epsilon_ratio") or 0.001)
        self.nms_iou_threshold = float(self.options.get("nms_iou_threshold") or 0.7)
        self.save_raw_mask = bool(self.options.get("save_raw_mask", True))
        self.save_overlay = bool(self.options.get("save_overlay", True))
        self.debug_dir: Optional[Path] = None
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

    def stage2_options(self) -> Dict[str, Any]:
        # Binary SAM3 masks: no GrabCut / color-quantize / morph fattening.
        return {
            "min_area": float(self.min_area),
            "epsilon_ratio": float(self.epsilon_ratio),
            "close_ksize": 0,
            "open_ksize": 0,
            "grabcut": False,
            "keep_largest": True,
        }

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=min(8.0, self.timeout))
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"status": "ok", "raw": data}

    def predict(
        self,
        image_path: str | Path,
        label_spec: LabelSpec,
        references: Optional[References] = None,
    ) -> Stage1Result:
        del references  # SAM3 text prompt only
        path = Path(image_path)
        img = imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        label, prompt = resolve_model_prompt(label_spec)

        payload = {
            "image_path": str(path.resolve()),
            "prompt": prompt,
            "score_threshold": self.score_threshold,
            "save_raw_mask": self.save_raw_mask,
            "save_overlay": self.save_overlay,
        }
        if self.debug_dir is not None:
            payload["run_dir"] = str(Path(self.debug_dir))
        url = f"{self.base_url}/segment"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(
                f"SAM3 Worker 无法连接 ({url}): {e}. "
                "请确认已用 sam3 Conda Python 启动 AutoLabel/sam3_worker/server.py"
            ) from e
        if resp.status_code >= 400:
            try:
                err = resp.json()
                msg = err.get("error") if isinstance(err, dict) else resp.text
            except Exception:
                msg = resp.text
            raise RuntimeError(f"SAM3 /segment failed ({resp.status_code}): {msg}")
        data = resp.json()
        raw_instances = list(data.get("instances") or [])
        raw_instances = nms_by_bbox(raw_instances, iou_threshold=self.nms_iou_threshold)

        instances: List[InstancePred] = []
        for i, item in enumerate(raw_instances, start=1):
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            area = int(item.get("area") or 0)
            mask_path = item.get("mask_path")
            if not mask_path:
                continue
            mask = _load_mask(str(mask_path), (h, w))
            area = int(np.count_nonzero(mask))
            if area < self.min_area:
                continue
            instances.append(
                InstancePred(
                    instance_id=i,
                    class_id=1,
                    class_name=label,
                    mask=mask,
                    score=float(score) if score is not None else None,
                )
            )

        transform = TransformMeta.identity_for(w, h)
        return Stage1Result(
            instances=instances,
            transform=transform,
            raw=data,
        )


@register_backend("sam3_local")
def _factory(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return Sam3LocalBackend(backend_id, options)
