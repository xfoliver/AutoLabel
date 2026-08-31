"""OpenAI-compatible /v1/images/edits backend (e.g. gpt-image-2 via NewAPI).

Returns a colored flat mask PNG; locally split into InstancePred list.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from autolabel.cv_io import imread
from autolabel.errors import ImageModerationBlocked
from autolabel.geometry.transform import build_stretch_meta
from autolabel.inference.base import BackendCapabilities, InferenceBackend, register_backend
from autolabel.mask.color_instances import colored_mask_to_instances
from autolabel.models import LabelSpec, References, Stage1Result


_MODERATION_MARKERS = (
    "moderation_blocked",
    "safety_violations",
    "rejected by the safety system",
    "image_generation_user_error",
)


def _looks_like_moderation_block(status_code: int, body: Any) -> bool:
    """True when OpenAI/NewAPI safety moderation rejected this single image."""
    if status_code not in {400, 403}:
        return False
    if isinstance(body, dict):
        code = str(body.get("code") or "").strip().lower()
        err_type = str(body.get("type") or "").strip().lower()
        msg = str(body.get("message") or "")
        text = f"{code} {err_type} {msg}".lower()
        if code == "moderation_blocked" or err_type == "image_generation_user_error":
            return True
    else:
        text = str(body or "").lower()
    return any(m in text for m in _MODERATION_MARKERS)


def build_mask_prompt(label_spec: LabelSpec, *, prompt_extra: str = "") -> str:
    label = (label_spec.class_names[0] if label_spec.class_names else None) or "object"
    desc = ""
    exclude: List[str] = []
    if label_spec.targets:
        t0 = label_spec.targets[0]
        label = t0.label or label
        desc = t0.description or ""
        exclude = list(t0.exclude or [])
    exclude_txt = ", ".join(exclude) if exclude else "(none)"
    extra = (prompt_extra or "").strip()
    extra_block = f"\nAdditional instructions:\n{extra}\n" if extra else ""
    return (
        "You are producing a segmentation mask image for industrial annotation.\n"
        f'Input is a photo. Segment EVERY distinct instance of: "{label}".\n'
        f"Description: {desc or '(none)'}\n"
        f"Exclude: {exclude_txt}\n"
        f"{extra_block}"
        "Output rules (strict):\n"
        "- Flat mask only: pure black background (#000000).\n"
        "- Each separate object instance MUST be a different solid pure RGB color "
        "(no shared colors between instances).\n"
        "- No anti-aliasing, gradients, shadows, borders, text, legends, or photorealism.\n"
        "- Tight pixel-accurate silhouette: follow the true object boundary closely.\n"
        "- No halo, glow, or outline stroke outside the object.\n"
        "- Cover the full object silhouette; do not leave holes unless the object has a real hole.\n"
        "- Respect Exclude: do not paint case decorations, accessories, or other non-targets.\n"
        "- If ZERO matching objects are present in the photo, output a PURE BLACK image only. "
        "Do NOT invent, hallucinate, or fill any colored regions.\n"
        "- Never paint non-target objects just to avoid an empty mask.\n"
    )


class OpenAIImageEditsBackend(InferenceBackend):
    """Call /v1/images/edits and convert colored mask → instances."""

    def __init__(self, backend_id: str = "gpt_image_mask", options: Optional[Dict[str, Any]] = None):
        self.id = backend_id
        self.options = options or {}
        self.base_url = str(self.options.get("base_url") or "").rstrip("/")
        endpoint = str(self.options.get("endpoint") or "/images/edits")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        self.endpoint = endpoint
        self.model = str(self.options.get("model") or "gpt-image-2")
        env_key = str(self.options.get("api_key_env") or "AUTOLABEL_VISION_KEY")
        self.api_key = str(self.options.get("api_key") or os.environ.get(env_key) or "")
        self.size = str(self.options.get("size") or "1024x1024")
        self.timeout = float(self.options.get("timeout") or 180)
        self.min_area = int(self.options.get("min_area") or 64)
        self.quantize = int(self.options.get("quantize") or 8)
        self.bg_luma_thresh = float(self.options.get("bg_luma_thresh") or 24)
        self.merge_fragments = bool(self.options.get("merge_fragments", True))
        self.merge_iou = float(self.options.get("merge_iou") or 0.15)
        self.merge_bbox_iou = float(self.options.get("merge_bbox_iou") or 0.20)
        self.merge_gap_px = int(self.options.get("merge_gap_px", 5))
        self.min_relative_area = float(self.options.get("min_relative_area") or 0.12)
        self.prompt_extra = str(self.options.get("prompt_extra") or "")
        self.epsilon_ratio = float(self.options.get("epsilon_ratio") or 0.0005)
        # Default 0: avoid morph close fattening edges before / after GrabCut.
        self.mask_close_ksize = int(self.options.get("mask_close_ksize", 0))
        self.mask_open_ksize = int(self.options.get("mask_open_ksize", 0))
        self.grabcut = bool(self.options.get("grabcut", True))
        self.grabcut_iters = int(self.options.get("grabcut_iters", 3))
        self.grabcut_pad_px = int(self.options.get("grabcut_pad_px", 8))
        self.grabcut_sure_fg_erode = int(self.options.get("grabcut_sure_fg_erode", 3))
        self.grabcut_sure_bg_dilate = int(self.options.get("grabcut_sure_bg_dilate", 8))
        caps = self.options.get("capabilities") or {}
        self._caps = BackendCapabilities(
            image_input=True,
            text_guided_segmentation=bool(caps.get("text_guided_segmentation", True)),
            instance_mask=bool(caps.get("instance_mask", True)),
            class_mask=bool(caps.get("class_mask", False)),
            confidence=bool(caps.get("confidence", False)),
            reference_images=bool(caps.get("reference_images", False)),
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def stage2_options(self) -> Dict[str, Any]:
        return {
            "min_area": float(self.min_area),
            "epsilon_ratio": float(self.epsilon_ratio),
            "close_ksize": int(self.mask_close_ksize),
            "open_ksize": int(self.mask_open_ksize),
            "grabcut": bool(self.grabcut),
            "grabcut_iters": int(self.grabcut_iters),
            "grabcut_pad_px": int(self.grabcut_pad_px),
            "grabcut_sure_fg_erode": int(self.grabcut_sure_fg_erode),
            "grabcut_sure_bg_dilate": int(self.grabcut_sure_bg_dilate),
        }

    def predict(
        self,
        image_path: str | Path,
        label_spec: LabelSpec,
        references: Optional[References] = None,
    ) -> Stage1Result:
        if not self.base_url:
            raise RuntimeError(f"Backend {self.id}: base_url is required")
        if not self.api_key:
            raise RuntimeError(
                f"Backend {self.id}: api_key missing "
                f"(set options.api_key or env {self.options.get('api_key_env') or 'AUTOLABEL_VISION_KEY'})"
            )

        path = Path(image_path)
        img = imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        oh, ow = img.shape[:2]

        prompt = build_mask_prompt(label_spec, prompt_extra=self.prompt_extra)
        url = f"{self.base_url}{self.endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Encode as PNG bytes for multipart
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("Failed to encode image as PNG for edits upload")
        png_bytes = buf.tobytes()

        files = {
            "image": (path.name if path.suffix else "image.png", io.BytesIO(png_bytes), "image/png"),
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            "n": "1",
            "size": self.size,
        }

        try:
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Backend {self.id}: request failed: {e}") from e

        if resp.status_code >= 400:
            body = (resp.text or "")[:800]
            if _looks_like_moderation_block(resp.status_code, body):
                raise ImageModerationBlocked(
                    f"Backend {self.id}: HTTP {resp.status_code}: {body}",
                    status_code=resp.status_code,
                    body=body,
                )
            raise RuntimeError(f"Backend {self.id}: HTTP {resp.status_code}: {body}")

        try:
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"Backend {self.id}: invalid JSON response") from e

        if isinstance(payload, dict) and payload.get("error"):
            err = payload.get("error")
            if _looks_like_moderation_block(resp.status_code, err):
                raise ImageModerationBlocked(
                    f"Backend {self.id}: API error: {err}",
                    status_code=resp.status_code,
                    body=err,
                )
            raise RuntimeError(f"Backend {self.id}: API error: {err}")

        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"Backend {self.id}: response missing data[]")

        item0 = items[0] if isinstance(items[0], dict) else {}
        b64 = item0.get("b64_json") or item0.get("b64")
        if not b64 and item0.get("url"):
            raise RuntimeError(
                f"Backend {self.id}: got image URL but b64_json is required "
                "(configure gateway to return b64_json)"
            )
        if not b64:
            raise RuntimeError(f"Backend {self.id}: data[0] missing b64_json")

        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        mask_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise RuntimeError(f"Backend {self.id}: failed to decode mask PNG")

        mh, mw = mask_bgr.shape[:2]
        class_name = (
            label_spec.class_names[0]
            if label_spec.class_names
            else (label_spec.targets[0].label if label_spec.targets else "object")
        )
        transform = build_stretch_meta(
            original_wh=(ow, oh),
            model_input_wh=(mw, mh),
            mask_wh=(mw, mh),
        )
        from autolabel.geometry.transform import original_min_area_to_mask

        # options.min_area is original-image pixels; convert to mask space.
        orig_min = float(self.min_area)
        mask_min = max(16, int(round(original_min_area_to_mask(orig_min, transform))))
        instances = colored_mask_to_instances(
            mask_bgr,
            class_name=str(class_name or "object"),
            class_id=1,
            min_area=mask_min,
            quantize=self.quantize,
            bg_luma_thresh=self.bg_luma_thresh,
            merge_fragments=self.merge_fragments,
            merge_iou=self.merge_iou,
            merge_bbox_iou=self.merge_bbox_iou,
            min_relative_area=self.min_relative_area,
            merge_gap_px=self.merge_gap_px,
        )
        # Empty is valid: target may be absent → pure-black mask → zero instances.
        # Downstream review buckets these as "empty"; do not fail the image.

        raw_meta = {
            "model": self.model,
            "size_requested": self.size,
            "mask_size": [mw, mh],
            "usage": payload.get("usage") if isinstance(payload, dict) else None,
            "output_format": payload.get("output_format") if isinstance(payload, dict) else None,
            "n_instances": len(instances),
            "empty_mask": len(instances) == 0,
            "min_area_original": orig_min,
            "min_area_mask": mask_min,
            "merge_fragments": self.merge_fragments,
            "merge_iou": self.merge_iou,
            "merge_bbox_iou": self.merge_bbox_iou,
            "merge_gap_px": self.merge_gap_px,
            "min_relative_area": self.min_relative_area,
        }
        return Stage1Result(instances=instances, transform=transform, raw=raw_meta)


@register_backend("openai_image_edits")
def _factory(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return OpenAIImageEditsBackend(backend_id, options)
