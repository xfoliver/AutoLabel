"""OpenAI-compatible /v1/images/edits backend (e.g. gpt-image-2 via NewAPI).

Returns a colored flat mask PNG; locally split into InstancePred list.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from autolabel.cv_io import imread
from autolabel.errors import ImageModerationBlocked
from autolabel.geometry.transform import build_letterbox_meta, letterbox_image_bgr
from autolabel.inference.base import BackendCapabilities, InferenceBackend, register_backend
from autolabel.mask.color_instances import colored_mask_to_instances
from autolabel.models import LabelSpec, References, Stage1Result

logger = logging.getLogger(__name__)


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
        "- Respect Exclude: do not paint listed exclusions or any other non-targets.\n"
        "- If ZERO matching objects are present in the photo, output a PURE BLACK image only. "
        "Do NOT invent, hallucinate, or fill any colored regions.\n"
        "- Never paint non-target objects just to avoid an empty mask.\n"
    )


def parse_size_wh(size: str) -> Tuple[int, int]:
    """Parse OpenAI-style ``1024x1024`` / ``1024`` into (width, height)."""
    text = str(size or "1024x1024").strip().lower().replace("*", "x").replace("×", "x")
    if "x" not in text:
        n = max(1, int(float(text)))
        return n, n
    a, b = text.split("x", 1)
    return max(1, int(float(a))), max(1, int(float(b)))


def request_proxies(base_url: str, http_proxy: str = "") -> Optional[Dict[str, Optional[str]]]:
    """Return ``requests`` proxies. Private/LAN hosts always bypass the proxy."""
    from urllib.parse import urlparse

    host = (urlparse(str(base_url or "")).hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.startswith("192.168.") or host.startswith("10."):
        return {"http": None, "https": None}
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return {"http": None, "https": None}
        except (IndexError, ValueError):
            pass
    proxy = str(http_proxy or "").strip()
    if not proxy:
        # Explicit empty: do not inherit HTTPS_PROXY. Clash on 7890
        # hangs gpt-image /images/edits (connect works, body never returns).
        return {"http": None, "https": None}
    return {"http": proxy, "https": proxy}


def prepare_upload_image(
    image_bgr: np.ndarray,
    *,
    size: str = "1024x1024",
    upload_format: str = "jpeg",
    jpeg_quality: int = 85,
    downscale: bool = True,
) -> Tuple[bytes, str, str, Tuple[int, int]]:
    """Letterbox the source photo onto the requested edits canvas and encode it.

    gpt-image-* generates a square canvas. Sending a non-square JPEG makes the
    model pad internally; mapping that mask back with stretch shifts objects
    sideways. We pad here with black so upload geometry == mask geometry.
    ``downscale`` is kept for config compatibility; the canvas is always
    ``size``. Returns ``(payload, filename, mime, (w, h))``.
    """
    del downscale  # canvas is always the requested size
    if image_bgr is None or np.asarray(image_bgr).size == 0:
        raise RuntimeError("Cannot encode empty image for edits upload")
    tw, th = parse_size_wh(size)
    canvas, _meta = letterbox_image_bgr(image_bgr, (tw, th), fill=0)
    h, w = canvas.shape[:2]

    fmt = str(upload_format or "jpeg").strip().lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        q = int(min(100, max(40, jpeg_quality)))
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("Failed to encode image as JPEG for edits upload")
        return buf.tobytes(), "image.jpg", "image/jpeg", (w, h)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG for edits upload")
    return buf.tobytes(), "image.png", "image/png", (w, h)


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
        self.quality = str(self.options.get("quality") or "").strip()
        self.upload_format = str(self.options.get("upload_format") or "jpeg").strip().lower()
        self.jpeg_quality = int(self.options.get("jpeg_quality") or 85)
        self.downscale_upload = bool(self.options.get("downscale_upload", True))
        if "http_proxy" in self.options:
            # Empty string means "direct"; do not fall back to env.
            self.http_proxy = str(self.options.get("http_proxy") or "").strip()
        else:
            self.http_proxy = str(
                os.environ.get("HTTPS_PROXY")
                or os.environ.get("https_proxy")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("http_proxy")
                or ""
            ).strip()
        self._log = logger.info
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

    def bind_log(self, log) -> None:
        if callable(log):
            self._log = log

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

        upload_bytes, upload_name, upload_mime, upload_wh = prepare_upload_image(
            img,
            size=self.size,
            upload_format=self.upload_format,
            jpeg_quality=self.jpeg_quality,
            downscale=self.downscale_upload,
        )
        files = {
            "image": (upload_name, io.BytesIO(upload_bytes), upload_mime),
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            "n": "1",
            "size": self.size,
        }
        if self.quality:
            data["quality"] = self.quality

        proxies = request_proxies(self.base_url, self.http_proxy)
        via = (proxies or {}).get("https") if proxies else None
        via_txt = via or "direct"
        self._log(
            f"[{self.id}] {path.name} → {self.model} {self.size} "
            f"via {via_txt}（云端出图，约 30～90 秒，进度条这段不会动）"
        )
        t0 = time.perf_counter()
        try:
            # Connect budget is short; generation wait stays ``timeout`` (default 180s).
            resp = requests.post(
                url,
                headers=headers,
                data=data,
                files=files,
                timeout=(15.0, self.timeout),
                proxies=proxies,
            )
        except requests.RequestException as e:
            elapsed = time.perf_counter() - t0
            raise RuntimeError(
                f"Backend {self.id}: request failed after {elapsed:.1f}s: {e}"
            ) from e
        elapsed = time.perf_counter() - t0
        logger.info(
            "[%s] %s upload=%dx%d %dKB quality=%s proxy=%s elapsed=%.1fs http=%s",
            self.id,
            path.name,
            upload_wh[0],
            upload_wh[1],
            len(upload_bytes) // 1024,
            self.quality or "-",
            (proxies or {}).get("https") or "direct",
            elapsed,
            getattr(resp, "status_code", "?"),
        )

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
        # Square canvas + non-square photo: letterbox, never anisotropic stretch.
        transform = build_letterbox_meta(
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
            "quality": self.quality or None,
            "upload_size": [upload_wh[0], upload_wh[1]],
            "upload_bytes": len(upload_bytes),
            "request_seconds": round(elapsed, 3),
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
        return Stage1Result(
            instances=instances,
            transform=transform,
            colored_mask=mask_bgr,
            raw=raw_meta,
        )


@register_backend("openai_image_edits")
def _factory(backend_id: str, options: Dict[str, Any]) -> InferenceBackend:
    return OpenAIImageEditsBackend(backend_id, options)
