"""Load SAM3 once and run text-prompted instance segmentation.

Must be executed by the SAM3 Conda Python (e.g. C:\\dlcv\\envs\\sam3\\python.exe).
Does not import AutoLabel. Does not write into the official sam3/ tree.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

# Official SAM3 checkout sits next to AutoLabel; never modify it.
_SAM3_ROOT = Path(__file__).resolve().parents[2] / "sam3"
if _SAM3_ROOT.is_dir() and str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _squeeze_mask(mask: Any) -> np.ndarray:
    arr = np.squeeze(_to_numpy(mask))
    if arr.ndim > 2:
        arr = arr[0]
    return arr


def _bbox_from_mask(binary: np.ndarray) -> List[float]:
    ys, xs = np.where(binary)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]


def _safe_stem(path: Path) -> str:
    stem = path.stem or "image"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[:80]


def _exif_orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
        if not exif:
            return None
        value = exif.get(274) or exif.get(0x0112)
        return int(value) if value is not None else None
    except Exception:
        return None


def open_image_normalized(path: Path) -> tuple[Image.Image, dict]:
    """Load JPEG/PNG in display orientation (EXIF transpose), never write back."""
    raw = Image.open(path)
    raw_size = (int(raw.size[0]), int(raw.size[1]))
    ori = _exif_orientation(raw)
    image = ImageOps.exif_transpose(raw) or raw
    if image.mode != "RGB":
        image = image.convert("RGB")
    meta = {
        "exif_orientation": ori,
        "pil_raw_size_wh": [raw_size[0], raw_size[1]],
        "normalized_size_wh": [int(image.size[0]), int(image.size[1])],
        "applied_exif_transpose": bool(ori not in (None, 0, 1) and tuple(image.size) != raw_size),
    }
    return image, meta


class Sam3Engine:
    """Load the SAM3 image model once; reuse for every /segment call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._loaded = False
        self._error = ""

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def error(self) -> str:
        return self._error

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_sam3_image_model(device=self._device)
                processor = Sam3Processor(model, device=self._device)
                self._model = model
                self._processor = processor
                self._loaded = True
                self._error = ""
            except Exception as e:
                self._error = str(e)
                self._loaded = False
                raise

    def segment(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        score_threshold: float = 0.55,
        save_dir: Optional[str | Path] = None,
        save_raw_mask: bool = True,
        save_overlay: bool = True,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")

        self.load()
        assert self._processor is not None

        image, orient_meta = open_image_normalized(path)
        width, height = image.size

        import torch

        with self._lock:
            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self._device == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with autocast_ctx:
                    if hasattr(self._processor, "set_confidence_threshold"):
                        self._processor.confidence_threshold = float(score_threshold)
                    state = self._processor.set_image(image)
                    output = self._processor.set_text_prompt(state=state, prompt=prompt)

            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")

        n = 0
        if masks is not None:
            n = int(len(masks))
        instances: List[Dict[str, Any]] = []
        mask_arrays: List[np.ndarray] = []

        for i in range(n):
            mask_arr = _squeeze_mask(masks[i])
            binary = mask_arr > 0.5
            score = None
            if scores is not None:
                try:
                    score = float(_to_numpy(scores[i]).reshape(-1)[0])
                except Exception:
                    score = None
            if score is not None and score < float(score_threshold):
                continue
            bbox: List[float]
            if boxes is not None:
                try:
                    box = _to_numpy(boxes[i]).reshape(-1).tolist()
                    bbox = [float(x) for x in box[:4]]
                except Exception:
                    bbox = _bbox_from_mask(binary)
            else:
                bbox = _bbox_from_mask(binary)
            area = int(np.count_nonzero(binary))
            if area <= 0:
                continue
            mask_arrays.append(binary.astype(np.uint8))
            instances.append(
                {
                    "score": score,
                    "bbox": bbox,
                    "area": area,
                    "mask_path": None,
                }
            )

        overlay_path = None
        result_json = None
        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = _safe_stem(path)
            for i, binary in enumerate(mask_arrays):
                mask_path = out_dir / f"{stem}_sam3_mask_{i}.png"
                Image.fromarray((binary > 0).astype(np.uint8) * 255).save(mask_path)
                instances[i]["mask_path"] = str(mask_path)
            if save_overlay:
                overlay_path = str(out_dir / f"{stem}_sam3_overlay.jpg")
                _write_overlay(image, mask_arrays, overlay_path)
            result_json = str(out_dir / f"{stem}_sam3_result.json")
            payload = {
                "prompt": prompt,
                "image_path": str(path),
                "image_size": [int(width), int(height)],
                "score_threshold": float(score_threshold),
                **orient_meta,
                "instances": [
                    {
                        "score": inst["score"],
                        "bbox": inst["bbox"],
                        "area": inst["area"],
                    }
                    for inst in instances
                ],
            }
            Path(result_json).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "prompt": prompt,
            "image_path": str(path),
            "image_size": [int(width), int(height)],
            "instances": instances,
            "overlay_path": overlay_path,
            "result_json": result_json,
            "masks": mask_arrays,
            **orient_meta,
        }


def _write_overlay(image: Image.Image, masks: Sequence[np.ndarray], dest: str) -> None:
    original = np.array(image).astype(np.float32)
    overlay = original.copy()
    colors = [
        np.array([255.0, 0.0, 0.0]),
        np.array([0.0, 220.0, 80.0]),
        np.array([50.0, 120.0, 255.0]),
        np.array([255.0, 200.0, 0.0]),
        np.array([200.0, 0.0, 220.0]),
    ]
    for i, mask in enumerate(masks):
        binary = np.squeeze(mask).astype(bool)
        if binary.shape[:2] != overlay.shape[:2]:
            continue
        color = colors[i % len(colors)]
        overlay[binary] = overlay[binary] * 0.45 + color * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(dest, quality=90)


_ENGINE: Optional[Sam3Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Sam3Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Sam3Engine()
        return _ENGINE
