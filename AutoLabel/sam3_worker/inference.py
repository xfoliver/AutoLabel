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


def _sam3_version() -> str:
    # AutoLabel no longer ships SAM3 weights; the worker always loads SAM 3.1.
    raw = (os.environ.get("SAM3_VERSION") or os.environ.get("SAM3_HF_VERSION") or "sam3.1").strip().lower()
    if raw in {"sam3.1", "sam31", "sam3_1", "sam3p1", "sam3"}:
        return "sam3.1"
    return "sam3.1"


def _checkpoint_filename(version: str) -> str:
    return "sam3.1_multiplex.pt" if version == "sam3.1" else "sam3.pt"


def _allow_hf_download() -> bool:
    return (os.environ.get("SAM3_ALLOW_HF") or "").strip().lower() in {"1", "true", "yes"}


def _discover_local_checkpoint(version: str) -> Path | None:
    """Find a local weight file. Never hits the network."""
    env = (os.environ.get("SAM3_CHECKPOINT") or "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"SAM3_CHECKPOINT not found: {path}")
    name = _checkpoint_filename(version)
    here = Path(__file__).resolve()
    cands = [here.parents[1] / "models" / name]
    if len(here.parents) >= 3:
        repo = here.parents[2]
        cands.append(repo / "models" / name)
        cands.append(repo / "dist" / "AutoLabel_Portable" / "models" / name)
    if _SAM3_ROOT is not None:
        parent = _SAM3_ROOT.parent
        cands.extend(
            [
                parent / "models" / name,
                parent / "AutoLabel" / "models" / name,
                parent / "dist" / "AutoLabel_Portable" / "models" / name,
            ]
        )
    seen: set[str] = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.is_file():
                return cand
        except Exception:
            continue
    return None


def _discover_sam3_root() -> Path | None:
    env = (os.environ.get("SAM3_ROOT") or "").strip()
    here = Path(__file__).resolve()
    candidates = []
    if env:
        candidates.append(Path(env))
    # Portable layout: AutoLabel_Portable/sam3_src  (worker lives in AutoLabel_Portable/sam3_worker)
    candidates.append(here.parents[1] / "sam3_src")
    candidates.append(here.parents[1] / "sam3")
    # Dev layout: repo/sam3 next to AutoLabel/
    if len(here.parents) >= 3:
        candidates.append(here.parents[2] / "sam3")
    for raw in candidates:
        path = raw.expanduser()
        if (path / "sam3" / "model_builder.py").is_file() or (path / "model_builder.py").is_file():
            return path
    return None


_SAM3_ROOT = _discover_sam3_root()
if _SAM3_ROOT is not None and str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))


def _to_numpy(x: Any) -> np.ndarray:
    """CPU numpy. CUDA bfloat16 cannot ``.numpy()`` until cast to float32."""
    detach = getattr(x, "detach", None)
    if callable(detach):
        try:
            t = detach()
            to_float = getattr(t, "float", None)
            if callable(to_float):
                t = to_float()
            to_cpu = getattr(t, "cpu", None)
            if callable(to_cpu):
                t = to_cpu()
            to_np = getattr(t, "numpy", None)
            if callable(to_np):
                return np.asarray(to_np())
        except Exception:
            pass
    return np.asarray(x)


def _squeeze_mask(mask: Any) -> np.ndarray:
    arr = np.squeeze(_to_numpy(mask))
    if arr.ndim > 2:
        arr = arr[0]
    return arr


def _scores_list(scores: Any, n: int) -> List[Optional[float]]:
    """Flatten SAM3 score tensor/list to n python floats (None if missing)."""
    out: List[Optional[float]] = [None] * max(int(n), 0)
    if scores is None or n <= 0:
        return out
    try:
        arr = np.asarray(_to_numpy(scores), dtype=np.float64).reshape(-1)
    except Exception:
        return out
    for i in range(min(int(n), int(arr.size))):
        val = float(arr[i])
        if np.isfinite(val):
            out[i] = val
    return out


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


def _xyxy_to_cxcywh_norm(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = max(x2 - x1, 1.0) / w
    bh = max(y2 - y1, 1.0) / h
    return [
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(bw, 1.0 / w), 1.0),
        min(max(bh, 1.0 / h), 1.0),
    ]


def _point_to_cxcywh_norm(point: Sequence[float], width: int, height: int, *, box_frac: float = 0.04) -> List[float]:
    x, y = float(point[0]), float(point[1])
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    bw = max(box_frac, 8.0 / w)
    bh = max(box_frac, 8.0 / h)
    return [
        min(max(x / w, 0.0), 1.0),
        min(max(y / h, 0.0), 1.0),
        min(bw, 1.0),
        min(bh, 1.0),
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


def _wire_interactive_backbone(model: Any) -> bool:
    """Point the SAM-1 interactive tracker at the image model's VL backbone.

    ``build_sam3_image_model(enable_inst_interactivity=True)`` builds a tracker
    with ``backbone=None`` on purpose: the SAM2 neck lives on the detector
    backbone (``add_sam2_neck=True``). Without this share, ``predictor.set_image``
    crashes: ``NoneType.forward_image``.
    """
    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None:
        return False
    tracker = getattr(predictor, "model", None)
    if tracker is None:
        return False
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return False
    if getattr(tracker, "backbone", None) is None:
        tracker.backbone = backbone
    return getattr(tracker, "backbone", None) is not None


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
        self._image_key = ""
        self._image_state: Optional[Dict[str, Any]] = None
        self._image = None
        self._orient_meta: Dict[str, Any] = {}
        self._sam3_version = _sam3_version()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def error(self) -> str:
        return self._error

    @property
    def sam3_version(self) -> str:
        return self._sam3_version

    @property
    def refine_mode(self) -> str:
        if not self._loaded or self._model is None:
            return ""
        predictor = getattr(self._model, "inst_interactive_predictor", None)
        if predictor is None:
            return "geometric"
        tracker = getattr(predictor, "model", None)
        if tracker is None or getattr(tracker, "backbone", None) is None:
            return "broken"
        return "interactive"

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                # Box/point refine needs the SAM-1 interactive head, not text grounding.
                version = _sam3_version()
                self._sam3_version = version
                ckpt_path = _discover_local_checkpoint(version)
                bpe = (os.environ.get("SAM3_BPE") or "").strip()
                bpe_path = Path(bpe) if bpe else None
                if bpe_path is None and _SAM3_ROOT is not None:
                    for cand in (
                        _SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
                        _SAM3_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz",
                    ):
                        if cand.is_file():
                            bpe_path = cand
                            break
                if bpe_path is not None and not bpe_path.is_file():
                    raise FileNotFoundError(f"SAM3_BPE not found: {bpe_path}")
                load_from_hf = ckpt_path is None
                if load_from_hf:
                    if not _allow_hf_download():
                        name = _checkpoint_filename(version)
                        raise FileNotFoundError(
                            f"本地 {version} 权重未找到（{name}）。"
                            "请放到 AutoLabel/models 或仓库根 models/。"
                            "需要联网下载时设置 SAM3_ALLOW_HF=1。"
                        )
                    if version == "sam3.1":
                        from sam3.model_builder import download_ckpt_from_hf

                        ckpt_path = Path(download_ckpt_from_hf(version="sam3.1"))
                        load_from_hf = False
                model = build_sam3_image_model(
                    device=self._device,
                    enable_inst_interactivity=True,
                    checkpoint_path=str(ckpt_path) if ckpt_path is not None else None,
                    load_from_HF=load_from_hf,
                    bpe_path=str(bpe_path) if bpe_path is not None else None,
                )
                if not _wire_interactive_backbone(model):
                    raise RuntimeError(
                        "SAM3 interactive predictor has no backbone after wiring. "
                        "Box refine cannot run; restart the worker after updating AutoLabel/sam3_worker."
                    )
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
                    # User slider is often 0.3–0.55; some classes score ~0.10.
                    # Discover at 0.05, let AutoLabel filter.
                    requested = float(score_threshold)
                    discovery = requested if requested <= 0.05 else 0.05
                    self._processor.confidence_threshold = float(discovery)
                    state = self._set_image_locked(path, image)
                    if hasattr(self._processor, "reset_all_prompts"):
                        self._processor.reset_all_prompts(state)
                    output = self._processor.set_text_prompt(state=state, prompt=prompt)

        requested = float(score_threshold)
        discovery = requested if requested <= 0.05 else 0.05
        return self._pack_output(
            path=path,
            image=image,
            orient_meta=orient_meta,
            output=output,
            score_threshold=discovery,
            save_dir=save_dir,
            save_raw_mask=save_raw_mask,
            save_overlay=save_overlay,
            extra={
                "prompt": prompt,
                "mode": "text",
                "requested_score_threshold": requested,
            },
        )

    def refine(
        self,
        *,
        image_path: str | Path,
        bbox: Optional[Sequence[float]] = None,
        points: Optional[Sequence[Sequence[float]]] = None,
        score_threshold: float = 0.30,
        save_dir: Optional[str | Path] = None,
        save_raw_mask: bool = True,
        save_overlay: bool = False,
    ) -> Dict[str, Any]:
        """Cut a mask from a box / points via SAM3 interactive predictor (SAM-1 task).

        ``add_geometric_prompt`` is the wrong API here: it re-runs whole-image
        grounding and ignores the box as a crop. Box refine must use
        ``inst_interactive_predictor.predict``.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        boxes_in = list(bbox) if bbox else []
        pts_in = [list(p) for p in (points or []) if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(boxes_in) < 4 and not pts_in:
            raise ValueError("refine requires bbox and/or points")

        self.load()
        assert self._model is not None
        predictor = getattr(self._model, "inst_interactive_predictor", None)
        if predictor is None:
            raise RuntimeError(
                "SAM3 interactive predictor is not enabled. Restart AutoLabel so the worker reloads with enable_inst_interactivity=True."
            )

        image, orient_meta = open_image_normalized(path)
        import torch

        box_arr = np.array([float(x) for x in boxes_in[:4]], dtype=np.float32) if len(boxes_in) >= 4 else None
        point_coords = np.array(pts_in, dtype=np.float32) if pts_in else None
        point_labels = np.ones((len(pts_in),), dtype=np.int32) if pts_in else None
        # A box is unambiguous; extra masks from multimask_output wander off-object.
        multimask = box_arr is None

        with self._lock:
            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self._device == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with autocast_ctx:
                    self._set_interactive_image_locked(path, image)
                    masks, ious, _low = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_arr,
                        multimask_output=multimask,
                        normalize_coords=True,
                    )

        packed_masks = []
        packed_scores = []
        if masks is None:
            masks = []
        arr = _to_numpy(masks)
        if arr.ndim == 2:
            arr = arr[None, ...]
        iou_arr = _to_numpy(ious).reshape(-1) if ious is not None else None
        for i in range(int(arr.shape[0])):
            packed_masks.append(arr[i])
            score = None
            if iou_arr is not None and i < iou_arr.shape[0]:
                try:
                    score = float(iou_arr[i])
                except Exception:
                    score = None
            packed_scores.append(score)

        fake_output = {"masks": packed_masks, "scores": packed_scores, "boxes": None}
        return self._pack_output(
            path=path,
            image=image,
            orient_meta=orient_meta,
            output=fake_output,
            score_threshold=score_threshold,
            save_dir=save_dir,
            save_raw_mask=save_raw_mask,
            save_overlay=save_overlay,
            extra={"prompt": "visual", "mode": "refine", "bbox": list(boxes_in), "points": pts_in},
        )

    def _set_interactive_image_locked(self, path: Path, image) -> None:
        predictor = getattr(self._model, "inst_interactive_predictor", None)
        if predictor is None:
            raise RuntimeError("SAM3 interactive predictor is missing")
        key = f"{path.resolve()}::interactive"
        if self._image_key == key and getattr(predictor, "_is_image_set", False):
            return
        predictor.set_image(image)
        self._image_key = key
        self._image_state = None
        self._image = image

    def _set_image_locked(self, path: Path, image):
        key = str(path.resolve())
        if self._image_key == key and self._image_state is not None:
            return self._image_state
        state = self._processor.set_image(image)
        self._image_key = key
        self._image_state = state
        self._image = image
        return state

    def _pack_output(
        self,
        *,
        path: Path,
        image,
        orient_meta: Dict[str, Any],
        output: Dict[str, Any],
        score_threshold: float,
        save_dir: Optional[str | Path],
        save_raw_mask: bool,
        save_overlay: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        extra = extra or {}
        prompt = str(extra.get("prompt") or "")
        mode = str(extra.get("mode") or "text")
        width, height = image.size
        masks = output.get("masks") if isinstance(output, dict) else None
        boxes = output.get("boxes") if isinstance(output, dict) else None
        scores = output.get("scores") if isinstance(output, dict) else None

        n = 0
        if masks is not None:
            n = int(len(masks))
        instances: List[Dict[str, Any]] = []
        mask_arrays: List[np.ndarray] = []
        score_vals = _scores_list(scores, n)

        for i in range(n):
            mask_arr = _squeeze_mask(masks[i])
            binary = mask_arr > 0.5
            score = score_vals[i] if i < len(score_vals) else None
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
            tag = "mask"
            if mode == "refine":
                box = extra.get("bbox") or [0, 0, 0, 0]
                tag = "refine_{}_{}_{}_{}".format(*(int(float(x)) for x in (list(box) + [0, 0, 0, 0])[:4]))
            if save_raw_mask:
                for i, binary in enumerate(mask_arrays):
                    mask_path = out_dir / f"{stem}_sam3_{tag}_{i}.png"
                    Image.fromarray((binary > 0).astype(np.uint8) * 255).save(mask_path)
                    instances[i]["mask_path"] = str(mask_path)
            if save_overlay:
                overlay_path = str(out_dir / f"{stem}_sam3_{tag}_overlay.jpg")
                _write_overlay(image, mask_arrays, overlay_path)
            result_json = str(out_dir / f"{stem}_sam3_{tag}_result.json")
            payload = {
                "prompt": prompt,
                "mode": mode,
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
            "mode": mode,
            "image_path": str(path),
            "image_size": [int(width), int(height)],
            "instances": instances,
            "overlay_path": overlay_path,
            "result_json": result_json,
            "masks": mask_arrays,
            **orient_meta,
            **{k: v for k, v in extra.items() if k not in {"prompt", "mode"}},
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
