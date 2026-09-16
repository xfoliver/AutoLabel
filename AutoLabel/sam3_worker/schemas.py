"""Request / response shapes for the SAM3 worker (documentation + helpers)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


WORKER_API_VERSION = "0.4"
WORKER_ENDPOINTS = ["/health", "/segment", "/refine"]
WORKER_REFINE_MODE = "interactive"


def health_payload(
    *,
    model_loaded: bool,
    device: str = "",
    error: str = "",
    refine_mode: str = "",
    sam3_version: str = "",
) -> Dict[str, Any]:
    status = "ok" if model_loaded else ("loading" if not error else "error")
    payload: Dict[str, Any] = {
        "status": status,
        "model_loaded": bool(model_loaded),
        "device": device,
        "api_version": WORKER_API_VERSION,
        "endpoints": list(WORKER_ENDPOINTS),
        "refine_mode": refine_mode or (WORKER_REFINE_MODE if model_loaded else ""),
        "sam3_version": sam3_version or "",
    }
    if error:
        payload["error"] = error
    return payload


def segment_request(
    *,
    image_path: str,
    prompt: str,
    score_threshold: float = 0.55,
    run_dir: str = "",
) -> Dict[str, Any]:
    return {
        "image_path": image_path,
        "prompt": prompt,
        "score_threshold": float(score_threshold),
        "run_dir": run_dir,
    }


def segment_response(
    *,
    prompt: str,
    image_path: str,
    image_size: List[int],
    instances: List[Dict[str, Any]],
    overlay_path: Optional[str] = None,
    result_json: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "image_path": image_path,
        "image_size": image_size,
        "instances": instances,
        "overlay_path": overlay_path,
        "result_json": result_json,
    }
