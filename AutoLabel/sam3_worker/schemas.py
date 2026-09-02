"""Request / response shapes for the SAM3 worker (documentation + helpers)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def health_payload(*, model_loaded: bool, device: str = "", error: str = "") -> Dict[str, Any]:
    status = "ok" if model_loaded else ("loading" if not error else "error")
    payload: Dict[str, Any] = {
        "status": status,
        "model_loaded": bool(model_loaded),
        "device": device,
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
