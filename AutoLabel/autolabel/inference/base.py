"""InferenceBackend protocol, capabilities, and registry."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from autolabel.models import LabelSpec, References, Stage1Result


@dataclass
class BackendCapabilities:
    image_input: bool = True
    text_guided_segmentation: bool = False
    instance_mask: bool = False
    class_mask: bool = False
    confidence: bool = False
    reference_images: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BackendCapabilities":
        data = data or {}
        return cls(
            image_input=bool(data.get("image_input", True)),
            text_guided_segmentation=bool(data.get("text_guided_segmentation", False)),
            instance_mask=bool(data.get("instance_mask", False)),
            class_mask=bool(data.get("class_mask", False)),
            confidence=bool(data.get("confidence", False)),
            reference_images=bool(data.get("reference_images", False)),
        )


class InferenceBackend(ABC):
    """Stage-1 vision backend: image + LabelSpec → per-instance masks."""

    id: str = "base"

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image_path: str | Path,
        label_spec: LabelSpec,
        references: Optional[References] = None,
    ) -> Stage1Result:
        raise NotImplementedError

    def stage2_options(self) -> Dict[str, Any]:
        """Optional kwargs for stage1_to_shapes / polygon refine (epsilon, morph, …)."""
        return {}


_REGISTRY: Dict[str, Callable[[str, Dict[str, Any]], InferenceBackend]] = {}


def register_backend(type_name: str):
    """Decorator to register a backend factory: (id, options) -> InferenceBackend."""

    def deco(factory: Callable[[str, Dict[str, Any]], InferenceBackend]):
        _REGISTRY[type_name] = factory
        return factory

    return deco


def list_backend_types() -> List[str]:
    _ensure_builtins()
    return sorted(_REGISTRY.keys())


def create_backend(type_name: str, backend_id: str, options: Optional[Dict[str, Any]] = None) -> InferenceBackend:
    if type_name not in _REGISTRY:
        _ensure_builtins()
    if type_name not in _REGISTRY:
        raise KeyError(f"Unknown backend type: {type_name}. Known: {list_backend_types()}")
    return _REGISTRY[type_name](backend_id, options or {})


def _ensure_builtins() -> None:
    from autolabel.inference import mock as _mock  # noqa: F401
    from autolabel.inference import openai_image_edits as _oie  # noqa: F401
    from autolabel.inference import sam3_local as _sam3  # noqa: F401


def load_backends_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load backends JSON; return dict with annotation_engine + backends list."""
    from autolabel.paths import backends_path, bundled_config_dir, ensure_user_config

    if path is None:
        ensure_user_config()
        for cand in (backends_path(), bundled_config_dir() / "backends.example.json"):
            if cand.exists():
                path = cand
                break
    if path is None:
        return {
            "annotation_engine": "sam3.1",
            "backends": [{"id": "sam3_1_local", "type": "sam3_local", "options": {}}],
        }
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("backends config must be a JSON object")
    from autolabel.engines import ENGINE_SAM31, normalize_engine

    data["annotation_engine"] = normalize_engine(
        data.get("annotation_engine") or data.get("active_backend") or ENGINE_SAM31,
        backends_config=data,
    )
    return data


def _backend_entry(cfg: Dict[str, Any], backend_id: str) -> Optional[Dict[str, Any]]:
    for b in cfg.get("backends") or []:
        if isinstance(b, dict) and b.get("id") == backend_id:
            return b
    return None


def _entry_by_type(cfg: Dict[str, Any], type_name: str) -> Optional[Dict[str, Any]]:
    for b in cfg.get("backends") or []:
        if isinstance(b, dict) and b.get("type") == type_name:
            return b
    return None


def _merge_sam3_options(
    cfg: Dict[str, Any],
    settings: Optional[Dict[str, Any]],
    *,
    version: str = "sam3.1",
) -> Dict[str, Any]:
    from autolabel.engines import BACKEND_SAM3, BACKEND_SAM31, ENGINE_SAM31

    del version
    entry = (
        _backend_entry(cfg, BACKEND_SAM31)
        or _backend_entry(cfg, BACKEND_SAM3)
        or _entry_by_type(cfg, "sam3_local")
        or {}
    )
    opt = dict(entry.get("options") or {})
    settings = settings or {}
    if settings.get("sam3_python"):
        opt["python_path"] = str(settings["sam3_python"])
    if settings.get("sam3_host"):
        opt["host"] = str(settings["sam3_host"])
    if settings.get("sam3_port"):
        try:
            opt["port"] = int(settings["sam3_port"])
        except (TypeError, ValueError):
            pass
    host = str(opt.get("host") or "127.0.0.1")
    port = int(opt.get("port") or 9892)
    opt["base_url"] = f"http://{host}:{port}"
    opt["sam3_version"] = ENGINE_SAM31
    return opt


def _merge_api_engine_options(
    cfg: Dict[str, Any],
    settings: Optional[Dict[str, Any]],
    backend_id: str,
) -> Dict[str, Any]:
    from autolabel.engines import BACKEND_CHAT_IMAGE, ENGINE_CHAT_IMAGE

    entry = _backend_entry(cfg, backend_id)
    if entry is None and backend_id in {BACKEND_CHAT_IMAGE, ENGINE_CHAT_IMAGE}:
        entry = _backend_entry(cfg, BACKEND_CHAT_IMAGE) or _entry_by_type(cfg, "openai_image_edits")
    entry = entry or {}
    opt = dict(entry.get("options") or {})
    settings = settings or {}
    # Legacy single-slot vision_* settings only overlay the default chat-image backend.
    if str(entry.get("id") or backend_id) in {BACKEND_CHAT_IMAGE, ENGINE_CHAT_IMAGE}:
        url = str(settings.get("vision_base_url") or opt.get("base_url") or "").strip()
        model = str(settings.get("vision_model") or opt.get("model") or "").strip()
        key = str(settings.get("vision_api_key") or opt.get("api_key") or "").strip()
        if url:
            opt["base_url"] = url
        if model:
            opt["model"] = model
        if key:
            opt["api_key"] = key
    return opt


def build_backend_from_config(
    config: Optional[Dict[str, Any]] = None,
    backend_id: Optional[str] = None,
    *,
    settings: Optional[Dict[str, Any]] = None,
    intent: Any = None,
    log: Optional[Callable[[str], None]] = None,
) -> InferenceBackend:
    """Build a backend from config.

    ``backend_id`` may be SAM3.1, a user-added API engine id, the legacy
    ``chat-image`` / ``gpt_image_mask`` names, or ``mock``.
    """
    from autolabel.engines import (
        BACKEND_SAM31,
        ENGINE_SAM31,
        is_sam_family,
        resolve_annotation_engine,
        sam3_version_for_engine,
    )

    del intent
    cfg = config or load_backends_config()
    requested = backend_id or cfg.get("annotation_engine") or cfg.get("active_backend") or ENGINE_SAM31
    requested = str(requested).strip() or ENGINE_SAM31

    entry = _backend_entry(cfg, requested)
    if entry is not None and str(entry.get("type") or "") in {"mock", "no_instance"}:
        backend = create_backend(str(entry.get("type") or "mock"), str(entry.get("id") or requested), entry.get("options") or {})
        if log and hasattr(backend, "bind_log"):
            backend.bind_log(log)
        return backend
    if requested == "mock":
        backend = create_backend("mock", "mock", {})
        if log and hasattr(backend, "bind_log"):
            backend.bind_log(log)
        return backend

    internal = resolve_annotation_engine(requested, settings=settings, backends_config=cfg)
    if is_sam_family(internal, cfg) or internal == BACKEND_SAM31:
        version = sam3_version_for_engine(requested)
        options = _merge_sam3_options(cfg, settings, version=version)
        backend = create_backend("sam3_local", BACKEND_SAM31, options)
    else:
        options = _merge_api_engine_options(cfg, settings, internal)
        backend = create_backend("openai_image_edits", internal, options)
    if log and hasattr(backend, "bind_log"):
        backend.bind_log(log)
    return backend
