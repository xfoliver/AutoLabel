"""InferenceBackend protocol, capabilities, and registry."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

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
    return sorted(_REGISTRY.keys())


def create_backend(type_name: str, backend_id: str, options: Optional[Dict[str, Any]] = None) -> InferenceBackend:
    if type_name not in _REGISTRY:
        # Lazy-import built-ins
        _ensure_builtins()
    if type_name not in _REGISTRY:
        raise KeyError(f"Unknown backend type: {type_name}. Known: {list_backend_types()}")
    return _REGISTRY[type_name](backend_id, options or {})


def _ensure_builtins() -> None:
    # Import side-effects register factories
    from autolabel.inference import mock as _mock  # noqa: F401
    from autolabel.inference import http_api as _http  # noqa: F401
    from autolabel.inference import dlcv_inference as _dlcv  # noqa: F401
    from autolabel.inference import openai_image_edits as _oie  # noqa: F401
    from autolabel.inference import sam3_local as _sam3  # noqa: F401


def load_backends_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load backends JSON; return dict with active_backend + backends list."""
    if path is None:
        # Default: AutoLabel/config/backends.json then example
        here = Path(__file__).resolve().parents[2] / "config"
        for name in ("backends.json", "backends.example.json"):
            cand = here / name
            if cand.exists():
                path = cand
                break
    if path is None:
        return {"active_backend": "mock", "backends": [{"id": "mock", "type": "mock", "options": {}}]}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("backends config must be a JSON object")
    return data


def build_backend_from_config(config: Optional[Dict[str, Any]] = None, backend_id: Optional[str] = None) -> InferenceBackend:
    cfg = config or load_backends_config()
    active = backend_id or cfg.get("active_backend") or "mock"
    backends = cfg.get("backends") or []
    entry = None
    for b in backends:
        if isinstance(b, dict) and b.get("id") == active:
            entry = b
            break
    if entry is None:
        # fallback mock
        return create_backend("mock", "mock", {})
    return create_backend(str(entry.get("type") or "mock"), str(entry.get("id") or active), entry.get("options") or {})
