"""Pluggable vision inference backends."""

from autolabel.inference.base import (
    BackendCapabilities,
    InferenceBackend,
    create_backend,
    list_backend_types,
    load_backends_config,
    register_backend,
)

__all__ = [
    "BackendCapabilities",
    "InferenceBackend",
    "create_backend",
    "list_backend_types",
    "load_backends_config",
    "register_backend",
]
