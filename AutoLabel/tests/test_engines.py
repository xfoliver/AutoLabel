"""User-facing engine catalog."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.engines import (  # noqa: E402
    BACKEND_CHAT_IMAGE,
    BACKEND_SAM31,
    ENGINE_CHAT_IMAGE,
    ENGINE_SAM3,
    ENGINE_SAM31,
    list_user_engine_choices,
    normalize_engine,
    resolve_annotation_engine,
    sam3_version_for_engine,
    slug_engine_id,
)
from autolabel.inference.base import build_backend_from_config  # noqa: E402
from autolabel.inference.mock import MockBackend  # noqa: E402
from autolabel.inference.openai_image_edits import OpenAIImageEditsBackend  # noqa: E402
from autolabel.inference.sam3_local import Sam3LocalBackend  # noqa: E402


def test_normalize_retired_engines():
    assert normalize_engine("auto") == ENGINE_SAM31
    assert normalize_engine("image") == ENGINE_SAM31
    assert normalize_engine("image_sam3") == ENGINE_SAM31
    assert normalize_engine("sam3") == ENGINE_SAM31
    assert normalize_engine("sam3_local") == ENGINE_SAM31
    assert normalize_engine("chat-image") == ENGINE_CHAT_IMAGE
    assert normalize_engine("gpt_image_mask") == ENGINE_CHAT_IMAGE
    assert normalize_engine("sam3.1") == ENGINE_SAM31
    assert normalize_engine("sam3_1_local") == ENGINE_SAM31
    assert normalize_engine("sam31") == ENGINE_SAM31


def test_normalize_keeps_custom_api_id():
    cfg = {
        "backends": [
            {
                "id": "my_gpt",
                "type": "openai_image_edits",
                "label": "GPT Image",
                "options": {"base_url": "https://api.example/v1", "model": "gpt-image-2"},
            }
        ]
    }
    assert normalize_engine("my_gpt", cfg) == "my_gpt"
    assert normalize_engine("gpt_image_mask", cfg) == ENGINE_CHAT_IMAGE
    cfg["backends"].append(
        {"id": "gpt_image_mask", "type": "openai_image_edits", "options": {"base_url": "http://x"}}
    )
    assert normalize_engine("chat-image", cfg) == BACKEND_CHAT_IMAGE
    assert normalize_engine("gpt_image_mask", cfg) == BACKEND_CHAT_IMAGE


def test_resolve_to_internal_ids():
    assert resolve_annotation_engine("sam3") == BACKEND_SAM31
    assert resolve_annotation_engine("sam3.1") == BACKEND_SAM31
    assert resolve_annotation_engine("chat-image") == BACKEND_CHAT_IMAGE
    assert resolve_annotation_engine(None) == BACKEND_SAM31
    cfg = {
        "backends": [
            {"id": "my_gpt", "type": "openai_image_edits", "options": {"base_url": "http://x"}},
        ]
    }
    assert resolve_annotation_engine("my_gpt", backends_config=cfg) == "my_gpt"


def test_sam3_version_always_31():
    assert sam3_version_for_engine("sam3") == ENGINE_SAM31
    assert sam3_version_for_engine("sam3.1") == ENGINE_SAM31
    assert sam3_version_for_engine(None) == ENGINE_SAM31


def test_user_engine_choices_builtin_only_by_default():
    ids = [eid for eid, _ in list_user_engine_choices()]
    assert ids == [ENGINE_SAM31]
    assert ENGINE_SAM3 not in ids
    assert ENGINE_CHAT_IMAGE not in ids


def test_user_engine_choices_includes_api_engines():
    cfg = {
        "backends": [
            {"id": "sam3_1_local", "type": "sam3_local", "options": {}},
            {
                "id": "gpt_image_mask",
                "type": "openai_image_edits",
                "label": "gpt-image-2",
                "options": {"base_url": "https://api.example/v1", "model": "gpt-image-2"},
            },
            {
                "id": "qwen_edit",
                "type": "openai_image_edits",
                "label": "Qwen Image",
                "options": {"base_url": "https://dashscope.example/v1", "model": "qwen-image-edit"},
            },
        ]
    }
    ids = [eid for eid, _ in list_user_engine_choices(cfg)]
    assert ids == [ENGINE_SAM31, BACKEND_CHAT_IMAGE, "qwen_edit"]


def test_build_sam31_and_chat_image():
    cfg = {
        "annotation_engine": "sam3.1",
        "backends": [
            {"id": "sam3_1_local", "type": "sam3_local", "options": {"port": 9892}},
            {"id": "gpt_image_mask", "type": "openai_image_edits", "options": {"base_url": "http://x", "api_key": "k"}},
        ],
    }
    sam31 = build_backend_from_config(cfg, "sam3.1")
    assert isinstance(sam31, Sam3LocalBackend)
    assert sam31.id == BACKEND_SAM31
    assert sam31.sam3_version == ENGINE_SAM31
    retired = build_backend_from_config(cfg, "sam3")
    assert isinstance(retired, Sam3LocalBackend)
    assert retired.id == BACKEND_SAM31
    assert retired.sam3_version == ENGINE_SAM31
    chat = build_backend_from_config(cfg, "chat-image", settings={"vision_base_url": "http://y", "vision_api_key": "kk"})
    assert isinstance(chat, OpenAIImageEditsBackend)
    assert chat.id == BACKEND_CHAT_IMAGE
    assert chat.options["base_url"] == "http://y"
    mock = build_backend_from_config(cfg, "mock")
    assert isinstance(mock, MockBackend)


def test_build_custom_api_engine():
    cfg = {
        "annotation_engine": "qwen_edit",
        "backends": [
            {
                "id": "qwen_edit",
                "type": "openai_image_edits",
                "options": {"base_url": "https://qwen.example/v1", "model": "qwen-image-edit", "api_key": "sk-q"},
            }
        ],
    }
    backend = build_backend_from_config(cfg, "qwen_edit")
    assert isinstance(backend, OpenAIImageEditsBackend)
    assert backend.id == "qwen_edit"
    assert backend.options["base_url"] == "https://qwen.example/v1"
    assert backend.options["model"] == "qwen-image-edit"


def test_load_normalizes_legacy_annotation_engine(tmp_path: Path):
    p = tmp_path / "backends.json"
    p.write_text('{"annotation_engine": "image_sam3", "backends": []}', encoding="utf-8")
    from autolabel.inference.base import load_backends_config

    data = load_backends_config(p)
    assert data["annotation_engine"] == ENGINE_SAM31


def test_slug_engine_id_avoids_reserved():
    assert slug_engine_id("SAM3.1") != ENGINE_SAM31
    assert slug_engine_id("chat-image") != ENGINE_CHAT_IMAGE
    first = slug_engine_id("GPT Image")
    second = slug_engine_id("GPT Image", [first])
    assert first != second
