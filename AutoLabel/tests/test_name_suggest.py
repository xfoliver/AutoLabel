"""AI name-suggest: Vision chat/completions → short SAM3 prompts. No SAM3."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.errors import NameSuggestError  # noqa: E402
from autolabel.name_suggest import (  # noqa: E402
    DEFAULT_GPT_VISION_MODEL,
    NAME_SUGGEST_SYSTEM,
    _chat_completions_url,
    chat_model_for_name_suggest,
    first_preview_image,
    looks_like_image_generation_model,
    looks_like_sam_prompt,
    parse_candidates,
    require_vision_credentials,
    resolve_vision_credentials,
    sample_name_suggest_images,
    sanitize_english,
    suggest_names_from_image,
)


def test_prompt_has_no_object_vocabulary():
    """Production prompt must not steer the model toward any named object."""
    blob = NAME_SUGGEST_SYSTEM.lower()
    for needle in ("wooden", "disk", "phone", "weld", "car", "person", "cup"):
        assert needle not in blob


def test_settings_vision_wins_over_backend():
    settings = {
        "vision_base_url": "https://vectorengine.ai/v1",
        "vision_model": "chat-image-2",
        "vision_api_key": "sk-settings",
    }
    backends = {
        "backends": [
            {
                "id": "gpt_image_mask",
                "type": "openai_image_edits",
                "options": {
                    "base_url": "https://other.example/v1",
                    "model": "gpt-image-2",
                    "api_key": "sk-backend",
                },
            }
        ]
    }
    creds = resolve_vision_credentials(settings, backends)
    assert creds["base_url"] == "https://vectorengine.ai/v1"
    assert creds["model"] == "gpt-4o"
    assert creds["configured_model"] == "chat-image-2"
    assert creds["api_key"] == "sk-settings"


def test_gpt_image_2_remaps_to_gpt4o():
    settings = {
        "vision_base_url": "https://vectorengine.ai/v1",
        "vision_model": "gpt-image-2",
        "vision_api_key": "sk-vision",
        "llm_base_url": "http://192.168.1.3:3000/v1",
        "llm_model": "grok-4.6",
        "llm_api_key": "sk-llm",
        "vision_chat_model": "qwen-vl-plus",
    }
    creds = require_vision_credentials(settings, {})
    assert creds["model"] == DEFAULT_GPT_VISION_MODEL
    assert creds["base_url"] == "https://vectorengine.ai/v1"
    assert creds["api_key"] == "sk-vision"
    assert creds["configured_model"] == "gpt-image-2"


def test_chat_model_for_name_suggest():
    assert chat_model_for_name_suggest("gpt-image-2") == "gpt-4o"
    assert chat_model_for_name_suggest("chat-image-2") == "gpt-4o"
    assert chat_model_for_name_suggest("dall-e-3") == "gpt-4o"
    assert chat_model_for_name_suggest("gpt-4o") == "gpt-4o"
    assert chat_model_for_name_suggest("gpt-4.1") == "gpt-4.1"
    assert chat_model_for_name_suggest("") == "gpt-4o"


def test_looks_like_image_generation_model():
    assert looks_like_image_generation_model("gpt-image-2")
    assert looks_like_image_generation_model("dall-e-3")
    assert looks_like_image_generation_model("qwen-image-edit-max")
    assert not looks_like_image_generation_model("gpt-4o")
    assert not looks_like_image_generation_model("qwen-vl-plus")
    assert not looks_like_image_generation_model("grok-4.6")


def test_fallback_to_chat_image_backend_options():
    backends = {
        "backends": [
            {
                "id": "gpt_image_mask",
                "options": {
                    "base_url": "https://api.example/v1",
                    "model": "gpt-image-2",
                    "api_key": "sk-b",
                },
            }
        ]
    }
    creds = resolve_vision_credentials({}, backends)
    assert creds["model"] == "gpt-4o"
    assert creds["configured_model"] == "gpt-image-2"
    assert creds["api_key"] == "sk-b"


def test_missing_credentials_raises():
    with pytest.raises(NameSuggestError) as ei:
        require_vision_credentials({}, {"backends": []})
    msg = str(ei.value)
    assert "视觉识别" in msg or "凭证" in msg
    assert "Base URL" in msg


def test_parse_keeps_whatever_short_names_the_model_returned():
    payload = {
        "candidates": [
            {"english": "alpha part", "chinese": "甲", "confidence": 0.92},
            {"english": "beta-part", "chinese": "乙", "confidence": 0.86},
            {"english": "gamma widget", "chinese": "丙", "confidence": 0.74},
            {
                "english": "a very long caption that is not a SAM prompt at all",
                "chinese": "过长",
                "confidence": 0.5,
            },
            {"english": "delta", "chinese": "丁", "confidence": 99},
            {"english": "alpha part", "chinese": "重复", "confidence": 0.1},
        ]
    }
    cands = parse_candidates(payload)
    assert [c.english for c in cands if c.english] == [
        "alpha part",
        "beta-part",
        "gamma widget",
        "a very long caption that is not a SAM prompt at all",
        "delta",
    ]
    assert cands[0].chinese == "甲"
    assert cands[0].confidence == pytest.approx(0.92)


def test_parse_candidates_from_markdown_fence():
    text = """```json
{"candidates": [{"english": "omega clip", "chinese": "夹", "confidence": 0.8}]}
```"""
    cands = parse_candidates(text)
    assert cands[0].english == "omega clip"


def test_parse_accepts_alternate_json_shapes():
    named = parse_candidates({"names": [{"name": "alpha part", "label_zh": "甲"}]})
    assert named[0].english == "alpha part"
    assert named[0].chinese == "甲"

    listed = parse_candidates([{"label": "beta-part"}, {"prompt": "gamma widget"}])
    assert [c.english for c in listed] == ["beta-part", "gamma widget"]

    plain = parse_candidates("omega clip")
    assert plain[0].english == "omega clip"

    zh_only = parse_candidates({"object": "圆形垫片"})
    assert zh_only[0].chinese == "圆形垫片"
    assert zh_only[0].english == ""

    mixed = parse_candidates("圆形垫片 gasket")
    assert mixed[0].english == "gasket"
    assert mixed[0].chinese == "圆形垫片"

    subject = parse_candidates('{"main_subject": "A person sitting at a table indoors"}')
    assert "person" in subject[0].english.lower()
    assert "sitting" in subject[0].english.lower()


def test_parse_keeps_sentence_descriptions():
    cands = parse_candidates(
        {
            "candidates": [
                {
                    "english": "a yellow labrador retriever standing on green grass",
                    "chinese": "草地上的黄色拉布拉多",
                    "confidence": 0.9,
                }
            ]
        }
    )
    assert "labrador" in cands[0].english.lower()
    assert "grass" in cands[0].english.lower()
    assert "拉布拉多" in cands[0].chinese


def test_parse_rejects_empty_and_image_payload():
    with pytest.raises(NameSuggestError):
        parse_candidates({"candidates": []})
    with pytest.raises(NameSuggestError):
        parse_candidates("   ")
    with pytest.raises(NameSuggestError) as ei:
        parse_candidates({"data": [{"b64_json": "aaaa"}]})
    assert "图像生成" in str(ei.value)


def test_first_preview_image_accepts_file_or_folder(tmp_path: Path):
    img = tmp_path / "shot.jpg"
    img.write_bytes(b"x")
    assert first_preview_image(img) == img
    folder = tmp_path / "pics"
    folder.mkdir()
    one = folder / "a.png"
    one.write_bytes(b"x")
    assert first_preview_image(folder) == one
    with pytest.raises(NameSuggestError):
        first_preview_image(tmp_path / "missing")


def test_sample_name_suggest_images_picks_up_to_five(tmp_path: Path):
    folder = tmp_path / "pics"
    folder.mkdir()
    files = []
    for i in range(8):
        p = folder / f"{i}.png"
        p.write_bytes(b"x")
        files.append(p)
    picked = sample_name_suggest_images(folder, count=5)
    assert len(picked) == 5
    assert len(set(picked)) == 5
    assert set(picked) <= set(files)
    fewer = sample_name_suggest_images(folder, count=5)
    assert len(fewer) == 5
    single = sample_name_suggest_images(files[0])
    assert single == [files[0]]
    tiny = tmp_path / "tiny"
    tiny.mkdir()
    a = tiny / "a.jpg"
    a.write_bytes(b"x")
    b = tiny / "b.jpg"
    b.write_bytes(b"x")
    both = sample_name_suggest_images(tiny, count=5)
    assert set(both) == {a, b}


def test_sanitize_is_format_only():
    assert sanitize_english("a alpha part") == "a alpha part"
    assert looks_like_sam_prompt("alpha part")
    assert looks_like_sam_prompt("odd-widget")
    assert looks_like_sam_prompt("a yellow dog standing on grass near a fence")
    long_sentence = "a " + "very " * 40 + "long yellow dog sitting on grass"
    assert looks_like_sam_prompt(long_sentence)
    assert sanitize_english(long_sentence) == long_sentence
    assert not looks_like_sam_prompt("中文名称")


def test_chat_completions_url():
    assert _chat_completions_url("https://x/v1") == "https://x/v1/chat/completions"
    assert _chat_completions_url("https://x/v1/") == "https://x/v1/chat/completions"
    assert _chat_completions_url("https://x/v1/chat/completions") == "https://x/v1/chat/completions"


def test_suggest_names_posts_vision_chat_not_images_edits(tmp_path: Path):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "preview.jpg"
    import cv2

    cv2.imwrite(str(img_path), img)
    settings = {
        "vision_base_url": "https://vision.example/v1",
        "vision_model": "chat-image-2",
        "vision_api_key": "sk-test",
    }
    returned = ["alpha part", "beta-part", "gamma widget"]
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "candidates": [
                                {"english": returned[0], "chinese": "甲", "confidence": 0.9},
                                {"english": returned[1], "chinese": "乙", "confidence": 0.7},
                                {"english": returned[2], "chinese": "丙", "confidence": 0.6},
                            ]
                        }
                    )
                }
            }
        ]
    }
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = body
    with patch("autolabel.name_suggest.requests.post", return_value=fake) as post:
        cands = suggest_names_from_image(img_path, settings=settings, backends_config={})
    assert [c.english for c in cands] == returned
    args, kwargs = post.call_args
    assert args[0] == "https://vision.example/v1/chat/completions"
    payload = kwargs["json"]
    assert payload["model"] == "gpt-4o"
    assert "/images/edits" not in args[0]
    system = payload["messages"][0]["content"].lower()
    for needle in ("wooden", "disk", "phone", "weld"):
        assert needle not in system
    content = payload["messages"][1]["content"]
    assert any(p.get("type") == "image_url" for p in content if isinstance(p, dict))


def test_suggest_names_posts_gpt4o_when_configured_gpt_image(tmp_path: Path):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "preview.jpg"
    import cv2

    cv2.imwrite(str(img_path), img)
    settings = {
        "vision_base_url": "https://vision.example/v1",
        "vision_model": "gpt-image-2",
        "vision_api_key": "sk-test",
        "vision_chat_model": "qwen-vl-plus",
    }
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"candidates": [{"english": "alpha part", "chinese": "甲", "confidence": 0.9}]}
                    )
                }
            }
        ]
    }
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = body
    with patch("autolabel.name_suggest.requests.post", return_value=fake) as post:
        cands = suggest_names_from_image(img_path, settings=settings, backends_config={})
    assert cands[0].english == "alpha part"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "gpt-4o"
    assert payload["model"] != "qwen-vl-plus"
    assert payload["model"] != "gpt-image-2"


def test_suggest_names_posts_all_sampled_images(tmp_path: Path):
    import cv2

    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))
        paths.append(p)
    settings = {
        "vision_base_url": "https://vision.example/v1",
        "vision_model": "gpt-4o",
        "vision_api_key": "sk-test",
    }
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"candidates": [{"english": "shared object", "chinese": "主体", "confidence": 0.9}]}
                    )
                }
            }
        ]
    }
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = body
    with patch("autolabel.name_suggest.requests.post", return_value=fake) as post:
        cands = suggest_names_from_image(paths, settings=settings, backends_config={})
    assert cands[0].english == "shared object"
    content = post.call_args.kwargs["json"]["messages"][1]["content"]
    n_img = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
    assert n_img == 5
    assert "5 photo" in content[0]["text"]


def test_upstream_busy_retries_then_explains(tmp_path: Path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "busy.jpg"
    import cv2

    cv2.imwrite(str(img_path), img)
    settings = {
        "vision_base_url": "https://vision.example/v1",
        "vision_model": "chat-image-2",
        "vision_api_key": "sk-test",
    }
    busy = MagicMock()
    busy.status_code = 503
    busy.json.return_value = {"error": {"message": "当前分组上游负载已饱和"}}
    busy.text = ""
    with patch("autolabel.name_suggest.time.sleep"):
        with patch("autolabel.name_suggest.requests.post", return_value=busy) as post:
            with pytest.raises(NameSuggestError) as ei:
                suggest_names_from_image(img_path, settings=settings)
    assert post.call_count == 3
    assert "太忙" in str(ei.value)


def test_http_error_surfaces():
    settings = {
        "vision_base_url": "https://vision.example/v1",
        "vision_model": "chat-image-2",
        "vision_api_key": "sk-test",
    }
    fake = MagicMock()
    fake.status_code = 401
    fake.json.return_value = {"error": {"message": "invalid key"}}
    fake.text = ""
    with patch("autolabel.name_suggest.imread", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
        with patch("autolabel.name_suggest.requests.post", return_value=fake):
            with pytest.raises(NameSuggestError) as ei:
                suggest_names_from_image("x.jpg", settings=settings)
    assert "401" in str(ei.value)


def test_unsupported_chat_explains_gpt_image(tmp_path: Path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "shot.jpg"
    import cv2

    cv2.imwrite(str(img_path), img)
    settings = {
        "vision_base_url": "https://vision.example/v1",
        "vision_model": "gpt-image-2",
        "vision_api_key": "sk-test",
    }
    fake = MagicMock()
    fake.status_code = 400
    fake.json.return_value = {"error": {"message": "the requested operation is unsupported"}}
    fake.text = ""
    with patch("autolabel.name_suggest.requests.post", return_value=fake):
        with pytest.raises(NameSuggestError) as ei:
            suggest_names_from_image(img_path, settings=settings)
    msg = str(ei.value)
    assert "400" in msg
    assert "看图" in msg
    assert "gpt-image-2" in msg
