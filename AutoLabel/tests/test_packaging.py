"""Packaging helpers (no GPU / no conda copy)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packaging"))

from copy_sam3_src import copy_sam3_src  # noqa: E402


def test_copy_sam3_src_keeps_train_eval_agent(tmp_path: Path):
    src = tmp_path / "sam3_repo"
    pkg = src / "sam3"
    (pkg / "assets").mkdir(parents=True)
    (pkg / "train").mkdir()
    (pkg / "eval").mkdir()
    (pkg / "agent").mkdir()
    (pkg / "model_builder.py").write_text("ok\n", encoding="utf-8")
    (pkg / "assets" / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"gz")
    (pkg / "train" / "trainer.py").write_text("ok\n", encoding="utf-8")
    (pkg / "eval" / "coco_eval.py").write_text("ok\n", encoding="utf-8")
    (pkg / "agent" / "client_sam3.py").write_text("ok\n", encoding="utf-8")
    (src / "LICENSE").write_text("MIT\n", encoding="utf-8")
    dest = tmp_path / "sam3_src"
    copy_sam3_src(src, dest)
    assert (dest / "sam3" / "model_builder.py").is_file()
    assert (dest / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz").is_file()
    assert (dest / "sam3" / "train" / "trainer.py").is_file()
    assert (dest / "sam3" / "eval" / "coco_eval.py").is_file()
    assert (dest / "sam3" / "agent" / "client_sam3.py").is_file()
    assert (dest / "LICENSE").is_file()


def test_assemble_refuses_secret_example(tmp_path: Path):
    from assemble_portable import SECRET_RE, _assert_examples_clean, _scan_secrets

    assert SECRET_RE.search('{"api_key": "sk-kbAe93oYOuzIdAgTrpWUzN4KwIco"}')
    _assert_examples_clean()
    example = REPO / "AutoLabel" / "config" / "backends.example.json"
    data = json.loads(example.read_text(encoding="utf-8"))
    ids = [b.get("id") for b in data["backends"]]
    assert "sam3_local" not in ids
    assert data.get("annotation_engine") == "sam3.1"
    assert "runtime" in str(data["backends"][0]["options"]["python_path"])
    sam31 = next(b for b in data["backends"] if b.get("id") == "sam3_1_local")
    assert sam31["type"] == "sam3_local"
    assert sam31["options"].get("sam3_version") == "sam3.1"
    vision = next(b for b in data["backends"] if b.get("id") == "gpt_image_mask")
    assert vision["options"].get("base_url") == ""
    assert vision["options"].get("model") == ""
    assert not vision["options"].get("api_key")
    settings = json.loads((REPO / "AutoLabel" / "config" / "settings.example.json").read_text(encoding="utf-8"))
    for field in (
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "vision_base_url",
        "vision_api_key",
        "vision_model",
    ):
        assert settings.get(field) == ""


def test_scan_secrets_rejects_dev_host(tmp_path: Path):
    from assemble_portable import _scan_secrets

    leak = tmp_path / "config"
    leak.mkdir()
    (leak / "note.md").write_text("proxy http://192.168.1.3:3000/v1\n", encoding="utf-8")
    try:
        _scan_secrets(tmp_path)
        raise AssertionError("expected developer host leak to fail the scan")
    except RuntimeError as exc:
        assert "192.168.1.3" in str(exc)


def test_extract_needles_skips_api_key_env():
    from assemble_portable import _extract_secret_needles

    found = _extract_secret_needles(
        {
            "api_key_env": "AUTOLABEL_VISION_KEY",
            "base_url": "https://vectorengine.ai/v1",
            "api_key": "sk-live-secret-value-123",
        }
    )
    assert "AUTOLABEL_VISION_KEY" not in found
    assert "https://vectorengine.ai/v1" in found
    assert "sk-live-secret-value-123" in found


def test_copy_model_packs_only_sam31(tmp_path: Path):
    from copy_model import copy_model

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    ckpt = src_dir / "sam3.1_multiplex.pt"
    ckpt.write_bytes(b"sam31-weights")
    (src_dir / "config.json").write_text("{}", encoding="utf-8")
    leftover_src = tmp_path / "keep" / "sam3.pt"
    leftover_src.parent.mkdir()
    leftover_src.write_bytes(b"do-not-delete")
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "sam3.pt").write_bytes(b"old-portable")
    (dest / "config.json").write_text("old", encoding="utf-8")
    out = copy_model(dest, ckpt)
    assert out == dest / "sam3.1_multiplex.pt"
    assert out.is_file()
    assert not (dest / "sam3.pt").exists()
    assert leftover_src.is_file()
    assert leftover_src.read_bytes() == b"do-not-delete"
