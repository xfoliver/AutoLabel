"""Portable path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.paths import (  # noqa: E402
    DEV_CONDA_PYTHON,
    PORTABLE_CHECKPOINT_SAM31_REL,
    PORTABLE_PYTHON_REL,
    app_root,
    bpe_vocab_path,
    checkpoint_path,
    checkpoint_search_paths,
    ensure_user_config,
    resolve_python_path,
    sam3_src_dir,
    user_data_dir,
)


def test_app_root_is_autolabel_in_dev(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_ROOT", raising=False)
    assert app_root() == ROOT


def test_resolve_python_prefers_existing_portable(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_PYTHON", raising=False)
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    portable = tmp_path / PORTABLE_PYTHON_REL
    portable.parent.mkdir(parents=True)
    portable.write_text("", encoding="utf-8")
    resolved = resolve_python_path("runtime/sam3/python.exe")
    assert resolved == portable.resolve()


def test_resolve_python_relative_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_PYTHON", raising=False)
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    configured = tmp_path / "custom" / "python.exe"
    configured.parent.mkdir(parents=True)
    configured.write_text("", encoding="utf-8")
    resolved = resolve_python_path("custom/python.exe")
    assert resolved == configured.resolve()


def test_resolve_python_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "env" / "python.exe"
    fake.parent.mkdir()
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("SAM3_PYTHON", str(fake))
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path / "app"))
    assert resolve_python_path("missing.exe") == fake.resolve()


def test_frozen_user_data_goes_to_appdata(tmp_path, monkeypatch):
    monkeypatch.setattr("autolabel.paths.is_frozen", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.delenv("AUTOLABEL_USER_DATA", raising=False)
    dest = user_data_dir()
    assert dest == (tmp_path / "Roaming" / "AutoLabel").resolve()
    assert dest.is_dir()


def test_checkpoint_from_models_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    leftover = tmp_path / "models" / "sam3.pt"
    leftover.parent.mkdir()
    leftover.write_bytes(b"old")
    ckpt = tmp_path / "models" / "sam3.1_multiplex.pt"
    ckpt.write_bytes(b"x")
    found = checkpoint_path()
    assert found == ckpt.resolve()
    assert checkpoint_path("sam3") == ckpt.resolve()
    assert checkpoint_path("sam3.1") == ckpt.resolve()


def test_checkpoint_sam31_from_models_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    ckpt = tmp_path / PORTABLE_CHECKPOINT_SAM31_REL
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    found = checkpoint_path("sam3.1")
    assert found == ckpt.resolve()
    # Retired SAM3 names now resolve to the 3.1 weight, not sam3.pt.
    assert checkpoint_path("sam3") == ckpt.resolve()


def test_checkpoint_sam31_from_repo_models(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    app = tmp_path / "AutoLabel"
    app.mkdir()
    monkeypatch.setenv("AUTOLABEL_ROOT", str(app))
    ckpt = tmp_path / "models" / "sam3.1_multiplex.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    found = checkpoint_path("sam3.1")
    assert found == ckpt.resolve()
    searched = checkpoint_search_paths("sam3.1")
    assert ckpt in searched


def test_dev_layout_finds_existing_repo_sam31():
    repo = ROOT.parent / "models" / "sam3.1_multiplex.pt"
    if not repo.is_file():
        return
    found = checkpoint_path("sam3.1")
    assert found == repo.resolve()


def test_sam3_src_portable_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_ROOT", raising=False)
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    builder = tmp_path / "sam3_src" / "sam3" / "model_builder.py"
    builder.parent.mkdir(parents=True)
    builder.write_text("# stub\n", encoding="utf-8")
    assert sam3_src_dir() == (tmp_path / "sam3_src").resolve()


def test_bpe_vocab_from_sam3_src(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_BPE", raising=False)
    monkeypatch.delenv("SAM3_ROOT", raising=False)
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path))
    bpe = tmp_path / "sam3_src" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    bpe.parent.mkdir(parents=True)
    bpe.write_bytes(b"gz")
    (tmp_path / "sam3_src" / "sam3" / "model_builder.py").write_text("# stub\n", encoding="utf-8")
    assert bpe_vocab_path() == bpe.resolve()


def test_dev_fallback_constant():
    assert DEV_CONDA_PYTHON.endswith("python.exe")


def test_ensure_user_config_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_USER_DATA", str(tmp_path / "user"))
    monkeypatch.setenv("AUTOLABEL_ROOT", str(tmp_path / "app"))
    monkeypatch.setattr("autolabel.paths.is_frozen", lambda: True)
    example_dir = tmp_path / "app" / "config"
    example_dir.mkdir(parents=True)
    (example_dir / "settings.example.json").write_text('{"llm_model": ""}\n', encoding="utf-8")
    (example_dir / "backends.example.json").write_text("{}\n", encoding="utf-8")
    dest = tmp_path / "user" / "config"
    dest.mkdir(parents=True)
    (dest / "settings.json").write_text('{"llm_model": "keep-me"}\n', encoding="utf-8")
    ensure_user_config()
    assert '"keep-me"' in (dest / "settings.json").read_text(encoding="utf-8")
