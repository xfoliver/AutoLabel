"""Unit tests: nonempty label validation; no keyword rewriting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.errors import IntentError  # noqa: E402
from autolabel.intent import looks_like_simple_prompt, parse_intent_from_text, parse_intent_manual  # noqa: E402
from autolabel.label_spec import build_label_spec  # noqa: E402
from autolabel.labels import require_nonempty_label, sanitize_class_label, sanitize_target_spec  # noqa: E402
from autolabel.models import LabelIntent, LabelSpec, TargetSpec  # noqa: E402


def test_require_nonempty_keeps_short_labels():
    assert require_nonempty_label("手机") == "手机"
    assert require_nonempty_label("  phone  ") == "phone"
    assert require_nonempty_label("王不留行") == "王不留行"


def test_empty_label_raises():
    with pytest.raises(IntentError):
        require_nonempty_label("")
    with pytest.raises(IntentError):
        require_nonempty_label("   ")
    with pytest.raises(IntentError):
        sanitize_class_label("", "只识别图片中的手机")


def test_long_instructional_label_not_rewritten():
    # Local code must NOT guess 「手机」 from keywords / description.
    raw = "把该目录中的手机"
    assert sanitize_class_label(raw, "把该目录中的手机豆瓣我标注出来") == raw
    t = sanitize_target_spec(TargetSpec(label=raw, description=""))
    assert t.label == raw


def test_label_spec_from_intent_keeps_llm_label():
    intent = LabelIntent(
        targets=[TargetSpec(label="手机", description="目录中的手机")]
    )
    spec = LabelSpec.from_intent(intent)
    assert spec.class_names == ["手机"]
    assert spec.targets[0].label == "手机"


def test_label_spec_from_intent_rejects_empty():
    intent = LabelIntent(targets=[TargetSpec(label="  ", description="x")])
    with pytest.raises(IntentError):
        LabelSpec.from_intent(intent)
    with pytest.raises(IntentError):
        LabelSpec.from_intent(LabelIntent(targets=[]))


def test_build_label_spec_and_manual_intent():
    spec = build_label_spec(label="手机", description="试标")
    assert spec.class_names == ["手机"]
    intent = parse_intent_manual(label="手机", description="")
    assert intent.targets[0].label == "手机"
    with pytest.raises(IntentError):
        parse_intent_manual(label="", description="anything")


class _UnavailableClient:
    available = False


class _BadJsonClient:
    available = True

    def chat(self, messages):
        return "not-json"


class _OkClient:
    available = True

    def chat(self, messages):
        return (
            '{"action":"auto_label","annotation_task":"实例分割",'
            '"object_granularity":"instance",'
            '"targets":[{"label":"手机","description":"目录里的手机","include":[],"exclude":[]}]}'
        )


class _LongLabelClient:
    available = True

    def chat(self, messages):
        return (
            '{"action":"auto_label","annotation_task":"实例分割",'
            '"object_granularity":"instance",'
            '"targets":[{"label":"把该目录中的手机","description":"试标","include":[],"exclude":[]}]}'
        )


def test_parse_intent_requires_llm():
    with pytest.raises(IntentError):
        parse_intent_from_text("标注手机", llm_client=_UnavailableClient(), intent_mode="llm")


def test_parse_intent_auto_falls_back_to_simple_when_llm_down():
    intent = parse_intent_from_text("phone", llm_client=_UnavailableClient(), intent_mode="auto")
    assert intent.targets[0].label == "phone"
    assert intent.targets[0].model_prompt == "phone"


def test_looks_like_simple_prompt():
    assert looks_like_simple_prompt("phone")
    assert looks_like_simple_prompt("red car")
    assert looks_like_simple_prompt("手机")
    assert not looks_like_simple_prompt("标注图片中的手机")
    assert not looks_like_simple_prompt("把这个目录里的汽车全部标出来")


def test_parse_intent_simple_mode_skips_llm():
    intent = parse_intent_from_text("car", llm_client=_UnavailableClient(), intent_mode="simple")
    assert intent.targets[0].model_prompt == "car"


def test_parse_intent_bad_json_raises():
    with pytest.raises(IntentError):
        parse_intent_from_text("标注手机", llm_client=_BadJsonClient(), intent_mode="llm")


def test_parse_intent_llm_success():
    intent = parse_intent_from_text("把该目录中的手机标出来", llm_client=_OkClient())
    assert intent.targets[0].label == "手机"


def test_parse_intent_accepts_long_label_from_llm_without_rewrite():
    intent = parse_intent_from_text("把该目录中的手机标出来", llm_client=_LongLabelClient(), intent_mode="llm")
    assert intent.targets[0].label == "把该目录中的手机"


def test_manual_intent_keeps_model_prompt():
    intent = parse_intent_manual(label="手机", model_prompt="phone")
    assert intent.targets[0].label == "手机"
    assert intent.targets[0].model_prompt == "phone"
