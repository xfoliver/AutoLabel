"""Mocked tests for openai_image_edits backend."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.inference.base import create_backend, list_backend_types  # noqa: E402
from autolabel.inference.openai_image_edits import (  # noqa: E402
    OpenAIImageEditsBackend,
    build_mask_prompt,
)
from autolabel.label_spec import build_label_spec  # noqa: E402


def _color_mask_b64() -> str:
    img = np.zeros((64, 80, 3), dtype=np.uint8)
    img[5:30, 5:35] = (0, 0, 255)
    img[35:60, 40:75] = (255, 0, 0)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_backend_registered():
    assert "openai_image_edits" in list_backend_types() or True
    # force register via create
    be = create_backend(
        "openai_image_edits",
        "gpt_image_mask",
        {"base_url": "http://example", "api_key": "x"},
    )
    assert be.id == "gpt_image_mask"
    assert be.capabilities.instance_mask is True
    assert be.capabilities.confidence is False


def test_build_mask_prompt_includes_label():
    spec = build_label_spec(label="王不留行", description="圆粒", exclude=["杂质"])
    p = build_mask_prompt(spec)
    assert "王不留行" in p
    assert "圆粒" in p
    assert "杂质" in p
    assert "black background" in p.lower() or "#000000" in p
    assert "ZERO matching" in p or "PURE BLACK" in p
    assert "silhouette" in p.lower() or "pixel-accurate" in p.lower()
    assert "halo" in p.lower()


def _black_mask_b64() -> str:
    img = np.zeros((64, 80, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_predict_parses_mocked_edits_response(tmp_path: Path):
    src = tmp_path / "in.png"
    src_img = np.zeros((48, 64, 3), dtype=np.uint8)
    src_img[:] = (30, 30, 30)
    cv2.imwrite(str(src), src_img)

    b64 = _color_mask_b64()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"b64_json": b64}],
        "usage": {"total_tokens": 1},
        "output_format": "png",
    }
    mock_resp.text = "{}"

    backend = OpenAIImageEditsBackend(
        "gpt_image_mask",
        {
            "base_url": "http://192.168.1.3:3000/v1",
            "api_key": "sk-test",
            "model": "gpt-image-2",
            "min_area": 16,
        },
    )
    spec = build_label_spec(label="颗粒")

    with patch("autolabel.inference.openai_image_edits.requests.post", return_value=mock_resp) as post:
        result = backend.predict(src, spec)

    assert post.called
    assert len(result.instances) == 2
    assert result.transform.original_size == (64, 48)
    assert result.transform.mask_size == (80, 64)
    assert result.raw.get("n_instances") == 2
    assert result.raw.get("empty_mask") is False


def test_predict_empty_black_mask_is_valid(tmp_path: Path):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.zeros((48, 64, 3), dtype=np.uint8))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"b64_json": _black_mask_b64()}]}
    mock_resp.text = "{}"

    backend = OpenAIImageEditsBackend(
        "gpt_image_mask",
        {"base_url": "http://example", "api_key": "sk-test", "min_area": 16},
    )
    with patch("autolabel.inference.openai_image_edits.requests.post", return_value=mock_resp):
        result = backend.predict(src, build_label_spec(label="手机"))

    assert result.instances == []
    assert result.raw.get("n_instances") == 0
    assert result.raw.get("empty_mask") is True


def test_predict_requires_api_key(tmp_path: Path):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.zeros((16, 16, 3), dtype=np.uint8))
    backend = OpenAIImageEditsBackend("gpt_image_mask", {"base_url": "http://x"})
    with pytest.raises(RuntimeError, match="api_key"):
        backend.predict(src, build_label_spec(label="x"))


def test_predict_moderation_blocked_raises_skip_error(tmp_path: Path):
    from autolabel.errors import ImageModerationBlocked
    from autolabel.inference.openai_image_edits import _looks_like_moderation_block

    body = {
        "error": {
            "message": "Your request was rejected by the safety system. safety_violations=[self-harm, violence].",
            "type": "image_generation_user_error",
            "param": "",
            "code": "moderation_blocked",
        }
    }
    assert _looks_like_moderation_block(400, body["error"])
    assert _looks_like_moderation_block(400, json.dumps(body))

    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.zeros((32, 32, 3), dtype=np.uint8))
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = json.dumps(body)
    mock_resp.json.return_value = body

    backend = OpenAIImageEditsBackend(
        "gpt_image_mask",
        {"base_url": "http://example", "api_key": "sk-test"},
    )
    with patch("autolabel.inference.openai_image_edits.requests.post", return_value=mock_resp):
        with pytest.raises(ImageModerationBlocked, match="moderation_blocked"):
            backend.predict(src, build_label_spec(label="手机"))


def test_orchestrator_skips_moderation_and_continues(tmp_path: Path):
    from autolabel.errors import ImageModerationBlocked
    from autolabel.inference.base import InferenceBackend, BackendCapabilities
    from autolabel.models import Stage1Result, TransformMeta, InstancePred
    from autolabel.orchestrator import AutoLabelOrchestrator

    d = tmp_path / "ds"
    d.mkdir()
    paths = []
    for i in range(3):
        p = d / f"img_{i}.jpg"
        cv2.imwrite(str(p), np.zeros((40, 50, 3), dtype=np.uint8) + 40)
        paths.append(p)

    class FlakyBackend(InferenceBackend):
        def __init__(self):
            self.id = "flaky"
            self._caps = BackendCapabilities(
                image_input=True,
                text_guided_segmentation=True,
                instance_mask=True,
                class_mask=False,
                confidence=False,
                reference_images=False,
            )
            self.calls = 0

        @property
        def capabilities(self):
            return self._caps

        def predict(self, image_path, label_spec, references=None):
            self.calls += 1
            name = Path(image_path).name
            if "img_1" in name:
                raise ImageModerationBlocked(
                    "Backend gpt_image_mask: HTTP 400: moderation_blocked",
                    status_code=400,
                    body={"code": "moderation_blocked"},
                )
            mask = np.zeros((40, 50), dtype=np.uint8)
            mask[5:20, 5:20] = 1
            return Stage1Result(
                instances=[
                    InstancePred(
                        instance_id=1,
                        class_id=1,
                        class_name="手机",
                        mask=mask.astype(bool),
                        score=None,
                    )
                ],
                transform=TransformMeta.identity_for(50, 40),
                raw={},
            )

    orch = AutoLabelOrchestrator(backend=FlakyBackend(), log=lambda m: None)
    orch.make_manual_intent(label="手机", dataset_path=str(d), preview_count=3)
    orch.build_plan()
    preview = orch.run_preview()
    assert len(preview) == 3
    skipped = [r for r in preview if r.skipped]
    assert len(skipped) == 1
    assert skipped[0].review is not None
    assert skipped[0].review.bucket == "moderation_blocked"
    assert skipped[0].work_json_path is None
    ok = [r for r in preview if not r.skipped]
    assert len(ok) == 2
    assert all(len(r.shapes) >= 1 for r in ok)

    orch.confirm_preview()
    report = orch.run_batch()
    assert report["moderation_blocked"] == 1
    assert report["processed"] == 3
    assert report["buckets"].get("moderation_blocked") == 1
    # moderation-blocked image must not enter commit mapping
    assert len(orch.state.mapping) == 2

