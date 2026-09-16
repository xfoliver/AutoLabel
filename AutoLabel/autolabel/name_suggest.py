"""Suggest SAM3 text-prompt names from sampled dataset images.

Vision-only: OpenAI-compatible chat/completions with one or more images.
This is not segmentation — it never calls SAM3 and never produces a mask.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import requests

from autolabel.cv_io import imread
from autolabel.errors import NameSuggestError
from autolabel.inference.openai_image_edits import request_proxies

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 5
SAMPLE_COUNT = 5
UPLOAD_MAX_EDGE = 1024
JPEG_QUALITY = 85

NAME_SUGGEST_SYSTEM = """You look at one or more photos from the same dataset and identify the main object they share for a text-guided segmenter.

Judge only from these images. Do not reuse names from examples, memory, or a fixed vocabulary.
If several photos are given, find the object they have in common. Ignore one-off backgrounds.
Write whatever wording you think will help the segmenter find that object. A word, a phrase, or a sentence are all fine. There is no length quota.

Output JSON only, no markdown:

{"candidates":[{"english":"<what you see>","chinese":"<中文>","confidence":0.0}]}

- Return 3 to 5 candidates, most likely first, each a different wording of the same main object.
- english: the prompt that will be sent to the segmenter.
- chinese: the same idea for display.
- confidence: a number from 0 to 1.
- Do not invent masks, boxes, coordinates, or class ids.
"""


@dataclass
class NameCandidate:
    english: str
    chinese: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "english": self.english,
            "chinese": self.chinese,
            "confidence": self.confidence,
        }


def _backend_entry(cfg: Optional[Dict[str, Any]], backend_id: str) -> Dict[str, Any]:
    for b in (cfg or {}).get("backends") or []:
        if isinstance(b, dict) and b.get("id") == backend_id:
            return b
    return {}


# /images/edits models reject chat/completions with HTTP 400 unsupported.
_IMAGE_GEN_MODEL_MARKERS = (
    "gpt-image",
    "chat-image",
    "dall-e",
    "dalle",
    "imagen",
    "flux",
    "stable-diffusion",
    "stable_diffusion",
    "midjourney",
    "image-edit",
    "image_edit",
    "qwen-image",
    "seedream",
    "ideogram",
    "recraft",
    "lumina-image",
    "pixverse-image",
)


DEFAULT_GPT_VISION_MODEL = "gpt-4o"


def looks_like_image_generation_model(model: str) -> bool:
    """True for image-edit / image-gen models that cannot do chat vision."""
    text = str(model or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _IMAGE_GEN_MODEL_MARKERS)


def chat_model_for_name_suggest(model: str) -> str:
    """gpt-image-* generates masks; name-suggest uses GPT-4o on the same API."""
    text = str(model or "").strip()
    if not text or looks_like_image_generation_model(text):
        return DEFAULT_GPT_VISION_MODEL
    return text


def resolve_vision_credentials(
    settings: Optional[Dict[str, Any]] = None,
    backends_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Pick chat/completions credentials for name-suggest.

    Settings win over backends.json gpt_image_mask options. This feature talks
    to chat/completions (image in, JSON names out) — not /images/edits.
    The UI keeps a single GPT model (usually gpt-image-2). Image-edit models
    are remapped to gpt-4o on the same Base URL / API Key.
    """
    settings = settings or {}
    from autolabel.engines import is_api_engine_entry, list_api_engine_entries, normalize_engine

    gpt = (_backend_entry(backends_config, "gpt_image_mask").get("options") or {})
    selected = None
    engine_id = normalize_engine(
        (backends_config or {}).get("annotation_engine"),
        backends_config,
    )
    for item in list_api_engine_entries(backends_config):
        if str(item.get("id") or "") == engine_id:
            selected = item
            break
    if selected is None:
        api_list = list_api_engine_entries(backends_config)
        selected = api_list[0] if api_list else None
    api_opt = (selected.get("options") if isinstance(selected, dict) else None) or {}
    if not api_opt and is_api_engine_entry(_backend_entry(backends_config, "gpt_image_mask")):
        api_opt = gpt
    selected_id = str(selected.get("id") or "") if isinstance(selected, dict) else ""
    use_legacy_overlay = engine_id == "sam3.1" or selected_id in {"", "gpt_image_mask", "chat-image"}
    if use_legacy_overlay:
        base_url = str(
            settings.get("vision_base_url") or api_opt.get("base_url") or gpt.get("base_url") or ""
        ).strip().rstrip("/")
        configured = str(
            settings.get("vision_model") or api_opt.get("model") or gpt.get("model") or ""
        ).strip()
        api_key = str(
            settings.get("vision_api_key") or api_opt.get("api_key") or gpt.get("api_key") or ""
        ).strip()
        http_proxy = str(api_opt.get("http_proxy") or gpt.get("http_proxy") or "").strip()
    else:
        base_url = str(api_opt.get("base_url") or "").strip().rstrip("/")
        configured = str(api_opt.get("model") or "").strip()
        api_key = str(api_opt.get("api_key") or "").strip()
        http_proxy = str(api_opt.get("http_proxy") or "").strip()
    model = chat_model_for_name_suggest(configured)
    source = "gpt_vision" if looks_like_image_generation_model(configured) or not configured else "vision"
    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "http_proxy": http_proxy,
        "source": source,
        "configured_model": configured,
    }


def require_vision_credentials(
    settings: Optional[Dict[str, Any]] = None,
    backends_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    creds = resolve_vision_credentials(settings, backends_config)
    missing = []
    if not creds["base_url"]:
        missing.append("Base URL")
    if not creds["model"]:
        missing.append("Model")
    if not creds["api_key"]:
        missing.append("API Key")
    if missing:
        raise NameSuggestError(
            "请先在设置里填写视觉识别模型凭证（"
            + "、".join(missing)
            + "）。「AI 识别名称」用同一套 API 看图。"
        )
    return creds


def _encode_image_jpeg(image_path: str | Path) -> Tuple[bytes, str]:
    path = Path(image_path)
    img = imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise NameSuggestError(f"无法读取预览图片：{path}")
    h, w = img.shape[:2]
    edge = max(h, w)
    if edge > UPLOAD_MAX_EDGE:
        scale = UPLOAD_MAX_EDGE / float(edge)
        img = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise NameSuggestError("无法编码预览图片")
    return buf.tobytes(), "image/jpeg"


def _data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _strip_markdown_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, count=1, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_value(text: str) -> Any:
    s = _strip_markdown_fence(text)
    if not s:
        return None
    try:
        data = json.loads(s)
        if isinstance(data, (dict, list, str)):
            return data
    except Exception:
        pass
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, s)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(data, (dict, list, str)):
            return data
    return None


_HAS_CJK = re.compile(r"[㐀-鿿]")


def sanitize_english(text: str) -> str:
    """Keep the model's wording. Only collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip())


def looks_like_sam_prompt(text: str) -> bool:
    """True when there is any usable english text from the model."""
    s = sanitize_english(text)
    return bool(s) and not _HAS_CJK.search(s)


def sanitize_chinese(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _first_str(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        if key not in mapping:
            continue
        val = mapping.get(key)
        if val is None or isinstance(val, (dict, list)):
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def _looks_like_image_generation(data: Any) -> bool:
    """True when the model returned an image-edits payload instead of names."""
    if not isinstance(data, dict):
        return False
    if any(k in data for k in ("b64_json", "revised_prompt")):
        return True
    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        return False
    first = rows[0]
    if not isinstance(first, dict):
        return False
    return any(k in first for k in ("b64_json", "url")) and not any(
        k in first for k in ("english", "chinese", "name", "label")
    )


def _candidate_from_text(text: str, confidence: float = 0.0) -> Optional[NameCandidate]:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return None
    latin = sanitize_english(re.sub(_HAS_CJK, " ", raw))
    cjk = sanitize_chinese("".join(_HAS_CJK.findall(raw)))
    if looks_like_sam_prompt(latin) or cjk:
        return NameCandidate(
            english=latin if looks_like_sam_prompt(latin) else "",
            chinese=cjk,
            confidence=confidence,
        )
    for piece in re.split(r"[,;/|、。；]+", raw):
        piece = piece.strip()
        if not piece or piece == raw:
            continue
        cand = _candidate_from_text(piece, confidence)
        if cand is not None:
            return cand
    return None


_SKIP_JSON_KEYS = {
    "id",
    "object",
    "model",
    "role",
    "type",
    "index",
    "created",
    "usage",
    "finish_reason",
    "b64_json",
    "url",
    "revised_prompt",
}


def _candidate_from_mapping(item: Dict[str, Any]) -> Optional[NameCandidate]:
    english_raw = _first_str(
        item,
        (
            "english",
            "en",
            "name_en",
            "label_en",
            "prompt",
            "model_prompt",
            "main_subject",
            "subject",
            "object_name",
            "object",
            "class",
            "label",
            "name",
            "title",
            "text",
            "answer",
            "caption",
            "description",
            "英文",
            "英文名",
        ),
    )
    chinese_raw = _first_str(
        item,
        ("chinese", "zh", "name_zh", "label_zh", "中文", "中文名", "名称", "主体", "目标"),
    )
    if not english_raw and not chinese_raw:
        for key, val in item.items():
            if str(key).lower() in _SKIP_JSON_KEYS:
                continue
            if isinstance(val, str) and val.strip():
                english_raw = val.strip()
                break
    if english_raw and _HAS_CJK.search(english_raw) and not chinese_raw:
        chinese_raw = english_raw
        english_raw = ""
    if chinese_raw and looks_like_sam_prompt(chinese_raw) and not english_raw:
        english_raw = chinese_raw
        chinese_raw = ""
    english = sanitize_english(english_raw)
    chinese = sanitize_chinese(chinese_raw)
    if not english and not chinese:
        return _candidate_from_text(english_raw or chinese_raw)
    return NameCandidate(
        english=english,
        chinese=chinese,
        confidence=_clamp_confidence(item.get("confidence", item.get("score", item.get("置信度", 0.0)))),
    )


def _iter_candidate_items(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        return [data]
    if not isinstance(data, dict):
        return []
    for key in (
        "candidates",
        "names",
        "suggestions",
        "items",
        "results",
        "objects",
        "labels",
        "output",
        "result",
        "data",
        "main_subject",
        "subject",
        "候选",
        "名称",
        "结果",
        "主体",
    ):
        val = data.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return [val]
        if isinstance(val, str) and val.strip():
            return [val]
    single = _candidate_from_mapping(data)
    return [data] if single is not None else []


def _clamp_confidence(value: Any) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    if n > 1.0 and n <= 100.0:
        n = n / 100.0
    return max(0.0, min(1.0, n))


def parse_candidates(payload: Any) -> List[NameCandidate]:
    """Accept whatever JSON/text the Vision model returned; keep short names only."""
    data = payload
    raw_preview = ""
    if isinstance(payload, str):
        raw_preview = payload.strip()[:240]
        data = _extract_json_value(payload)
        if data is None:
            # Plain text: one name per line, or a single phrase.
            lines = [ln.strip(" -*\t") for ln in payload.splitlines() if ln.strip()]
            data = lines if len(lines) > 1 else payload.strip()
    elif isinstance(payload, (dict, list)):
        try:
            raw_preview = json.dumps(payload, ensure_ascii=False)[:240]
        except Exception:
            raw_preview = str(payload)[:240]

    if _looks_like_image_generation(data):
        raise NameSuggestError(
            "当前模型返回的是图像生成结果，不是识别名称。"
            "请在设置里把 Model 换成支持看图对话的 Vision 模型（同一套 Base URL / API Key）。"
        )

    seen = set()
    out: List[NameCandidate] = []
    for item in _iter_candidate_items(data):
        cand = None
        if isinstance(item, dict):
            cand = _candidate_from_mapping(item)
        elif isinstance(item, str):
            cand = _candidate_from_text(item)
        if cand is None or (not cand.english and not cand.chinese):
            continue
        key = (cand.english or cand.chinese).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= MAX_CANDIDATES:
            break
    if not out:
        extra = f"\n模型原文：{raw_preview}" if raw_preview else ""
        raise NameSuggestError("Vision 没有返回可用的目标名称，请重试或手动填写" + extra)
    return out


def _chat_completions_url(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return root + "/chat/completions"
    return root + "/v1/chat/completions"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in {"text", "output_text"} or "text" in p:
                    parts.append(str(p.get("text") or ""))
                elif p.get("type") == "image_url":
                    continue
                else:
                    parts.append(_content_to_text(p))
        return "\n".join(x for x in parts if x)
    return str(content)


def _error_detail(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)
        return str(body)
    return str(body or "")


def _looks_like_upstream_busy(status_code: int, detail: Any) -> bool:
    text = str(detail or "")
    if status_code in {429, 502, 503}:
        return True
    markers = ("负载已饱和", "上游负载", "rate limit", "too many requests", "overloaded", "capacity")
    return any(m in text.lower() if m.isascii() else m in text for m in markers)


def _looks_like_unsupported_operation(status_code: int, detail: Any) -> bool:
    if status_code not in {400, 404, 405, 422}:
        return False
    text = str(detail or "").lower()
    markers = (
        "unsupported",
        "not supported",
        "does not support",
        "operation is not",
        "unknown operation",
        "invalid model",
        "model_not_found",
        "not a chat",
        "only supports",
        "image generation",
        "images/edits",
    )
    return any(m in text for m in markers)


def _unsupported_chat_error(model: str, status_code: int, detail: str) -> NameSuggestError:
    extra = f"\n网关原文：{detail}" if detail else ""
    return NameSuggestError(
        f"Vision HTTP {status_code}：当前模型「{model}」不支持看图对话。"
        "「AI 识别名称」走 GPT 看图（chat/completions）。设置里填 gpt-image-2 时会自动改用 gpt-4o。"
        + extra
    )


def _extract_message_text(body: Any) -> str:
    if not isinstance(body, dict):
        return str(body or "")
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = ch0.get("message") if isinstance(ch0.get("message"), dict) else {}
        for key in ("content", "reasoning_content", "parsed"):
            text = _content_to_text(msg.get(key))
            if text.strip():
                return text
        for key in ("text", "content"):
            text = _content_to_text(ch0.get(key))
            if text.strip():
                return text
    for key in ("output_text", "content", "text"):
        text = _content_to_text(body.get(key))
        if text.strip():
            return text
    return ""


def _as_image_paths(image_path: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(image_path, (str, Path)):
        paths = [Path(image_path)]
    else:
        paths = [Path(p) for p in image_path]
    if not paths:
        raise NameSuggestError("没有可识别的图片")
    return paths


def suggest_names_from_image(
    image_path: str | Path | Sequence[str | Path],
    *,
    settings: Optional[Dict[str, Any]] = None,
    backends_config: Optional[Dict[str, Any]] = None,
    timeout: float = 90.0,
) -> List[NameCandidate]:
    """Call the Settings Vision model with one or more photos; return 3–5 names."""
    creds = require_vision_credentials(settings, backends_config)
    paths = _as_image_paths(image_path)
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"These {len(paths)} photo(s) are from the same dataset. "
                "Identify the main object they share. "
                "Return 3-5 candidates as JSON. You choose the wording."
            ),
        }
    ]
    for path in paths:
        image_bytes, mime = _encode_image_jpeg(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        )
    url = _chat_completions_url(creds["base_url"])
    payload = {
        "model": creds["model"],
        "temperature": 0.3,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": NAME_SUGGEST_SYSTEM},
            {"role": "user", "content": content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json",
    }
    proxies = request_proxies(creds["base_url"], creds.get("http_proxy") or "")
    last_error = ""
    resp = None
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=float(timeout),
                proxies=proxies,
            )
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise NameSuggestError(f"Vision 请求失败：{e}") from e

        body: Any
        try:
            body = resp.json()
        except Exception:
            body = (resp.text or "")[:400]

        if resp.status_code < 400:
            break
        detail = _error_detail(body)
        last_error = f"Vision HTTP {resp.status_code}: {detail}"
        if attempt < 2 and _looks_like_upstream_busy(resp.status_code, detail):
            logger.info("name-suggest upstream busy, retry %s", attempt + 1)
            time.sleep(1.5 * (attempt + 1))
            continue
        if _looks_like_upstream_busy(resp.status_code, detail):
            raise NameSuggestError(
                "当前 Vision 线路太忙（网关：上游负载已饱和）。"
                "稍等几秒再点一次「AI 识别名称」，或换一个同网关下更空闲的看图模型。"
            )
        if _looks_like_unsupported_operation(resp.status_code, detail) or (
            resp.status_code == 400 and looks_like_image_generation_model(creds["model"])
        ):
            raise _unsupported_chat_error(creds["model"], resp.status_code, detail)
        raise NameSuggestError(last_error)
    else:
        raise NameSuggestError(last_error or "Vision 请求失败")

    if resp is None:
        raise NameSuggestError(last_error or "Vision 请求失败")
    try:
        body = resp.json()
    except Exception:
        body = (resp.text or "")[:400]

    text = _extract_message_text(body)
    if not text.strip():
        preview = ""
        try:
            preview = json.dumps(body, ensure_ascii=False)[:240] if isinstance(body, dict) else str(body)[:240]
        except Exception:
            preview = str(body)[:240]
        raise NameSuggestError("Vision 返回空内容" + (f"\n接口原文：{preview}" if preview else ""))
    logger.info(
        "name-suggest source=%s model=%s images=%s chars=%s",
        creds.get("source") or "vision",
        creds["model"],
        len(paths),
        len(text),
    )
    return parse_candidates(text)


def sample_name_suggest_images(
    dataset_path: str | Path,
    count: int = SAMPLE_COUNT,
) -> List[Path]:
    """Randomly pick up to ``count`` images from a file or folder."""
    from autolabel.dataset import is_image_file, list_images

    path = Path(dataset_path)
    if path.is_file():
        if is_image_file(path):
            return [path]
        raise NameSuggestError(f"不是图片文件：{path.name}")
    if not path.is_dir():
        raise NameSuggestError("请先选择图片文件夹，或填入一张图片的路径")
    images = [Path(p) for p in list_images(path)]
    if not images:
        raise NameSuggestError("当前文件夹没有可预览的图片")
    k = min(max(int(count), 1), len(images))
    if k == len(images):
        return images
    return random.sample(images, k)


def first_preview_image(dataset_path: str | Path) -> Path:
    return sample_name_suggest_images(dataset_path, count=1)[0]
