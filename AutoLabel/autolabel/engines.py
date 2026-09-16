"""User-facing annotation engines (UI / CLI catalog).

SAM 3.1 is the only built-in engine. Extra engines are OpenAI-compatible
image APIs the user adds (base URL / key / model). Old ``sam3`` configs map
onto SAM 3.1. Old ``chat-image`` / ``gpt_image_mask`` names still resolve.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

ENGINE_SAM3 = "sam3"  # retired alias → SAM 3.1
ENGINE_SAM31 = "sam3.1"
ENGINE_CHAT_IMAGE = "chat-image"

USER_ENGINES = (ENGINE_SAM31,)

BACKEND_SAM3 = "sam3_local"  # retired; leftover configs still parse
BACKEND_SAM31 = "sam3_1_local"
BACKEND_CHAT_IMAGE = "gpt_image_mask"

API_ENGINE_TYPE = "openai_image_edits"

_ENGINE_LABELS = {
    ENGINE_SAM31: "SAM3.1",
    ENGINE_CHAT_IMAGE: "chat-image",
}

_ENGINE_EXPLANATIONS = {
    ENGINE_SAM31: "本机 SAM 3.1（facebook/sam3.1）。用 models/sam3.1_multiplex.pt，无需 API。按文本提示做实例分割。",
    ENGINE_CHAT_IMAGE: "通过 Base URL / API Key / 模型调用的云端视觉标注。不经过本机 SAM3.1。",
}

_ALIASES = {
    ENGINE_SAM3: ENGINE_SAM31,
    "sam3_local": ENGINE_SAM31,
    ENGINE_SAM31: ENGINE_SAM31,
    "sam31": ENGINE_SAM31,
    "sam3_1": ENGINE_SAM31,
    "sam3p1": ENGINE_SAM31,
    "sam3_1_local": ENGINE_SAM31,
    ENGINE_CHAT_IMAGE: ENGINE_CHAT_IMAGE,
    "chat_image": ENGINE_CHAT_IMAGE,
    "gpt_image_mask": ENGINE_CHAT_IMAGE,
    "openai_image_edits": ENGINE_CHAT_IMAGE,
    "auto": ENGINE_SAM31,
    "image": ENGINE_SAM31,
    "image_sam3": ENGINE_SAM31,
}

_RESERVED_IDS = {
    ENGINE_SAM3,
    ENGINE_SAM31,
    ENGINE_CHAT_IMAGE,
    BACKEND_SAM3,
    BACKEND_SAM31,
    BACKEND_CHAT_IMAGE,
    "mock",
    "sam31",
    "sam3_1",
    "sam3p1",
    "chat_image",
    "openai_image_edits",
    "auto",
    "image",
    "image_sam3",
}


def _backend_entry(cfg: Optional[Dict[str, Any]], backend_id: str) -> Optional[Dict[str, Any]]:
    want = str(backend_id or "").strip()
    if not want:
        return None
    want_l = want.lower()
    found: Optional[Dict[str, Any]] = None
    chat_entry: Optional[Dict[str, Any]] = None
    for item in (cfg or {}).get("backends") or []:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id") or "")
        if eid == want or eid.lower() == want_l:
            found = item
            break
        if eid == BACKEND_CHAT_IMAGE or eid.lower() == ENGINE_CHAT_IMAGE:
            chat_entry = item
    if found is not None:
        return found
    if want_l in {ENGINE_CHAT_IMAGE, "chat_image", BACKEND_CHAT_IMAGE, "openai_image_edits"}:
        return chat_entry
    return None


def is_api_engine_entry(entry: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(entry, dict):
        return False
    etype = str(entry.get("type") or "").strip()
    eid = str(entry.get("id") or "").strip()
    if etype == API_ENGINE_TYPE:
        return True
    return eid in {BACKEND_CHAT_IMAGE, ENGINE_CHAT_IMAGE}


def api_engine_configured(entry: Optional[Dict[str, Any]]) -> bool:
    if not is_api_engine_entry(entry):
        return False
    opt = entry.get("options") if isinstance(entry, dict) else {}
    if not isinstance(opt, dict):
        opt = {}
    return bool(str(opt.get("base_url") or "").strip())


def api_engine_label(entry: Optional[Dict[str, Any]]) -> str:
    if not isinstance(entry, dict):
        return ENGINE_CHAT_IMAGE
    label = str(entry.get("label") or "").strip()
    if label:
        return label
    opt = entry.get("options") if isinstance(entry.get("options"), dict) else {}
    model = str(opt.get("model") or "").strip()
    if model:
        return model
    eid = str(entry.get("id") or "").strip()
    if eid in {BACKEND_CHAT_IMAGE, ENGINE_CHAT_IMAGE}:
        return ENGINE_CHAT_IMAGE
    return eid or ENGINE_CHAT_IMAGE


def list_api_engine_entries(cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in (cfg or {}).get("backends") or []:
        if not is_api_engine_entry(item):
            continue
        eid = str(item.get("id") or "").strip()
        if not eid:
            continue
        if api_engine_configured(item) or str(item.get("label") or "").strip():
            out.append(item)
    return out


def engine_label(engine: str, backends_config: Optional[Dict[str, Any]] = None) -> str:
    key = normalize_engine(engine, backends_config)
    if key == ENGINE_SAM31:
        return _ENGINE_LABELS[ENGINE_SAM31]
    entry = _backend_entry(backends_config, key)
    if entry is None and key == ENGINE_CHAT_IMAGE:
        entry = _backend_entry(backends_config, BACKEND_CHAT_IMAGE)
    if is_api_engine_entry(entry):
        return api_engine_label(entry)
    return _ENGINE_LABELS.get(key, str(engine))


def engine_explanation(engine: str, backends_config: Optional[Dict[str, Any]] = None) -> str:
    key = normalize_engine(engine, backends_config)
    if key == ENGINE_SAM31:
        return _ENGINE_EXPLANATIONS[ENGINE_SAM31]
    entry = _backend_entry(backends_config, key)
    if entry is None and key == ENGINE_CHAT_IMAGE:
        entry = _backend_entry(backends_config, BACKEND_CHAT_IMAGE)
    if is_api_engine_entry(entry):
        opt = entry.get("options") if isinstance(entry.get("options"), dict) else {}
        model = str(opt.get("model") or "").strip()
        url = str(opt.get("base_url") or "").strip()
        bits = ["通过 API 调用的视觉标注引擎。"]
        if model:
            bits.append(f"模型 {model}。")
        if url:
            bits.append(f"接口 {url}。")
        return "".join(bits)
    return _ENGINE_EXPLANATIONS.get(key, "通过 Base URL / API Key / 模型调用的云端视觉标注。")


def is_user_engine(engine: str, backends_config: Optional[Dict[str, Any]] = None) -> bool:
    key = normalize_engine(engine, backends_config)
    if key == ENGINE_SAM31:
        return True
    if key in {ENGINE_CHAT_IMAGE, BACKEND_CHAT_IMAGE}:
        return True
    return is_api_engine_entry(_backend_entry(backends_config, key))


def is_sam_family(engine: Optional[str], backends_config: Optional[Dict[str, Any]] = None) -> bool:
    return normalize_engine(engine, backends_config) == ENGINE_SAM31


def sam3_version_for_engine(engine: Optional[str] = None) -> str:
    del engine
    return ENGINE_SAM31


def normalize_engine(raw: Optional[str], backends_config: Optional[Dict[str, Any]] = None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ENGINE_SAM31
    key = text.lower()
    entry = _backend_entry(backends_config, text)
    if is_api_engine_entry(entry):
        return str(entry.get("id") or text)
    if key == ENGINE_SAM31:
        return ENGINE_SAM31
    alias = _ALIASES.get(key)
    if alias:
        return alias
    return ENGINE_SAM31


def resolve_annotation_engine(
    raw: Optional[str] = None,
    *,
    intent: Any = None,
    prompt: str = "",
    settings: Optional[Dict[str, Any]] = None,
    backends_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Map a UI/CLI engine name to an internal backend id."""
    del intent, prompt, settings
    engine = normalize_engine(raw, backends_config)
    if engine == ENGINE_SAM31:
        return BACKEND_SAM31
    if engine == ENGINE_CHAT_IMAGE:
        return BACKEND_CHAT_IMAGE
    entry = _backend_entry(backends_config, engine)
    if is_api_engine_entry(entry):
        return str(entry.get("id") or BACKEND_CHAT_IMAGE)
    return BACKEND_SAM31


def list_user_engine_choices(
    backends_config: Optional[Dict[str, Any]] = None,
) -> Iterable[Tuple[str, str]]:
    items: List[Tuple[str, str]] = [(ENGINE_SAM31, _ENGINE_LABELS[ENGINE_SAM31])]
    for entry in list_api_engine_entries(backends_config):
        eid = str(entry.get("id") or "").strip()
        if not eid or eid == BACKEND_SAM31:
            continue
        items.append((eid, api_engine_label(entry)))
    return items


def slug_engine_id(name: str, existing: Optional[Iterable[str]] = None) -> str:
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "").strip()).strip("_").lower()
    if not raw:
        raw = "api_engine"
    if raw in _RESERVED_IDS or raw.startswith("sam3"):
        raw = f"api_{raw}"
    taken = {str(x).strip() for x in (existing or []) if str(x).strip()}
    taken |= _RESERVED_IDS
    candidate = raw
    n = 2
    while candidate in taken:
        candidate = f"{raw}_{n}"
        n += 1
    return candidate
