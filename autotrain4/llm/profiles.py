"""LLM Profile 存储、迁移与 active 解析。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .presets import (
    PROVIDER_CUSTOM,
    PROVIDER_DEEPSEEK,
    get_preset,
    normalize_provider,
)
from .types import LLMProfile, new_profile_id
from .urlutil import normalize_base_url, normalize_protocol

KEY_PROFILES = "llm_profiles"
KEY_ACTIVE_ID = "llm_active_id"
_LEGACY_KEYS = ("llm_base_url", "llm_api_key", "llm_model", "llm_protocol")


def _guess_provider(base_url: str, model: str, protocol: str) -> str:
    u = (base_url or "").lower()
    m = (model or "").lower()
    if "deepseek" in u or m.startswith("deepseek"):
        return PROVIDER_DEEPSEEK
    if "openai.com" in u:
        return "openai"
    if "anthropic.com" in u or m.startswith("claude"):
        return "anthropic"
    if "x.ai" in u or m.startswith("grok"):
        return "xai"
    if "moonshot" in u or m.startswith("moonshot") or "kimi" in m:
        return "kimi"
    if normalize_protocol(protocol) == "anthropic":
        return "anthropic"
    return PROVIDER_CUSTOM


def _legacy_flat_to_profile(settings: Dict[str, Any]) -> Optional[LLMProfile]:
    base = str(settings.get("llm_base_url") or "").strip()
    key = str(settings.get("llm_api_key") or "").strip()
    model = str(settings.get("llm_model") or "").strip()
    proto = str(settings.get("llm_protocol") or "auto").strip() or "auto"
    if not (base or key or model):
        return None
    provider = _guess_provider(base, model, proto)
    preset = get_preset(provider)
    name = str(preset.get("name") or "默认 LLM")
    if model:
        name = f"{name} ({model})"
    return LLMProfile(
        id=new_profile_id(),
        name=name,
        provider=provider,
        base_url=base,
        api_key=key,
        model=model or str(preset.get("default_model") or ""),
        protocol=normalize_protocol(proto),
    )


def _parse_profiles(raw: Any) -> List[LLMProfile]:
    if not isinstance(raw, list):
        return []
    out: List[LLMProfile] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(LLMProfile.from_dict(item))
    return out


def _find_active(profiles: List[LLMProfile], active_id: str) -> Optional[LLMProfile]:
    if not profiles:
        return None
    for p in profiles:
        if p.id == active_id and p.enabled:
            return p
    for p in profiles:
        if p.enabled:
            return p
    return profiles[0]


def _mirror_profile_to_flat(settings: Dict[str, Any], profile: Optional[LLMProfile]) -> None:
    if profile is None:
        for k in _LEGACY_KEYS:
            settings.setdefault(k, "" if k != "llm_protocol" else "auto")
        return
    settings["llm_base_url"] = profile.base_url
    settings["llm_api_key"] = profile.api_key
    settings["llm_model"] = profile.model
    settings["llm_protocol"] = resolve_profile_protocol(profile)


def migrate_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化 settings：
    - 无 llm_profiles 时从旧扁平字段生成一条
    - 校正 active_id
    - 同步扁平镜像字段
    """
    out = dict(settings or {})
    profiles = _parse_profiles(out.get(KEY_PROFILES))
    if not profiles:
        legacy = _legacy_flat_to_profile(out)
        if legacy:
            profiles = [legacy]

    active_id = str(out.get(KEY_ACTIVE_ID) or "").strip()
    active = _find_active(profiles, active_id)
    if active is not None:
        active_id = active.id
    else:
        active_id = ""

    out[KEY_PROFILES] = [p.to_dict() for p in profiles]
    out[KEY_ACTIVE_ID] = active_id
    _mirror_profile_to_flat(out, active)
    return out


def list_profiles(settings: Dict[str, Any]) -> List[LLMProfile]:
    data = migrate_settings(settings)
    return _parse_profiles(data.get(KEY_PROFILES))


def get_active_profile(settings: Dict[str, Any]) -> Optional[LLMProfile]:
    data = migrate_settings(dict(settings or {}))
    profiles = _parse_profiles(data.get(KEY_PROFILES))
    return _find_active(profiles, str(data.get(KEY_ACTIVE_ID) or ""))


def resolve_profile_protocol(profile: LLMProfile) -> str:
    """显式 openai/anthropic 优先；DeepSeek 官方强制 openai；否则 auto。"""
    explicit = normalize_protocol(profile.protocol)
    base = normalize_base_url(profile.base_url)
    if base.rstrip("/").endswith("api.deepseek.com"):
        return "openai"
    if explicit in {"openai", "anthropic"}:
        return explicit
    # auto：运行时探测；非 custom 也可在 UI 预填 provider 默认协议
    return "auto"


def upsert_profile(
    settings: Dict[str, Any],
    profile: LLMProfile,
    *,
    make_active: bool = False,
) -> Dict[str, Any]:
    data = migrate_settings(settings)
    profiles = _parse_profiles(data.get(KEY_PROFILES))
    replaced = False
    for i, p in enumerate(profiles):
        if p.id == profile.id:
            profiles[i] = profile
            replaced = True
            break
    if not replaced:
        profiles.append(profile)
    data[KEY_PROFILES] = [p.to_dict() for p in profiles]
    if make_active or not data.get(KEY_ACTIVE_ID):
        data[KEY_ACTIVE_ID] = profile.id
    active = _find_active(profiles, str(data.get(KEY_ACTIVE_ID) or ""))
    if active is not None:
        data[KEY_ACTIVE_ID] = active.id
    _mirror_profile_to_flat(data, active)
    return data


def delete_profile(settings: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    data = migrate_settings(settings)
    profiles = [p for p in _parse_profiles(data.get(KEY_PROFILES)) if p.id != profile_id]
    data[KEY_PROFILES] = [p.to_dict() for p in profiles]
    if str(data.get(KEY_ACTIVE_ID) or "") == profile_id:
        data[KEY_ACTIVE_ID] = profiles[0].id if profiles else ""
    active = _find_active(profiles, str(data.get(KEY_ACTIVE_ID) or ""))
    if active is not None:
        data[KEY_ACTIVE_ID] = active.id
    else:
        data[KEY_ACTIVE_ID] = ""
    _mirror_profile_to_flat(data, active)
    return data


def set_active_profile(settings: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    data = migrate_settings(settings)
    profiles = _parse_profiles(data.get(KEY_PROFILES))
    ids = {p.id for p in profiles}
    if profile_id not in ids:
        raise ValueError(f"profile 不存在: {profile_id}")
    data[KEY_ACTIVE_ID] = profile_id
    active = _find_active(profiles, profile_id)
    _mirror_profile_to_flat(data, active)
    return data


def profiles_to_settings_fragment(settings: Dict[str, Any]) -> Dict[str, Any]:
    data = migrate_settings(settings)
    keys = [KEY_PROFILES, KEY_ACTIVE_ID, *_LEGACY_KEYS]
    return {k: copy.deepcopy(data.get(k)) for k in keys if k in data}


def apply_preset_defaults(profile: LLMProfile, *, overwrite_empty: bool = True) -> LLMProfile:
    preset = get_preset(profile.provider)
    p = LLMProfile.from_dict(profile.to_dict())
    p.provider = normalize_provider(p.provider)
    if (overwrite_empty or not p.base_url) and not p.base_url:
        p.base_url = str(preset.get("default_base_url") or "")
    if (overwrite_empty or not p.model) and not p.model:
        p.model = str(preset.get("default_model") or "")
    if not p.protocol:
        p.protocol = str(preset.get("default_protocol") or "auto")
    return p
