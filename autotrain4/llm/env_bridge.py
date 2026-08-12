"""环境变量桥接：Profile / 扁平配置 ↔ AUTOTRAIN_LLM_*。"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .presets import PROVIDER_DEEPSEEK, get_preset
from .profiles import get_active_profile, resolve_profile_protocol
from .types import LLMProfile, LLMRuntimeConfig
from .urlutil import normalize_base_url, normalize_protocol

ENV_BASE = "AUTOTRAIN_LLM_BASE_URL"
ENV_KEY = "AUTOTRAIN_LLM_API_KEY"
ENV_MODEL = "AUTOTRAIN_LLM_MODEL"
ENV_PROTOCOL = "AUTOTRAIN_LLM_PROTOCOL"
ENV_PROFILE = "AUTOTRAIN_LLM_PROFILE"  # 仅日志/调试

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


def runtime_from_profile(profile: LLMProfile) -> LLMRuntimeConfig:
    return LLMRuntimeConfig.from_profile(profile, resolve_profile_protocol(profile))


def runtime_from_settings(settings: Optional[Mapping[str, Any]]) -> LLMRuntimeConfig:
    profile = get_active_profile(dict(settings or {}))
    if profile is not None:
        return runtime_from_profile(profile)
    # 无 profile：尝试扁平字段
    base = str((settings or {}).get("llm_base_url") or "").strip()
    key = str((settings or {}).get("llm_api_key") or "").strip()
    model = str((settings or {}).get("llm_model") or "").strip()
    proto = normalize_protocol(str((settings or {}).get("llm_protocol") or "auto"))
    return LLMRuntimeConfig(
        base_url=base,
        api_key=key,
        model=model,
        protocol=proto,
    )


def runtime_from_environ(env: Optional[Mapping[str, str]] = None) -> LLMRuntimeConfig:
    """从环境变量解析（训练子进程路径）。"""
    e = env if env is not None else os.environ
    env_base = (e.get(ENV_BASE) or "").strip()
    env_key = (e.get(ENV_KEY) or "").strip()
    env_model = (e.get(ENV_MODEL) or "").strip()
    env_proto = e.get(ENV_PROTOCOL)

    ds_key = (e.get("DEEPSEEK_API_KEY") or "").strip()
    ds_model = (e.get("DEEPSEEK_MODEL") or "").strip()

    resolved_key = env_key or ds_key
    resolved_base = env_base
    if not resolved_base and ds_key and not env_key and not env_base:
        resolved_base = DEEPSEEK_BASE_URL
    resolved_model = (
        env_model
        or ds_model
        or (DEEPSEEK_DEFAULT_MODEL if resolved_base.rstrip("/").endswith("api.deepseek.com") else "")
        or "gpt-4o-mini"
    )
    if env_proto is not None and str(env_proto).strip() != "":
        resolved_proto = normalize_protocol(env_proto)
    elif resolved_base.rstrip("/").endswith("api.deepseek.com"):
        resolved_proto = "openai"
    else:
        resolved_proto = "auto"

    return LLMRuntimeConfig(
        base_url=normalize_base_url(resolved_base),
        api_key=resolved_key,
        model=resolved_model,
        protocol=resolved_proto,
        profile_name=(e.get(ENV_PROFILE) or "").strip(),
    )


def apply_runtime_to_environ(
    runtime: LLMRuntimeConfig,
    env: Optional[MutableMapping[str, str]] = None,
    *,
    clear_deepseek: bool = True,
) -> MutableMapping[str, str]:
    """写入 AUTOTRAIN_LLM_*；可选清除 DEEPSEEK_* 避免抢路由。"""
    target: MutableMapping[str, str] = env if env is not None else os.environ
    mapping = {
        ENV_BASE: (runtime.base_url or "").strip(),
        ENV_KEY: (runtime.api_key or "").strip(),
        ENV_MODEL: (runtime.model or "").strip(),
        ENV_PROTOCOL: (runtime.protocol or "auto").strip() or "auto",
    }
    for ek, val in mapping.items():
        if val:
            target[ek] = val
        else:
            target.pop(ek, None)
    if runtime.profile_name:
        target[ENV_PROFILE] = runtime.profile_name
    else:
        target.pop(ENV_PROFILE, None)

    if clear_deepseek and (runtime.base_url or runtime.api_key):
        target.pop("DEEPSEEK_API_KEY", None)
        target.pop("DEEPSEEK_MODEL", None)
    return target


def apply_settings_to_environ(settings: Mapping[str, Any]) -> None:
    """UI 保存后：把 active profile 同步到当前进程环境。"""
    apply_runtime_to_environ(runtime_from_settings(settings), os.environ, clear_deepseek=True)


def inject_llm_env_for_subprocess(
    base_env: Optional[Mapping[str, str]],
    settings: Mapping[str, Any],
) -> Dict[str, str]:
    """TrainWorker：复制环境并注入 LLM。"""
    env = dict(base_env or os.environ)
    apply_runtime_to_environ(runtime_from_settings(settings), env, clear_deepseek=True)
    return env
