"""
autotrain4.llm — 统一 LLM 层。

分层：
  业务 → LLMClient.chat() → Adapter → OpenAI / Anthropic
  配置 → LLMProfile（多份）→ active → env AUTOTRAIN_LLM_*
"""

from __future__ import annotations

from .client import LLMClient
from .env_bridge import (
    apply_settings_to_environ,
    inject_llm_env_for_subprocess,
    runtime_from_settings,
)
from .presets import list_presets, normalize_provider
from .profiles import (
    delete_profile,
    get_active_profile,
    list_profiles,
    migrate_settings,
    set_active_profile,
    upsert_profile,
)
from .types import LLMProfile, new_profile_id
from .urlutil import (
    AnthropicHTTPError,
    anthropic_base_url,
    anthropic_messages_url,
    looks_like_protocol_mismatch,
    normalize_base_url,
    normalize_protocol,
    openai_base_url,
)

__all__ = [
    "LLMClient",
    "LLMProfile",
    "AnthropicHTTPError",
    "new_profile_id",
    "list_presets",
    "normalize_provider",
    "list_profiles",
    "get_active_profile",
    "migrate_settings",
    "set_active_profile",
    "upsert_profile",
    "delete_profile",
    "apply_settings_to_environ",
    "inject_llm_env_for_subprocess",
    "runtime_from_settings",
    "normalize_base_url",
    "normalize_protocol",
    "openai_base_url",
    "anthropic_base_url",
    "anthropic_messages_url",
    "looks_like_protocol_mismatch",
]
