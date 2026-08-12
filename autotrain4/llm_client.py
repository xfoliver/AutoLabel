"""
兼容 shim（已弃用路径）。

历史代码仍可：
    from autotrain4.llm_client import LLMClient

**新代码请统一**：
    from autotrain4.llm import LLMClient, migrate_settings, ...

本模块仅 re-export，不再承载实现。后续大版本可能移除。
"""

from __future__ import annotations

from autotrain4.llm import (
    AnthropicHTTPError,
    LLMClient,
    anthropic_base_url,
    anthropic_messages_url,
    looks_like_protocol_mismatch,
    normalize_base_url,
    normalize_protocol,
    openai_base_url,
)
from autotrain4.llm.env_bridge import DEEPSEEK_BASE_URL, DEEPSEEK_DEFAULT_MODEL

__all__ = [
    "LLMClient",
    "AnthropicHTTPError",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_DEFAULT_MODEL",
    "normalize_base_url",
    "normalize_protocol",
    "openai_base_url",
    "anthropic_base_url",
    "anthropic_messages_url",
    "looks_like_protocol_mismatch",
]
