"""内置 LLM Provider 预设：默认 URL / 协议 / 模型提示。"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class ProviderPreset(TypedDict, total=False):
    id: str
    name: str
    default_base_url: str
    default_protocol: str  # auto | openai | anthropic
    default_model: str
    model_hint: str
    base_url_hint: str
    notes: str


PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_XAI = "xai"
PROVIDER_KIMI = "kimi"
PROVIDER_CUSTOM = "custom"

# 兼容旧别名 → 标准 provider id
PROVIDER_ALIASES: Dict[str, str] = {
    "deepseek": PROVIDER_DEEPSEEK,
    "openai": PROVIDER_OPENAI,
    "open_ai": PROVIDER_OPENAI,
    "gpt": PROVIDER_OPENAI,
    "anthropic": PROVIDER_ANTHROPIC,
    "claude": PROVIDER_ANTHROPIC,
    "xai": PROVIDER_XAI,
    "grok": PROVIDER_XAI,
    "kimi": PROVIDER_KIMI,
    "moonshot": PROVIDER_KIMI,
    "custom": PROVIDER_CUSTOM,
    "cc": PROVIDER_CUSTOM,
    "cc-switch": PROVIDER_CUSTOM,
    "cc_switch": PROVIDER_CUSTOM,
    "newapi": PROVIDER_CUSTOM,
    "new-api": PROVIDER_CUSTOM,
    "gateway": PROVIDER_CUSTOM,
}


PRESETS: Dict[str, ProviderPreset] = {
    PROVIDER_DEEPSEEK: {
        "id": PROVIDER_DEEPSEEK,
        "name": "DeepSeek",
        "default_base_url": "https://api.deepseek.com",
        "default_protocol": "openai",
        "default_model": "deepseek-chat",
        "model_hint": "deepseek-chat / deepseek-reasoner",
        "base_url_hint": "https://api.deepseek.com",
        "notes": "官方 OpenAI 兼容接口",
    },
    PROVIDER_OPENAI: {
        "id": PROVIDER_OPENAI,
        "name": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "default_protocol": "openai",
        "default_model": "gpt-4o-mini",
        "model_hint": "gpt-4o-mini / gpt-4o",
        "base_url_hint": "https://api.openai.com/v1",
        "notes": "官方或任意 /v1 Chat Completions 兼容源",
    },
    PROVIDER_ANTHROPIC: {
        "id": PROVIDER_ANTHROPIC,
        "name": "Anthropic / Claude",
        "default_base_url": "https://api.anthropic.com",
        "default_protocol": "anthropic",
        "default_model": "claude-sonnet-4-5",
        "model_hint": "claude-sonnet-4-5 / claude-opus-4",
        "base_url_hint": "https://api.anthropic.com 或网关根地址",
        "notes": "Anthropic Messages；经 NewAPI/CC Switch 转发 Claude 时也选此项",
    },
    PROVIDER_XAI: {
        "id": PROVIDER_XAI,
        "name": "xAI / Grok",
        "default_base_url": "https://api.x.ai/v1",
        "default_protocol": "openai",
        "default_model": "grok-4",
        "model_hint": "grok-4 / grok-3",
        "base_url_hint": "https://api.x.ai/v1 或兼容网关",
        "notes": "官方多为 OpenAI 兼容；若网关仅 Messages 可在高级设置改协议",
    },
    PROVIDER_KIMI: {
        "id": PROVIDER_KIMI,
        "name": "Kimi / Moonshot",
        "default_base_url": "https://api.moonshot.cn/v1",
        "default_protocol": "openai",
        "default_model": "moonshot-v1-8k",
        "model_hint": "moonshot-v1-8k / moonshot-v1-32k",
        "base_url_hint": "https://api.moonshot.cn/v1",
        "notes": "OpenAI 兼容",
    },
    PROVIDER_CUSTOM: {
        "id": PROVIDER_CUSTOM,
        "name": "自定义 / 网关",
        "default_base_url": "",
        "default_protocol": "auto",
        "default_model": "",
        "model_hint": "网关中的模型名",
        "base_url_hint": "http://host:port 或 http://host:port/v1",
        "notes": "NewAPI、CC Switch、OneAPI 等；默认自动探测协议",
    },
}


def normalize_provider(value: Optional[str]) -> str:
    text = (value or PROVIDER_CUSTOM).strip().lower()
    return PROVIDER_ALIASES.get(text, PROVIDER_CUSTOM if text not in PRESETS else text)


def list_presets() -> List[ProviderPreset]:
    order = [
        PROVIDER_DEEPSEEK,
        PROVIDER_OPENAI,
        PROVIDER_ANTHROPIC,
        PROVIDER_XAI,
        PROVIDER_KIMI,
        PROVIDER_CUSTOM,
    ]
    return [PRESETS[i] for i in order]


def get_preset(provider_id: Optional[str]) -> ProviderPreset:
    pid = normalize_provider(provider_id)
    return PRESETS.get(pid, PRESETS[PROVIDER_CUSTOM])


def default_protocol_for_provider(provider_id: Optional[str]) -> str:
    return str(get_preset(provider_id).get("default_protocol") or "auto")
