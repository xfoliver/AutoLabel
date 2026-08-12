"""LLM 配置数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import uuid


def new_profile_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class LLMProfile:
    """一份可保存、可切换的 LLM 连接配置。"""

    id: str
    name: str
    provider: str = "custom"  # presets.PROVIDER_* 或 custom
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # auto | openai | anthropic；空则按 provider 默认
    protocol: str = "auto"
    timeout: int = 60
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LLMProfile":
        d = dict(data or {})
        pid = str(d.get("id") or "").strip() or new_profile_id()
        name = str(d.get("name") or "").strip() or "未命名"
        try:
            timeout = int(d.get("timeout") or 60)
        except (TypeError, ValueError):
            timeout = 60
        return cls(
            id=pid,
            name=name,
            provider=str(d.get("provider") or "custom").strip().lower() or "custom",
            base_url=str(d.get("base_url") or "").strip(),
            api_key=str(d.get("api_key") or "").strip(),
            model=str(d.get("model") or "").strip(),
            protocol=str(d.get("protocol") or "auto").strip().lower() or "auto",
            timeout=max(5, timeout),
            enabled=bool(d.get("enabled", True)),
        )

    def display_label(self) -> str:
        """下拉框展示：名称 + 模型。"""
        model = self.model.strip()
        return f"{self.name} · {model}" if model else self.name


@dataclass
class LLMRuntimeConfig:
    """解析后的运行时连接参数（无 UI 元数据）。"""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    protocol: str = "auto"
    timeout: int = 60
    profile_id: str = ""
    profile_name: str = ""
    provider: str = "custom"

    @classmethod
    def from_profile(cls, profile: LLMProfile, resolved_protocol: str) -> "LLMRuntimeConfig":
        return cls(
            base_url=profile.base_url,
            api_key=profile.api_key,
            model=profile.model,
            protocol=resolved_protocol,
            timeout=profile.timeout,
            profile_id=profile.id,
            profile_name=profile.name,
            provider=profile.provider,
        )
