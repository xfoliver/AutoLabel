"""
统一 LLM 客户端。

业务只依赖 chat() / extract_intent 等；协议由 Adapter 处理。
配置来源：构造参数 > settings active profile > 环境变量。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from .adapters import AnthropicMessagesAdapter, OpenAICompatAdapter, split_system_messages
from .env_bridge import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    runtime_from_environ,
    runtime_from_settings,
)
from .tasks import extract_intent as _extract_intent
from .tasks import match_categories as _match_categories
from .tasks import safe_parse_json
from .tasks import select_dataset as _select_dataset
from .types import LLMRuntimeConfig
from .urlutil import (
    AnthropicHTTPError,
    anthropic_base_url,
    anthropic_messages_url,
    looks_like_protocol_mismatch,
    normalize_base_url,
    normalize_protocol,
    openai_base_url,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI / Anthropic 双协议客户端，对外统一 chat()。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        protocol: Optional[str] = None,
        *,
        runtime: Optional[LLMRuntimeConfig] = None,
    ):
        if runtime is not None:
            cfg = runtime
        elif any(v is not None for v in (base_url, api_key, model, protocol)):
            env_cfg = runtime_from_environ()
            cfg = LLMRuntimeConfig(
                base_url=normalize_base_url(base_url if base_url is not None else env_cfg.base_url),
                api_key=(api_key if api_key is not None else env_cfg.api_key) or "",
                model=(model if model is not None else env_cfg.model) or env_cfg.model,
                protocol=normalize_protocol(
                    protocol if protocol is not None else env_cfg.protocol
                ),
                timeout=timeout,
                profile_id=env_cfg.profile_id,
                profile_name=env_cfg.profile_name,
                provider=env_cfg.provider,
            )
            if not cfg.model:
                cfg.model = (
                    DEEPSEEK_DEFAULT_MODEL
                    if cfg.base_url.rstrip("/").endswith("api.deepseek.com")
                    else "gpt-4o-mini"
                )
        else:
            cfg = runtime_from_environ()
            cfg.timeout = timeout

        # DeepSeek 官方强制 openai
        if cfg.base_url.rstrip("/").endswith("api.deepseek.com"):
            cfg.protocol = "openai"

        self.base_url = normalize_base_url(cfg.base_url)
        self.api_key = (cfg.api_key or "").strip()
        self.model = (cfg.model or "").strip() or "gpt-4o-mini"
        self.timeout = int(cfg.timeout or timeout or 60)
        self.protocol = normalize_protocol(cfg.protocol)
        self.profile_id = cfg.profile_id
        self.profile_name = cfg.profile_name
        self.provider = cfg.provider

        self._openai_adapter: Optional[OpenAICompatAdapter] = None
        self._anthropic_adapter: Optional[AnthropicMessagesAdapter] = None
        self._resolved_protocol: Optional[str] = (
            None if self.protocol == "auto" else self.protocol
        )

    # --- 工厂 ---

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any], timeout: int = 60) -> "LLMClient":
        rt = runtime_from_settings(settings)
        if timeout and not rt.timeout:
            rt.timeout = timeout
        elif timeout:
            rt.timeout = timeout
        return cls(runtime=rt)

    @classmethod
    def from_environ(cls, timeout: int = 60) -> "LLMClient":
        rt = runtime_from_environ()
        rt.timeout = timeout
        return cls(runtime=rt)

    @classmethod
    def deepseek(
        cls,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> "LLMClient":
        import os

        return cls(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            model=model or os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL),
            timeout=timeout,
            protocol="openai",
        )

    # --- 状态 ---

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    @property
    def active_protocol(self) -> str:
        return self._resolved_protocol or self.protocol

    def _openai(self) -> OpenAICompatAdapter:
        if self._openai_adapter is None:
            self._openai_adapter = OpenAICompatAdapter(
                self.base_url, self.api_key, self.model, self.timeout
            )
        return self._openai_adapter

    def _anthropic(self) -> AnthropicMessagesAdapter:
        if self._anthropic_adapter is None:
            self._anthropic_adapter = AnthropicMessagesAdapter(
                self.base_url, self.api_key, self.model, self.timeout
            )
        return self._anthropic_adapter

    # 兼容旧测试私有方法名
    def _chat_openai(self, messages: List[Dict[str, str]], temperature: float) -> str:
        return self._openai().chat(messages, temperature)

    def _chat_anthropic(self, messages: List[Dict[str, str]], temperature: float) -> str:
        return self._anthropic().chat(messages, temperature)

    def _ensure_openai(self) -> Any:
        return self._openai()._ensure()

    @staticmethod
    def _split_system_messages(messages):
        return split_system_messages(messages)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
        if not self.available:
            raise RuntimeError("LLM 未配置 base_url / api_key，无法调用")

        preferred = self._resolved_protocol or self.protocol
        if preferred == "openai":
            return self._chat_openai(messages, temperature)
        if preferred == "anthropic":
            return self._chat_anthropic(messages, temperature)

        try:
            result = self._chat_openai(messages, temperature)
            self._resolved_protocol = "openai"
            return result
        except Exception as openai_exc:
            if not looks_like_protocol_mismatch(openai_exc):
                raise
            logger.info(
                "OpenAI 兼容调用失败（%s），尝试 Anthropic Messages 协议…",
                openai_exc,
            )
            try:
                result = self._chat_anthropic(messages, temperature)
            except Exception as anth_exc:
                raise RuntimeError(
                    "LLM auto 探测失败：OpenAI 与 Anthropic 均不可用。"
                    f" OpenAI 错误: {openai_exc}; Anthropic 错误: {anth_exc}"
                ) from anth_exc
            self._resolved_protocol = "anthropic"
            logger.info(
                "已切换到 Anthropic Messages 协议（base=%s）",
                anthropic_base_url(self.base_url),
            )
            return result

    def test_connection(self) -> Dict[str, Any]:
        """发送最短请求，返回 {ok, protocol, model, message, error?}。"""
        if not self.available:
            return {
                "ok": False,
                "protocol": self.protocol,
                "model": self.model,
                "message": "未配置 Base URL 或 API Key",
                "error": "not_configured",
            }
        try:
            text = self.chat(
                [{"role": "user", "content": "Reply with exactly: ok"}],
                temperature=0.0,
            )
            return {
                "ok": True,
                "protocol": self.active_protocol,
                "model": self.model,
                "message": f"连接成功 · 模型 {self.model} · 协议 {self.active_protocol}",
                "sample": (text or "")[:80],
            }
        except Exception as exc:
            return {
                "ok": False,
                "protocol": self.active_protocol,
                "model": self.model,
                "message": f"连接失败: {exc}",
                "error": str(exc)[:500],
            }

    def extract_intent(self, prompt: str) -> Optional[Dict[str, Any]]:
        return _extract_intent(self, prompt)

    def select_dataset(
        self,
        prompt: str,
        intent: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _select_dataset(self, prompt, intent, candidates)

    def match_categories(
        self,
        user_categories: List[str],
        dataset_categories: List[str],
    ) -> Optional[Dict[str, Optional[str]]]:
        if not self.available:
            return None
        return _match_categories(self, user_categories, dataset_categories)

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
        parsed = safe_parse_json(text)
        return parsed if isinstance(parsed, dict) else None
