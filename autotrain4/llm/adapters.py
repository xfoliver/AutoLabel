"""传输适配器：OpenAI Chat Completions / Anthropic Messages。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .urlutil import (
    ANTHROPIC_VERSION,
    AnthropicHTTPError,
    anthropic_messages_url,
    openai_base_url,
)

logger = logging.getLogger(__name__)


def split_system_messages(
    messages: Sequence[Dict[str, str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Anthropic：system 独立；仅保留 user/assistant 轮次。"""
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip().lower()
        content = msg.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            if content.strip():
                system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        converted.append({"role": role, "content": content})

    if not converted:
        converted = [{"role": "user", "content": "\n".join(system_parts) or "."}]
        return "", converted
    if converted[0]["role"] != "user":
        converted.insert(0, {"role": "user", "content": "."})
    return "\n\n".join(system_parts), converted


class OpenAICompatAdapter:
    """openai SDK → chat.completions。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("未安装 openai 包，无法使用 LLM 功能") from e
        if not self.api_key:
            raise RuntimeError("LLM API key 未配置")
        base = openai_base_url(self.base_url)
        self._client = openai.OpenAI(
            base_url=base,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        logger.debug("OpenAI adapter base_url=%s model=%s", base, self.model)
        return self._client

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""


class AnthropicMessagesAdapter:
    """requests POST /v1/messages（不用 anthropic SDK，兼容部分网关）。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._session: Any = None

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "未安装 requests 包，无法使用 Anthropic Messages 协议。请执行: pip install requests"
            ) from e
        self._session = requests.Session()
        return self._session

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError("LLM API key 未配置")

        system, anth_messages = split_system_messages(messages)
        url = anthropic_messages_url(self.base_url)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system:
            payload["system"] = system

        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

        session = self._ensure_session()
        logger.debug("Anthropic adapter POST %s model=%s", url, self.model)
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as exc:
            raise AnthropicHTTPError(f"Anthropic 请求失败: {exc}") from exc

        text_body = resp.text or ""
        if resp.status_code >= 400:
            raise AnthropicHTTPError(
                f"Anthropic HTTP {resp.status_code}: {text_body[:800]}",
                status_code=resp.status_code,
                body=text_body,
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise AnthropicHTTPError(
                f"Anthropic 响应非 JSON: {text_body[:400]}",
                status_code=resp.status_code,
                body=text_body,
            ) from exc

        if not isinstance(data, dict):
            raise AnthropicHTTPError(
                f"Anthropic 响应格式异常: {type(data).__name__}",
                status_code=resp.status_code,
                body=data,
            )

        err = data.get("error")
        if err:
            raise AnthropicHTTPError(
                f"Anthropic error: {err}",
                status_code=resp.status_code,
                body=data,
            )

        parts: List[str] = []
        for block in data.get("content") or []:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
