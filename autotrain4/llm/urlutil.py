"""Base URL / protocol 规范化与协议探测。"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

# Anthropic Messages API 版本头
ANTHROPIC_VERSION = "2023-06-01"

# auto 探测时认为「OpenAI 路径不对口、值得试 Anthropic」的信号
_PROTOCOL_HINT_STATUS = frozenset({400, 404, 405, 415, 422, 501, 502, 503})
_PROTOCOL_HINT_MARKERS = (
    "not supported",
    "unknown endpoint",
    "no route",
    "invalid url",
    "does not exist",
    "not found",
    "unsupported",
    "only support",
    "anthropic",
    "messages api",
    "chat/completions",
    "invalid model",
    "model_not_found",
    "channel not found",
    "group not found",
    "bad request",
)


def normalize_base_url(url: str) -> str:
    """去掉末尾斜杠；无 scheme 时补 http://。"""
    text = (url or "").strip()
    if not text:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = "http://" + text
    return text.rstrip("/")


def normalize_protocol(value: Optional[str]) -> str:
    """归一化为 auto | openai | anthropic。"""
    text = (value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "openai": "openai",
        "open_ai": "openai",
        "oai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "cc": "anthropic",
        "cc-switch": "anthropic",
        "cc_switch": "anthropic",
        "newapi": "anthropic",
        "new-api": "anthropic",
    }
    if text in aliases:
        return aliases[text]
    if text in {"auto", "openai", "anthropic"}:
        return text
    return "auto"


def _origin_and_prefix(base_url: str) -> Tuple[str, str]:
    """从 base_url 拆出 origin 与 path 前缀（保留 /v1 等）。"""
    normalized = normalize_base_url(base_url)
    parsed = urlparse(normalized)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    prefix = (parsed.path or "").rstrip("/")
    return origin, prefix


def openai_base_url(base_url: str) -> str:
    """OpenAI SDK 的 base_url：尽量以 /v1 结尾。"""
    origin, prefix = _origin_and_prefix(base_url)
    if not origin.startswith("http"):
        return normalize_base_url(base_url)
    if not prefix:
        return f"{origin}/v1"
    if prefix.endswith("/v1") or re.search(r"/v\d+$", prefix):
        return f"{origin}{prefix}"
    return f"{origin}{prefix}/v1"


def anthropic_base_url(base_url: str) -> str:
    """
    Anthropic 请求根（不含 /v1/messages）。

    实际请求为 ``{base}/v1/messages``：
    - 输入 ``http://host:3000``     → ``http://host:3000``
    - 输入 ``http://host:3000/v1``  → ``http://host:3000``（去掉末尾 /v1）
    """
    origin, prefix = _origin_and_prefix(base_url)
    if not origin.startswith("http"):
        return normalize_base_url(base_url)
    if not prefix:
        return origin
    if prefix.endswith("/v1"):
        prefix = prefix[: -len("/v1")]
    elif re.search(r"/v\d+$", prefix):
        prefix = re.sub(r"/v\d+$", "", prefix)
    return f"{origin}{prefix}" if prefix else origin


def anthropic_messages_url(base_url: str) -> str:
    """完整 Messages 端点：``{anthropic_base_url}/v1/messages``。"""
    return f"{anthropic_base_url(base_url)}/v1/messages"


def _error_status_and_text(exc: BaseException) -> Tuple[Optional[int], str]:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    parts: List[str] = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            parts.append(resp.text)  # type: ignore[attr-defined]
        except Exception:
            pass
    return status if isinstance(status, int) else None, " ".join(parts).lower()


def looks_like_protocol_mismatch(exc: BaseException) -> bool:
    """OpenAI 调用失败时，是否值得再试 Anthropic。"""
    status, text = _error_status_and_text(exc)
    if status in _PROTOCOL_HINT_STATUS:
        return True
    if status == 401:
        return any(m in text for m in ("channel", "route", "endpoint", "model"))
    return any(marker in text for marker in _PROTOCOL_HINT_MARKERS)


class AnthropicHTTPError(RuntimeError):
    """requests 直连 Anthropic 时的 HTTP/业务错误，带 status_code 供协议探测。"""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
