"""AutoTrain 子进程与桌面前端之间的结构化输出协议。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


CHAT_RESPONSE_MARKER = "[CHAT_RESPONSE_JSON] "
# 阶段 C：结构化事件前缀。TrainWorker 优先 parse，旧 [RUN_DIR] 等作 fallback。
AT4_MARKER = "__AT4__"


def encode_chat_response(content: str) -> str:
    """把多行助手回复编码为可安全按行传输的 JSON 消息。"""
    return CHAT_RESPONSE_MARKER + json.dumps(content, ensure_ascii=False)


def decode_chat_response(line: str) -> Optional[str]:
    """解码助手回复；非聊天协议行返回 ``None``。"""
    if not line.startswith(CHAT_RESPONSE_MARKER):
        return None
    payload = line[len(CHAT_RESPONSE_MARKER):]
    try:
        content = json.loads(payload)
    except json.JSONDecodeError:
        # 兼容协议行被旧版本直接输出为纯文本的情况。
        return payload
    return content if isinstance(content, str) else str(content)


def encode_at4_event(event_type: str, **fields: Any) -> str:
    """编码一条结构化 IPC 行：``__AT4__{"type":"...","path":...}``。"""
    payload: Dict[str, Any] = {"type": event_type}
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    return AT4_MARKER + json.dumps(payload, ensure_ascii=False)


def decode_at4_event(line: str) -> Optional[Dict[str, Any]]:
    """解码 ``__AT4__`` JSON 事件；非协议行返回 ``None``。"""
    text = (line or "").strip()
    if not text.startswith(AT4_MARKER):
        return None
    payload = text[len(AT4_MARKER) :]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def emit_run_dir(path: Any) -> str:
    """便捷：run 目录事件（与旧 ``[RUN_DIR]`` 并行输出时使用）。"""
    return encode_at4_event("run_dir", path=str(path))


def emit_stage(name: str, **fields: Any) -> str:
    """便捷：阶段事件，如 train_done。"""
    return encode_at4_event("stage", name=name, **fields)
