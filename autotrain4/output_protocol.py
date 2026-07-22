"""AutoTrain 子进程与桌面前端之间的结构化输出协议。"""

import json
from typing import Optional


CHAT_RESPONSE_MARKER = "[CHAT_RESPONSE_JSON] "


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
