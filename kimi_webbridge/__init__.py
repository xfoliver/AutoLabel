"""
Kimi WebBridge 适配层

为 auto_train2.0.py 提供统一的浏览器自动化接口，底层可替换为：
- HTTP API 适配器
- WebSocket 适配器
- 本地 SDK 适配器
- 测试桩

使用：
    from kimi_webbridge import create_webbridge
    bridge = create_webbridge(config_dict)
    bridge.navigate("http://localhost:8842")
    bridge.execute_task("点击'训练参数'菜单，设置 max_epochs=12，点击保存和开始训练")
"""

from .base import BaseWebBridge, WebBridgeError, WebBridgeResult
from .playwright_adapter import PlaywrightWebBridgeAdapter
from .webview2_cdp_adapter import WebView2CDPWebBridgeAdapter

__all__ = [
    "BaseWebBridge",
    "WebBridgeError",
    "WebBridgeResult",
    "PlaywrightWebBridgeAdapter",
    "WebView2CDPWebBridgeAdapter",
    "create_webbridge",
]


def create_webbridge(config: dict):
    """根据配置创建 WebBridge 适配器实例。"""
    adapter_type = config.get("adapter", "playwright").lower()
    if adapter_type == "playwright":
        return PlaywrightWebBridgeAdapter(config)
    if adapter_type == "webview2_cdp":
        return WebView2CDPWebBridgeAdapter(config)
    raise WebBridgeError(
        f"不支持的 WebBridge adapter 类型: {adapter_type}（当前支持 playwright / webview2_cdp）"
    )
