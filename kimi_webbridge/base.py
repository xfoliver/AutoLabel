"""
Kimi WebBridge 抽象基类

所有适配器必须实现以下接口，供 auto_train2.0.py 调用。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class WebBridgeError(Exception):
    """WebBridge 调用异常。"""
    pass


@dataclass
class WebBridgeResult:
    """WebBridge 执行结果。"""
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    screenshot_path: Optional[str] = None


class BaseWebBridge(ABC):
    """Kimi WebBridge 抽象基类。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "")
        self.timeout = int(config.get("timeout", 120))
        self.screenshot_dir = config.get("screenshot_dir", "./webbridge_screenshots")
        self.session_id: Optional[str] = None

    @abstractmethod
    def connect(self) -> None:
        """建立与 WebBridge 服务的连接/会话。"""
        pass

    @abstractmethod
    def navigate(self, url: str) -> WebBridgeResult:
        """让浏览器导航到指定 URL。"""
        pass

    @abstractmethod
    def execute_task(self, task_description: str) -> WebBridgeResult:
        """
        让 Kimi Agent 执行一段自然语言描述的任务。
        例如："在训练参数页设置 max_epochs=12，batch_size=8，点击保存，然后点击开始训练"
        """
        pass

    @abstractmethod
    def click(self, element_description: str) -> WebBridgeResult:
        """点击页面上匹配描述的控件。"""
        pass

    @abstractmethod
    def fill(self, element_description: str, value: str) -> WebBridgeResult:
        """在页面上匹配描述的输入框中填入值。"""
        pass

    @abstractmethod
    def screenshot(self, name: str = "screenshot") -> WebBridgeResult:
        """截取当前页面。"""
        pass

    @abstractmethod
    def get_page_state(self) -> WebBridgeResult:
        """获取当前页面状态（URL、可交互元素列表等）。"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭会话/浏览器。"""
        pass

    def apply_hyperparameters(self, hyperparameters: Dict[str, Any]) -> "WebBridgeResult":
        """（可选）在训练参数页批量设置超参。需要真实浏览器能力的适配器应重写。"""
        raise NotImplementedError(
            "当前适配器不支持直接设置训练超参，请使用 execute_task() 或切换适配器"
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
