"""UI 等比例缩放工具。"""

from __future__ import annotations


class UIScale:
    """以设计稿分辨率换算缩放系数，驱动字体 / 间距 / 固定尺寸同步缩放。"""

    REF_W = 1280
    REF_H = 820
    MIN_FACTOR = 0.72
    MAX_FACTOR = 1.55
    # 小于该差值不重绘，避免 resize 抖动
    EPSILON = 0.015

    def __init__(self) -> None:
        self.factor = 1.0

    def update_from_size(self, width: int, height: int) -> bool:
        if width <= 0 or height <= 0:
            return False
        raw = min(width / self.REF_W, height / self.REF_H)
        factor = max(self.MIN_FACTOR, min(self.MAX_FACTOR, raw))
        if abs(factor - self.factor) < self.EPSILON:
            return False
        self.factor = factor
        return True

    def px(self, value: float, minimum: int = 1) -> int:
        return max(minimum, int(round(value * self.factor)))

    def font(self, value: float, minimum: int = 8) -> int:
        return max(minimum, int(round(value * self.factor)))

    def qss_px(self, value: float, minimum: int = 0) -> str:
        return f"{self.px(value, minimum)}px"
