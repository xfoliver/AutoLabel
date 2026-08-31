"""AutoLabel UI styles（字号 / 圆角 / 内边距随 UIScale 生成）。"""

from __future__ import annotations

from typing import Any, Optional

from ui.widgets.ui_scale import UIScale


def build_stylesheet(scale: Optional[UIScale] = None) -> str:
    """按当前缩放因子生成主窗口 / 对话框 QSS。"""
    s = scale or UIScale()
    f = s.font
    p = s.qss_px
    return f"""
QMainWindow, QDialog {{
    background: #f5f6fb;
    color: #192033;
    font-size: {f(12)}px;
}}
QLabel#Title {{
    font-size: {f(16)}px;
    font-weight: 700;
    color: #192033;
}}
QLabel#Hint {{
    color: #7b8499;
    font-size: {f(11)}px;
}}
QTextEdit, QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {{
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: {p(6, 4)};
    padding: {p(4, 2)} {p(6, 3)};
    selection-background-color: #6c5ce7;
    font-size: {f(12)}px;
    min-height: {p(22, 18)};
}}
QPushButton {{
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: {p(6, 4)};
    padding: {p(4, 2)} {p(10, 6)};
    font-size: {f(12)}px;
    min-height: {p(24, 20)};
}}
QPushButton:hover {{ background: #f0f2fa; }}
QPushButton#Primary {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #8b7cf6);
    color: white;
    border: none;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #5a4bd6; }}
QPushButton#Danger {{
    background: #fff5f5;
    color: #c0392b;
    border: 1px solid #f5c6c6;
}}
QGroupBox {{
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: {p(8, 6)};
    margin-top: {p(8, 4)};
    padding: {p(6, 4)};
    font-weight: 600;
    font-size: {f(11)}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {p(8, 4)};
    padding: 0 {p(4, 2)};
    color: #5c6578;
}}
QStatusBar {{ background: #ffffff; font-size: {f(11)}px; }}
QProgressBar {{
    border: 1px solid #e2e6ef;
    border-radius: {p(4, 2)};
    text-align: center;
    background: #ffffff;
    min-height: {p(12, 8)};
    font-size: {f(10)}px;
}}
QProgressBar::chunk {{
    background: #6c5ce7;
    border-radius: {p(4, 2)};
}}
QListWidget, QScrollArea {{
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: {p(6, 4)};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
"""


APP_QSS = build_stylesheet()


def apply_window_styles(window: Any) -> None:
    """应用主题；字号 / 圆角 / 内边距随窗口等比例缩放。"""
    s = getattr(window, "_ui_scale", None) or UIScale()
    window.setStyleSheet(build_stylesheet(s))
