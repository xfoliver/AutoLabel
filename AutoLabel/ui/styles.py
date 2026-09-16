"""AutoLabel UI styles — matches the frontend demo (light purple)."""

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
    background: #f6f8fc;
    color: #1f2937;
    font-size: {f(13)}px;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
}}
QWidget#Root {{
    background: #f6f8fc;
}}
QWidget#TopBar {{
    background: rgba(255,255,255,235);
    border-bottom: 1px solid #e8ebf2;
}}
QLabel#BrandTitle {{
    font-size: {f(22)}px;
    font-weight: 800;
    color: #1f2937;
    letter-spacing: -0.4px;
}}
QLabel#BrandSub {{
    color: #a1a9b6;
    font-size: {f(13)}px;
    padding-left: {p(12, 6)};
    border-left: 1px solid #e8ebf2;
}}
QLabel#Title {{
    font-size: {f(22)}px;
    font-weight: 800;
    color: #1f2937;
}}
QLabel#PageTitle {{
    font-size: {f(22)}px;
    font-weight: 800;
    color: #1f2937;
}}
QLabel#Subtitle {{
    color: #6e7a8a;
    font-size: {f(13)}px;
}}
QLabel#Notice {{
    background: #f6f3ff;
    color: #4a3d78;
    border: 1px solid #e0d8ff;
    border-left: {p(4, 3)} solid #6c4cff;
    border-radius: {p(10, 6)};
    padding: {p(10, 6)} {p(14, 8)};
    font-size: {f(12)}px;
}}
QLabel#Hint {{
    color: #7b8798;
    font-size: {f(12)}px;
}}
QLabel#SectionTitle, QLabel#PageTitle, QLabel#BrandTitle {{
    min-width: 0px;
}}
QLabel#SectionTitle {{
    font-size: {f(16)}px;
    font-weight: 750;
    color: #1f2937;
}}
QLabel#ReadyPill {{
    color: #1a9f62;
    background: #eaf9f1;
    border-radius: {p(10, 6)};
    padding: {p(6, 3)} {p(10, 6)};
    font-weight: 700;
    font-size: {f(12)}px;
}}
QLabel#NumberBadge {{
    color: white;
    background: #6c4cff;
    border-radius: {p(15, 10)};
    font-weight: 800;
    font-size: {f(13)}px;
    qproperty-alignment: AlignCenter;
}}
QLabel#StepCircle {{
    color: #6c4cff;
    border: 2px solid #6c4cff;
    border-radius: {p(13, 8)};
    font-weight: 800;
    font-size: {f(12)}px;
    qproperty-alignment: AlignCenter;
}}
QWidget#Sidebar {{
    background: rgba(255,255,255,200);
    border-right: 1px solid #e8ebf2;
    min-width: 0px;
}}
QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    min-width: 0px;
    background: transparent;
}}
QWidget#Card, QFrame#Card {{
    background: rgba(255,255,255,230);
    border: 1px solid #e8ebf2;
    border-radius: {p(16, 10)};
    min-width: 0px;
}}
QWidget#BackendCard {{
    background: #ffffff;
    border: 1px solid #e8ebf2;
    border-radius: {p(16, 10)};
}}
QPushButton {{
    background: #ffffff;
    border: 1px solid #dfe3eb;
    border-radius: {p(10, 6)};
    padding: {p(8, 4)} {p(14, 8)};
    font-size: {f(13)}px;
    color: #344054;
    min-height: {p(28, 22)};
}}
QPushButton:hover {{
    background: #fbfaff;
    border-color: #c9c0ff;
    color: #6146ed;
}}
QPushButton:disabled {{
    color: #a3acbb;
    background: #f7f8fb;
}}
QPushButton#Primary, QPushButton#StartBtn {{
    background: #6b48ef;
    color: white;
    border: none;
    font-weight: 800;
    font-size: {f(15)}px;
    border-radius: {p(12, 8)};
    min-height: {p(44, 32)};
}}
QPushButton#Primary:hover, QPushButton#StartBtn:hover {{
    background: #5a3ad8;
}}
QPushButton#StartBtn {{
    min-height: {p(52, 40)};
    min-width: {p(140, 110)};
    font-size: {f(16)}px;
}}
QPushButton#Ghost {{
    background: transparent;
    border: none;
    color: #2e3745;
    font-weight: 650;
}}
QPushButton#Ghost:hover {{
    background: #f2f4f8;
}}
QPushButton#Secondary {{
    background: #f6f3ff;
    border: 1px solid #c9c0ff;
    border-radius: {p(10, 6)};
    padding: {p(8, 4)} {p(14, 8)};
    color: #5a3ad8;
    font-weight: 750;
    font-size: {f(13)}px;
    min-height: {p(32, 26)};
}}
QPushButton#Secondary:hover {{
    background: #ece6ff;
    border-color: #6c4cff;
    color: #4a2fd0;
}}
QPushButton#Secondary:pressed {{
    background: #e0d8ff;
}}
QPushButton#Secondary:disabled {{
    color: #a3acbb;
    background: #f7f8fb;
    border-color: #e8ebf2;
}}
QPushButton#Nav {{
    background: transparent;
    border: none;
    border-radius: {p(12, 8)};
    text-align: left;
    padding: {p(10, 6)} {p(16, 8)};
    color: #3f4a5a;
    font-size: {f(14)}px;
    font-weight: 600;
    min-height: {p(44, 32)};
    min-width: 0px;
}}
QPushButton#Nav:hover {{
    background: #f6f5ff;
    color: #6146ed;
}}
QPushButton#Nav:checked {{
    background: #f2efff;
    color: #6248f1;
    font-weight: 750;
    border-right: 4px solid #6c4cff;
}}
QPushButton#Chip {{
    background: #ffffff;
    border: 1px solid #e1e4ec;
    border-radius: {p(10, 6)};
    padding: {p(6, 3)} {p(12, 6)};
    color: #536071;
    font-size: {f(12)}px;
    min-height: {p(26, 20)};
    min-width: 0px;
}}
QPushButton#Chip:hover, QPushButton#Chip:checked {{
    border-color: #bdb2ff;
    background: #f6f3ff;
    color: #6248ee;
}}
QPushButton#Danger {{
    background: #fff5f5;
    color: #c0392b;
    border: 1px solid #f5c6c6;
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #ffffff;
    border: 1px solid #dfe3eb;
    border-radius: {p(11, 6)};
    padding: {p(8, 4)} {p(12, 6)};
    selection-background-color: #6c4cff;
    font-size: {f(13)}px;
    min-height: {p(28, 22)};
    min-width: 0px;
    color: #344054;
}}
QTextEdit:focus, QLineEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid #a99cff;
}}
QSlider::groove:horizontal {{
    height: 5px;
    background: #e1e5ec;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: #6c4cff;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: #7557f7;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QProgressBar {{
    border: none;
    border-radius: {p(8, 4)};
    text-align: center;
    background: #eceef4;
    min-height: {p(10, 6)};
    font-size: {f(11)}px;
    color: #6749ef;
}}
QProgressBar::chunk {{
    background: #6d4af1;
    border-radius: {p(8, 4)};
}}
QStatusBar, QWidget#Footer {{
    background: #ffffff;
    border-top: 1px solid #e8ebf2;
    color: #303a47;
    font-size: {f(12)}px;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QListWidget {{
    background: #ffffff;
    border: 1px solid #e8ebf2;
    border-radius: {p(10, 6)};
}}
QGroupBox {{
    background: rgba(255,255,255,230);
    border: 1px solid #e8ebf2;
    border-radius: {p(16, 10)};
    margin-top: {p(22, 16)};
    padding-top: {p(16, 12)};
    padding-right: {p(12, 6)};
    padding-bottom: {p(12, 6)};
    padding-left: {p(12, 6)};
    font-weight: 600;
    font-size: {f(12)}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {p(12, 6)};
    padding: 0 {p(8, 4)};
    color: #5c6578;
}}
QLabel#Toast {{
    background: #1f2937;
    color: white;
    border-radius: {p(11, 6)};
    padding: {p(10, 6)} {p(14, 8)};
    font-size: {f(12)}px;
}}
"""


APP_QSS = build_stylesheet()


def apply_window_styles(window: Any) -> None:
    """应用主题；字号 / 圆角 / 内边距随窗口等比例缩放。"""
    s = getattr(window, "_ui_scale", None) or UIScale()
    window.setStyleSheet(build_stylesheet(s))
