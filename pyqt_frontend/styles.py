"""MainWindow 主题 QSS（随 UIScale 生成）。"""

from __future__ import annotations

from typing import Any

from PyQt5.QtGui import QColor, QPalette

from widgets.ui_scale import UIScale


def build_main_stylesheet(scale: UIScale) -> str:
    """按当前缩放因子生成主窗口 QSS。"""
    s = scale
    f = s.font
    p = s.qss_px
    return f"""
            QMainWindow {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(108,92,231,0.16),
                    stop:0.28 transparent,
                    stop:1 #f4f6fb
                );
            }}
            QWidget#mainArea {{
                background: transparent;
            }}

            /* 侧边栏宽度由 setFixedWidth 控制，这里只保留视觉样式 */
            QWidget#sidebar {{
                background: rgba(255, 255, 255, 0.86);
                border-right: 1px solid rgba(232, 235, 242, 0.9);
            }}
            QLabel#brandMark {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #a879ff);
                color: white;
                font-weight: 800;
                font-size: {f(15)}px;
                border-radius: {p(12, 8)};
            }}
            QLabel#brandTitle {{
                font-size: {f(17)}px;
                font-weight: bold;
                color: #192033;
            }}
            QLabel#brandSub {{
                font-size: {f(12)}px;
                color: #7b8499;
                margin-top: {p(2, 0)};
            }}
            QPushButton#navBtn, QPushButton#navBtnActive {{
                border: none;
                background: transparent;
                color: #5e6678;
                text-align: left;
                padding: {p(12, 6)} {p(14, 8)};
                border-radius: {p(12, 8)};
                font-size: {f(14)}px;
            }}
            QPushButton#navBtn:hover {{
                background: rgba(108, 92, 231, 0.06);
            }}
            QPushButton#navBtnActive {{
                background: #f0edff;
                color: #6c5ce7;
                font-weight: bold;
            }}
            QFrame#sideCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f0edff, stop:1 #faf9ff);
                border: 1px solid #e5defe;
                border-radius: {p(16, 10)};
                padding: {p(6, 4)};
            }}
            QLabel#sideCardTitle {{
                font-size: {f(14)}px;
                font-weight: bold;
                color: #192033;
            }}
            QLabel#sideCardDesc {{
                font-size: {f(12)}px;
                color: #7b8499;
                line-height: 1.5;
            }}
            QLabel#sideCardBadge {{
                font-size: {f(12)}px;
                font-weight: bold;
                color: #1fa971;
                background: #e9f8f2;
                border-radius: 999px;
                padding: {p(5, 3)} {p(10, 6)};
            }}

            /* 顶栏 */
            QLabel#pageTitle {{
                font-size: {f(28)}px;
                font-weight: bold;
                color: #ffffff;
            }}
            QLabel#pageSubtitle {{
                font-size: {f(14)}px;
                color: #7b8499;
                margin-top: {p(4, 2)};
            }}
            QPushButton#primaryBtn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #8b7cf6);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: {p(12, 8)};
                padding: {p(10, 6)} {p(16, 10)};
                font-size: {f(14)}px;
            }}
            QPushButton#ghostBtn, QPushButton#tinyGhostBtn {{
                background: #eef0f7;
                color: #4d5566;
                border: none;
                border-radius: {p(12, 8)};
                padding: {p(10, 6)} {p(14, 8)};
                font-size: {f(13)}px;
            }}
            QPushButton#tinyGhostBtn {{
                padding: {p(6, 4)} {p(10, 6)};
                font-size: {f(12)}px;
            }}

            /* 卡片通用 */
            QFrame#card {{
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(232, 235, 242, 0.96);
                border-radius: {p(20, 12)};
                padding: {p(8, 4)};
            }}
            QLabel#cardTitle {{
                font-size: {f(16)}px;
                font-weight: bold;
                color: #192033;
            }}
            QLabel#tag {{
                font-size: {f(12)}px;
                font-weight: bold;
                color: #6c5ce7;
                background: #f2f0ff;
                border-radius: 999px;
                padding: {p(5, 3)} {p(10, 6)};
            }}
            QLabel#sectionTitle {{
                font-size: {f(14)}px;
                font-weight: bold;
                color: #192033;
            }}

            /* 输入框 */
            QPlainTextEdit#promptInput {{
                background-color: #fbfcff;
                border: 1px solid #e5e8f0;
                border-radius: {p(17, 10)};
                padding: {p(14, 8)};
                font-size: {f(16)}px;
            }}
            QPlainTextEdit#promptInput:focus {{
                border: 1px solid #b8afff;
            }}

            /* 建议气泡 */
            QPushButton#suggestionChip {{
                background-color: #f7f5ff;
                color: #5347b4;
                border: 1px solid #e6e1ff;
                border-radius: 999px;
                padding: {p(7, 4)} {p(14, 8)};
                font-size: {f(13)}px;
            }}
            QPushButton#suggestionChip:hover {{
                background-color: #efebff;
                border: 1px solid #d6cdff;
            }}
            QWidget#suggestionsContainer {{
                background: transparent;
            }}

            /* 按钮 */
            QPushButton#runBtn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #7b66ef);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: {p(14, 8)};
                padding: {p(13, 8)};
                font-size: {f(15)}px;
            }}
            QPushButton#runBtn:disabled {{
                opacity: 0.65;
            }}
            QPushButton#secondaryBtn {{
                background: #eef0f6;
                color: #4f586b;
                border: none;
                border-radius: {p(14, 8)};
                padding: {p(13, 8)};
                font-size: {f(14)}px;
            }}
            QPushButton#secondaryBtn:hover {{
                background: #e2e6f0;
                color: #374151;
            }}
            QPushButton#secondaryBtn:pressed {{
                background: #d5dae8;
                color: #1f2937;
                padding-top: {p(14, 9)};
                padding-bottom: {p(12, 7)};
            }}
            QPushButton#softBtn {{
                background: #f4efff;
                color: #6555c7;
                border: none;
                border-radius: {p(14, 8)};
                padding: {p(13, 8)};
                font-size: {f(14)}px;
            }}

            /* 指标 */
            QFrame#metric {{
                background: #f8f9fc;
                border: 1px solid #e8ebf2;
                border-radius: {p(14, 8)};
                padding: {p(8, 4)};
            }}
            QLabel#metricLabel {{
                font-size: {f(12)}px;
                color: #7b8499;
            }}
            QLabel#metricValue {{
                font-size: {f(20)}px;
                font-weight: bold;
                color: #192033;
            }}

            /* 日志 */
            QLabel#statusBadge {{
                font-size: {f(12)}px;
                font-weight: bold;
                color: #6b7280;
                background: #e5e7eb;
                border-radius: 999px;
                padding: {p(5, 3)} {p(10, 6)};
            }}
            QLabel#statusBadgeRunning {{
                font-size: {f(12)}px;
                font-weight: bold;
                color: #6c5ce7;
                background: #eeeaff;
                border-radius: 999px;
                padding: {p(5, 3)} {p(10, 6)};
            }}
            QLabel#statusBadgeSuccess {{
                font-size: {f(12)}px;
                font-weight: bold;
                color: #1fa971;
                background: #e9f8f2;
                border-radius: 999px;
                padding: {p(5, 3)} {p(10, 6)};
            }}
            QLabel#statusBadgeError {{
                font-size: {f(12)}px;
                font-weight: bold;
                color: #db5a65;
                background: #fdecee;
                border-radius: 999px;
                padding: {p(5, 3)} {p(10, 6)};
            }}
            QWidget#logPanel {{
                background: transparent;
                border: 1px solid #dde2ec;
                border-radius: {p(16, 10)};
            }}
            QFrame#logStageRail {{
                background: #f7f8fc;
                border: none;
                border-top-left-radius: {p(15, 10)};
                border-bottom-left-radius: {p(15, 10)};
            }}
            QLabel#logRailTitle {{
                color: #929bad;
                font-size: {f(10)}px;
                font-weight: 800;
                letter-spacing: 1px;
                padding: {p(2, 1)} {p(5, 3)} {p(6, 3)} {p(5, 3)};
            }}
            QWidget#logStageHost,
            QScrollArea#logStageScroll,
            QScrollArea#logStageScroll > QWidget > QWidget {{
                background: transparent;
                border: none;
            }}
            QPushButton#logStageNav,
            QPushButton#logStageNavActive {{
                background: transparent;
                color: #697386;
                font-size: {f(11)}px;
                font-weight: 600;
                border: none;
                border-radius: {p(9, 6)};
                padding: {p(9, 5)} {p(8, 4)};
                text-align: left;
            }}
            QPushButton#logStageNav:hover {{
                background: #eff1f7;
                color: #414b5f;
            }}
            QPushButton#logStageNavActive {{
                background: #ebe8ff;
                color: #6655d9;
                font-weight: 800;
            }}
            QFrame#logConsole {{
                background: #0d1423;
                border: none;
                border-top-right-radius: {p(15, 10)};
                border-bottom-right-radius: {p(15, 10)};
            }}
            QFrame#logConsoleToolbar {{
                background: #151e2f;
                border: none;
                border-bottom: 1px solid #222d42;
                border-top-right-radius: {p(15, 10)};
            }}
            QLabel#consoleDotRed,
            QLabel#consoleDotAmber,
            QLabel#consoleDotGreen {{
                border-radius: {p(4, 2)};
                background: #ff6b72;
            }}
            QLabel#consoleDotAmber {{
                background: #f2bd55;
            }}
            QLabel#consoleDotGreen {{
                background: #38c98d;
            }}
            QLabel#logConsoleTitle {{
                color: #8390a5;
                font-size: {f(10)}px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#logConsoleLive {{
                color: #4dd4a1;
                background: #122d2a;
                border: 1px solid #1e473e;
                border-radius: {p(8, 4)};
                padding: {p(3, 2)} {p(7, 4)};
                font-size: {f(9)}px;
                font-weight: 800;
            }}
            QPlainTextEdit#logConsoleOutput {{
                background: #0d1423;
                border: none;
                border-bottom-right-radius: {p(15, 10)};
                padding: {p(14, 8)} {p(16, 10)};
                font-family: "Consolas", "SFMono-Regular", monospace;
                font-size: {f(12)}px;
                color: #c7d0df;
                selection-background-color: #6655d9;
                selection-color: #ffffff;
            }}
            QPlainTextEdit#logConsoleOutput QScrollBar:vertical {{
                background: #0d1423;
                width: {p(10, 6)};
                margin: {p(4, 2)} {p(2, 1)} {p(4, 2)} 0;
            }}
            QPlainTextEdit#logConsoleOutput QScrollBar::handle:vertical {{
                background: #344057;
                border-radius: {p(4, 2)};
                min-height: {p(28, 16)};
            }}
            QPlainTextEdit#logConsoleOutput QScrollBar::add-line:vertical,
            QPlainTextEdit#logConsoleOutput QScrollBar::sub-line:vertical {{
                height: 0;
            }}

            /* 快捷模板 */
            QFrame#quickItem {{
                background: #fbfcff;
                border: 1px solid #e8ebf2;
                border-radius: {p(14, 8)};
                padding: {p(8, 4)};
            }}
            QLabel#quickTitle {{
                font-size: {f(13)}px;
                font-weight: bold;
                color: #192033;
            }}
            QLabel#quickDesc {{
                font-size: {f(12)}px;
                color: #7b8499;
                margin-top: {p(4, 2)};
            }}

            /* 关闭浏览器（训练需求按钮行） */
            QPushButton#closeBrowserBtn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #8b7cf6);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: {p(14, 8)};
                padding: {p(13, 8)};
                font-size: {f(14)}px;
            }}
            QPushButton#closeBrowserBtn:disabled {{
                background: #c4b5fd;
                color: #f5f3ff;
            }}
            QPushButton#closeBrowserBtn:hover:!disabled {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5b4cdb, stop:1 #7c6cf0);
            }}
"""


def apply_main_styles(window: Any) -> None:
    """应用 Web 前端风格主题；字号/圆角/内边距随窗口等比例缩放。"""
    s = getattr(window, "_ui_scale", None) or UIScale()
    window.setStyleSheet(build_main_stylesheet(s))

    # 显式设置输入框调色板，确保 IME 预编辑拼音颜色正常
    if hasattr(window, "prompt_input"):
        palette = window.prompt_input.palette()
        palette.setColor(QPalette.Base, QColor("#fbfcff"))
        palette.setColor(QPalette.Text, QColor("#1f2937"))
        palette.setColor(QPalette.PlaceholderText, QColor("#9ca3af"))
        palette.setColor(QPalette.Highlight, QColor("#c4b5fd"))
        palette.setColor(QPalette.HighlightedText, QColor("#1f2937"))
        window.prompt_input.setPalette(palette)
