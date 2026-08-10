"""
AutoTrain 4.0 PyQt5 桌面前端

启动：
    python pyqt_frontend/main.py

功能：
- 提供一个输入框，用户直接输入自然语言训练需求。
- 点击「开始训练」后，后台调用 auto_train4.0.py 并实时显示日志。
- 训练成功后由桌面端弹出固定选项：生成 report / 对比 baseline / 结束
  （不走 4.0 CLI 的 LLM + input，避免嵌套 stdin 卡死）。
- 前后端分离：界面逻辑在本文件，执行逻辑在 backend.py。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import markdown
import psutil

from PyQt5.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend import PostTrainWorker, TrainWorker, resolve_post_train_context
from suggestions import SmartSuggestionEngine as SuggestionEngine


# ---------------------------------------------------------------------------
# UI 等比例缩放
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 可点击的模板卡片
# ---------------------------------------------------------------------------
class ClickableFrame(QFrame):
    """可点击的卡片容器，点击时发射携带索引的信号。"""

    clicked = pyqtSignal(int)

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._index)
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# 快捷模板管理弹窗
# ---------------------------------------------------------------------------
class QuickTemplatesDialog(QDialog):
    """管理用户保存的快捷模板：查看完整 prompt、删除。"""

    def __init__(
        self,
        templates: list,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("管理快捷模板")
        self.setMinimumWidth(420)
        self._templates = list(templates)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("已保存的快捷模板")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #192033;")
        layout.addWidget(title)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(10)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setAlignment(Qt.AlignTop)
        layout.addWidget(self._list_widget)

        self._render_list()

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _render_list(self) -> None:
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._templates:
            empty = QLabel("暂无自定义模板")
            empty.setStyleSheet("color: #7b8499; padding: 20px;")
            empty.setAlignment(Qt.AlignCenter)
            self._list_layout.addWidget(empty)
            return

        for i, (title, desc, prompt) in enumerate(self._templates):
            row = QFrame()
            row.setStyleSheet(
                "QFrame { background: #f8f9fc; border: 1px solid #e8ebf2; border-radius: 12px; padding: 4px; }"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setSpacing(10)
            row_layout.setContentsMargins(12, 10, 12, 10)

            text_col = QVBoxLayout()
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #192033;")
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("font-size: 12px; color: #7b8499;")
            text_col.addWidget(title_lbl)
            text_col.addWidget(desc_lbl)
            row_layout.addLayout(text_col, 1)

            view_btn = QPushButton("查看")
            view_btn.setStyleSheet(
                "background: #eef0f7; color: #4d5566; border: none; border-radius: 8px; padding: 6px 12px;"
            )
            view_btn.setCursor(Qt.PointingHandCursor)
            view_btn.clicked.connect(lambda _, p=prompt: self._view_prompt(p))

            del_btn = QPushButton("删除")
            del_btn.setStyleSheet(
                "background: #fdecee; color: #db5a65; border: none; border-radius: 8px; padding: 6px 12px;"
            )
            del_btn.setCursor(Qt.PointingHandCursor)
            del_btn.clicked.connect(lambda _, idx=i: self._delete_template(idx))

            row_layout.addWidget(view_btn)
            row_layout.addWidget(del_btn)
            self._list_layout.addWidget(row)

        self._list_layout.addStretch()

    def _view_prompt(self, prompt: str) -> None:
        QMessageBox.information(self, "模板内容", prompt)

    def _delete_template(self, index: int) -> None:
        if 0 <= index < len(self._templates):
            self._templates.pop(index)
            self._render_list()

    def get_templates(self) -> list:
        """返回删除操作后的最新模板列表。"""
        return self._templates


# ---------------------------------------------------------------------------
# 保存模板弹窗：支持自定义标题和 prompt
# ---------------------------------------------------------------------------
class SaveTemplateDialog(QDialog):
    """保存当前 prompt 为快捷模板，允许用户自定义标题。"""

    def __init__(
        self,
        default_prompt: str,
        suggested_title: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("保存为模板")
        self.setMinimumWidth(460)
        self._default_prompt = default_prompt
        self._suggested_title = suggested_title
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title_lbl = QLabel("保存为快捷模板")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #192033;")
        layout.addWidget(title_lbl)

        # 标题输入
        name_lbl = QLabel("模板标题")
        name_lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
        layout.addWidget(name_lbl)
        self.title_input = QLineEdit(self._suggested_title)
        self.title_input.setPlaceholderText("给模板起个名字")
        self.title_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #e8ebf2;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #b8afff;
            }
        """)
        layout.addWidget(self.title_input)

        # Prompt 输入
        prompt_lbl = QLabel("Prompt 内容")
        prompt_lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
        layout.addWidget(prompt_lbl)
        self.prompt_input = QPlainTextEdit(self._default_prompt)
        self.prompt_input.setPlaceholderText("输入训练需求...")
        self.prompt_input.setMaximumHeight(120)
        self.prompt_input.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid #e8ebf2;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QPlainTextEdit:focus {
                border: 1px solid #b8afff;
            }
        """)
        layout.addWidget(self.prompt_input)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #eef0f7;
                color: #4d5566;
                border: none;
                border-radius: 10px;
                padding: 8px 16px;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("保存")
        save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #8b7cf6);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 16px;
                font-weight: bold;
            }
        """)
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def get_template(self) -> Optional[tuple]:
        """返回 (title, desc, prompt)，如果用户取消则返回 None。"""
        title = self.title_input.text().strip()
        prompt = self.prompt_input.toPlainText().strip()
        if not title or not prompt:
            return None
        desc = "自定义模板"
        upper = prompt.upper()
        for t in ["OCR", "CLS", "DET", "DETR", "ISEG", "SSEG"]:
            if t in upper:
                desc = f"{t} 任务模板"
                break
        return (title, desc, prompt)


# ---------------------------------------------------------------------------
# 系统设置弹窗
# ---------------------------------------------------------------------------
class SettingsDialog(QDialog):
    """AutoTrain 自动化参数配置弹窗。"""

    def __init__(
        self,
        settings: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.setMinimumWidth(460)
        self._settings = dict(settings)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("⚙ 系统设置")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #192033;")
        layout.addWidget(title)

        self.fields: dict = {}
        fields_config = [
            ("backend_url", "推理后端地址", "http://127.0.0.1:9887"),
            ("frontend_url", "前端地址", "http://localhost:8842"),
            ("platform_exe", "壳端 EXE 路径", r"C:\dlcv\Lib\site-packages\dlcv_train\DLCV AI Training Platform-win_x64.exe"),
            ("cdp_port", "CDP 调试端口", "9222"),
        ]

        for key, label, placeholder in fields_config:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
            layout.addWidget(lbl)
            input_widget = QLineEdit(str(self._settings.get(key, "")))
            input_widget.setPlaceholderText(placeholder)
            input_widget.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #e8ebf2;
                    border-radius: 10px;
                    padding: 8px 12px;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 1px solid #b8afff;
                }
            """)
            self.fields[key] = input_widget
            layout.addWidget(input_widget)

        adapter_lbl = QLabel("默认 Adapter")
        adapter_lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
        layout.addWidget(adapter_lbl)
        self.adapter_combo = QComboBox()
        self.adapter_combo.addItems(["playwright", "webview2_cdp", "stub", "skill", "http"])
        self.adapter_combo.setCurrentText(self._settings.get("adapter", "playwright"))
        layout.addWidget(self.adapter_combo)

        # LLM 配置分隔
        llm_sep = QLabel("🤖 LLM 配置")
        llm_sep.setStyleSheet("font-size: 14px; font-weight: bold; color: #192033; margin-top: 10px;")
        layout.addWidget(llm_sep)

        line_edit_style = """
            QLineEdit {
                border: 1px solid #e8ebf2;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #b8afff;
            }
        """

        # Base URL
        lbl = QLabel("Base URL")
        lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
        layout.addWidget(lbl)
        base_input = QLineEdit(str(self._settings.get("llm_base_url", "")))
        base_input.setPlaceholderText("https://api.moonshot.cn/v1")
        base_input.setStyleSheet(line_edit_style)
        self.fields["llm_base_url"] = base_input
        layout.addWidget(base_input)

        # API Key（带显示/隐藏切换）
        lbl = QLabel("API Key")
        lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
        layout.addWidget(lbl)
        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        key_row.setContentsMargins(0, 0, 0, 0)
        self.fields["llm_api_key"] = QLineEdit(str(self._settings.get("llm_api_key", "")))
        self.fields["llm_api_key"].setPlaceholderText("sk-...")
        self.fields["llm_api_key"].setEchoMode(QLineEdit.Password)
        self.fields["llm_api_key"].setStyleSheet(line_edit_style)
        key_row.addWidget(self.fields["llm_api_key"])
        self.toggle_key_btn = QPushButton("显示")
        self.toggle_key_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_key_btn.setStyleSheet("""
            QPushButton {
                background: #eef0f7;
                color: #4d5566;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                font-size: 12px;
            }
        """)
        self.toggle_key_btn.setCheckable(True)
        self.toggle_key_btn.clicked.connect(self._toggle_key_visibility)
        key_row.addWidget(self.toggle_key_btn)
        layout.addLayout(key_row)

        # Model
        lbl = QLabel("Model")
        lbl.setStyleSheet("color: #7b8499; font-size: 12px;")
        layout.addWidget(lbl)
        model_input = QLineEdit(str(self._settings.get("llm_model", "")))
        model_input.setPlaceholderText("moonshot-v1-8k")
        model_input.setStyleSheet(line_edit_style)
        self.fields["llm_model"] = model_input
        layout.addWidget(model_input)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #eef0f7;
                color: #4d5566;
                border: none;
                border-radius: 10px;
                padding: 8px 16px;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("保存")
        save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #8b7cf6);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 16px;
                font-weight: bold;
            }
        """)
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def get_settings(self) -> dict:
        """返回用户确认后的设置字典。"""
        result = {key: widget.text().strip() for key, widget in self.fields.items()}
        result["adapter"] = self.adapter_combo.currentText()
        return result

    def _toggle_key_visibility(self) -> None:
        """切换 API Key 的显示/隐藏。"""
        if self.toggle_key_btn.isChecked():
            self.fields["llm_api_key"].setEchoMode(QLineEdit.Normal)
            self.toggle_key_btn.setText("隐藏")
        else:
            self.fields["llm_api_key"].setEchoMode(QLineEdit.Password)
            self.toggle_key_btn.setText("显示")


# ---------------------------------------------------------------------------
# Web 风格执行控制台
# ---------------------------------------------------------------------------
class CollapsibleLogPanel(QWidget):
    """Web 风格阶段控制台，保留原有 append/clear 调用接口。"""

    _STAGE_MARKERS = [
        (r"={5,}\s*(.+?)\s*={5,}", lambda m: m.group(1).strip()),
        (r"\[新运行\].*", lambda m: "新运行"),
        (r"\[开始\].*", lambda m: "开始"),
        (r"\[完成\].*", lambda m: "完成"),
        (r"\[错误\].*", lambda m: "错误"),
        (r"\[训练后\].*", lambda m: "训练后"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._stages: list[Dict[str, Any]] = []
        self._all_lines: list[str] = []
        self._current_stage: Optional[Dict[str, Any]] = None
        self._selected_stage: Optional[int] = None
        self._ui_scale: Optional[UIScale] = None

        root = QHBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        self._root_layout = root

        rail = QFrame()
        rail.setObjectName("logStageRail")
        self._rail = rail
        rail_layout = QVBoxLayout(rail)
        rail_layout.setSpacing(7)
        rail_layout.setContentsMargins(12, 14, 12, 14)
        self._rail_layout = rail_layout

        rail_title = QLabel("EXECUTION FLOW")
        rail_title.setObjectName("logRailTitle")
        rail_layout.addWidget(rail_title)

        self._all_button = QPushButton("◉  全部输出")
        self._all_button.setObjectName("logStageNavActive")
        self._all_button.setCursor(Qt.PointingHandCursor)
        self._all_button.setFocusPolicy(Qt.NoFocus)
        self._all_button.clicked.connect(lambda: self._select_stage(None))
        rail_layout.addWidget(self._all_button)

        stage_host = QWidget()
        stage_host.setObjectName("logStageHost")
        self._stage_layout = QVBoxLayout(stage_host)
        self._stage_layout.setSpacing(5)
        self._stage_layout.setContentsMargins(0, 0, 0, 0)
        self._stage_layout.setAlignment(Qt.AlignTop)

        stage_scroll = QScrollArea()
        stage_scroll.setObjectName("logStageScroll")
        stage_scroll.setWidgetResizable(True)
        stage_scroll.setFrameShape(QFrame.NoFrame)
        stage_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stage_scroll.setWidget(stage_host)
        rail_layout.addWidget(stage_scroll, 1)

        root.addWidget(rail)

        console = QFrame()
        console.setObjectName("logConsole")
        console_layout = QVBoxLayout(console)
        console_layout.setSpacing(0)
        console_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QFrame()
        toolbar.setObjectName("logConsoleToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 9, 14, 9)
        toolbar_layout.setSpacing(8)
        self._toolbar_layout = toolbar_layout
        self._console_dots: list[QLabel] = []

        for color_name in ("red", "amber", "green"):
            dot = QLabel()
            dot.setObjectName(f"consoleDot{color_name.title()}")
            self._console_dots.append(dot)
            toolbar_layout.addWidget(dot)

        console_title = QLabel("AUTOTRAIN / LIVE CONSOLE")
        console_title.setObjectName("logConsoleTitle")
        toolbar_layout.addSpacing(5)
        toolbar_layout.addWidget(console_title)
        toolbar_layout.addStretch()

        live = QLabel("● LIVE")
        live.setObjectName("logConsoleLive")
        toolbar_layout.addWidget(live)
        console_layout.addWidget(toolbar)

        self._console = QPlainTextEdit()
        self._console.setObjectName("logConsoleOutput")
        self._console.setReadOnly(True)
        self._console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._console.setPlaceholderText(
            "$ 等待任务启动…\n\n训练过程、状态变化和错误信息将在这里实时显示。"
        )
        console_layout.addWidget(self._console, 1)
        root.addWidget(console, 1)

        self.apply_ui_scale(UIScale())

    def apply_ui_scale(self, scale: UIScale) -> None:
        """按当前缩放因子刷新控制台内部固定尺寸。"""
        self._ui_scale = scale
        self._rail.setFixedWidth(scale.px(174, 120))
        self._rail_layout.setSpacing(scale.px(7, 4))
        self._rail_layout.setContentsMargins(
            scale.px(12, 8),
            scale.px(14, 8),
            scale.px(12, 8),
            scale.px(14, 8),
        )
        self._stage_layout.setSpacing(scale.px(5, 3))
        self._toolbar_layout.setContentsMargins(
            scale.px(14, 8),
            scale.px(9, 6),
            scale.px(14, 8),
            scale.px(9, 6),
        )
        self._toolbar_layout.setSpacing(scale.px(8, 4))
        for dot in self._console_dots:
            size = scale.px(8, 6)
            dot.setFixedSize(size, size)
        self.setMinimumHeight(scale.px(300, 180))

    def clear(self) -> None:
        while self._stage_layout.count():
            item = self._stage_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._stages.clear()
        self._all_lines.clear()
        self._current_stage = None
        self._selected_stage = None
        self._console.clear()
        self._refresh_nav_styles()

    def append(self, line: str) -> None:
        title = self._detect_stage_title(line)
        if title:
            self._start_stage(title)

        if self._current_stage is None:
            self._start_stage("日志")

        self._all_lines.append(line)
        self._current_stage["lines"].append(line)

        if self._selected_stage is None:
            self._console.appendPlainText(line)
        elif self._selected_stage == len(self._stages) - 1:
            self._console.appendPlainText(line)

        scrollbar = self._console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _detect_stage_title(self, line: str) -> Optional[str]:
        import re

        for pattern, extractor in self._STAGE_MARKERS:
            match = re.search(pattern, line)
            if match:
                return extractor(match)[:40] or "阶段"
        return None

    def _start_stage(self, title: str) -> None:
        if self._current_stage is not None:
            if self._current_stage["state"] != "error":
                self._current_stage["state"] = "done"
            self._update_stage_button(self._current_stage)

        index = len(self._stages)
        button = QPushButton()
        button.setCursor(Qt.PointingHandCursor)
        button.setFocusPolicy(Qt.NoFocus)
        button.clicked.connect(lambda _checked=False, i=index: self._select_stage(i))

        stage: Dict[str, Any] = {
            "title": title,
            "lines": [],
            "button": button,
            "state": "error" if "错误" in title else "running",
            "index": index,
        }
        self._stages.append(stage)
        self._stage_layout.addWidget(button)
        self._current_stage = stage
        self._update_stage_button(stage)
        self._refresh_nav_styles()

    def _select_stage(self, index: Optional[int]) -> None:
        self._selected_stage = index
        lines = self._all_lines if index is None else self._stages[index]["lines"]
        self._console.setPlainText("\n".join(lines))
        scrollbar = self._console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self._refresh_nav_styles()

    def _refresh_nav_styles(self) -> None:
        self._all_button.setObjectName(
            "logStageNavActive" if self._selected_stage is None else "logStageNav"
        )
        self._repolish(self._all_button)
        for stage in self._stages:
            button = stage["button"]
            selected = stage["index"] == self._selected_stage
            button.setObjectName("logStageNavActive" if selected else "logStageNav")
            self._repolish(button)

    def _update_stage_button(self, stage: Dict[str, Any]) -> None:
        prefix = {"running": "●", "done": "✓", "error": "×"}.get(stage["state"], "○")
        number = stage["index"] + 1
        title = stage["title"].replace("AUTO TRAIN 4.0", "AUTO TRAIN").strip()
        if len(title) > 20:
            title = title[:19] + "…"
        stage["button"].setText(f"{prefix}  {number:02d}  {title}")

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


# ---------------------------------------------------------------------------
# 训练后选择弹窗（蓝紫渐变 + 白卡片，与主界面同风格）
# ---------------------------------------------------------------------------
class PostTrainDialog(QDialog):
    """训练完成后的后续操作选择。

    choice():
        "report"  | 生成 inference report
        "compare" | 与 baseline 对比
        "skip"    | 结束
    """

    def __init__(self, context: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._choice = "skip"
        self._drag_pos: Optional[QPoint] = None

        # 无边框圆角弹窗，避免系统标题栏/图标抢风格
        self.setWindowTitle("训练完成")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedWidth(620)
        self.setMaximumHeight(520)

        self._build_ui(context)
        self._apply_styles()
        self.adjustSize()
        if self.height() > 520:
            self.setFixedHeight(520)
        self._center_on_parent()

    def choice(self) -> str:
        return self._choice

    # -- 拖拽无边框窗口 -----------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            geo = parent.frameGeometry()
            center = geo.center()
            fg = self.frameGeometry()
            fg.moveCenter(center)
            self.move(fg.topLeft())
        else:
            screen = QApplication.desktop().availableGeometry(self)
            fg = self.frameGeometry()
            fg.moveCenter(screen.center())
            self.move(fg.topLeft())

    @staticmethod
    def _display_path(path: Optional[str]) -> tuple:
        """返回 (显示文本, tooltip 完整路径)。路径过长只显示文件名。"""
        if not path:
            return "未找到", ""
        p = Path(path)
        name = p.name or str(path)
        # 短路径直接展示
        if len(path) <= 48:
            return path, path
        return name, path

    def _add_info_row(
        self,
        layout: QVBoxLayout,
        key: str,
        value: str,
        tooltip: str = "",
    ) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)

        k = QLabel(key)
        k.setObjectName("infoKey")
        k.setFixedWidth(72)
        k.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        v = QLabel(value)
        v.setObjectName("infoVal")
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if tooltip:
            v.setToolTip(tooltip)
            v.setCursor(Qt.IBeamCursor)

        row.addWidget(k, 0, Qt.AlignTop)
        row.addWidget(v, 1)
        layout.addLayout(row)

    def _build_ui(self, context: Dict[str, Any]) -> None:
        # 外层透明，真正可见的是 shell（渐变圆角）
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)  # 给阴影留边
        outer.setSpacing(0)

        shell = QFrame()
        shell.setObjectName("shell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(18, 18, 18, 16)
        shell_layout.setSpacing(12)

        # 阴影
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 60))
        shell.setGraphicsEffect(shadow)

        # 顶栏：标题 + 关闭
        top = QHBoxLayout()
        top.setContentsMargins(4, 0, 0, 0)
        title = QLabel("训练已完成")
        title.setObjectName("dlgTitle")
        top.addWidget(title)
        top.addStretch()

        btn_x = QPushButton("×")
        btn_x.setObjectName("closeX")
        btn_x.setFixedSize(28, 28)
        btn_x.setCursor(Qt.PointingHandCursor)
        btn_x.setToolTip("关闭")
        btn_x.clicked.connect(lambda: self._select("skip"))
        top.addWidget(btn_x)
        shell_layout.addLayout(top)

        subtitle = QLabel("接下来要做什么？")
        subtitle.setObjectName("dlgSubtitle")
        shell_layout.addWidget(subtitle)

        # 白色信息卡片
        card = QFrame()
        card.setObjectName("infoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(8)

        new_show, new_tip = self._display_path(context.get("new_dvp"))
        ds_show, ds_tip = self._display_path(context.get("dataset_dir"))
        base_raw = context.get("baseline_dvp")
        if base_raw:
            base_show, base_tip = self._display_path(base_raw)
        else:
            base_show, base_tip = "未找到（对比不可用）", ""

        self._add_info_row(card_layout, "任务类型", context.get("task_type") or "未知")
        self._add_info_row(card_layout, "新模型", new_show, new_tip)
        self._add_info_row(card_layout, "测试集", ds_show, ds_tip)
        self._add_info_row(card_layout, "Baseline", base_show, base_tip)

        shell_layout.addWidget(card)

        hint = QLabel("路径过长时仅显示文件名，鼠标悬停可查看完整路径")
        hint.setObjectName("hint")
        shell_layout.addWidget(hint)

        # 底部三按钮横向
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_report = QPushButton("生成 inference report")
        btn_report.setObjectName("btnPrimary")
        btn_report.setCursor(Qt.PointingHandCursor)
        btn_report.setMinimumHeight(38)
        btn_report.clicked.connect(lambda: self._select("report"))

        btn_compare = QPushButton("与 baseline 对比")
        btn_compare.setObjectName("btnSecondary")
        btn_compare.setCursor(Qt.PointingHandCursor)
        btn_compare.setMinimumHeight(38)
        has_baseline = bool(context.get("baseline_dvp"))
        btn_compare.setEnabled(has_baseline)
        if not has_baseline:
            btn_compare.setToolTip("未找到对应 baseline，无法对比")
        btn_compare.clicked.connect(lambda: self._select("compare"))

        btn_skip = QPushButton("结束")
        btn_skip.setObjectName("btnGhost")
        btn_skip.setCursor(Qt.PointingHandCursor)
        btn_skip.setMinimumHeight(38)
        btn_skip.setFixedWidth(88)
        btn_skip.clicked.connect(lambda: self._select("skip"))

        btn_row.addWidget(btn_report, 2)
        btn_row.addWidget(btn_compare, 2)
        btn_row.addWidget(btn_skip, 0)
        shell_layout.addLayout(btn_row)

        outer.addWidget(shell)

    def _select(self, action: str) -> None:
        self._choice = action
        if action == "skip":
            self.reject()
        else:
            self.accept()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QFrame#shell {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea,
                    stop:0.55 #764ba2,
                    stop:1 #f093fb
                );
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.35);
            }
            QLabel#dlgTitle {
                font-size: 21px;
                font-weight: bold;
                color: #ffffff;
                letter-spacing: 0.5px;
            }
            QLabel#dlgSubtitle {
                font-size: 13px;
                color: rgba(255, 255, 255, 0.92);
                padding-left: 2px;
            }
            QLabel#hint {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.75);
                padding-left: 2px;
            }
            QFrame#infoCard {
                background: rgba(255, 255, 255, 0.94);
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.65);
            }
            QLabel#infoKey {
                font-size: 12px;
                font-weight: bold;
                color: #6d28d9;
            }
            QLabel#infoVal {
                font-size: 12px;
                color: #374151;
            }
            QPushButton#closeX {
                background: rgba(255, 255, 255, 0.22);
                color: #ffffff;
                border: none;
                border-radius: 14px;
                font-size: 16px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton#closeX:hover {
                background: rgba(255, 255, 255, 0.38);
            }
            QPushButton#btnPrimary {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea,
                    stop:1 #764ba2
                );
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton#btnPrimary:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a6fd6,
                    stop:1 #6a4191
                );
            }
            QPushButton#btnPrimary:pressed {
                background: #5b21b6;
            }
            QPushButton#btnSecondary {
                background: #ffffff;
                color: #6d28d9;
                font-size: 14px;
                font-weight: bold;
                border: 1.5px solid #7c3aed;
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton#btnSecondary:hover {
                background: #f5f3ff;
                border: 1.5px solid #6d28d9;
                color: #5b21b6;
            }
            QPushButton#btnSecondary:disabled {
                background: #f3f4f6;
                color: #9ca3af;
                border: 1.5px solid #e5e7eb;
            }
            QPushButton#btnGhost {
                background: rgba(255, 255, 255, 0.55);
                color: #4b5563;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid rgba(255, 255, 255, 0.75);
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton#btnGhost:hover {
                background: rgba(255, 255, 255, 0.85);
                color: #1f2937;
            }
            """
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoTrain 4.0")
        self.setMinimumSize(780, 600)
        self.resize(UIScale.REF_W, UIScale.REF_H)
        self.worker: Optional[TrainWorker] = None
        self.post_worker: Optional[PostTrainWorker] = None
        self._ui_scale = UIScale()
        self._last_compact = False

        from autotrain4.llm_client import LLMClient

        self.suggestion_engine = SuggestionEngine(llm_client=LLMClient())
        self.suggestion_engine.on_llm_result(self._on_llm_suggestions)

        self._ime_composing = False
        self._suggestion_debounce = QTimer(self)
        self._suggestion_debounce.setSingleShot(True)
        self._suggestion_debounce.timeout.connect(self._do_update_suggestions)
        self._scale_debounce = QTimer(self)
        self._scale_debounce.setSingleShot(True)
        self._scale_debounce.timeout.connect(self._apply_responsive_scale)

        # 空输入时点击「刷新建议」会轮换这几组默认模板
        self._default_suggestion_sets = [
            ["训练 OCR 模型", "训练 CLS 模型", "训练 DET 模型", "训练 DETR 模型", "训练 ISEG 模型"],
            ["使用随机旋转增强", "训练完成后自动测试", "对比 Baseline 模型", "导出 ONNX", "Epoch 设置为 20"],
            ["不使用数据增强", "生成推理报告", "批量训练多个模型", "设置 batch_size=8", "训练完成后关闭浏览器"],
        ]
        self._current_suggestion_set_index = 0

        # 快捷模板：默认 + 用户自定义（持久化到 JSON），最多 8 个
        self.MAX_QUICK_TEMPLATES = 8
        self._quick_templates_file = Path(__file__).parent / "quick_templates.json"
        self._quick_templates = self._load_quick_templates()

        # 系统设置持久化
        self._settings_file = Path(__file__).parent / "settings.json"
        self._settings = self._load_settings()

        self._setup_ui()
        self._apply_responsive_scale(force=True)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # 拖拽缩放时防抖，避免每像素重算样式
        self._scale_debounce.start(40)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_responsive_scale(force=True)

    def _apply_responsive_scale(self, force: bool = False) -> None:
        """窗口尺寸变化时，等比例缩放字体、间距、固定控件尺寸。"""
        changed = self._ui_scale.update_from_size(self.width(), self.height())
        compact = self.width() < 1080
        if not force and not changed and compact == self._last_compact:
            return
        self._last_compact = compact
        self._apply_layout_metrics(compact)
        self._apply_styles()
        if hasattr(self, "log_panel"):
            self.log_panel.apply_ui_scale(self._ui_scale)
        if hasattr(self, "chat_response_box"):
            self._apply_chat_box_style()
        # 模板卡片内部 spacing 也随缩放
        self._refresh_template_item_metrics()

    def _apply_layout_metrics(self, compact: bool) -> None:
        s = self._ui_scale
        gap = s.px(16 if compact else 22, 10)
        side = s.px(200 if compact else 250, 160)

        if hasattr(self, "_sidebar"):
            self._sidebar.setFixedWidth(side)
            self._sidebar_layout.setSpacing(s.px(18 if compact else 24, 10))
            self._sidebar_layout.setContentsMargins(
                s.px(14 if compact else 18, 8),
                s.px(18 if compact else 24, 10),
                s.px(14 if compact else 18, 8),
                s.px(18 if compact else 24, 10),
            )
        if hasattr(self, "_brand_mark"):
            mark = s.px(34 if compact else 38, 28)
            self._brand_mark.setFixedSize(mark, mark)
        if hasattr(self, "_nav_layout"):
            self._nav_layout.setSpacing(s.px(6 if compact else 8, 4))
        if hasattr(self, "_status_layout"):
            self._status_layout.setSpacing(s.px(6 if compact else 8, 4))

        if hasattr(self, "_main_vlayout"):
            self._main_vlayout.setSpacing(gap)
            self._main_vlayout.setContentsMargins(
                s.px(18 if compact else 28, 12),
                s.px(16 if compact else 24, 10),
                s.px(20 if compact else 34, 12),
                s.px(20 if compact else 32, 12),
            )
        if hasattr(self, "_content_layout"):
            self._content_layout.setSpacing(gap)
        if hasattr(self, "_topbar_layout"):
            self._topbar_layout.setSpacing(s.px(10 if compact else 12, 6))
        if hasattr(self, "_left_col_layout"):
            self._left_col_layout.setSpacing(gap)
        if hasattr(self, "_right_col_layout"):
            self._right_col_layout.setSpacing(gap)
        if hasattr(self, "_prompt_card_layout"):
            self._prompt_card_layout.setSpacing(s.px(12 if compact else 16, 8))
        if hasattr(self, "prompt_input"):
            self.prompt_input.setMaximumHeight(s.px(100 if compact else 125, 72))
            self.prompt_input.setMinimumHeight(s.px(72 if compact else 90, 56))
        if hasattr(self, "suggestions_container"):
            self.suggestions_container.setMinimumHeight(s.px(30 if compact else 36, 24))
        if hasattr(self, "suggestions_layout"):
            self.suggestions_layout.setSpacing(s.px(8 if compact else 10, 4))
        if hasattr(self, "_sugg_header_layout"):
            self._sugg_header_layout.setSpacing(s.px(6 if compact else 8, 4))
        if hasattr(self, "_btn_row_layout"):
            self._btn_row_layout.setSpacing(s.px(8 if compact else 12, 6))
        if hasattr(self, "_metrics_layout"):
            self._metrics_layout.setSpacing(s.px(8 if compact else 12, 6))
        for layout in getattr(self, "_metric_item_layouts", []):
            layout.setSpacing(s.px(4 if compact else 6, 2))
        if hasattr(self, "_tpl_grid_layout") and self._tpl_grid_layout is not None:
            self._tpl_grid_layout.setSpacing(s.px(8 if compact else 10, 4))
        if hasattr(self, "chat_response_box"):
            self.chat_response_box.setMinimumHeight(s.px(120 if compact else 150, 90))

    def _refresh_template_item_metrics(self) -> None:
        if not hasattr(self, "quick_templates_grid"):
            return
        s = self._ui_scale
        layout = self.quick_templates_grid.layout()
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            inner = widget.layout()
            if inner is not None:
                inner.setSpacing(s.px(4, 2))
                inner.setContentsMargins(
                    s.px(8, 4),
                    s.px(8, 4),
                    s.px(8, 4),
                    s.px(8, 4),
                )

    def _apply_chat_box_style(self) -> None:
        s = self._ui_scale
        self.chat_response_box.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background: #f7f5ff;
                border: 1px solid #e6e1ff;
                border-radius: {s.qss_px(14, 8)};
                padding: {s.qss_px(14, 8)};
                font-size: {s.font(15)}px;
                color: #4c1d95;
                line-height: 1.7;
            }}
            """
        )

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        self._root_layout = root

        # ---------------- 左侧边栏 ----------------
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # ---------------- 主内容区 ----------------
        main_area = QWidget()
        main_area.setObjectName("mainArea")
        main_vlayout = QVBoxLayout(main_area)
        main_vlayout.setSpacing(22)
        main_vlayout.setContentsMargins(28, 24, 34, 32)
        self._main_vlayout = main_vlayout

        # 顶栏
        topbar = self._build_topbar()
        main_vlayout.addWidget(topbar)

        # 两列内容
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setSpacing(22)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setAlignment(Qt.AlignTop)
        self._content_layout = content_layout

        left_col = self._build_left_column()
        content_layout.addWidget(left_col, 2)

        right_col = self._build_right_column()
        content_layout.addWidget(right_col, 1)

        main_vlayout.addWidget(content, 1)
        root.addWidget(main_area, 1)

        # 初始化建议气泡
        self._do_update_suggestions()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        self._sidebar = sidebar
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(24)
        layout.setContentsMargins(18, 24, 18, 24)
        self._sidebar_layout = layout

        # 品牌
        brand = QHBoxLayout()
        brand_mark = QLabel("AT")
        brand_mark.setObjectName("brandMark")
        brand_mark.setAlignment(Qt.AlignCenter)
        brand_mark.setFixedSize(38, 38)
        self._brand_mark = brand_mark
        brand.addWidget(brand_mark)

        brand_text = QVBoxLayout()
        brand_title = QLabel("AutoTrain 4.0")
        brand_title.setObjectName("brandTitle")
        brand_sub = QLabel("DLCV AI Automation")
        brand_sub.setObjectName("brandSub")
        brand_text.addWidget(brand_title)
        brand_text.addWidget(brand_sub)
        brand.addLayout(brand_text)
        brand.addStretch()
        layout.addLayout(brand)

        # 导航
        nav = QVBoxLayout()
        nav.setSpacing(8)
        self._nav_layout = nav
        nav_items = [
            ("⌂", "工作台", True, None),
            ("⚙", "系统设置", False, self._open_settings),
        ]
        for icon, name, active, handler in nav_items:
            btn = QPushButton(f"{icon}  {name}")
            btn.setObjectName("navBtnActive" if active else "navBtn")
            btn.setCursor(Qt.PointingHandCursor)
            if handler is not None:
                btn.clicked.connect(handler)
            nav.addWidget(btn)
        layout.addLayout(nav)

        layout.addStretch()

        # 状态卡片
        status_card = QFrame()
        status_card.setObjectName("sideCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setSpacing(8)
        self._status_layout = status_layout
        status_title = QLabel("平台状态")
        status_title.setObjectName("sideCardTitle")
        status_desc = QLabel("Backend 9887 与 Frontend 8842 均已连接。")
        status_desc.setObjectName("sideCardDesc")
        status_desc.setWordWrap(True)
        status_badge = QLabel("运行正常")
        status_badge.setObjectName("sideCardBadge")
        status_layout.addWidget(status_title)
        status_layout.addWidget(status_desc)
        status_layout.addWidget(status_badge)
        layout.addWidget(status_card)

        return sidebar

    def _build_topbar(self) -> QWidget:
        topbar = QWidget()
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setSpacing(12)
        topbar_layout.setContentsMargins(0, 0, 0, 0)
        self._topbar_layout = topbar_layout

        title_col = QVBoxLayout()
        title = QLabel("自动化训练工作台")
        title.setObjectName("pageTitle")
        subtitle = QLabel("用自然语言创建、监控并管理你的视觉训练任务。")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        topbar_layout.addLayout(title_col, 1)

        topbar_layout.addStretch()

        doc_btn = QPushButton("查看文档")
        doc_btn.setObjectName("ghostBtn")
        doc_btn.setCursor(Qt.PointingHandCursor)
        doc_btn.clicked.connect(self._open_readme)
        topbar_layout.addWidget(doc_btn)

        return topbar

    def _build_left_column(self) -> QWidget:
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setSpacing(22)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setAlignment(Qt.AlignTop)
        self._left_col_layout = col_layout

        # 训练需求卡片
        prompt_card = QFrame()
        prompt_card.setObjectName("card")
        card_layout = QVBoxLayout(prompt_card)
        card_layout.setSpacing(16)
        self._prompt_card_layout = card_layout

        header = QHBoxLayout()
        header_title = QLabel("训练需求")
        header_title.setObjectName("cardTitle")
        header.addWidget(header_title)
        header.addStretch()
        tag = QLabel("AI Intent")
        tag.setObjectName("tag")
        header.addWidget(tag)
        card_layout.addLayout(header)

        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText(
            "例如：训练一个 cls 模型来辨别王不留行和白芷"
        )
        self.prompt_input.setMaximumHeight(125)
        self.prompt_input.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.prompt_input.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.prompt_input.setInputMethodHints(Qt.ImhNone)
        self.prompt_input.textChanged.connect(self._on_input_text_changed)
        self.prompt_input.installEventFilter(self)
        card_layout.addWidget(self.prompt_input)

        sugg_header = QHBoxLayout()
        sugg_header.setSpacing(8)
        self._sugg_header_layout = sugg_header
        sugg_title = QLabel("猜你想说")
        sugg_title.setObjectName("sectionTitle")
        sugg_header.addWidget(sugg_title)
        sugg_header.addStretch()
        refresh_btn = QPushButton("刷新建议")
        refresh_btn.setObjectName("tinyGhostBtn")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_suggestions)
        sugg_header.addWidget(refresh_btn)
        card_layout.addLayout(sugg_header)

        self.suggestions_container = QWidget()
        self.suggestions_container.setObjectName("suggestionsContainer")
        self.suggestions_container.setMinimumHeight(36)
        self.suggestions_layout = QHBoxLayout(self.suggestions_container)
        self.suggestions_layout.setSpacing(10)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestions_layout.setAlignment(Qt.AlignLeft)
        card_layout.addWidget(self.suggestions_container)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._btn_row_layout = btn_row
        self.start_btn = QPushButton("▶ 开始训练")
        self.start_btn.setObjectName("runBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_run)

        self.clear_btn = QPushButton("清空任务")
        self.clear_btn.setObjectName("softBtn")
        self.clear_btn.clicked.connect(self._clear_logs)

        save_tpl_btn = QPushButton("保存为模板")
        save_tpl_btn.setObjectName("secondaryBtn")
        save_tpl_btn.clicked.connect(self._save_current_as_template)

        btn_row.addWidget(self.start_btn, 2)
        btn_row.addWidget(self.clear_btn, 1)
        btn_row.addWidget(save_tpl_btn, 1)
        card_layout.addLayout(btn_row)

        # 指标概览
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self._metrics_layout = metrics
        self._metric_item_layouts: list = []
        self.metric_labels = {}
        for key, label in [
            ("task_type", "任务类型"),
            ("dataset", "匹配数据集"),
            ("estimate", "预计耗时"),
        ]:
            m = QFrame()
            m.setObjectName("metric")
            mv = QVBoxLayout(m)
            mv.setSpacing(6)
            self._metric_item_layouts.append(mv)
            ms = QLabel(label)
            ms.setObjectName("metricLabel")
            mv.addWidget(ms)
            mv_val = QLabel("-")
            mv_val.setObjectName("metricValue")
            mv_val.setWordWrap(True)
            self.metric_labels[key] = mv_val
            mv.addWidget(mv_val)
            metrics.addWidget(m)
        card_layout.addLayout(metrics)

        col_layout.addWidget(prompt_card)

        # 运行日志卡片
        log_card = QFrame()
        log_card.setObjectName("card")
        log_layout = QVBoxLayout(log_card)
        log_header = QHBoxLayout()
        log_title = QLabel("执行控制台")
        log_title.setObjectName("cardTitle")
        log_header.addWidget(log_title)
        log_header.addStretch()
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusBadge")
        log_header.addWidget(self.status_label)
        log_layout.addLayout(log_header)

        self.log_panel = CollapsibleLogPanel()
        self.log_panel.setObjectName("logPanel")
        self.log_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_panel)
        col_layout.addWidget(log_card, 1)

        return col

    def _build_right_column(self) -> QWidget:
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setSpacing(22)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setAlignment(Qt.AlignTop)
        self._right_col_layout = col_layout

        # 快捷模板
        tpl_card = QFrame()
        tpl_card.setObjectName("card")
        tpl_layout = QVBoxLayout(tpl_card)
        tpl_header = QHBoxLayout()
        tpl_title = QLabel("快捷模板")
        tpl_title.setObjectName("cardTitle")
        tpl_header.addWidget(tpl_title)
        tpl_header.addStretch()
        manage_btn = QPushButton("管理")
        manage_btn.setObjectName("tinyGhostBtn")
        manage_btn.setCursor(Qt.PointingHandCursor)
        manage_btn.clicked.connect(self._open_template_manager)
        tpl_header.addWidget(manage_btn)
        tpl_tag = QLabel("效率工具")
        tpl_tag.setObjectName("tag")
        tpl_header.addWidget(tpl_tag)
        tpl_layout.addLayout(tpl_header)

        self.quick_templates_grid = QWidget()
        self._tpl_grid_layout = None
        self._render_quick_templates()
        tpl_layout.addWidget(self.quick_templates_grid)
        col_layout.addWidget(tpl_card)

        # 本次任务
        task_card = QFrame()
        task_card.setObjectName("card")
        task_layout = QVBoxLayout(task_card)
        task_header = QHBoxLayout()
        task_title = QLabel("本次任务")
        task_title.setObjectName("cardTitle")
        task_header.addWidget(task_title)
        task_header.addStretch()
        self.task_status = QLabel("未开始")
        self.task_status.setObjectName("statusBadgeWaiting")
        task_header.addWidget(self.task_status)
        task_layout.addLayout(task_header)

        self.close_browser_btn = QPushButton("关闭浏览器")
        self.close_browser_btn.setObjectName("closeBrowserBtn")
        self.close_browser_btn.setEnabled(False)
        self.close_browser_btn.setCursor(Qt.PointingHandCursor)
        self.close_browser_btn.clicked.connect(self._close_browser)
        task_layout.addWidget(self.close_browser_btn)
        col_layout.addWidget(task_card)

        # 助手回复卡片（永久显示在右侧）
        self.chat_response_card = QFrame()
        self.chat_response_card.setObjectName("card")
        chat_layout = QVBoxLayout(self.chat_response_card)
        chat_header = QHBoxLayout()
        chat_title = QLabel("💬 助手回复")
        chat_title.setObjectName("cardTitle")
        chat_header.addWidget(chat_title)
        chat_header.addStretch()
        chat_tag = QLabel("通用对话")
        chat_tag.setObjectName("tag")
        chat_header.addWidget(chat_tag)
        chat_layout.addLayout(chat_header)

        self.chat_response_box = QPlainTextEdit()
        self.chat_response_box.setReadOnly(True)
        self.chat_response_box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.chat_response_box.setPlaceholderText("暂无回复，输入闲聊或问题后将在此显示助手回答。")
        self._apply_chat_box_style()
        chat_layout.addWidget(self.chat_response_box)
        col_layout.addWidget(self.chat_response_card)

        return col

    def _render_quick_templates(self) -> None:
        """根据 self._quick_templates 重新渲染快捷模板网格。"""
        # 如果已有布局，清空旧子控件，复用该布局
        existing_layout = self.quick_templates_grid.layout()
        if existing_layout is not None:
            while existing_layout.count():
                item = existing_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            grid_layout = existing_layout
        else:
            grid_layout = QGridLayout(self.quick_templates_grid)
            grid_layout.setSpacing(10)
            grid_layout.setContentsMargins(0, 0, 0, 0)
        self._tpl_grid_layout = grid_layout
        s = getattr(self, "_ui_scale", UIScale())

        for i, (title, desc, prompt) in enumerate(self._quick_templates[: self.MAX_QUICK_TEMPLATES]):
            item = ClickableFrame(i)
            item.setObjectName("quickItem")
            iv = QVBoxLayout(item)
            iv.setSpacing(s.px(4, 2))
            iv.setContentsMargins(s.px(8, 4), s.px(8, 4), s.px(8, 4), s.px(8, 4))
            it = QLabel(title)
            it.setObjectName("quickTitle")
            it.setWordWrap(True)
            idesc = QLabel(desc)
            idesc.setObjectName("quickDesc")
            idesc.setWordWrap(True)
            iv.addWidget(it)
            iv.addWidget(idesc)
            item.clicked.connect(self._apply_quick_template)
            grid_layout.addWidget(item, i // 2, i % 2)

    def _load_quick_templates(self) -> list:
        """从 JSON 加载快捷模板，没有则返回默认模板。"""
        defaults = [
            ("OCR 基础训练", "标准参数", "训练一个 OCR 模型，使用标准参数"),
            ("DET 快速实验", "低 Epoch", "训练一个 DET 目标检测模型，epoch 设为 5"),
            ("Baseline 对比", "自动评估", "训练一个 CLS 分类模型并对比 baseline"),
            ("批量任务", "多模型队列", "依次训练 OCR、CLS、DET 三个模型"),
        ]
        if not self._quick_templates_file.exists():
            return defaults
        try:
            data = json.loads(self._quick_templates_file.read_text(encoding="utf-8"))
            loaded = [
                (item.get("title", ""), item.get("desc", ""), item.get("prompt", ""))
                for item in data
                if item.get("title") and item.get("prompt")
            ]
            return loaded if loaded else defaults
        except Exception:
            return defaults

    def _save_quick_templates(self) -> None:
        """把当前模板列表持久化到 JSON。"""
        try:
            data = [
                {"title": t, "desc": d, "prompt": p}
                for t, d, p in self._quick_templates
            ]
            self._quick_templates_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存模板：{e}")

    def _load_settings(self) -> dict:
        """从 JSON 加载系统设置，没有则返回默认值；LLM 配置会额外回退读取环境变量。"""
        defaults = {
            "backend_url": "http://127.0.0.1:9887",
            "frontend_url": "http://localhost:8842",
            "platform_exe": r"C:\dlcv\Lib\site-packages\dlcv_train\DLCV AI Training Platform-win_x64.exe",
            "cdp_port": "9222",
            "adapter": "playwright",
            "llm_base_url": "https://api.moonshot.cn/v1",
            "llm_api_key": "",
            "llm_model": "moonshot-v1-8k",
        }

        # 从环境变量读取当前实际生效的 LLM 配置
        env_llm = {
            "llm_base_url": os.environ.get("AUTOTRAIN_LLM_BASE_URL", ""),
            "llm_api_key": os.environ.get("AUTOTRAIN_LLM_API_KEY", ""),
            "llm_model": os.environ.get("AUTOTRAIN_LLM_MODEL", ""),
        }
        # DeepSeek 专用环境变量兜底
        if not env_llm["llm_api_key"]:
            env_llm["llm_api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
        if not env_llm["llm_model"]:
            env_llm["llm_model"] = os.environ.get("DEEPSEEK_MODEL", "")
        if not env_llm["llm_base_url"] and env_llm["llm_api_key"] and "deepseek" in env_llm["llm_api_key"].lower():
            env_llm["llm_base_url"] = "https://api.deepseek.com"

        # 过滤掉空值，避免覆盖有意义的默认值
        env_llm = {k: v for k, v in env_llm.items() if v}
        defaults.update(env_llm)

        if not self._settings_file.exists():
            return defaults
        try:
            data = json.loads(self._settings_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # JSON 中显式保存的值优先级最高
                merged = {**defaults, **data}
                return merged
            return defaults
        except Exception:
            return defaults

    def _save_settings(self) -> None:
        """把当前设置持久化到 JSON。"""
        try:
            self._settings_file.write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存设置：{e}")

    def _open_settings(self) -> None:
        """打开系统设置弹窗。"""
        dialog = SettingsDialog(self._settings, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._settings = dialog.get_settings()
            self._save_settings()

    def _apply_quick_template(self, index: int) -> None:
        """点击快捷模板时，把对应 prompt 填入输入框。"""
        if 0 <= index < len(self._quick_templates):
            _, _, prompt = self._quick_templates[index]
            self.prompt_input.setPlainText(prompt)
            self._update_metrics_from_prompt()
            self._do_update_suggestions()

    def _save_current_as_template(self) -> None:
        """把当前输入框内容保存为新的快捷模板，弹出弹窗让用户自定义标题。"""
        text = self.prompt_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请先输入训练需求再保存为模板")
            return

        if len(self._quick_templates) >= self.MAX_QUICK_TEMPLATES:
            QMessageBox.warning(
                self,
                "模板数量已达上限",
                f"最多保存 {self.MAX_QUICK_TEMPLATES} 个模板，请通过「管理」删除不再使用的模板。",
            )
            return

        suggested_title = text[:12] + ("..." if len(text) > 12 else "")
        dialog = SaveTemplateDialog(text, suggested_title, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        template = dialog.get_template()
        if template is None:
            QMessageBox.information(self, "提示", "标题和 Prompt 不能为空")
            return

        self._quick_templates.append(template)
        self._save_quick_templates()
        self._render_quick_templates()
        QMessageBox.information(self, "保存成功", f"已保存模板：{template[0]}")

    def _open_template_manager(self) -> None:
        """打开快捷模板管理弹窗，支持查看和删除。"""
        dialog = QuickTemplatesDialog(self._quick_templates, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._quick_templates = dialog.get_templates()
            self._save_quick_templates()
            self._render_quick_templates()

    def _open_readme(self) -> None:
        """读取项目 README.md，转换为 HTML 后在弹窗中展示。"""
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            QMessageBox.warning(self, "查看文档", "未找到 README.md")
            return

        try:
            md_text = readme_path.read_text(encoding="utf-8")
            html_body = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
            # 某些 QTextEdit 渲染会把相邻 <pre> 拆成多条，先合并
            html_body = self._merge_adjacent_pre_blocks(html_body)
        except Exception as e:
            QMessageBox.warning(self, "查看文档", f"读取文档失败：{e}")
            return

        self._show_documentation_dialog(html_body)

    @staticmethod
    def _merge_adjacent_pre_blocks(html: str) -> str:
        """合并相邻的 <pre>...</pre> 代码块为一块，避免一行一条黑条。"""
        def merge_match(m: re.Match) -> str:
            inner = m.group(1).strip("\n")
            # 去掉中间 </pre>...<pre> 的标签残留
            inner = re.sub(r"</code>\s*</pre>\s*<pre>\s*<code[^>]*>", "\n", inner)
            inner = re.sub(r"</pre>\s*<pre>", "\n", inner)
            return f"<pre><code>{inner}</code></pre>"

        # 匹配由 </pre> 和 <pre> 连接的连续代码区域
        pattern = re.compile(
            r"<pre>\s*<code[^>]*>(.*?)</code>\s*</pre>(?:\s*<pre>\s*<code[^>]*>.*?</code>\s*</pre>)+",
            re.DOTALL,
        )
        return pattern.sub(merge_match, html)

    def _show_documentation_dialog(self, html_body: str) -> None:
        """用 QTextEdit 展示渲染后的文档 HTML。"""
        dialog = QDialog(self)
        dialog.setWindowTitle("AutoTrain 4.0 使用文档")
        dialog.setMinimumSize(760, 600)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        header = QLabel("📖 使用文档")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #192033;")
        layout.addWidget(header)

        # 文档内容
        doc_edit = QTextEdit()
        doc_edit.setReadOnly(True)
        doc_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e8ebf2;
                border-radius: 16px;
                background: #ffffff;
                padding: 16px;
                font-size: 14px;
                line-height: 1.6;
            }
        """)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
                color: #192033;
                line-height: 1.7;
            }}
            h1, h2, h3 {{
                color: #6c5ce7;
                font-weight: 700;
                margin-top: 1.4em;
                margin-bottom: 0.5em;
            }}
            h1 {{ font-size: 28px; border-bottom: 2px solid #f0edff; padding-bottom: 8px; }}
            h2 {{ font-size: 22px; }}
            h3 {{ font-size: 18px; }}
            a {{ color: #6c5ce7; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            code {{
                background: #f4f6fb;
                padding: 2px 6px;
                border-radius: 5px;
                font-family: Consolas, "SFMono-Regular", monospace;
                color: #db5a65;
                font-size: 13px;
            }}
            pre {{
                background: #f4f6fb;
                color: #1f2937;
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid #e8ebf2;
                overflow-x: auto;
                font-family: Consolas, "SFMono-Regular", monospace;
                font-size: 13px;
                line-height: 1.6;
                margin: 8px 0;
                white-space: pre-wrap;
                word-break: break-all;
            }}
            pre code {{
                background: transparent;
                color: inherit;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 12px 0;
            }}
            th, td {{
                border: 1px solid #e8ebf2;
                padding: 10px 12px;
                text-align: left;
            }}
            th {{
                background: #f0edff;
                color: #6c5ce7;
                font-weight: 700;
            }}
            tr:nth-child(even) {{ background: #f8f9fc; }}
            ul, ol {{ padding-left: 24px; }}
            li {{ margin: 6px 0; }}
            hr {{
                border: none;
                border-top: 1px solid #e8ebf2;
                margin: 24px 0;
            }}
            blockquote {{
                border-left: 4px solid #c4b5fd;
                margin: 12px 0;
                padding: 8px 16px;
                background: #f7f5ff;
                color: #5347b4;
            }}
        </style>
        </head>
        <body>
        {html_body}
        </body>
        </html>
        """
        doc_edit.setHtml(html)
        layout.addWidget(doc_edit, 1)

        close_btn = QPushButton("关闭")
        close_btn.setObjectName("secondaryBtn")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dialog.exec_()

    def _apply_styles(self) -> None:
        """应用 Web 前端风格主题；字号/圆角/内边距随窗口等比例缩放。"""
        s = getattr(self, "_ui_scale", None) or UIScale()
        f = s.font
        p = s.qss_px

        self.setStyleSheet(
            f"""
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
            QLabel#statusBadge,
            QLabel#statusBadgeWaiting {{
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

            /* 关闭浏览器 */
            QPushButton#closeBrowserBtn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6c5ce7, stop:1 #8b7cf6);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: {p(12, 8)};
                padding: {p(12, 8)};
                font-size: {f(14)}px;
            }}
            QPushButton#closeBrowserBtn:disabled {{
                background: #c4b5fd;
            }}
            """
        )

        # 显式设置输入框调色板，确保 IME 预编辑拼音颜色正常
        if hasattr(self, "prompt_input"):
            palette = self.prompt_input.palette()
            palette.setColor(QPalette.Base, QColor("#fbfcff"))
            palette.setColor(QPalette.Text, QColor("#1f2937"))
            palette.setColor(QPalette.PlaceholderText, QColor("#9ca3af"))
            palette.setColor(QPalette.Highlight, QColor("#c4b5fd"))
            palette.setColor(QPalette.HighlightedText, QColor("#1f2937"))
            self.prompt_input.setPalette(palette)

    def _on_input_text_changed(self) -> None:
        """输入框内容变化时触发防抖。中文 IME 组字期间不刷新，英文输入实时刷新。"""
        self._suggestion_debounce.stop()
        self._update_metrics_from_prompt()
        if self._ime_composing:
            return
        self._suggestion_debounce.start(400)

    def _update_metrics_from_prompt(self) -> None:
        """根据 prompt 内容更新任务类型和预计耗时指标；清空输入时不重置。"""
        text = self.prompt_input.toPlainText().strip()
        if not text:
            return

        # 任务类型识别
        upper = text.upper()
        task_types = ["DETR", "ISEG", "SSEG", "OCR", "CLS", "DET"]
        detected = None
        for t in task_types:
            if t in upper:
                detected = t
                break
        if detected:
            self.metric_labels["task_type"].setText(detected)

        # 预计耗时：根据类型给一个经验值
        estimates = {
            "OCR": "约 3 分钟",
            "CLS": "约 4 分钟",
            "DET": "约 6 分钟",
            "DETR": "约 8 分钟",
            "ISEG": "约 10 分钟",
            "SSEG": "约 10 分钟",
        }
        if detected:
            self.metric_labels["estimate"].setText(estimates.get(detected, "-"))

    def _update_metrics_from_log(self, line: str) -> None:
        """根据日志输出更新任务类型、匹配数据集、预计耗时指标。"""
        # 任务类型：优先精确匹配 task_type=
        m = re.search(r"task_type\s*=\s*([A-Z]+)", line)
        if m:
            self.metric_labels["task_type"].setText(m.group(1))

        # 任务类型：从日志里的中文任务描述推断
        task_type_map = {
            "分类": "CLS",
            "目标检测": "DET",
            "实例分割": "ISEG",
            "语义分割": "SSEG",
            "文字识别": "OCR",
            "文本识别": "OCR",
            "旋转框": "ROTATE",
            "关键点": "KPT",
        }
        for cn, en in task_type_map.items():
            if cn in line:
                self.metric_labels["task_type"].setText(en)
                break

        # 任务类型：从项目名 / URL / project_id 里的类型代码推断
        m = re.search(r"[/-](\d-)?(OCR|CLS|DET|DETR|ISEG|SSEG|ROTATE)[/_-]", line.upper())
        if m:
            self.metric_labels["task_type"].setText(m.group(2))

        # 匹配数据集：从 dataset_dir 路径里提取
        m = re.search(r"dataset_dir\s*=\s*(.+?)(?:\s|$)", line)
        if m:
            path = Path(m.group(1).strip().strip('"'))
            if path.name == "训练集" or path.name == "train":
                ds_name = path.parent.name
            else:
                ds_name = path.name
            self.metric_labels["dataset"].setText(ds_name)

        # 匹配数据集：从 "创建项目: X-CLS-..." 提取
        m = re.search(r"创建项目[:：]\s*([\w\-]+)", line)
        if m:
            self.metric_labels["dataset"].setText(m.group(1))

        # 训练进度 -> 更新预计剩余时间
        m = re.search(r"训练进度[:：]\s*(\d+(?:\.\d+)?)%", line)
        if m:
            pct = float(m.group(1))
            if pct > 0 and pct < 100:
                total_sec = self._estimate_total_seconds()
                remain = int(total_sec * (100 - pct) / 100)
                self.metric_labels["estimate"].setText(f"剩余约 {remain} 秒")
            elif pct >= 100:
                self.metric_labels["estimate"].setText("即将完成")

    def _estimate_total_seconds(self) -> int:
        """根据当前任务类型估算总训练时长（秒）。"""
        task_type = self.metric_labels["task_type"].text()
        mapping = {
            "OCR": 180,
            "CLS": 240,
            "DET": 360,
            "DETR": 480,
            "ISEG": 600,
            "SSEG": 600,
        }
        return mapping.get(task_type, 300)

    def eventFilter(self, obj, event):
        """监听 IME 组字事件，确保在组字期间不读取/刷新输入框。"""
        if obj is self.prompt_input and event.type() == QEvent.InputMethod:
            preedit = event.preeditString()
            commit = event.commitString()

            if preedit:
                self._ime_composing = True
                self._suggestion_debounce.stop()
            else:
                was_composing = self._ime_composing
                self._ime_composing = False
                if commit or was_composing:
                    QTimer.singleShot(50, self._do_update_suggestions)

        elif obj is self.prompt_input and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self._suggestion_debounce.stop()
                self._do_update_suggestions()
        # 必须交给父类默认处理，否则 IME 组字/候选窗口会失效
        return super().eventFilter(obj, event)

    def _do_update_suggestions(self) -> None:
        """Refresh suggestion chips based on input text."""
        if self._ime_composing:
            return
        text = self.prompt_input.toPlainText()
        suggestions = self.suggestion_engine.suggest(text)
        self._render_suggestions(suggestions)

    def _refresh_suggestions(self) -> None:
        """点击「刷新建议」：空输入时轮换默认模板，有输入时清缓存重新请求。"""
        if self._ime_composing:
            return
        text = self.prompt_input.toPlainText().strip()
        if not text:
            self._current_suggestion_set_index = (self._current_suggestion_set_index + 1) % len(self._default_suggestion_sets)
            self._render_suggestions(self._default_suggestion_sets[self._current_suggestion_set_index])
        else:
            # 清掉 LLM 缓存，强制重新生成建议
            self.suggestion_engine._cache.clear()
            self._do_update_suggestions()

    def _on_llm_suggestions(self, text: str, suggestions: list) -> None:
        """LLM 异步建议到达时，与本地建议混合后渲染。"""
        current = self.prompt_input.toPlainText()
        if current.strip().lower() != text.strip().lower():
            return

        def _blend_and_render():
            local = self.suggestion_engine.suggest(current)
            blended: list = []
            seen: set = set()
            for s in suggestions + local:
                key = s.strip()
                if key and key not in seen:
                    seen.add(key)
                    blended.append(key)
            self._render_suggestions(blended[:5])

        QTimer.singleShot(0, _blend_and_render)

    def _render_suggestions(self, suggestions: list) -> None:
        """Render suggestion chip buttons inline below input."""
        while self.suggestions_layout.count():
            item = self.suggestions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        cleaned: list = []
        for phrase in suggestions[:5]:
            phrase = phrase.strip().replace("\n", " ").replace("\r", "")
            phrase = "".join(ch for ch in phrase if ch.isprintable() or ch in " ")
            phrase = phrase.strip()[:24]
            if phrase:
                cleaned.append(phrase)

        if not cleaned:
            self.suggestions_container.setVisible(False)
            return

        self.suggestions_container.setVisible(True)
        for phrase in cleaned:
            btn = QPushButton(phrase)
            btn.setObjectName("suggestionChip")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, p=phrase: self._apply_suggestion(p))
            self.suggestions_layout.addWidget(btn)

        self.suggestions_layout.addStretch()

    def _apply_suggestion(self, suggestion: str) -> None:
        """Append suggestion to input and refresh."""
        self._suggestion_debounce.stop()
        current = self.prompt_input.toPlainText()
        new_text = self.suggestion_engine.take_suggestion(current, suggestion)
        self.prompt_input.setPlainText(new_text)
        cursor = self.prompt_input.textCursor()
        cursor.movePosition(cursor.End)
        self.prompt_input.setTextCursor(cursor)
        self.prompt_input.setFocus()
        self._do_update_suggestions()

    def _start_run(self) -> None:
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请输入训练需求")
            return

        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.post_worker and self.post_worker.isRunning():
            self.post_worker.stop()

        self.log_panel.clear()
        self.log_panel.append(f"[新运行] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.chat_response_box.clear()
        self.start_btn.setEnabled(False)
        self.close_browser_btn.setEnabled(False)
        self._set_status("运行中", "running")

        self.worker = TrainWorker(prompt, self._settings, self)
        self.worker.log_received.connect(self._append_log)
        self.worker.chat_response.connect(self._show_chat_response)
        self.worker.status_changed.connect(self._set_status)
        self.worker.finished_signal.connect(self._on_train_finished)
        self.worker.prompt_for_close.connect(self._enable_close_browser)
        self.worker.process_started.connect(self._enable_close_browser)
        self.worker.start()

        # 先把指标从当前 prompt 里解析出来，再清空输入框
        self._update_metrics_from_prompt()
        self.prompt_input.clear()
        self.prompt_input.setFocus()

    def _enable_close_browser(self) -> None:
        adapter = self._settings.get("adapter", "playwright").lower()
        self.close_browser_btn.setText("关闭应用" if adapter == "webview2_cdp" else "关闭浏览器")
        self.close_browser_btn.setEnabled(True)

    def _close_browser(self) -> None:
        """关闭保持打开的应用：网页端按 Enter，壳端直接结束 EXE 进程。"""
        adapter = self._settings.get("adapter", "playwright").lower()
        if adapter == "webview2_cdp":
            self._close_shell_app()
        elif self.worker:
            self.worker.send_enter()
            self._append_log("[操作] 已请求关闭浏览器，等待后台进程退出...")
            self._set_status("关闭中", "running")
        self.close_browser_btn.setEnabled(False)

    def _close_shell_app(self) -> None:
        """终止壳端 EXE 进程，并通过关闭信号让训练子进程干净退出。"""
        exe_path = self._settings.get("platform_exe", "")
        killed = []

        # 1) 结束壳端 EXE 进程
        if exe_path:
            exe_name = Path(exe_path).name.lower()
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    proc_name = (proc.info.get("name") or "").lower()
                    proc_exe = (proc.info.get("exe") or "").lower()
                    if exe_name in proc_name or exe_path.lower() in proc_exe:
                        proc.terminate()
                        killed.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        # 2) 等待 graceful terminate，超时则强制 kill
        gone, alive = psutil.wait_procs(killed, timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed:
            self._append_log(f"[操作] 已关闭壳端应用，PID: {', '.join(str(p.pid) for p in killed)}")
        else:
            self._append_log("[操作] 未找到运行中的壳端应用")

        # 3) 写关闭信号文件，让 auto_train2.0 干净退出，从而触发 post-train 弹窗
        if self.worker:
            self.worker.send_enter()
            self._append_log("[操作] 已发送关闭信号，等待后台进程退出...")

        self._set_status("关闭中", "running")

    def _clear_logs(self) -> None:
        self.log_panel.clear()
        self._set_status("就绪", "")
        self.close_browser_btn.setEnabled(False)
        self.chat_response_box.clear()
        for key in self.metric_labels:
            self.metric_labels[key].setText("-")
        self._update_metrics_from_prompt()

    def _append_log(self, line: str) -> None:
        self.log_panel.append(line)
        self._update_metrics_from_log(line)

    def _show_chat_response(self, answer: str) -> None:
        """显示由结构化子进程协议传来的完整多行回复。"""
        self.chat_response_box.setPlainText(answer)
        self.chat_response_card.setVisible(True)

    def _set_status(self, text: str, style_class: str) -> None:
        """统一更新日志区状态 badge 和右侧本次任务状态。"""
        name_map = {
            "running": "statusBadgeRunning",
            "success": "statusBadgeSuccess",
            "error": "statusBadgeError",
            "": "statusBadge",
        }
        obj_name = name_map.get(style_class, "statusBadge")
        for lbl in (self.status_label, self.task_status):
            lbl.setText(text)
            lbl.setObjectName(obj_name)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def _on_train_finished(
        self, code: int, elapsed: float, training_completed: bool = False, run_dir: str = ""
    ) -> None:
        self.close_browser_btn.setEnabled(False)

        if code != 0:
            self.start_btn.setEnabled(True)
            if code == -1:
                self._set_status("失败", "error")
            else:
                self._append_log(f"[完成] 进程退出码: {code}，耗时 {elapsed} 秒")
                self._set_status("失败", "error")
            self._show_run_dir_popup(run_dir, success=False)
            return

        # 成功结束：仅「真正完成模型训练」才弹 post-train 选择窗
        # 测试最新模型 / 闲聊 / 仅生成配置 等：只更新状态，不弹窗
        if training_completed:
            self._append_log(f"[完成] 训练成功，耗时 {elapsed} 秒")
            self._set_status("训练成功", "success")
            self._show_run_dir_popup(run_dir, success=True)
            self._offer_post_train_actions()
        else:
            self._append_log(f"[完成] 任务结束，耗时 {elapsed} 秒（非训练流程，跳过后续选择）")
            self.start_btn.setEnabled(True)
            self._set_status("成功", "success")
            self._show_run_dir_popup(run_dir, success=True)

    def _show_run_dir_popup(self, run_dir: str, success: bool = True) -> None:
        if not run_dir:
            return
        tip = QMessageBox(self)
        tip.setWindowTitle("训练信息已保存" if success else "训练失败")
        tip.setIcon(QMessageBox.Information if success else QMessageBox.Warning)
        tip.setText(f"本次训练信息已保存到：\n{run_dir}")
        tip.setStandardButtons(QMessageBox.Ok)
        tip.exec_()

    def _offer_post_train_actions(self) -> None:
        """仅在模型训练成功后弹出后续操作选择（自定义蓝紫弹窗）。"""
        context, err = resolve_post_train_context()
        if err or not context:
            self._append_log(f"[训练后] 无法准备评估上下文: {err}")
            self.start_btn.setEnabled(True)
            return

        detail = (
            f"任务: {context.get('task_type') or '未知'} | "
            f"模型: {Path(context.get('new_dvp', '')).name} | "
            f"baseline: {Path(context['baseline_dvp']).name if context.get('baseline_dvp') else '无'}"
        )
        self._append_log("[训练后] 请选择后续操作…")
        self._append_log(detail)

        dlg = PostTrainDialog(context, self)
        result = dlg.exec_()
        action = dlg.choice() if result == QDialog.Accepted else "skip"

        if action == "skip":
            self._append_log("[训练后] 用户选择结束，跳过评估")
            self.start_btn.setEnabled(True)
            self._set_status("成功", "success")
            return

        self._start_post_train(action, context)

    def _start_post_train(self, action: str, context: Dict[str, Any]) -> None:
        self.start_btn.setEnabled(False)
        self._set_status("评估中", "running")

        self.post_worker = PostTrainWorker(action, context, parent=self)
        self.post_worker.log_received.connect(self._append_log)
        self.post_worker.status_changed.connect(self._set_status)
        self.post_worker.finished_signal.connect(self._on_post_train_finished)
        self.post_worker.start()

    def _on_post_train_finished(self, success: bool, message: str) -> None:
        """评估/对比结束后只更新状态与日志，不再弹出 post-train 选择窗。"""
        self.start_btn.setEnabled(True)
        if success:
            self._set_status("成功", "success")
            self._append_log(f"[训练后] 报告路径: {message}")
            tip = QMessageBox(self)
            tip.setWindowTitle("评估完成")
            tip.setIcon(QMessageBox.Information)
            tip.setText("报告已保存")
            tip.setInformativeText(message)
            tip.setStandardButtons(QMessageBox.Ok)
            tip.exec_()
        else:
            self._set_status("失败", "error")
            tip = QMessageBox(self)
            tip.setWindowTitle("评估失败")
            tip.setIcon(QMessageBox.Warning)
            tip.setText("训练后评估失败")
            tip.setInformativeText(
                f"{message}\n\n请确认推理服务 http://127.0.0.1:9890 已启动。"
            )
            tip.setStandardButtons(QMessageBox.Ok)
            tip.exec_()

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.post_worker and self.post_worker.isRunning():
            self.post_worker.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    # 应用级默认字体；各控件字号由 MainWindow 等比例缩放覆盖
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
