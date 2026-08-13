"""执行控制台（阶段侧栏 + 实时日志）— 与主工作台浅色卡片风格统一。"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from widgets.ui_scale import UIScale


class CollapsibleLogPanel(QWidget):
    """阶段控制台，保留原有 append/clear 调用接口。"""

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
        self._log_filter: str = "all"  # all | warn | error
        self._level_formats: Dict[str, QTextCharFormat] = {}

        root = QHBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        self._root_layout = root

        # ── 左侧：Execution Flow ─────────────────────────────
        rail = QFrame()
        rail.setObjectName("logStageRail")
        self._rail = rail
        rail_layout = QVBoxLayout(rail)
        rail_layout.setSpacing(7)
        rail_layout.setContentsMargins(12, 14, 12, 14)
        self._rail_layout = rail_layout

        rail_eyebrow = QLabel("EXECUTION FLOW")
        rail_eyebrow.setObjectName("logRailTitle")
        rail_layout.addWidget(rail_eyebrow)

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

        # ── 右侧：浅色结构化日志 ─────────────────────────────
        console = QFrame()
        console.setObjectName("logConsole")
        console_layout = QVBoxLayout(console)
        console_layout.setSpacing(0)
        console_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QFrame()
        toolbar.setObjectName("logConsoleToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 10, 14, 10)
        toolbar_layout.setSpacing(8)
        self._toolbar_layout = toolbar_layout

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_block.setContentsMargins(0, 0, 0, 0)
        console_eyebrow = QLabel("LIVE OUTPUT")
        console_eyebrow.setObjectName("logConsoleEyebrow")
        console_title = QLabel("运行日志")
        console_title.setObjectName("logConsoleTitle")
        title_block.addWidget(console_eyebrow)
        title_block.addWidget(console_title)
        toolbar_layout.addLayout(title_block)
        toolbar_layout.addStretch()

        # 级别筛选
        self._filter_buttons: Dict[str, QPushButton] = {}
        filter_wrap = QFrame()
        filter_wrap.setObjectName("logFilterGroup")
        filter_row = QHBoxLayout(filter_wrap)
        filter_row.setContentsMargins(3, 3, 3, 3)
        filter_row.setSpacing(2)
        self._filter_row = filter_row
        for key, label in (("all", "全部"), ("warn", "警告"), ("error", "错误")):
            btn = QPushButton(label)
            btn.setObjectName("logFilterBtnActive" if key == "all" else "logFilterBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _c=False, k=key: self._set_log_filter(k))
            self._filter_buttons[key] = btn
            filter_row.addWidget(btn)
        toolbar_layout.addWidget(filter_wrap)

        self._live_badge = QLabel("● 待命")
        self._live_badge.setObjectName("logConsoleLiveIdle")
        toolbar_layout.addWidget(self._live_badge)

        console_layout.addWidget(toolbar)

        self._console = QPlainTextEdit()
        self._console.setObjectName("logConsoleOutput")
        self._console.setReadOnly(True)
        self._console.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._console.setUndoRedoEnabled(False)
        self._console.setPlaceholderText(
            "等待任务启动…\n\n训练过程、状态变化和错误信息将在这里实时显示。"
        )
        mono = QFont("Cascadia Code")
        if not mono.exactMatch():
            mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self._console.setFont(mono)
        console_layout.addWidget(self._console, 1)

        footer = QFrame()
        footer.setObjectName("logConsoleFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 8, 14, 8)
        footer_layout.setSpacing(10)
        self._footer_layout = footer_layout

        stream_dot = QLabel()
        stream_dot.setObjectName("logStreamDot")
        self._stream_dot = stream_dot
        footer_layout.addWidget(stream_dot)

        self._stream_label = QLabel("stream idle")
        self._stream_label.setObjectName("logConsoleFooterText")
        footer_layout.addWidget(self._stream_label)
        footer_layout.addStretch()

        self._line_count_label = QLabel("0 lines")
        self._line_count_label.setObjectName("logConsoleFooterText")
        footer_layout.addWidget(self._line_count_label)

        console_layout.addWidget(footer)
        root.addWidget(console, 1)

        self._init_level_formats()
        self.apply_ui_scale(UIScale())
        self._refresh_footer()

    def _init_level_formats(self) -> None:
        def fmt(color: str, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.DemiBold)
            return f

        self._level_formats = {
            "info": fmt("#4a5568"),
            "success": fmt("#15966d", bold=True),
            "warn": fmt("#c47d12", bold=True),
            "error": fmt("#d94d62", bold=True),
            "meta": fmt("#8b7cf6"),
            "dim": fmt("#9aa3b5"),
        }

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
            scale.px(10, 6),
            scale.px(14, 8),
            scale.px(10, 6),
        )
        self._toolbar_layout.setSpacing(scale.px(8, 4))
        self._filter_row.setContentsMargins(
            scale.px(3, 2),
            scale.px(3, 2),
            scale.px(3, 2),
            scale.px(3, 2),
        )
        self._filter_row.setSpacing(scale.px(2, 1))
        self._footer_layout.setContentsMargins(
            scale.px(14, 8),
            scale.px(8, 5),
            scale.px(14, 8),
            scale.px(8, 5),
        )
        dot = scale.px(6, 4)
        self._stream_dot.setFixedSize(dot, dot)
        mono = self._console.font()
        mono.setPointSize(scale.font(11, 9))
        self._console.setFont(mono)
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
        self._set_live(False)
        self._refresh_nav_styles()
        self._refresh_footer()

    def append(self, line: str) -> None:
        title = self._detect_stage_title(line)
        if title:
            self._start_stage(title)

        if self._current_stage is None:
            self._start_stage("日志")

        self._all_lines.append(line)
        self._current_stage["lines"].append(line)
        self._set_live(True)

        if self._line_matches_filter(line) and (
            self._selected_stage is None
            or self._selected_stage == len(self._stages) - 1
        ):
            self._append_styled_line(line)

        self._refresh_footer()
        scrollbar = self._console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_running(self, running: bool) -> None:
        """外部可同步运行状态到 LIVE 徽标。"""
        self._set_live(running)

    # ── internal ────────────────────────────────────────────

    def _set_live(self, running: bool) -> None:
        if running:
            self._live_badge.setText("● LIVE")
            self._live_badge.setObjectName("logConsoleLive")
            self._stream_label.setText("stream connected")
            self._stream_dot.setObjectName("logStreamDotOn")
        else:
            self._live_badge.setText("● 待命")
            self._live_badge.setObjectName("logConsoleLiveIdle")
            self._stream_label.setText("stream idle")
            self._stream_dot.setObjectName("logStreamDot")
        self._repolish(self._live_badge)
        self._repolish(self._stream_dot)

    def _refresh_footer(self) -> None:
        n = len(self._all_lines)
        self._line_count_label.setText(f"{n} line{'s' if n != 1 else ''}")

    def _detect_stage_title(self, line: str) -> Optional[str]:
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
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self._render_lines(lines)
        self._refresh_nav_styles()

    def _set_log_filter(self, key: str) -> None:
        if key == self._log_filter:
            return
        self._log_filter = key
        for k, btn in self._filter_buttons.items():
            btn.setObjectName("logFilterBtnActive" if k == key else "logFilterBtn")
            self._repolish(btn)
        lines = (
            self._all_lines
            if self._selected_stage is None
            else self._stages[self._selected_stage]["lines"]
        )
        self._render_lines(lines)

    def _line_matches_filter(self, line: str) -> bool:
        level = self._detect_level(line)
        if self._log_filter == "error":
            return level == "error"
        if self._log_filter == "warn":
            return level in ("warn", "error")
        return True

    def _render_lines(self, lines: list[str]) -> None:
        self._console.clear()
        filtered = [ln for ln in lines if self._line_matches_filter(ln)]
        if not filtered:
            cursor = self._console.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(
                "当前筛选条件下没有日志",
                self._level_formats["dim"],
            )
        else:
            for i, line in enumerate(filtered):
                self._append_styled_line(line, line_no=i + 1, leading_newline=i > 0)
        scrollbar = self._console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_styled_line(
        self,
        line: str,
        *,
        line_no: Optional[int] = None,
        leading_newline: Optional[bool] = None,
    ) -> None:
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)

        plain = self._console.toPlainText()
        if leading_newline is None:
            leading_newline = bool(plain)
        if leading_newline and plain:
            cursor.insertText("\n")
            plain = self._console.toPlainText()

        if line_no is None:
            line_no = 1 if not plain.strip() else plain.count("\n") + 1

        level = self._detect_level(line)
        fmt = self._level_formats.get(level, self._level_formats["info"])
        num_fmt = self._level_formats["dim"]
        cursor.insertText(f"{line_no:02d}  ", num_fmt)
        cursor.insertText(line, fmt)
        self._console.setTextCursor(cursor)

    @staticmethod
    def _detect_level(line: str) -> str:
        upper = line.upper()
        if re.search(r"ERROR|失败|\[错误\]|Traceback|Exception", line, re.I):
            return "error"
        if re.search(r"WARN|警告", line, re.I):
            return "warn"
        if re.search(r"DONE|完成|成功|\[完成\]", line, re.I):
            return "success"
        if re.search(r"\[新运行\]|\[开始\]|={5,}", line):
            return "meta"
        if upper.startswith("[DEBUG]") or upper.startswith("[TRACE]"):
            return "dim"
        return "info"

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
