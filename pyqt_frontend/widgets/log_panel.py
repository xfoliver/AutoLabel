"""Web 风格执行控制台（阶段侧栏 + 实时日志）。"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from widgets.ui_scale import UIScale


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
