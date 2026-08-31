"""Simple log panel that follows UIScale."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPlainTextEdit, QSizePolicy, QVBoxLayout, QWidget

from ui.widgets.ui_scale import UIScale


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui_scale: Optional[UIScale] = None
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(5000)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.apply_ui_scale(UIScale())

    def apply_ui_scale(self, scale: UIScale) -> None:
        self._ui_scale = scale
        font = self.view.font()
        font.setFamily("Cascadia Code")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(scale.font(10))
        self.view.setFont(font)
        self.setMinimumHeight(scale.px(100, 72))

    def append(self, text: str) -> None:
        self.view.appendPlainText(text)

    def clear(self) -> None:
        self.view.clear()
