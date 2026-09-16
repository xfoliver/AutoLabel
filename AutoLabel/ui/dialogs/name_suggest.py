"""Candidate list for AI name-suggest. Choosing one only fills the target field."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.styles import apply_window_styles
from ui.widgets.ui_scale import UIScale


class NameSuggestDialog(QDialog):
    """Show 3–5 Vision name candidates. 「使用」 returns the English SAM prompt."""

    def __init__(self, candidates: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 识别名称")
        self.setMinimumSize(420, 280)
        self.resize(520, 380)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self._ui_scale = getattr(parent, "_ui_scale", None) or UIScale()
        self.selected_english: Optional[str] = None

        title = QLabel("根据随机抽样图建议的视觉描述")
        title.setObjectName("Title")
        title.setWordWrap(True)
        hint = QLabel("点击「使用」把模型给出的英文填进目标框，发给 SAM3.1。不会开始标注，可再手动修改。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)

        list_host = QWidget()
        list_l = QVBoxLayout(list_host)
        list_l.setContentsMargins(0, 0, 8, 0)
        list_l.setSpacing(8)
        self._use_buttons: List[QPushButton] = []
        for item in candidates:
            english = str(item.get("english") or "").strip()
            chinese = str(item.get("chinese") or "").strip()
            try:
                conf = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            fill = english or chinese
            row = QWidget()
            row.setObjectName("Card")
            hl = QHBoxLayout(row)
            hl.setContentsMargins(12, 10, 12, 10)
            hl.setSpacing(10)
            text = QLabel()
            if chinese and english:
                text.setText(f"{chinese}    {english}\n置信度 {conf:.2f}")
            else:
                text.setText(f"{fill}\n置信度 {conf:.2f}")
            text.setWordWrap(True)
            text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            use = QPushButton("使用")
            use.setObjectName("Primary")
            use.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            use.clicked.connect(lambda _=False, e=fill: self._use(e))
            self._use_buttons.append(use)
            hl.addWidget(text, 1)
            hl.addWidget(use, 0)
            list_l.addWidget(row)
        list_l.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(list_host)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        apply_window_styles(self)

    def _use(self, english: str) -> None:
        self.selected_english = (english or "").strip()
        if self.selected_english:
            self.accept()
        else:
            self.reject()
