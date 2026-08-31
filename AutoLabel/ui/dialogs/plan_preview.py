"""Plan preview dialog."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)

from autolabel.plan import format_plan_text
from ui.styles import apply_window_styles
from ui.widgets.ui_scale import UIScale


class PlanPreviewDialog(QDialog):
    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.setWindowTitle("执行计划确认")
        self.setMinimumSize(380, 280)
        self.resize(560, 400)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self._result = "cancel"
        self._ui_scale = getattr(parent, "_ui_scale", None) or UIScale()

        title = QLabel("Plan Preview — 烧算力前请确认")
        title.setObjectName("Title")
        title.setWordWrap(True)
        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(format_plan_text(plan))
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        buttons = QDialogButtonBox()
        self.btn_preview = buttons.addButton("开始试标", QDialogButtonBox.AcceptRole)
        self.btn_edit = buttons.addButton("修改意图", QDialogButtonBox.ActionRole)
        self.btn_cancel = buttons.addButton("取消", QDialogButtonBox.RejectRole)

        self.btn_preview.clicked.connect(self._accept_preview)
        self.btn_edit.clicked.connect(self._edit)
        self.btn_cancel.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(title)
        hint = QLabel("确认目标语义 / 数据集 / Backend 无误后再试标。试标未确认不会全量。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(body, 1)
        layout.addWidget(buttons)
        apply_window_styles(self)

    def _accept_preview(self):
        self._result = "preview"
        self.accept()

    def _edit(self):
        self._result = "edit"
        self.accept()

    @property
    def choice(self) -> str:
        return self._result
