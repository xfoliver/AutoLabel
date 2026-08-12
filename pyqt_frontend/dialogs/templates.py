"""快捷模板相关对话框与可点击卡片。"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


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
