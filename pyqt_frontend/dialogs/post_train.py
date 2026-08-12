"""训练完成后的后续操作选择弹窗。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


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
