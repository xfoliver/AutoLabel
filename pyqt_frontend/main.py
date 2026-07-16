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

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend import PostTrainWorker, TrainWorker, resolve_post_train_context


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
        self.setMinimumSize(900, 700)
        self.worker: Optional[TrainWorker] = None
        self.post_worker: Optional[PostTrainWorker] = None

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setSpacing(18)
        layout.setContentsMargins(40, 40, 40, 40)

        # 标题
        title = QLabel("AutoTrain 4.0")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("用自然语言描述训练需求，一键启动 DLCV AI 训练平台")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # 输入区
        prompt_label = QLabel("训练需求")
        prompt_label.setObjectName("inputLabel")
        layout.addWidget(prompt_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText(
            "例如：训练一个 cls 模型来辨别王不留行和白芷"
        )
        self.prompt_input.setMaximumHeight(120)
        layout.addWidget(self.prompt_input)

        # 按钮区
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.start_btn = QPushButton("开始训练")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_run)

        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self._clear_logs)

        self.close_browser_btn = QPushButton("关闭浏览器")
        self.close_browser_btn.setObjectName("closeBrowserBtn")
        self.close_browser_btn.setEnabled(False)
        self.close_browser_btn.setCursor(Qt.PointingHandCursor)
        self.close_browser_btn.clicked.connect(self._close_browser)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.close_browser_btn)
        layout.addLayout(btn_layout)

        # 日志区
        log_header = QHBoxLayout()
        log_title = QLabel("运行日志")
        log_title.setObjectName("logTitle")
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")
        log_header.addWidget(log_title)
        log_header.addStretch()
        log_header.addWidget(self.status_label)
        layout.addLayout(log_header)

        self.log_output = QTextEdit()
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        # 底部说明
        footer = QLabel(
            "后台调用 auto_train4.0.py；训练成功后可生成 inference report 或对比 baseline"
        )
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

    def _apply_styles(self) -> None:
        """应用白色卡片 + 蓝紫渐变风格，与 Web 前端保持一致，字体整体放大两号。"""
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea,
                    stop:0.5 #764ba2,
                    stop:1 #f093fb
                );
            }
            QLabel#title {
                font-size: 44px;
                font-weight: bold;
                color: #4c1d95;
            }
            QLabel#subtitle {
                font-size: 18px;
                color: #581c87;
                margin-bottom: 10px;
            }
            QLabel#inputLabel {
                font-size: 18px;
                font-weight: bold;
                color: #374151;
            }
            QTextEdit#promptInput {
                background-color: #ffffff;
                border: 2px solid #e5e7eb;
                border-radius: 16px;
                padding: 16px;
                font-size: 18px;
                color: #1f2937;
            }
            QTextEdit#promptInput:focus {
                border: 2px solid #667eea;
                background-color: #ffffff;
                color: #1f2937;
            }
            QPushButton#startBtn {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea,
                    stop:1 #764ba2
                );
                color: white;
                font-size: 19px;
                font-weight: bold;
                border: none;
                border-radius: 12px;
                padding: 14px 26px;
            }
            QPushButton#startBtn:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a6fd6,
                    stop:1 #6a4191
                );
            }
            QPushButton#startBtn:disabled {
                background: #c4b5fd;
            }
            QPushButton#clearBtn,
            QPushButton#closeBrowserBtn {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea,
                    stop:1 #764ba2
                );
                color: white;
                font-size: 19px;
                font-weight: bold;
                border: none;
                border-radius: 12px;
                padding: 14px 26px;
            }
            QPushButton#clearBtn:hover,
            QPushButton#closeBrowserBtn:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a6fd6,
                    stop:1 #6a4191
                );
            }
            QPushButton#clearBtn:disabled,
            QPushButton#closeBrowserBtn:disabled {
                background: #c4b5fd;
            }
            QLabel#logTitle {
                font-size: 18px;
                font-weight: bold;
                color: #374151;
            }
            QLabel#statusLabel {
                font-size: 16px;
                font-weight: bold;
                color: #6b7280;
                background: #e5e7eb;
                border-radius: 12px;
                padding: 6px 14px;
            }
            QTextEdit#logOutput {
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 16px;
                font-family: "Consolas", "SFMono-Regular", monospace;
                font-size: 16px;
                color: #1f2937;
            }
            QLabel#footer {
                font-size: 16px;
                color: #581c87;
            }
            """
        )

        # 显式设置输入框调色板，防止系统深色主题或 Qt 默认调色板导致文字/背景颜色异常
        palette = self.prompt_input.palette()
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.Text, QColor("#1f2937"))
        palette.setColor(QPalette.PlaceholderText, QColor("#9ca3af"))
        self.prompt_input.setPalette(palette)

    def _start_run(self) -> None:
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请输入训练需求")
            return

        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.post_worker and self.post_worker.isRunning():
            self.post_worker.stop()

        self.log_output.append("=" * 60)
        self.log_output.append(f"[新运行] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.start_btn.setEnabled(False)
        self.close_browser_btn.setEnabled(False)
        self._set_status("运行中", "running")

        self.worker = TrainWorker(prompt, self)
        self.worker.log_received.connect(self._append_log)
        self.worker.status_changed.connect(self._set_status)
        self.worker.finished_signal.connect(self._on_train_finished)
        self.worker.prompt_for_close.connect(self._enable_close_browser)
        self.worker.process_started.connect(self._enable_close_browser)
        self.worker.start()

        self.prompt_input.clear()
        self.prompt_input.setFocus()

    def _enable_close_browser(self) -> None:
        self.close_browser_btn.setEnabled(True)

    def _close_browser(self) -> None:
        """关闭保持打开的浏览器，并等待后台进程收尾后恢复 UI。"""
        if self.worker:
            self.worker.send_enter()
            self._append_log("[操作] 已请求关闭浏览器，等待后台进程退出...")
            self._set_status("关闭中", "running")
        self.close_browser_btn.setEnabled(False)

    def _clear_logs(self) -> None:
        self.log_output.clear()
        self._set_status("就绪", "")
        self.close_browser_btn.setEnabled(False)

    def _append_log(self, line: str) -> None:
        self.log_output.append(line)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_status(self, text: str, style_class: str) -> None:
        self.status_label.setText(text)
        base_style = """
            QLabel#statusLabel {
                font-size: 12px;
                font-weight: bold;
                border-radius: 12px;
                padding: 4px 12px;
            }
        """
        if style_class == "running":
            self.status_label.setStyleSheet(
                base_style
                + "QLabel#statusLabel { color: #1e40af; background: #dbeafe; }"
            )
        elif style_class == "success":
            self.status_label.setStyleSheet(
                base_style
                + "QLabel#statusLabel { color: #065f46; background: #d1fae5; }"
            )
        elif style_class == "error":
            self.status_label.setStyleSheet(
                base_style
                + "QLabel#statusLabel { color: #991b1b; background: #fee2e2; }"
            )
        else:
            self.status_label.setStyleSheet(
                base_style
                + "QLabel#statusLabel { color: #6b7280; background: #e5e7eb; }"
            )

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
    font = QFont("Microsoft YaHei", 14)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
