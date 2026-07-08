"""
AutoTrain 4.0 PyQt5 桌面前端

启动：
    python pyqt_frontend/main.py

功能：
- 提供一个输入框，用户直接输入自然语言训练需求。
- 点击「开始训练」后，后台调用 auto_train4.0.py 并实时显示日志。
- 前后端分离：界面逻辑在本文件，训练执行逻辑在 backend.py 的 TrainWorker。
"""

import sys
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend import TrainWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoTrain 4.0")
        self.setMinimumSize(900, 700)
        self.worker: Optional[TrainWorker] = None

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

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.clear_btn)
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
        footer = QLabel("后台调用 auto_train4.0.py --prompt \"...\" 执行训练")
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
                font-size: 38px;
                font-weight: bold;
                color: #4c1d95;
            }
            QLabel#subtitle {
                font-size: 16px;
                color: #6b7280;
                margin-bottom: 10px;
            }
            QLabel#inputLabel {
                font-size: 16px;
                font-weight: bold;
                color: #374151;
            }
            QTextEdit#promptInput {
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 16px;
                padding: 16px;
                font-size: 16px;
                color: #1f2937;
            }
            QTextEdit#promptInput:focus {
                border: 2px solid #667eea;
            }
            QPushButton#startBtn {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea,
                    stop:1 #764ba2
                );
                color: white;
                font-size: 17px;
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
            QPushButton#clearBtn {
                background: #f3f4f6;
                color: #4b5563;
                font-size: 17px;
                font-weight: bold;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 14px 26px;
            }
            QPushButton#clearBtn:hover {
                background: #e5e7eb;
            }
            QLabel#logTitle {
                font-size: 16px;
                font-weight: bold;
                color: #374151;
            }
            QLabel#statusLabel {
                font-size: 14px;
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
                font-size: 14px;
                color: #1f2937;
            }
            QLabel#footer {
                font-size: 14px;
                color: #9ca3af;
            }
            """
        )

    def _start_run(self) -> None:
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请输入训练需求")
            return

        if self.worker and self.worker.isRunning():
            self.worker.stop()

        self.log_output.clear()
        self.start_btn.setEnabled(False)
        self._set_status("运行中", "running")

        self.worker = TrainWorker(prompt, self)
        self.worker.log_received.connect(self._append_log)
        self.worker.status_changed.connect(self._set_status)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _clear_logs(self) -> None:
        self.log_output.clear()
        self._set_status("就绪", "")

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
                base_style + "QLabel#statusLabel { color: #1e40af; background: #dbeafe; }"
            )
        elif style_class == "success":
            self.status_label.setStyleSheet(
                base_style + "QLabel#statusLabel { color: #065f46; background: #d1fae5; }"
            )
        elif style_class == "error":
            self.status_label.setStyleSheet(
                base_style + "QLabel#statusLabel { color: #991b1b; background: #fee2e2; }"
            )
        else:
            self.status_label.setStyleSheet(
                base_style + "QLabel#statusLabel { color: #6b7280; background: #e5e7eb; }"
            )

    def _on_finished(self, code: int, elapsed: float) -> None:
        self.start_btn.setEnabled(True)
        if code == 0:
            self._append_log(f"[完成] 训练成功，耗时 {elapsed} 秒")
            self._set_status("成功", "success")
        elif code == -1:
            self._set_status("失败", "error")
        else:
            self._append_log(f"[完成] 进程退出码: {code}，耗时 {elapsed} 秒")
            self._set_status("失败", "error")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
