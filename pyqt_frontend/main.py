"""
AutoTrain 4.0 PyQt5 桌面前端

启动：
    python pyqt_frontend/main.py

功能：
- 提供一个输入框，用户直接输入自然语言训练需求。
- 点击「开始训练」后，后台调用 auto_train4.0.py 并实时显示日志。
- 训练成功后由桌面端弹出固定选项：生成 report / 对比 baseline / 结束
  （不走 4.0 CLI 的 LLM + input，避免嵌套 stdin 卡死）。
- 右侧「数据质检」为独立模块（data_quality_panel），不介入训练子进程。
- 训练完成 / 质检结束 / 训练后评估后，根据结果自动生成助手总结（模板 + 可选 LLM）。
- 前后端分离：界面在 app_window / widgets / dialogs，执行逻辑在 backend.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

# 保证以脚本方式启动时，pyqt_frontend/ 在 sys.path 中
_FRONTEND_DIR = Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

from app_window import MainWindow  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    # 应用级默认字体；各控件字号由 MainWindow 等比例缩放覆盖
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)

    window = MainWindow()
    # 只有主窗口关闭才退出应用；设置等子对话框已设 WA_QuitOnClose=False
    window.setAttribute(Qt.WA_QuitOnClose, True)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
