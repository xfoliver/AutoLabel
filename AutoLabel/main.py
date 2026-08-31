#!/usr/bin/env python
"""AutoLabel PyQt entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for p in (str(ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from ui.app_window import MainWindow


def main() -> int:
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AutoLabel")
    # 应用级默认字体；各控件字号由 MainWindow 等比例缩放覆盖
    app.setFont(QFont("Microsoft YaHei", 9))
    win = MainWindow()
    win.setAttribute(Qt.WA_QuitOnClose, True)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
