"""UIScale + stylesheet + compact layout tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from ui.styles import APP_QSS, build_stylesheet  # noqa: E402
from ui.widgets.ui_scale import UIScale  # noqa: E402


def test_scale_clamps_and_epsilon():
    s = UIScale()
    assert s.factor == 1.0
    # 设计稿尺寸本身就是 1.0，EPSILON 内不重绘
    assert not s.update_from_size(UIScale.REF_W, UIScale.REF_H)
    assert not s.update_from_size(UIScale.REF_W + 8, UIScale.REF_H)

    assert s.update_from_size(200, 200)
    assert s.factor == UIScale.MIN_FACTOR
    # 窗口变大不再放大控件，只伸展空白
    assert s.update_from_size(4000, 4000)
    assert s.factor == UIScale.MAX_FACTOR


def test_px_font_qss():
    s = UIScale()
    s.factor = 0.8
    assert s.px(10) == 8
    assert s.font(10) == 8
    assert s.qss_px(8) == "6px"
    assert s.px(0.1, minimum=1) == 1
    assert s.font(1, minimum=8) == 8


def test_stylesheet_follows_factor():
    s = UIScale()
    s.factor = 1.0
    css1 = build_stylesheet(s)
    s.factor = 0.8
    css2 = build_stylesheet(s)
    assert "font-size: 22px" in css1  # Title / PageTitle
    assert "font-size: 18px" in css2  # Title 22 * 0.8
    assert APP_QSS
    assert "QPushButton#Primary" in APP_QSS
    assert "QPushButton#StartBtn" in APP_QSS


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_compact_metrics(qapp):
    from ui.app_window import MainWindow

    win = MainWindow()
    win.resize(1180, 760)
    win._apply_responsive_scale(force=True)
    assert win._last_compact is False
    assert win._sidebar.width() >= 200
    assert win.pages.count() == 4
    assert win.ed_prompt.maximumHeight() >= 56
    assert hasattr(win, "btn_plan")

    win.resize(720, 520)
    win._apply_responsive_scale(force=True)
    assert win._last_compact is True
    assert win._sidebar.width() <= 120
    assert "font-size:" in win.styleSheet()
    assert "QPushButton#StartBtn" in win.styleSheet()
    win.close()


def test_preview_gallery_column_count(qapp):
    from ui.dialogs.preview_gallery import PreviewGalleryDialog

    class _R:
        preview_overlay_path = ""
        image_path = "missing.png"
        skipped = False
        shapes = []
        review = None

    dlg = PreviewGalleryDialog([_R() for _ in range(5)])
    dlg.resize(480, 360)
    dlg._relayout_cards()
    narrow = dlg._column_count()
    dlg.resize(980, 560)
    wide = dlg._column_count()
    assert wide >= narrow
    assert narrow >= 1
    dlg.close()
