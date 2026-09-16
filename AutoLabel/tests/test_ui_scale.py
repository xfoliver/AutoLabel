"""UIScale + stylesheet + compact layout tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.styles import APP_QSS, build_stylesheet  # noqa: E402
from ui.widgets.flow_layout import FlowLayout  # noqa: E402
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
    assert "QLabel#Notice" in APP_QSS


def test_flow_layout_wraps(qapp):
    from PyQt5.QtWidgets import QLabel, QWidget

    host = QWidget()
    flow = FlowLayout(host, spacing=8)
    for i in range(6):
        lab = QLabel(f"chip-{i}")
        lab.setMinimumWidth(80)
        flow.addWidget(lab)
    wide = flow.heightForWidth(800)
    narrow = flow.heightForWidth(180)
    assert narrow > wide
    assert wide > 0
    host.close()


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_compact_metrics(qapp):
    from PyQt5.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout

    from ui.app_window import MainWindow

    win = MainWindow()
    win.resize(1180, 760)
    win.show()
    qapp.processEvents()
    win._apply_responsive_scale(force=True)
    assert win._last_compact is False
    assert win._sidebar.width() >= 200
    assert win._nav_btns["new"].text() == "新建任务"
    assert win.pages.count() == 4
    assert win.sl_area.maximum() == 16384
    assert not hasattr(win, "_chip_btns")
    assert win.ed_prompt.maximumHeight() >= 56
    assert hasattr(win, "btn_plan")
    assert hasattr(win, "btn_ai_name")
    assert win.btn_ai_name.text() == "AI 识别名称"
    assert "LabelMe" in win.lbl_review_notice.text()
    assert win.lbl_review_notice.wordWrap()
    assert win._page_new.findChild(QScrollArea) is not None
    assert any(lay.direction() == QHBoxLayout.LeftToRight for lay, *_ in win._stackable_layouts)

    win.resize(720, 520)
    qapp.processEvents()
    win._apply_responsive_scale(force=True)
    assert win._last_compact is True
    # 窄窗仍保留可读侧栏，不再缩成空白 88px 轨道
    assert 160 <= win._sidebar.width() <= 240
    assert win._nav_btns["new"].text() == "新建任务"
    assert "font-size:" in win.styleSheet()
    assert "QPushButton#StartBtn" in win.styleSheet()
    assert any(lay.direction() == QVBoxLayout.TopToBottom for lay, *_ in win._stackable_layouts)
    assert win.ed_prompt.sizePolicy().horizontalPolicy() == win.ed_prompt.sizePolicy().Expanding
    win.close()


def test_attach_worker_does_not_destroy_running_thread(qapp):
    from PyQt5.QtCore import QThread

    from ui.app_window import MainWindow

    win = MainWindow()
    first = QThread(win)
    first.start()
    assert first.isRunning()
    win.worker = first
    second = QThread(win)
    win._attach_worker(second)
    assert win.worker is second
    # Old thread is retired (quit + deferred delete), never terminate()'d.
    first.wait(2000)
    qapp.processEvents()
    assert not first.isRunning()
    win.close()
    qapp.processEvents()


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


def test_stylesheet_groupbox_title_has_clearance():
    s = UIScale()
    s.factor = 1.0
    css = build_stylesheet(s)
    assert "QGroupBox::title" in css
    assert "subcontrol-position: top left" in css
    assert "margin-top: 22px" in css
    assert "padding-top: 16px" in css


def test_settings_groupbox_title_does_not_overlap_fields(qapp, tmp_path):
    from ui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(
        {},
        {},
        tmp_path / "settings.json",
        tmp_path / "backends.json",
    )
    dlg.resize(560, 520)
    dlg.show()
    qapp.processEvents()
    dlg.adjustSize()
    qapp.processEvents()

    for box, field in (
        (dlg.llm_box, dlg.ed_base),
        (dlg.sam3_box, dlg.ed_sam3_python),
        (dlg.vision_box, dlg.ed_vision_url),
    ):
        box.setVisible(True)
        qapp.processEvents()
        title_h = box.fontMetrics().height()
        top = field.mapTo(box, field.rect().topLeft()).y()
        assert top >= title_h + 8, (
            f"{box.title()!r} field top={top} overlaps title height={title_h}"
        )
    dlg.close()


def test_settings_vision_box_visible_on_sam3(qapp, tmp_path):
    from autolabel.engines import ENGINE_SAM31
    from ui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(
        {},
        {"annotation_engine": ENGINE_SAM31},
        tmp_path / "settings.json",
        tmp_path / "backends.json",
    )
    dlg.show()
    qapp.processEvents()
    assert dlg.cmb_engine.currentData() == ENGINE_SAM31
    assert dlg.vision_box.isVisible()
    assert dlg.vision_box.title() == "视觉识别模型"
    assert not hasattr(dlg, "ed_vision_chat_model")
    ids = [dlg.cmb_engine.itemData(i) for i in range(dlg.cmb_engine.count())]
    assert ENGINE_SAM31 in ids
    assert "chat-image" not in ids
    assert dlg.btn_add_engine is not None
    dlg.close()


def test_settings_lists_saved_api_engine(qapp, tmp_path):
    from autolabel.engines import ENGINE_SAM31
    from ui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(
        {},
        {
            "annotation_engine": "qwen_edit",
            "backends": [
                {
                    "id": "qwen_edit",
                    "type": "openai_image_edits",
                    "label": "Qwen Image",
                    "options": {"base_url": "https://api.example/v1", "model": "qwen-image-edit", "api_key": "sk"},
                }
            ],
        },
        tmp_path / "settings.json",
        tmp_path / "backends.json",
    )
    dlg.show()
    qapp.processEvents()
    ids = [dlg.cmb_engine.itemData(i) for i in range(dlg.cmb_engine.count())]
    assert ids[0] == ENGINE_SAM31
    assert "qwen_edit" in ids
    assert dlg.cmb_engine.currentData() == "qwen_edit"
    assert dlg.ed_vision_url.text() == "https://api.example/v1"
    dlg.close()


def test_ai_name_fills_label_without_starting_sam3(qapp):
    from ui.app_window import MainWindow
    from ui.dialogs.name_suggest import NameSuggestDialog

    win = MainWindow()
    win.ed_label.setText("old")
    started = {"n": 0}
    win.on_start = lambda: started.__setitem__("n", started["n"] + 1)
    win.apply_suggested_name("alpha part")
    assert win.ed_label.text() == "alpha part"
    assert win.ed_prompt.toPlainText() == "alpha part"
    assert started["n"] == 0
    sentence = "a yellow labrador retriever standing on green grass near a fence"
    win.apply_suggested_name(sentence)
    assert win.ed_label.text() == sentence
    assert win.ed_prompt.toPlainText() == sentence
    assert started["n"] == 0
    assert win.worker is None or not win.worker.isRunning()

    dlg = NameSuggestDialog(
        [
            {"english": "beta-part", "chinese": "乙", "confidence": 0.86},
            {"english": "alpha part", "chinese": "甲", "confidence": 0.92},
        ],
        parent=win,
    )
    assert len(dlg._use_buttons) == 2
    dlg._use("beta-part")
    assert dlg.selected_english == "beta-part"
    dlg.close()
    win.close()
