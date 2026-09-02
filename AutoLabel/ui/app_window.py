"""AutoLabel main window — layout aligned with the frontend demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autolabel.dataset import list_images
from autolabel.inference.base import build_backend_from_config, load_backends_config
from autolabel.orchestrator import AutoLabelOrchestrator
from autolabel.sam3_worker_ctl import ensure_worker, probe_health, worker_options, worker_url
from ui.backend_worker import LabelWorker
from ui.dialogs.preview_gallery import PreviewGalleryDialog
from ui.dialogs.settings_dialog import SettingsDialog
from ui.styles import apply_window_styles
from ui.widgets.log_panel import LogPanel
from ui.widgets.ui_scale import UIScale

AUTO_LABEL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUTO_LABEL_ROOT.parent

CHIP_TARGETS = (
    ("手机", "phone"),
    ("汽车", "car"),
    ("人", "person"),
    ("电脑", "laptop"),
    ("杯子", "cup"),
)


class _BootWorker(QThread):
    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)
    log_received = pyqtSignal(str)

    def __init__(self, options: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.options = options

    def run(self) -> None:
        try:
            data = ensure_worker(
                self.options,
                log=lambda m: self.log_received.emit(str(m)),
                timeout=float(self.options.get("startup_timeout") or 600),
            )
            self.ready.emit(data if isinstance(data, dict) else {"status": "ok"})
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoLabel")
        self.setMinimumSize(880, 560)
        self.resize(max(UIScale.REF_W, 1180), max(UIScale.REF_H, 760))
        self._ui_scale = UIScale()
        self._last_compact = False
        self._scale_debounce = QTimer(self)
        self._scale_debounce.setSingleShot(True)
        self._scale_debounce.timeout.connect(self._apply_responsive_scale)

        self.settings_path = AUTO_LABEL_ROOT / "config" / "settings.json"
        self.backends_path = AUTO_LABEL_ROOT / "config" / "backends.json"
        self.settings = self._load_settings()
        self.backends_config = load_backends_config(self.backends_path)
        self.orch = self._make_orch()
        self.worker: Optional[LabelWorker] = None
        self._boot: Optional[_BootWorker] = None
        self._image_count = 0
        self._preview_results = []

        self._build_ui()
        self._refresh_backend_chip()
        self._apply_responsive_scale(force=True)
        # Offscreen tests / CI must not block on a 9892 health probe.
        import os

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QTimer.singleShot(200, self._probe_sam3_on_boot)

    def _load_settings(self) -> Dict[str, Any]:
        for path in (
            self.settings_path,
            PROJECT_ROOT / "pyqt_frontend" / "settings.json",
            AUTO_LABEL_ROOT / "config" / "settings.example.json",
        ):
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return {}

    def _make_orch(self) -> AutoLabelOrchestrator:
        backend = build_backend_from_config(self.backends_config)
        return AutoLabelOrchestrator(
            backends_config=self.backends_config,
            backend=backend,
            settings=self.settings,
            log=lambda m: None,
        )

    # ----------------------------------------------------------------- layout
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._root_layout = root

        root.addWidget(self._build_topbar())

        main = QWidget()
        main_l = QHBoxLayout(main)
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.setSpacing(0)
        self._main_layout = main_l
        main_l.addWidget(self._build_sidebar())
        main_l.addWidget(self._build_content(), 1)
        root.addWidget(main, 1)
        root.addWidget(self._build_footer())

        self._toast = QLabel("", self)
        self._toast.setObjectName("Toast")
        self._toast.hide()
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._toast.hide)

        self._overlay = self._build_overlay(central)

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(72)
        self._topbar = bar
        row = QHBoxLayout(bar)
        row.setContentsMargins(24, 0, 20, 0)
        row.setSpacing(12)
        self._top_layout = row

        logo = QLabel("✦")
        logo.setFixedSize(30, 30)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "background:#6c4cff;color:white;border-radius:9px;font-weight:800;font-size:14px;"
        )
        title = QLabel("AutoLabel")
        title.setObjectName("BrandTitle")
        sub = QLabel("自动标注工具")
        sub.setObjectName("BrandSub")
        self.lbl_brand_sub = sub
        row.addWidget(logo)
        row.addWidget(title)
        row.addWidget(sub)
        row.addStretch(1)

        self.lbl_ready = QLabel("●  就绪")
        self.lbl_ready.setObjectName("ReadyPill")
        btn_settings = QPushButton("⚙  设置")
        btn_settings.setObjectName("Ghost")
        btn_settings.clicked.connect(self.open_settings)
        row.addWidget(self.lbl_ready)
        row.addWidget(btn_settings)
        return bar

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(300)
        self._sidebar = side
        col = QVBoxLayout(side)
        col.setContentsMargins(18, 28, 16, 16)
        col.setSpacing(8)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_btns = {}
        for key, text in (
            ("new", "新建任务"),
            ("preview", "预览标注"),
            ("batch", "批量处理"),
            ("export", "结果导出"),
        ):
            btn = QPushButton(text)
            btn.setObjectName("Nav")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self._show_page(k))
            self._nav_group.addButton(btn)
            self._nav_btns[key] = btn
            col.addWidget(btn)
        self._nav_btns["new"].setChecked(True)

        col.addStretch(1)

        card = QWidget()
        card.setObjectName("BackendCard")
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(16, 16, 16, 16)
        h = QLabel("后端信息")
        h.setObjectName("SectionTitle")
        self.lbl_backend = QLabel("Backend: —")
        self.lbl_backend.setObjectName("Hint")
        self.lbl_backend.setWordWrap(True)
        self.lbl_conn = QLabel("状态: 未检查")
        self.lbl_conn.setObjectName("Hint")
        self.lbl_conn.setWordWrap(True)
        check = QPushButton("检查连接")
        check.clicked.connect(self.check_connection)
        self.btn_check = check
        card_l.addWidget(h)
        card_l.addWidget(self.lbl_backend)
        card_l.addWidget(self.lbl_conn)
        card_l.addWidget(check)
        col.addWidget(card)

        ver = QLabel("AutoLabel v1.0.0")
        ver.setObjectName("Hint")
        ver.setAlignment(Qt.AlignCenter)
        self.lbl_version = ver
        col.addWidget(ver)
        return side

    def _build_content(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        self._page_new = self._build_page_new()
        self._page_preview = self._build_page_preview()
        self._page_batch = self._build_page_batch()
        self._page_export = self._build_page_export()
        self.pages.addWidget(self._page_new)
        self.pages.addWidget(self._page_preview)
        self.pages.addWidget(self._page_batch)
        self.pages.addWidget(self._page_export)
        lay.addWidget(self.pages)
        return wrap

    def _card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        return card

    def _section_label(self, number: str, title: str, subtitle: str) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        badge = QLabel(number)
        badge.setObjectName("NumberBadge")
        badge.setFixedSize(31, 31)
        badge.setAlignment(Qt.AlignCenter)
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(4)
        h = QLabel(title)
        h.setObjectName("SectionTitle")
        p = QLabel(subtitle)
        p.setObjectName("Hint")
        p.setWordWrap(True)
        texts.addWidget(h)
        texts.addWidget(p)
        row.addWidget(badge, 0, Qt.AlignTop)
        row.addLayout(texts, 1)
        box.setMinimumWidth(220)
        return box

    def _build_page_new(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(36, 28, 36, 20)
        self._content_layout = outer

        scroll_host = QWidget()
        col = QVBoxLayout(scroll_host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)

        title_row = QHBoxLayout()
        step = QLabel("1")
        step.setObjectName("StepCircle")
        step.setFixedSize(27, 27)
        step.setAlignment(Qt.AlignCenter)
        h1 = QLabel("新建任务")
        h1.setObjectName("PageTitle")
        title_row.addWidget(step)
        title_row.addWidget(h1)
        title_row.addStretch(1)
        col.addLayout(title_row)
        sub = QLabel("只需三步，开始自动标注")
        sub.setObjectName("Subtitle")
        col.addWidget(sub)

        # 1 folder
        c1 = self._card()
        r1 = QHBoxLayout(c1)
        r1.setContentsMargins(24, 22, 24, 22)
        r1.setSpacing(20)
        r1.addWidget(self._section_label("1", "选择图片文件夹", "选择需要标注的图片所在文件夹"), 0)
        folder_col = QVBoxLayout()
        folder_row = QHBoxLayout()
        self.ed_dataset = QLineEdit()
        self.ed_dataset.setPlaceholderText("请选择文件夹")
        self.ed_dataset.textChanged.connect(self._on_folder_changed)
        browse = QPushButton("浏览")
        browse.setFixedWidth(100)
        browse.clicked.connect(self.browse_dataset)
        folder_row.addWidget(self.ed_dataset, 1)
        folder_row.addWidget(browse)
        folder_col.addLayout(folder_row)
        r1.addLayout(folder_col, 1)
        col.addWidget(c1)

        # 2 prompt
        c2 = self._card()
        r2 = QHBoxLayout(c2)
        r2.setContentsMargins(24, 22, 24, 22)
        r2.setSpacing(20)
        r2.addWidget(self._section_label("2", "告诉我你要标注什么", "用自然语言描述要标注的目标"), 0)
        target_col = QVBoxLayout()
        target_col.setSpacing(8)
        self.ed_prompt = QTextEdit()
        self.ed_prompt.setPlaceholderText("例：标注图片中的手机")
        self.ed_prompt.setMaximumHeight(100)
        self.ed_prompt.textChanged.connect(self._on_prompt_changed)
        self._nl_layout = target_col
        target_col.addWidget(self.ed_prompt)
        self.lbl_chars = QLabel("0/100")
        self.lbl_chars.setObjectName("Hint")
        self.lbl_chars.setAlignment(Qt.AlignRight)
        target_col.addWidget(self.lbl_chars)
        or_l = QLabel("或者直接输入目标名称（如：phone, car, person）")
        or_l.setObjectName("Hint")
        target_col.addWidget(or_l)
        self.ed_label = QLineEdit()
        self.ed_label.setPlaceholderText("例如：phone")
        self.ed_label.textChanged.connect(self._sync_preview_meta)
        target_col.addWidget(self.ed_label)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        lab = QLabel("常用目标：")
        lab.setObjectName("Hint")
        chips.addWidget(lab)
        self._chip_btns = []
        for zh, en in CHIP_TARGETS:
            chip = QPushButton(f"{zh} ({en})")
            chip.setObjectName("Chip")
            chip.setCheckable(True)
            chip.clicked.connect(lambda _=False, e=en, z=zh, b=chip: self._apply_chip(e, z, b))
            self._chip_btns.append(chip)
            chips.addWidget(chip)
        custom = QPushButton("自定义")
        custom.setObjectName("Chip")
        custom.clicked.connect(self._clear_chips)
        chips.addWidget(custom)
        chips.addStretch(1)
        target_col.addLayout(chips)
        r2.addLayout(target_col, 1)
        col.addWidget(c2)

        # 3 advanced
        c3 = self._card()
        r3 = QHBoxLayout(c3)
        r3.setContentsMargins(24, 22, 24, 22)
        r3.setSpacing(20)
        r3.addWidget(self._section_label("3", "高级选项（可选）", "调整标注参数（新手可跳过）"), 0)
        adv = QHBoxLayout()
        adv.setSpacing(36)
        conf_box = QVBoxLayout()
        conf_lab = QLabel("置信度阈值")
        conf_lab.setObjectName("Hint")
        conf_row = QHBoxLayout()
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(10, 95)
        self.sl_conf.setValue(55)
        self.sl_conf.setSingleStep(5)
        self.lbl_conf = QLabel("0.55")
        self.lbl_conf.setFixedWidth(48)
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setObjectName("Hint")
        self.sl_conf.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.sl_conf, 1)
        conf_row.addWidget(self.lbl_conf)
        conf_box.addWidget(conf_lab)
        conf_box.addLayout(conf_row)
        area_box = QVBoxLayout()
        area_lab = QLabel("最小目标面积")
        area_lab.setObjectName("Hint")
        area_row = QHBoxLayout()
        self.sl_area = QSlider(Qt.Horizontal)
        self.sl_area.setRange(16, 512)
        self.sl_area.setValue(64)
        self.sl_area.setSingleStep(16)
        self.lbl_area = QLabel("64 px")
        self.lbl_area.setFixedWidth(56)
        self.lbl_area.setAlignment(Qt.AlignCenter)
        self.lbl_area.setObjectName("Hint")
        self.sl_area.valueChanged.connect(self._on_area_changed)
        area_row.addWidget(self.sl_area, 1)
        area_row.addWidget(self.lbl_area)
        area_box.addWidget(area_lab)
        area_box.addLayout(area_row)
        adv.addLayout(conf_box, 1)
        adv.addLayout(area_box, 1)
        r3.addLayout(adv, 1)
        col.addWidget(c3)

        # preview + start
        c4 = self._card()
        r4 = QHBoxLayout(c4)
        r4.setContentsMargins(22, 18, 22, 18)
        r4.setSpacing(20)
        meta = QVBoxLayout()
        pt = QLabel("任务预览")
        pt.setObjectName("SectionTitle")
        self.lbl_preview_folder = QLabel("文件夹：未选择")
        self.lbl_preview_count = QLabel("预计处理：0 张图片")
        self.lbl_preview_target = QLabel("目标：未设置")
        self.lbl_preview_backend = QLabel("后端：—")
        for w in (
            self.lbl_preview_folder,
            self.lbl_preview_count,
            self.lbl_preview_target,
            self.lbl_preview_backend,
        ):
            w.setObjectName("Hint")
            w.setWordWrap(True)
        meta.addWidget(pt)
        meta.addWidget(self.lbl_preview_folder)
        meta.addWidget(self.lbl_preview_count)
        meta.addWidget(self.lbl_preview_target)
        meta.addWidget(self.lbl_preview_backend)
        start_col = QVBoxLayout()
        self.btn_plan = QPushButton("开始标注")
        self.btn_plan.setObjectName("StartBtn")
        self.btn_plan.clicked.connect(self.on_start)
        hint = QLabel("点击开始，先试标若干张，确认后再全量")
        hint.setObjectName("Hint")
        hint.setAlignment(Qt.AlignCenter)
        start_col.addWidget(self.btn_plan)
        start_col.addWidget(hint)
        r4.addLayout(meta, 1)
        r4.addLayout(start_col, 0)
        col.addWidget(c4)
        col.addStretch(1)

        outer.addWidget(scroll_host, 1)

        # compatibility aliases used by older tests
        self._actions_layout = start_col
        self.ed_desc = QLineEdit()
        self.ed_exclude = QLineEdit()
        self.sp_preview_value = 5
        return page

    def _build_page_preview(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(36, 28, 36, 20)
        title = QLabel("预览标注")
        title.setObjectName("PageTitle")
        self.lbl_preview_empty = QLabel("试标完成后，这里会显示 overlay / mask。先在「新建任务」开始标注。")
        self.lbl_preview_empty.setObjectName("Hint")
        self.lbl_preview_empty.setWordWrap(True)
        self.lbl_preview_empty.setAlignment(Qt.AlignCenter)
        empty = self._card()
        el = QVBoxLayout(empty)
        el.setContentsMargins(40, 60, 40, 60)
        emoji = QLabel("👁")
        emoji.setAlignment(Qt.AlignCenter)
        emoji.setStyleSheet("font-size: 40px;")
        el.addWidget(emoji)
        el.addWidget(self.lbl_preview_empty)
        self._preview_empty_card = empty
        lay.addWidget(title)
        lay.addWidget(empty, 1)
        btns = QHBoxLayout()
        self.btn_open_gallery = QPushButton("打开试标画廊")
        self.btn_open_gallery.clicked.connect(self._open_gallery)
        self.btn_open_gallery.setEnabled(False)
        self.btn_to_batch = QPushButton("确认并批量标注")
        self.btn_to_batch.setObjectName("Primary")
        self.btn_to_batch.clicked.connect(self._start_batch)
        self.btn_to_batch.setEnabled(False)
        btns.addStretch(1)
        btns.addWidget(self.btn_open_gallery)
        btns.addWidget(self.btn_to_batch)
        lay.addLayout(btns)
        return page

    def _build_page_batch(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(36, 28, 36, 20)
        title = QLabel("批量处理")
        title.setObjectName("PageTitle")
        sub = QLabel("任务队列、进度和日志")
        sub.setObjectName("Subtitle")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        actions = QHBoxLayout()
        self._actions_batch = actions
        self.btn_commit = QPushButton("Commit 到数据集")
        self.btn_commit.clicked.connect(self.on_commit)
        self.btn_commit.setEnabled(False)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        actions.addWidget(self.btn_commit)
        actions.addWidget(self.btn_stop)
        actions.addStretch(1)
        self.log_panel = LogPanel()
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addWidget(self.progress)
        lay.addLayout(actions)
        lay.addWidget(self.log_panel, 1)
        return page

    def _build_page_export(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(36, 28, 36, 20)
        title = QLabel("结果导出")
        title.setObjectName("PageTitle")
        card = self._card()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 28, 28, 28)
        self.lbl_export = QLabel("完成全量标注后，可将 work JSON Commit 为图片旁路 LabelMe 文件。")
        self.lbl_export.setObjectName("Hint")
        self.lbl_export.setWordWrap(True)
        btn = QPushButton("Commit 到数据集")
        btn.setObjectName("Primary")
        btn.clicked.connect(self.on_commit)
        self.btn_export_commit = btn
        cl.addWidget(self.lbl_export)
        cl.addSpacing(12)
        cl.addWidget(btn, 0, Qt.AlignLeft)
        cl.addStretch(1)
        lay.addWidget(title)
        lay.addWidget(card, 1)
        return page

    def _build_footer(self) -> QWidget:
        foot = QWidget()
        foot.setObjectName("Footer")
        foot.setFixedHeight(40)
        self._footer = foot
        row = QHBoxLayout(foot)
        row.setContentsMargins(28, 0, 28, 0)
        self.lbl_footer_status = QLabel("就绪")
        self.lbl_footer_count = QLabel("共 0 张图片")
        self.lbl_gpu = QLabel("GPU: —")
        row.addWidget(self.lbl_footer_status)
        row.addStretch(1)
        row.addWidget(self.lbl_footer_count)
        div = QLabel("|")
        div.setObjectName("Hint")
        row.addWidget(div)
        row.addWidget(self.lbl_gpu)
        self.status = self.lbl_footer_status  # compat
        return foot

    def _build_overlay(self, parent: QWidget) -> QWidget:
        overlay = QFrame(parent)
        overlay.setStyleSheet("background: rgba(246,248,252,210);")
        overlay.hide()
        box = QFrame(overlay)
        box.setObjectName("Card")
        box.setFixedWidth(460)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(28, 24, 28, 24)
        top = QHBoxLayout()
        h = QLabel("正在自动标注")
        h.setObjectName("SectionTitle")
        self.lbl_overlay_pct = QLabel("0%")
        self.lbl_overlay_pct.setStyleSheet("color:#6749ef;font-weight:800;")
        top.addWidget(h)
        top.addStretch(1)
        top.addWidget(self.lbl_overlay_pct)
        self.overlay_bar = QProgressBar()
        self.overlay_bar.setTextVisible(False)
        self.overlay_bar.setFixedHeight(9)
        self.lbl_overlay_status = QLabel("正在初始化…")
        self.lbl_overlay_status.setObjectName("Hint")
        bl.addLayout(top)
        bl.addWidget(self.overlay_bar)
        bl.addWidget(self.lbl_overlay_status)
        self._overlay_box = box
        return overlay

    def _show_overlay(self, text: str) -> None:
        self.lbl_overlay_status.setText(text)
        self.overlay_bar.setValue(0)
        self.lbl_overlay_pct.setText("0%")
        self._overlay.show()
        self._overlay.raise_()
        self._place_overlay()

    def _hide_overlay(self) -> None:
        self._overlay.hide()

    def _place_overlay(self) -> None:
        if not hasattr(self, "_overlay"):
            return
        geo = self.centralWidget().rect()
        self._overlay.setGeometry(geo)
        box = self._overlay_box
        box.move((geo.width() - box.width()) // 2, max(80, (geo.height() - 160) // 2))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._scale_debounce.start(40)
        self._place_overlay()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_responsive_scale(force=True)
        self._place_overlay()

    def _apply_responsive_scale(self, force: bool = False) -> None:
        changed = self._ui_scale.update_from_size(self.width(), self.height())
        compact = self.width() < 980
        if not force and not changed and compact == self._last_compact:
            return
        self._last_compact = compact
        self._apply_layout_metrics(compact)
        apply_window_styles(self)
        if hasattr(self, "log_panel"):
            self.log_panel.apply_ui_scale(self._ui_scale)

    def _apply_layout_metrics(self, compact: bool) -> None:
        s = self._ui_scale
        if hasattr(self, "_sidebar"):
            self._sidebar.setFixedWidth(s.px(88 if compact else 300, 72))
            for btn in self._nav_btns.values():
                btn.setText("" if compact else btn.property("full_text") or btn.text())
            if not compact:
                names = {
                    "new": "新建任务",
                    "preview": "预览标注",
                    "batch": "批量处理",
                    "export": "结果导出",
                }
                for k, v in names.items():
                    self._nav_btns[k].setText(v)
            if hasattr(self, "lbl_version"):
                self.lbl_version.setVisible(not compact)
        if hasattr(self, "ed_prompt"):
            self.ed_prompt.setMinimumHeight(s.px(56 if compact else 72, 48))
            self.ed_prompt.setMaximumHeight(s.px(88 if compact else 110, 56))
        if hasattr(self, "lbl_brand_sub"):
            self.lbl_brand_sub.setVisible(not compact)
        if hasattr(self, "_content_layout"):
            m = 18 if compact else 36
            self._content_layout.setContentsMargins(m, 20, m, 16)

    # ------------------------------------------------------------- interactions
    def _show_page(self, key: str) -> None:
        idx = {"new": 0, "preview": 1, "batch": 2, "export": 3}.get(key, 0)
        self.pages.setCurrentIndex(idx)
        btn = self._nav_btns.get(key)
        if btn:
            btn.setChecked(True)

    def toast(self, msg: str) -> None:
        self._toast.setText(msg)
        self._toast.adjustSize()
        geo = self.rect()
        self._toast.move(geo.width() - self._toast.width() - 28, geo.height() - 72)
        self._toast.show()
        self._toast.raise_()
        self._toast_timer.start(1900)

    def _active_backend_id(self) -> str:
        return str(self.backends_config.get("active_backend") or self.orch.backend.id)

    def _refresh_backend_chip(self) -> None:
        active = self._active_backend_id()
        caps = self.orch.backend.capabilities.to_dict()
        self.lbl_backend.setText(f"Backend: {active}")
        extra = "instance_mask" if caps.get("instance_mask") else "no-mask"
        self.lbl_preview_backend.setText(f"后端：{active}  ·  {extra}")
        opt = getattr(self.orch.backend, "options", {}) or {}
        if hasattr(self.orch.backend, "score_threshold"):
            self.sl_conf.setValue(int(round(float(self.orch.backend.score_threshold) * 100)))
        if hasattr(self.orch.backend, "min_area"):
            self.sl_area.setValue(int(self.orch.backend.min_area))
        del opt

    def _on_folder_changed(self, text: str) -> None:
        path = Path(text.strip()) if text.strip() else None
        if path and path.is_dir():
            images = list_images(path)
            self._image_count = len(images)
            self.ed_dataset.setStyleSheet("")
            self.lbl_preview_folder.setText(f"文件夹：{path.name}")
            self.lbl_preview_count.setText(f"预计处理：{self._image_count} 张图片")
            self.lbl_footer_count.setText(f"共 {self._image_count} 张图片")
        else:
            self._image_count = 0
            self.lbl_preview_folder.setText("文件夹：未选择")
            self.lbl_preview_count.setText("预计处理：0 张图片")
            self.lbl_footer_count.setText("共 0 张图片")
        self._sync_preview_meta()

    def _on_prompt_changed(self) -> None:
        text = self.ed_prompt.toPlainText()
        if len(text) > 100:
            self.ed_prompt.blockSignals(True)
            self.ed_prompt.setPlainText(text[:100])
            cursor = self.ed_prompt.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.ed_prompt.setTextCursor(cursor)
            self.ed_prompt.blockSignals(False)
            text = text[:100]
        self.lbl_chars.setText(f"{len(text)}/100")
        self._sync_preview_meta()

    def _sync_preview_meta(self) -> None:
        target = self.ed_prompt.toPlainText().strip() or self.ed_label.text().strip()
        self.lbl_preview_target.setText(f"目标：{target or '未设置'}")

    def _apply_chip(self, en: str, zh: str, btn: QPushButton) -> None:
        for c in self._chip_btns:
            c.setChecked(c is btn)
        self.ed_label.setText(en)
        if not self.ed_prompt.toPlainText().strip():
            self.ed_prompt.setPlainText(f"标注图片中的{zh}")
        self._sync_preview_meta()

    def _clear_chips(self) -> None:
        for c in self._chip_btns:
            c.setChecked(False)
        self.ed_label.setFocus()

    def _on_conf_changed(self, value: int) -> None:
        v = max(0.10, min(0.95, value / 100.0))
        self.lbl_conf.setText(f"{v:.2f}")
        if hasattr(self.orch.backend, "score_threshold"):
            self.orch.backend.score_threshold = v

    def _on_area_changed(self, value: int) -> None:
        self.lbl_area.setText(f"{value} px")
        if hasattr(self.orch.backend, "min_area"):
            self.orch.backend.min_area = int(value)

    def browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.ed_dataset.setText(path)
            self.toast(f"已读取 {self._image_count} 张图片")

    def open_settings(self) -> None:
        dlg = SettingsDialog(
            self.settings,
            self.backends_config,
            self.settings_path,
            self.backends_path,
            parent=self,
        )
        if dlg.exec_() == SettingsDialog.Accepted:
            self.settings = dlg.settings
            self.backends_config = dlg.backends_config
            self.orch = self._make_orch()
            self._refresh_backend_chip()
            if hasattr(self, "log_panel"):
                self.log_panel.append("设置已保存并重新加载 backend")
            self.toast("设置已保存")
            self._probe_sam3_on_boot()

    def check_connection(self) -> None:
        self.btn_check.setEnabled(False)
        self.btn_check.setText("检查中…")
        QTimer.singleShot(50, self._do_check_connection)

    def _do_check_connection(self) -> None:
        try:
            if self._active_backend_id() == "sam3_local" or type(self.orch.backend).__name__ == "Sam3LocalBackend":
                opt = getattr(self.orch.backend, "options", None) or worker_options(self.backends_config)
                ok, data = probe_health(worker_url(opt), timeout=3.0)
                if ok and data.get("model_loaded"):
                    self.lbl_conn.setText(f"状态: 已连接  ({data.get('device') or 'ok'})")
                    self.lbl_gpu.setText(f"GPU: {'可用' if str(data.get('device')) == 'cuda' else data.get('device') or '—'}")
                    self.toast("SAM3 后端连接正常")
                elif ok:
                    self.lbl_conn.setText("状态: 模型加载中")
                    self.toast("Worker 在线，模型尚未加载完成")
                else:
                    self.lbl_conn.setText("状态: 未连接")
                    self.toast("SAM3 Worker 未运行，正在启动…")
                    self._start_sam3_worker()
            else:
                self.lbl_conn.setText(f"状态: {self._active_backend_id()} 已加载")
                self.toast(f"当前 backend：{self._active_backend_id()}")
        except Exception as e:
            self.lbl_conn.setText("状态: 失败")
            self.toast(str(e)[:80])
        finally:
            self.btn_check.setEnabled(True)
            self.btn_check.setText("检查连接")

    def _probe_sam3_on_boot(self) -> None:
        if self._active_backend_id() != "sam3_local" and type(self.orch.backend).__name__ != "Sam3LocalBackend":
            self.lbl_conn.setText(f"状态: {self._active_backend_id()}")
            return
        opt = getattr(self.orch.backend, "options", None) or worker_options(self.backends_config)
        ok, data = probe_health(worker_url(opt), timeout=1.2)
        if ok and data.get("model_loaded"):
            self._on_worker_ready(data)
            return
        if ok:
            self.lbl_conn.setText("状态: 模型加载中")
            self.lbl_ready.setText("●  加载中")
            return
        self.lbl_conn.setText("状态: 未连接（开始标注时自动启动）")
        self.lbl_ready.setText("●  待启动")

    def _start_sam3_worker(self) -> None:
        if self._boot and self._boot.isRunning():
            return
        self.lbl_conn.setText("状态: 启动 Worker…")
        self.lbl_ready.setText("●  连接中")
        opt = getattr(self.orch.backend, "options", None) or worker_options(self.backends_config)
        self._boot = _BootWorker(dict(opt), self)
        self._boot.log_received.connect(lambda m: self.log_panel.append(m) if hasattr(self, "log_panel") else None)
        self._boot.ready.connect(self._on_worker_ready)
        self._boot.failed.connect(self._on_worker_failed)
        self._boot.start()

    def _on_worker_ready(self, data: dict) -> None:
        device = str(data.get("device") or "")
        self.lbl_conn.setText(f"状态: 已连接  ({device or 'ok'})")
        self.lbl_ready.setText("●  就绪")
        self.lbl_footer_status.setText("就绪")
        self.lbl_gpu.setText("GPU: 可用" if device == "cuda" else f"GPU: {device or '—'}")
        self.toast("SAM3 Worker 已就绪")

    def _on_worker_failed(self, err: str) -> None:
        self.lbl_conn.setText("状态: 启动失败")
        self.lbl_ready.setText("●  未连接")
        self.lbl_footer_status.setText("SAM3 未就绪")
        QMessageBox.warning(self, "SAM3 Worker 启动失败", err)

    def _target_text(self) -> str:
        return self.ed_prompt.toPlainText().strip() or self.ed_label.text().strip()

    def _prepare_intent(self) -> bool:
        dataset = self.ed_dataset.text().strip()
        if not dataset:
            self.toast("请先选择图片文件夹")
            return False
        if not Path(dataset).is_dir():
            QMessageBox.warning(self, "路径无效", "请选择存在的图片文件夹")
            return False
        target = self._target_text()
        if not target:
            self.toast("请先填写标注目标")
            return False
        prompt = self.ed_prompt.toPlainText().strip()
        try:
            if prompt:
                self.orch.parse_prompt(prompt, dataset_path=dataset)
                if self.orch.state.intent:
                    self.orch.state.intent.preview_count = 5
                    short = self.ed_label.text().strip()
                    if short and self.orch.state.intent.targets:
                        from autolabel.intent import looks_like_simple_prompt

                        t0 = self.orch.state.intent.targets[0]
                        t0.model_prompt = short
                        if looks_like_simple_prompt(short) and not looks_like_simple_prompt(t0.label):
                            t0.label = short
            else:
                self.orch.make_manual_intent(
                    label=self.ed_label.text().strip(),
                    description="",
                    dataset_path=dataset,
                    exclude=[],
                    preview_count=5,
                    model_prompt=self.ed_label.text().strip(),
                )
        except Exception as e:
            QMessageBox.warning(self, "意图解析失败", str(e))
            return False
        if hasattr(self.orch.backend, "score_threshold"):
            self.orch.backend.score_threshold = self.sl_conf.value() / 100.0
        if hasattr(self.orch.backend, "min_area"):
            self.orch.backend.min_area = int(self.sl_area.value())
        return True

    def _set_busy(self, busy: bool) -> None:
        self.btn_plan.setEnabled(not busy)
        self.btn_commit.setEnabled(not busy and bool(self.orch.state.mapping))
        self.btn_export_commit.setEnabled(not busy and bool(self.orch.state.mapping))
        self.btn_stop.setEnabled(busy)
        self.btn_to_batch.setEnabled(not busy and bool(self._preview_results))

    def on_start(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self._prepare_intent():
            return
        if hasattr(self, "log_panel"):
            self.log_panel.append("—— 构建 Plan + 试标 ——")
        self._set_busy(True)
        self._show_overlay("正在解析任务并初始化…")
        self.worker = LabelWorker("plan", self.orch, dataset_path=self.ed_dataset.text().strip())
        self.worker.log_received.connect(self._on_log)
        self.worker.status_changed.connect(self._on_status)
        self.worker.plan_ready.connect(self._on_plan_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: None)
        self.worker.start()

    def on_plan(self) -> None:
        self.on_start()

    def _on_log(self, msg: str) -> None:
        if hasattr(self, "log_panel"):
            self.log_panel.append(msg)

    def _on_status(self, s: str) -> None:
        self.lbl_footer_status.setText(s)
        mapping = {
            "building_plan": "正在构建执行计划…",
            "plan_ready": "计划已就绪",
            "previewing": "正在识别并生成 mask…",
            "preview_ready": "试标完成",
            "batching": "正在全量标注…",
            "batch_done": "全量完成",
        }
        if s in mapping:
            self.lbl_overlay_status.setText(mapping[s])

    def _on_plan_ready(self, plan) -> None:
        self.lbl_overlay_status.setText(
            f"计划确认：{plan.pending_count} 张待处理，开始试标 {plan.preview_count} 张…"
        )
        self._start_preview()

    def _start_preview(self) -> None:
        self._set_busy(True)
        self.progress.setValue(0)
        self.overlay_bar.setValue(0)
        self.worker = LabelWorker("preview", self.orch)
        self.worker.log_received.connect(self._on_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_preview_ready(self, results) -> None:
        self._preview_results = list(results or [])
        self._hide_overlay()
        self.btn_open_gallery.setEnabled(True)
        self.btn_to_batch.setEnabled(True)
        n = len(self._preview_results)
        self.lbl_preview_empty.setText(f"试标完成 {n} 张。打开画廊查看 overlay，确认后再全量。")
        self._show_page("preview")
        self.toast(f"试标完成：{n} 张")
        self._open_gallery()

    def _open_gallery(self) -> None:
        if not self._preview_results:
            self.toast("还没有试标结果")
            return
        dlg = PreviewGalleryDialog(self._preview_results, self)
        if dlg.exec_() != PreviewGalleryDialog.Accepted:
            return
        if dlg.choice == "retry":
            self._show_page("new")
            self.toast("请修改描述或设置后重试")
            return
        if dlg.choice == "batch":
            self._start_batch()

    def _start_batch(self) -> None:
        if not self._preview_results:
            self.toast("请先完成试标")
            return
        self._show_page("batch")
        self._show_overlay("正在全量标注…")
        if hasattr(self, "log_panel"):
            self.log_panel.append("—— 全量标注 ——")
        self._set_busy(True)
        self.progress.setValue(0)
        self.worker = LabelWorker("batch", self.orch)
        self.worker.log_received.connect(self._on_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.batch_finished.connect(self._on_batch_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_batch_finished(self, report) -> None:
        self._hide_overlay()
        self.btn_commit.setEnabled(True)
        self.btn_export_commit.setEnabled(True)
        self.lbl_export.setText(
            f"处理 {report.get('processed')} 张\n"
            f"buckets={report.get('buckets')}\n"
            f"work: {report.get('run_dir')}\n\n"
            "确认无误后点击 Commit，写入图片旁路 JSON。"
        )
        self._show_page("export")
        self.toast(f"完成：已处理 {report.get('processed')} 张图片")

    def on_commit(self) -> None:
        if not self.orch.state.mapping:
            self.toast("请先完成全量标注")
            return
        ret = QMessageBox.question(
            self,
            "确认 Commit",
            "将 work/ 中的 JSON 写入图片旁路（默认跳过已有 JSON）。继续？",
        )
        if ret != QMessageBox.Yes:
            return
        self._set_busy(True)
        self.worker = LabelWorker("commit", self.orch)
        self.worker.log_received.connect(self._on_log)
        self.worker.commit_finished.connect(self._on_commit_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_commit_finished(self, summary) -> None:
        msg = (
            f"写入 {summary.get('committed_count')}，跳过 {summary.get('skipped_count')}，"
            f"错误 {summary.get('error_count')}"
        )
        self.lbl_export.setText("Commit 完成。\n" + msg)
        self.toast("Commit 完成")
        QMessageBox.information(self, "Commit 完成", msg)

    def on_stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if hasattr(self, "log_panel"):
                self.log_panel.append("已请求停止…")
            self.toast("已请求停止")

    def _on_progress(self, cur: int, total: int, path: str) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(cur)
            self.overlay_bar.setMaximum(total)
            self.overlay_bar.setValue(cur)
            pct = int(round(100.0 * cur / total))
            self.lbl_overlay_pct.setText(f"{pct}%")
        name = Path(path).name
        self.lbl_footer_status.setText(f"{cur}/{total} {name}")
        self.lbl_overlay_status.setText(f"正在识别并生成 mask：{name}")

    def _on_failed(self, err: str) -> None:
        self._hide_overlay()
        if hasattr(self, "log_panel"):
            self.log_panel.append(f"[ERROR] {err}")
        QMessageBox.critical(self, "失败", err)
        self._set_busy(False)
