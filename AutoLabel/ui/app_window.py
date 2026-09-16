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
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
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
from autolabel.sam3_worker_ctl import (
    ensure_worker,
    probe_health,
    stop_owned_worker,
    worker_options,
    worker_url,
)
from autolabel.paths import (
    app_root,
    backends_path,
    bundled_config_dir,
    ensure_user_config,
    settings_path,
)
from ui.backend_worker import LabelWorker, NameSuggestWorker, _qt_alive
from ui.dialogs.name_suggest import NameSuggestDialog
from ui.dialogs.preview_gallery import PreviewGalleryDialog
from ui.dialogs.settings_dialog import SettingsDialog
from ui.styles import apply_window_styles
from ui.widgets.flow_layout import FlowLayout, VScrollHost, layout_height_for_width
from ui.widgets.log_panel import LogPanel
from ui.widgets.ui_scale import UIScale

AUTO_LABEL_ROOT = app_root()


class _Card(QFrame):
    """Card whose layout height follows available width (wrap / stack)."""

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return self.layout() is not None

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        lay = self.layout()
        if lay is None:
            return super().heightForWidth(width)
        return layout_height_for_width(lay, width)


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
        self.setMinimumSize(720, 520)
        self.resize(max(UIScale.REF_W, 1180), max(UIScale.REF_H, 760))
        self._ui_scale = UIScale()
        self._last_compact = False
        self._scale_debounce = QTimer(self)
        self._scale_debounce.setSingleShot(True)
        self._scale_debounce.timeout.connect(self._apply_responsive_scale)

        self.settings_path = settings_path()
        self.backends_path = backends_path()
        self.settings = self._load_settings()
        self.backends_config = load_backends_config(self.backends_path)
        self.orch = self._make_orch()
        self.worker: Optional[LabelWorker] = None
        self._name_suggest: Optional[NameSuggestWorker] = None
        self._boot: Optional[_BootWorker] = None
        self._retired_threads: list = []
        self._preview_after_plan = False
        self._pending_plan = None
        self._image_count = 0
        self._preview_results = []
        self._stackable_layouts: list = []
        self._page_layouts: list = []
        self._scaled_fixed: list = []
        self._stack_policies: dict = {}

        self._build_ui()
        self._refresh_backend_chip()
        self._apply_responsive_scale(force=True)
        # Offscreen tests / CI must not block on a 9892 health probe.
        import os

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QTimer.singleShot(200, self._probe_sam3_on_boot)

    def _load_settings(self) -> Dict[str, Any]:
        ensure_user_config()
        for path in (
            self.settings_path,
            bundled_config_dir() / "settings.example.json",
            AUTO_LABEL_ROOT / "config" / "settings.example.json",
        ):
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return {}

    def _make_orch(self) -> AutoLabelOrchestrator:
        from autolabel.engines import normalize_engine

        engine = normalize_engine(
            self.backends_config.get("annotation_engine") or self.backends_config.get("active_backend"),
            self.backends_config,
        )
        return AutoLabelOrchestrator(
            backends_config=self.backends_config,
            backend=build_backend_from_config(
                self.backends_config,
                engine,
                settings=self.settings,
            ),
            backend_id=engine,
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
        bar.setMinimumHeight(56)
        self._topbar = bar
        row = QHBoxLayout(bar)
        row.setContentsMargins(24, 0, 20, 0)
        row.setSpacing(12)
        self._top_layout = row

        logo = QLabel("✦")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "background:#6c4cff;color:white;border-radius:9px;font-weight:800;font-size:14px;"
        )
        self._logo = logo
        self._scaled_fixed.append((logo, 30, 30, 24))
        title = QLabel("AutoLabel")
        title.setObjectName("BrandTitle")
        title.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sub = QLabel("自动标注工具")
        sub.setObjectName("BrandSub")
        self.lbl_brand_sub = sub
        row.addWidget(logo, 0, Qt.AlignVCenter)
        row.addWidget(title, 0, Qt.AlignVCenter)
        row.addWidget(sub, 0, Qt.AlignVCenter)
        row.addStretch(1)

        self.lbl_ready = QLabel("●  就绪")
        self.lbl_ready.setObjectName("ReadyPill")
        self.lbl_ready.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        btn_settings = QPushButton("⚙  设置")
        btn_settings.setObjectName("Ghost")
        btn_settings.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        btn_settings.clicked.connect(self.open_settings)
        self.btn_settings = btn_settings
        row.addWidget(self.lbl_ready, 0, Qt.AlignVCenter)
        row.addWidget(btn_settings, 0, Qt.AlignVCenter)
        return bar

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setMinimumWidth(168)
        self._sidebar = side
        col = QVBoxLayout(side)
        col.setContentsMargins(18, 28, 16, 16)
        col.setSpacing(8)
        self._sidebar_layout = col

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
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setProperty("full_text", text)
            btn.setToolTip(text)
            btn.clicked.connect(lambda _=False, k=key: self._show_page(k))
            self._nav_group.addButton(btn)
            self._nav_btns[key] = btn
            col.addWidget(btn)
        self._nav_btns["new"].setChecked(True)

        col.addStretch(1)

        card = QWidget()
        card.setObjectName("BackendCard")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(16, 16, 16, 16)
        self._backend_card = card
        self._backend_card_layout = card_l
        h = QLabel("后端信息")
        h.setObjectName("SectionTitle")
        h.setWordWrap(True)
        self.lbl_backend = QLabel("引擎: —")
        self.lbl_backend.setObjectName("Hint")
        self.lbl_backend.setWordWrap(True)
        self.lbl_conn = QLabel("状态: 未检查")
        self.lbl_conn.setObjectName("Hint")
        self.lbl_conn.setWordWrap(True)
        check = QPushButton("检查连接")
        check.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        wrap.setMinimumWidth(0)
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        self.pages.setMinimumWidth(0)
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
        card = _Card()
        card.setObjectName("Card")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.setMinimumWidth(0)
        return card

    def _section_label(self, number: str, title: str, subtitle: str) -> QWidget:
        box = QWidget()
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        box.setMinimumWidth(0)
        box.setProperty("stack_section", True)
        box.setMaximumWidth(280)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        badge = QLabel(number)
        badge.setObjectName("NumberBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._scaled_fixed.append((badge, 31, 31, 24))
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(4)
        h = QLabel(title)
        h.setObjectName("SectionTitle")
        h.setWordWrap(True)
        h.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        h.setMinimumWidth(0)
        p = QLabel(subtitle)
        p.setObjectName("Hint")
        p.setWordWrap(True)
        p.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        p.setMinimumWidth(0)
        texts.addWidget(h)
        texts.addWidget(p)
        row.addWidget(badge, 0, Qt.AlignTop)
        row.addLayout(texts, 1)
        return box

    def _stackable_row(self, parent: QWidget, *, margins=(24, 22, 24, 22), spacing=20):
        """Horizontal on wide windows, vertical when the content column is tight."""
        lay = QHBoxLayout(parent)
        ml, mt, mr, mb = margins
        lay.setContentsMargins(ml, mt, mr, mb)
        lay.setSpacing(spacing)
        self._stackable_layouts.append((lay, spacing, margins))
        return lay

    def _scroll_page(self, inner: QWidget, margins=(36, 28, 36, 20)) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        ml, mt, mr, mb = margins
        outer.setContentsMargins(ml, mt, mr, mb)
        self._page_layouts.append((outer, margins))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(0)
        scroll.setWidget(inner)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(scroll, 1)
        return page

    def _build_page_new(self) -> QWidget:
        scroll_host = VScrollHost()
        col = QVBoxLayout(scroll_host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)
        col.setSizeConstraint(QLayout.SetNoConstraint)
        self._new_page_layout = col

        title_row = QHBoxLayout()
        step = QLabel("1")
        step.setObjectName("StepCircle")
        step.setAlignment(Qt.AlignCenter)
        self._scaled_fixed.append((step, 27, 27, 22))
        h1 = QLabel("新建任务")
        h1.setObjectName("PageTitle")
        h1.setWordWrap(True)
        title_row.addWidget(step, 0, Qt.AlignVCenter)
        title_row.addWidget(h1, 1)
        col.addLayout(title_row)
        sub = QLabel("只需三步，开始自动标注")
        sub.setObjectName("Subtitle")
        sub.setWordWrap(True)
        col.addWidget(sub)
        notice = QLabel(
            "自动标注可显著提高标注效率，但不能保证结果完全准确。"
            "请在 LabelMe 中对生成结果进行人工复核，重点检查漏标、误标及边界偏差。"
        )
        notice.setObjectName("Notice")
        notice.setWordWrap(True)
        notice.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_review_notice = notice
        col.addWidget(notice)

        # 1 folder
        c1 = self._card()
        r1 = self._stackable_row(c1)
        sec1 = self._section_label("1", "选择图片文件夹", "选择需要标注的图片所在文件夹")
        r1.addWidget(sec1, 0)
        folder_wrap = QWidget()
        folder_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        folder_wrap.setMinimumWidth(0)
        folder_col = QVBoxLayout(folder_wrap)
        folder_col.setContentsMargins(0, 0, 0, 0)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        self.ed_dataset = QLineEdit()
        self.ed_dataset.setPlaceholderText("请选择文件夹")
        self.ed_dataset.setMinimumWidth(0)
        self.ed_dataset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ed_dataset.textChanged.connect(self._on_folder_changed)
        browse = QPushButton("浏览")
        browse.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        browse.clicked.connect(self.browse_dataset)
        self.btn_browse = browse
        folder_row.addWidget(self.ed_dataset, 1)
        folder_row.addWidget(browse, 0)
        folder_col.addLayout(folder_row)
        r1.addWidget(folder_wrap, 1)
        col.addWidget(c1)

        # 2 prompt
        c2 = self._card()
        r2 = self._stackable_row(c2)
        r2.addWidget(self._section_label("2", "告诉我你要标注什么", "用自然语言描述要标注的目标"), 0)
        target_wrap = QWidget()
        target_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        target_wrap.setMinimumWidth(0)
        target_col = QVBoxLayout(target_wrap)
        target_col.setContentsMargins(0, 0, 0, 0)
        target_col.setSpacing(8)
        self.ed_prompt = QTextEdit()
        self.ed_prompt.setPlaceholderText("用自然语言描述要标注的目标")
        self.ed_prompt.setMinimumWidth(0)
        self.ed_prompt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ed_prompt.textChanged.connect(self._on_prompt_changed)
        self._nl_layout = target_col
        target_col.addWidget(self.ed_prompt)
        self.lbl_chars = QLabel("0")
        self.lbl_chars.setObjectName("Hint")
        self.lbl_chars.setAlignment(Qt.AlignRight)
        target_col.addWidget(self.lbl_chars)
        or_l = QLabel("或者直接输入目标名称 / 短语 / 短句")
        or_l.setObjectName("Hint")
        or_l.setWordWrap(True)
        target_col.addWidget(or_l)
        label_row = QHBoxLayout()
        label_row.setSpacing(8)
        self.ed_label = QLineEdit()
        self.ed_label.setPlaceholderText("目标名称或视觉描述")
        self.ed_label.setMinimumWidth(0)
        self.ed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ed_label.textChanged.connect(self._sync_preview_meta)
        self.btn_ai_name = QPushButton("AI 识别名称")
        self.btn_ai_name.setObjectName("Secondary")
        self.btn_ai_name.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_ai_name.setCursor(Qt.PointingHandCursor)
        self.btn_ai_name.setToolTip("看当前预览图，由模型自己写出给 SAM3.1 的英文 prompt")
        self.btn_ai_name.clicked.connect(self.on_ai_name)
        label_row.addWidget(self.ed_label, 1)
        label_row.addWidget(self.btn_ai_name, 0)
        target_col.addLayout(label_row)
        r2.addWidget(target_wrap, 1)
        col.addWidget(c2)

        # 3 advanced
        c3 = self._card()
        r3 = self._stackable_row(c3)
        r3.addWidget(self._section_label("3", "高级选项（可选）", "调整标注参数（新手可跳过）"), 0)
        adv_host = QWidget()
        adv_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        adv_host.setMinimumWidth(0)
        adv = QHBoxLayout(adv_host)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(24)
        self._adv_layout = adv
        self._stackable_layouts.append((adv, 24, (0, 0, 0, 0)))
        conf_box = QVBoxLayout()
        conf_lab = QLabel("置信度阈值")
        conf_lab.setObjectName("Hint")
        conf_row = QHBoxLayout()
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(10, 95)
        self.sl_conf.setValue(55)
        self.sl_conf.setSingleStep(5)
        self.sl_conf.setMinimumWidth(80)
        self.lbl_conf = QLabel("0.55")
        self.lbl_conf.setMinimumWidth(40)
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setObjectName("Hint")
        self.sl_conf.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.sl_conf, 1)
        conf_row.addWidget(self.lbl_conf, 0)
        conf_box.addWidget(conf_lab)
        conf_box.addLayout(conf_row)
        area_box = QVBoxLayout()
        area_lab = QLabel("最小目标面积")
        area_lab.setObjectName("Hint")
        area_row = QHBoxLayout()
        self.sl_area = QSlider(Qt.Horizontal)
        self.sl_area.setRange(16, 16384)
        self.sl_area.setValue(64)
        self.sl_area.setSingleStep(16)
        self.sl_area.setMinimumWidth(80)
        self.lbl_area = QLabel("64 px")
        self.lbl_area.setMinimumWidth(64)
        self.lbl_area.setAlignment(Qt.AlignCenter)
        self.lbl_area.setObjectName("Hint")
        self.sl_area.valueChanged.connect(self._on_area_changed)
        area_row.addWidget(self.sl_area, 1)
        area_row.addWidget(self.lbl_area, 0)
        area_box.addWidget(area_lab)
        area_box.addLayout(area_row)
        adv.addLayout(conf_box, 1)
        adv.addLayout(area_box, 1)
        r3.addWidget(adv_host, 1)
        col.addWidget(c3)

        # preview + start
        c4 = self._card()
        r4 = self._stackable_row(c4, margins=(22, 18, 22, 18))
        meta_w = QWidget()
        meta_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        meta_w.setMinimumWidth(0)
        meta = QVBoxLayout(meta_w)
        meta.setContentsMargins(0, 0, 0, 0)
        pt = QLabel("任务预览")
        pt.setObjectName("SectionTitle")
        self.lbl_preview_folder = QLabel("文件夹：未选择")
        self.lbl_preview_count = QLabel("预计处理：0 张图片")
        self.lbl_preview_target = QLabel("目标：未设置")
        self.lbl_preview_backend = QLabel("引擎：—")
        for w in (
            self.lbl_preview_folder,
            self.lbl_preview_count,
            self.lbl_preview_target,
            self.lbl_preview_backend,
        ):
            w.setObjectName("Hint")
            w.setWordWrap(True)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        meta.addWidget(pt)
        meta.addWidget(self.lbl_preview_folder)
        meta.addWidget(self.lbl_preview_count)
        meta.addWidget(self.lbl_preview_target)
        meta.addWidget(self.lbl_preview_backend)
        start_w = QWidget()
        start_w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        start_w.setMaximumWidth(260)
        start_col = QVBoxLayout(start_w)
        start_col.setContentsMargins(0, 0, 0, 0)
        self.btn_plan = QPushButton("开始标注")
        self.btn_plan.setObjectName("StartBtn")
        self.btn_plan.setMinimumWidth(140)
        self.btn_plan.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_plan.clicked.connect(self.on_start)
        hint = QLabel("点击开始，先试标若干张，确认后再全量")
        hint.setObjectName("Hint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        start_col.addWidget(self.btn_plan)
        start_col.addWidget(hint)
        r4.addWidget(meta_w, 1)
        r4.addWidget(start_w, 0)
        col.addWidget(c4)
        col.addStretch(1)

        page = self._scroll_page(scroll_host)
        self._content_layout = page.layout()

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
        self._page_layouts.append((lay, (36, 28, 36, 20)))
        title = QLabel("预览标注")
        title.setObjectName("PageTitle")
        title.setWordWrap(True)
        self.lbl_preview_empty = QLabel("试标完成后，这里会显示 overlay / mask。先在「新建任务」开始标注。")
        self.lbl_preview_empty.setObjectName("Hint")
        self.lbl_preview_empty.setWordWrap(True)
        self.lbl_preview_empty.setAlignment(Qt.AlignCenter)
        empty = self._card()
        empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        el = QVBoxLayout(empty)
        el.setContentsMargins(24, 40, 24, 40)
        emoji = QLabel("👁")
        emoji.setAlignment(Qt.AlignCenter)
        emoji.setStyleSheet("font-size: 40px;")
        el.addWidget(emoji)
        el.addWidget(self.lbl_preview_empty)
        self._preview_empty_card = empty
        lay.addWidget(title)
        lay.addWidget(empty, 1)
        btns_host = QWidget()
        btns = FlowLayout(btns_host, spacing=8)
        self.btn_open_gallery = QPushButton("打开试标画廊")
        self.btn_open_gallery.clicked.connect(self._open_gallery)
        self.btn_open_gallery.setEnabled(False)
        self.btn_to_batch = QPushButton("确认并批量标注")
        self.btn_to_batch.setObjectName("Primary")
        self.btn_to_batch.clicked.connect(self._start_batch)
        self.btn_to_batch.setEnabled(False)
        btns.addWidget(self.btn_open_gallery)
        btns.addWidget(self.btn_to_batch)
        lay.addWidget(btns_host)
        return page

    def _build_page_batch(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(36, 28, 36, 20)
        self._page_layouts.append((lay, (36, 28, 36, 20)))
        title = QLabel("批量处理")
        title.setObjectName("PageTitle")
        title.setWordWrap(True)
        sub = QLabel("任务队列、进度和日志")
        sub.setObjectName("Subtitle")
        sub.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMinimumWidth(0)
        actions_host = QWidget()
        actions = FlowLayout(actions_host, spacing=8)
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
        self.log_panel = LogPanel()
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addWidget(self.progress)
        lay.addWidget(actions_host)
        lay.addWidget(self.log_panel, 1)
        return page

    def _build_page_export(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(36, 28, 36, 20)
        self._page_layouts.append((lay, (36, 28, 36, 20)))
        title = QLabel("结果导出")
        title.setObjectName("PageTitle")
        title.setWordWrap(True)
        card = self._card()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 28, 28, 28)
        self.lbl_export = QLabel("完成全量标注后，可将 work JSON Commit 为图片旁路 LabelMe 文件。")
        self.lbl_export.setObjectName("Hint")
        self.lbl_export.setWordWrap(True)
        self.lbl_export.setTextInteractionFlags(Qt.TextSelectableByMouse)
        btn = QPushButton("Commit 到数据集")
        btn.setObjectName("Primary")
        btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
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
        foot.setMinimumHeight(32)
        self._footer = foot
        row = QHBoxLayout(foot)
        row.setContentsMargins(28, 0, 28, 0)
        self.lbl_footer_status = QLabel("就绪")
        self.lbl_footer_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.lbl_footer_count = QLabel("共 0 张图片")
        self.lbl_gpu = QLabel("GPU: —")
        row.addWidget(self.lbl_footer_status, 1)
        row.addWidget(self.lbl_footer_count, 0)
        div = QLabel("|")
        div.setObjectName("Hint")
        self._footer_div = div
        row.addWidget(div)
        row.addWidget(self.lbl_gpu, 0)
        self.status = self.lbl_footer_status  # compat
        return foot

    def _build_overlay(self, parent: QWidget) -> QWidget:
        overlay = QFrame(parent)
        overlay.setStyleSheet("background: rgba(246,248,252,210);")
        overlay.hide()
        box = QFrame(overlay)
        box.setObjectName("Card")
        box.setMinimumWidth(240)
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
        # Plan / LLM / SAM3 startup has no image count yet — use a busy bar,
        # otherwise the UI sits at 0% and looks frozen.
        self.overlay_bar.setRange(0, 0)
        self.progress.setRange(0, 0)
        self.lbl_overlay_pct.setText("…")
        self._overlay.show()
        self._overlay.raise_()
        self._place_overlay()

    def _hide_overlay(self) -> None:
        self._overlay.hide()
        self._reset_progress_idle()

    def _reset_progress_idle(self) -> None:
        self.overlay_bar.setRange(0, 100)
        self.overlay_bar.setValue(0)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_overlay_pct.setText("0%")

    def _place_overlay(self) -> None:
        if not hasattr(self, "_overlay"):
            return
        geo = self.centralWidget().rect()
        self._overlay.setGeometry(geo)
        box = self._overlay_box
        max_w = max(240, geo.width() - 48)
        box.setFixedWidth(min(460, max_w))
        box.adjustSize()
        bw = box.width()
        bh = max(box.sizeHint().height(), 120)
        box.setFixedHeight(bh)
        box.move(max(12, (geo.width() - bw) // 2), max(48, (geo.height() - bh) // 2))

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
        compact = self.width() < 1080
        if not force and not changed and compact == self._last_compact:
            self._sync_stack_direction()
            return
        self._last_compact = compact
        self._apply_layout_metrics(compact)
        apply_window_styles(self)
        if hasattr(self, "log_panel"):
            self.log_panel.apply_ui_scale(self._ui_scale)
        self._sync_stack_direction()
        self._place_overlay()

    def _content_column_width(self) -> int:
        if hasattr(self, "pages") and self.pages.width() > 80:
            return self.pages.width()
        side = self._sidebar.width() if hasattr(self, "_sidebar") else 0
        if side <= 0:
            side = 280
        return max(0, self.width() - side)

    def _sync_stack_direction(self) -> None:
        stack = self._content_column_width() < 640
        for lay, spacing, _margins in self._stackable_layouts:
            direction = QVBoxLayout.TopToBottom if stack else QHBoxLayout.LeftToRight
            if lay.direction() != direction:
                lay.setDirection(direction)
            lay.setSpacing(max(10, spacing // 2) if stack else spacing)
            lay.invalidate()
            for i in range(lay.count()):
                item = lay.itemAt(i)
                w = item.widget() if item is not None else None
                if w is None:
                    continue
                key = id(w)
                if key not in self._stack_policies:
                    self._stack_policies[key] = w.sizePolicy()
                if stack:
                    w.setMaximumWidth(16777215)
                    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                else:
                    w.setSizePolicy(self._stack_policies[key])
                    if w.property("stack_section"):
                        w.setMaximumWidth(280)
                    else:
                        w.setMaximumWidth(16777215)
        if hasattr(self, "_page_new"):
            host = self._page_new.findChild(VScrollHost)
            if host is not None:
                host.updateGeometry()
            self._page_new.updateGeometry()

    def _apply_layout_metrics(self, compact: bool) -> None:
        s = self._ui_scale
        if hasattr(self, "_sidebar"):
            self._sidebar.setFixedWidth(s.px(200 if compact else 280, 168))
            names = {
                "new": "新建任务",
                "preview": "预览标注",
                "batch": "批量处理",
                "export": "结果导出",
            }
            for k, v in names.items():
                btn = self._nav_btns.get(k)
                if btn is not None:
                    btn.setText(v)
            if hasattr(self, "_sidebar_layout"):
                self._sidebar_layout.setContentsMargins(
                    s.px(12 if compact else 18, 8),
                    s.px(18 if compact else 28, 12),
                    s.px(10 if compact else 16, 8),
                    s.px(12 if compact else 16, 8),
                )
            if hasattr(self, "lbl_version"):
                self.lbl_version.setVisible(not compact)
        if hasattr(self, "_topbar"):
            self._topbar.setFixedHeight(s.px(56 if compact else 64, 48))
            self._top_layout.setContentsMargins(
                s.px(14 if compact else 24, 10),
                0,
                s.px(12 if compact else 20, 8),
                0,
            )
        if hasattr(self, "_footer"):
            self._footer.setFixedHeight(s.px(32 if compact else 36, 28))
            self._footer.layout().setContentsMargins(
                s.px(16 if compact else 28, 10),
                0,
                s.px(16 if compact else 28, 10),
                0,
            )
        for widget, w, h, minimum in self._scaled_fixed:
            widget.setFixedSize(s.px(w, minimum), s.px(h, minimum))
        if hasattr(self, "ed_prompt"):
            self.ed_prompt.setMinimumHeight(s.px(56 if compact else 72, 48))
            self.ed_prompt.setMaximumHeight(s.px(88 if compact else 110, 56))
        if hasattr(self, "lbl_brand_sub"):
            self.lbl_brand_sub.setVisible(not compact)
        if hasattr(self, "btn_browse"):
            self.btn_browse.setMinimumWidth(s.px(72 if compact else 88, 64))
        if hasattr(self, "btn_ai_name"):
            self.btn_ai_name.setMinimumWidth(s.px(100 if compact else 112, 88))
        m = 16 if compact else 32
        for lay, _base in self._page_layouts:
            lay.setContentsMargins(m, 16 if compact else 24, m, 12 if compact else 16)
        if hasattr(self, "_new_page_layout"):
            self._new_page_layout.setSpacing(s.px(10 if compact else 14, 8))
        for lay, spacing, margins in self._stackable_layouts:
            ml, mt, mr, mb = margins
            pad = 14 if compact else ml
            vpad = 14 if compact else mt
            lay.setContentsMargins(pad, vpad, pad, 14 if compact else mb)
        if hasattr(self, "_backend_card_layout"):
            pad = s.px(12 if compact else 16, 8)
            self._backend_card_layout.setContentsMargins(pad, pad, pad, pad)

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
        x = max(12, geo.width() - self._toast.width() - 28)
        y = max(12, geo.height() - self._toast.height() - 56)
        self._toast.move(x, y)
        self._toast.show()
        self._toast.raise_()
        self._toast_timer.start(1900)

    def _active_backend_id(self) -> str:
        return str(
            self.backends_config.get("annotation_engine")
            or self.backends_config.get("active_backend")
            or getattr(self.orch.backend, "id", "")
        )

    def _engine_display(self) -> str:
        from autolabel.engines import engine_label

        bid = str(getattr(self.orch.backend, "id", "") or self._active_backend_id())
        return engine_label(bid, self.backends_config)

    def _refresh_backend_chip(self) -> None:
        label = self._engine_display()
        caps = self.orch.backend.capabilities.to_dict()
        self.lbl_backend.setText(f"引擎: {label}")
        extra = "instance_mask" if caps.get("instance_mask") else "no-mask"
        self.lbl_preview_backend.setText(f"引擎：{label}  ·  {extra}")
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
            self.lbl_preview_folder.setToolTip(str(path))
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
        self.lbl_chars.setText(str(len(text)))
        self._sync_preview_meta()

    def _sync_preview_meta(self) -> None:
        target = self.ed_prompt.toPlainText().strip() or self.ed_label.text().strip()
        self.lbl_preview_target.setText(f"目标：{target or '未设置'}")

    def _apply_score_threshold(self, value: float) -> None:
        backend = self.orch.backend
        if hasattr(backend, "score_threshold"):
            backend.score_threshold = float(value)

    def _on_conf_changed(self, value: int) -> None:
        v = max(0.10, min(0.95, value / 100.0))
        self.lbl_conf.setText(f"{v:.2f}")
        self._apply_score_threshold(v)

    def _on_area_changed(self, value: int) -> None:
        self.lbl_area.setText(f"{value} px")
        if hasattr(self.orch.backend, "min_area"):
            self.orch.backend.min_area = int(value)

    def browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.ed_dataset.setText(path)
            self.toast(f"已读取 {self._image_count} 张图片")

    def on_ai_name(self) -> None:
        """Suggest SAM3 prompt names from the current preview image. Does not start labeling."""
        if self._worker_running():
            self.toast("标注进行中，请稍后再识别名称")
            return
        if self._worker_running(self._name_suggest):
            return
        dataset = self.ed_dataset.text().strip()
        if not dataset:
            QMessageBox.information(self, "AI 识别名称", "请先选择图片文件夹，或把一张图片路径填进输入框。")
            return
        path = Path(dataset)
        if not path.exists():
            QMessageBox.information(self, "AI 识别名称", "路径不存在，请重新选择图片文件夹。")
            return
        from autolabel.errors import NameSuggestError
        from autolabel.name_suggest import require_vision_credentials, sample_name_suggest_images

        try:
            require_vision_credentials(self.settings, self.backends_config)
            images = sample_name_suggest_images(dataset)
        except NameSuggestError as e:
            QMessageBox.information(self, "AI 识别名称", str(e))
            return
        self.btn_ai_name.setEnabled(False)
        self.btn_ai_name.setText("识别中…")
        names = "、".join(p.name for p in images[:3])
        extra = f" 等{len(images)}张" if len(images) > 3 else ""
        self.toast(f"随机抽 {len(images)} 张：{names}{extra}")
        worker = NameSuggestWorker(
            [str(p) for p in images],
            dict(self.settings),
            dict(self.backends_config),
            parent=self,
        )
        worker.finished_ok.connect(self._on_ai_name_ready)
        worker.failed.connect(self._on_ai_name_failed)
        worker.finished.connect(self._on_ai_name_thread_finished)
        old = self._name_suggest
        self._name_suggest = worker
        self._retire_thread(old)
        worker.start()

    def _on_ai_name_ready(self, candidates: list) -> None:
        dlg = NameSuggestDialog(list(candidates or []), parent=self)
        if dlg.exec_() == NameSuggestDialog.Accepted and dlg.selected_english:
            self.apply_suggested_name(dlg.selected_english)
            self.toast(f"已填入：{dlg.selected_english}")

    def _on_ai_name_failed(self, err: str) -> None:
        QMessageBox.warning(self, "AI 识别名称失败", err)

    def _on_ai_name_thread_finished(self) -> None:
        if hasattr(self, "btn_ai_name"):
            self.btn_ai_name.setText("AI 识别名称")
            busy = self._worker_running()
            self.btn_ai_name.setEnabled(not busy)

    def apply_suggested_name(self, english: str) -> None:
        """Fill the target fields only. Never auto-starts SAM3."""
        text = (english or "").strip()
        if not text:
            return
        self.ed_label.setText(text)
        self.ed_prompt.blockSignals(True)
        self.ed_prompt.setPlainText(text)
        self.ed_prompt.blockSignals(False)
        self.lbl_chars.setText(str(len(text)))
        self._sync_preview_meta()

    def open_settings(self) -> None:
        if self._worker_running():
            QMessageBox.information(self, "任务进行中", "请先停止当前标注任务，再修改设置。")
            return
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
            if self._needs_sam3():
                opt = getattr(self.orch.backend, "options", None) or worker_options(self.backends_config)
                ok, data = probe_health(worker_url(opt), timeout=3.0)
                if ok and data.get("model_loaded"):
                    self.lbl_conn.setText(f"状态: 已连接  ({data.get('device') or 'ok'})")
                    self.lbl_gpu.setText(f"GPU: {'可用' if str(data.get('device')) == 'cuda' else data.get('device') or '—'}")
                    self.toast("SAM3.1 后端连接正常")
                elif ok:
                    self.lbl_conn.setText("状态: 模型加载中")
                    self.toast("Worker 在线，模型尚未加载完成")
                else:
                    self.lbl_conn.setText("状态: 未连接")
                    self.toast("SAM3.1 Worker 未运行，正在启动…")
                    self._start_sam3_worker()
            else:
                self.lbl_conn.setText(f"状态: {self._engine_display()} 已加载")
                self.toast(f"当前引擎：{self._engine_display()}")
        except Exception as e:
            self.lbl_conn.setText("状态: 失败")
            self.toast(str(e)[:80])
        finally:
            self.btn_check.setEnabled(True)
            self.btn_check.setText("检查连接")

    def _needs_sam3(self) -> bool:
        bid = str(getattr(self.orch.backend, "id", "") or "")
        return bid in {"sam3_local", "sam3", "sam3_1_local", "sam3.1"} or type(self.orch.backend).__name__ == "Sam3LocalBackend"

    def _probe_sam3_on_boot(self) -> None:
        if not self._needs_sam3():
            self.lbl_conn.setText(f"状态: {self._engine_display()}")
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
        if self._worker_running(self._boot):
            return
        self.lbl_conn.setText("状态: 启动 Worker…")
        self.lbl_ready.setText("●  连接中")
        opt = getattr(self.orch.backend, "options", None) or worker_options(self.backends_config)
        boot = _BootWorker(dict(opt), self)
        boot.log_received.connect(lambda m: self.log_panel.append(m) if hasattr(self, "log_panel") else None)
        boot.ready.connect(self._on_worker_ready)
        boot.failed.connect(self._on_worker_failed)
        old = self._boot
        self._boot = boot
        self._retire_thread(old)
        boot.start()

    def _on_worker_ready(self, data: dict) -> None:
        device = str(data.get("device") or "")
        self.lbl_conn.setText(f"状态: 已连接  ({device or 'ok'})")
        self.lbl_ready.setText("●  就绪")
        self.lbl_footer_status.setText(f"{self._engine_display()} Ready")
        self.lbl_gpu.setText("GPU: 可用" if device == "cuda" else f"GPU: {device or '—'}")
        self.toast(f"{self._engine_display()} Ready")

    def _on_worker_failed(self, err: str) -> None:
        self.lbl_conn.setText("状态: 启动失败")
        self.lbl_ready.setText("●  未连接")
        self.lbl_footer_status.setText(f"{self._engine_display()} 未就绪")
        QMessageBox.warning(self, f"{self._engine_display()} Worker 启动失败", err)

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
        self._apply_score_threshold(self.sl_conf.value() / 100.0)
        if hasattr(self.orch.backend, "min_area"):
            self.orch.backend.min_area = int(self.sl_area.value())
        return True

    def _set_busy(self, busy: bool) -> None:
        self.btn_plan.setEnabled(not busy)
        self.btn_commit.setEnabled(not busy and bool(self.orch.state.mapping))
        self.btn_export_commit.setEnabled(not busy and bool(self.orch.state.mapping))
        self.btn_stop.setEnabled(busy)
        self.btn_to_batch.setEnabled(not busy and bool(self._preview_results))
        if hasattr(self, "btn_ai_name"):
            suggesting = self._worker_running(self._name_suggest)
            self.btn_ai_name.setEnabled(not busy and not suggesting)

    def _worker_running(self, thread=None) -> bool:
        obj = self.worker if thread is None else thread
        if not _qt_alive(obj):
            return False
        try:
            return bool(obj.isRunning())
        except RuntimeError:
            return False

    def _retire_thread(self, thread) -> None:
        """Disconnect a QThread and drop it after it actually finishes.

        Never terminate() — a live HTTP call inside LabelWorker.run() would
        then emit into a deleted C++ object (RuntimeError: wrapped C/C++
        object of type LabelWorker has been deleted).
        """
        if thread is None or not _qt_alive(thread):
            return
        for name in (
            "ready",
            "failed",
            "log_received",
            "plan_ready",
            "preview_ready",
            "batch_finished",
            "commit_finished",
            "status_changed",
            "progress",
            "finished_ok",
            "finished",
        ):
            sig = getattr(thread, name, None)
            if sig is None:
                continue
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass
        if hasattr(thread, "stop"):
            try:
                thread.stop()
            except RuntimeError:
                return

        def _drop() -> None:
            try:
                self._retired_threads.remove(thread)
            except ValueError:
                pass
            if _qt_alive(thread):
                thread.deleteLater()

        try:
            running = thread.isRunning()
        except RuntimeError:
            return
        if running:
            # Detach from the window so closeEvent cannot delete the C++ QThread
            # while HTTP is still in flight. quit() only affects event-loop
            # threads; LabelWorker.run() ignores it and exits on stop_flag.
            try:
                thread.setParent(None)
            except RuntimeError:
                pass
            if thread not in self._retired_threads:
                self._retired_threads.append(thread)
            try:
                thread.finished.connect(_drop)
            except (TypeError, RuntimeError):
                QTimer.singleShot(0, _drop)
            try:
                thread.quit()
            except RuntimeError:
                pass
            return
        # Already finished (e.g. plan → preview handoff). Delete after this slot.
        QTimer.singleShot(0, _drop)

    def _attach_worker(self, worker: LabelWorker) -> LabelWorker:
        old = self.worker
        self.worker = worker
        self._retire_thread(old)
        return worker

    def on_start(self) -> None:
        if self._worker_running():
            return
        if not self._prepare_intent():
            return
        if hasattr(self, "log_panel"):
            self.log_panel.append("—— 构建 Plan + 试标 ——")
        self._set_busy(True)
        self._show_overlay("正在解析意图…")
        self._preview_after_plan = True
        self._pending_plan = None
        worker = self._attach_worker(
            LabelWorker(
                "plan",
                self.orch,
                parent=self,
                dataset_path=self.ed_dataset.text().strip(),
                prompt=self.ed_prompt.toPlainText().strip(),
                short_label=self.ed_label.text().strip(),
                preview_count=5,
            )
        )
        worker.log_received.connect(self._on_log)
        worker.status_changed.connect(self._on_status)
        worker.plan_ready.connect(self._on_plan_ready)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_plan_thread_finished)
        worker.start()

    def on_plan(self) -> None:
        self.on_start()

    def _on_log(self, msg: str) -> None:
        text = str(msg).strip()
        if hasattr(self, "log_panel"):
            self.log_panel.append(text)
        if text and hasattr(self, "_overlay") and self._overlay.isVisible():
            self.lbl_overlay_status.setText(text[:180])

    def _on_status(self, s: str) -> None:
        self.lbl_footer_status.setText(s)
        mapping = {
            "parsing_intent": "正在用 LLM 解析意图…",
            "building_plan": "正在启动标注引擎并构建计划…",
            "plan_ready": "计划已就绪",
            "previewing": "正在识别并生成 mask…",
            "preview_ready": "试标完成",
            "batching": "正在全量标注…",
            "batch_done": "全量完成",
        }
        if s in mapping:
            self.lbl_overlay_status.setText(mapping[s])

    def _on_plan_ready(self, plan) -> None:
        self._pending_plan = plan
        self.lbl_overlay_status.setText(
            f"计划确认：{plan.pending_count} 张待处理，开始试标 {plan.preview_count} 张…"
        )

    def _on_plan_thread_finished(self) -> None:
        if not self._preview_after_plan or self._pending_plan is None:
            self._set_busy(False)
            return
        self._preview_after_plan = False
        # Defer so this LabelWorker's finished slot can return before we
        # retire / deleteLater it and start the preview thread.
        QTimer.singleShot(0, self._start_preview)

    def _start_preview(self) -> None:
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.overlay_bar.setRange(0, 0)
        self.lbl_overlay_pct.setText("…")
        worker = self._attach_worker(LabelWorker("preview", self.orch, parent=self))
        worker.log_received.connect(self._on_log)
        worker.progress.connect(self._on_progress)
        worker.preview_ready.connect(self._on_preview_ready)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_idle)
        worker.start()

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
        self.progress.setRange(0, 0)
        self.overlay_bar.setRange(0, 0)
        self.lbl_overlay_pct.setText("…")
        worker = self._attach_worker(LabelWorker("batch", self.orch, parent=self))
        worker.log_received.connect(self._on_log)
        worker.progress.connect(self._on_progress)
        worker.batch_finished.connect(self._on_batch_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_idle)
        worker.start()

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
        worker = self._attach_worker(LabelWorker("commit", self.orch, parent=self))
        worker.log_received.connect(self._on_log)
        worker.commit_finished.connect(self._on_commit_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_idle)
        worker.start()

    def _on_commit_finished(self, summary) -> None:
        msg = (
            f"写入 {summary.get('committed_count')}，跳过 {summary.get('skipped_count')}，"
            f"错误 {summary.get('error_count')}"
        )
        self.lbl_export.setText("Commit 完成。\n" + msg)
        self.toast("Commit 完成")
        QMessageBox.information(self, "Commit 完成", msg)

    def _on_worker_idle(self) -> None:
        self._set_busy(False)

    def on_stop(self) -> None:
        if self._worker_running():
            self.worker.stop()
            if hasattr(self, "log_panel"):
                self.log_panel.append("已请求停止…")
            self.toast("已请求停止")

    def _on_progress(self, cur: int, total: int, path: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(cur)
            self.overlay_bar.setRange(0, total)
            self.overlay_bar.setValue(cur)
            pct = int(round(100.0 * cur / total))
            self.lbl_overlay_pct.setText(f"{pct}%")
        name = Path(path).name
        self.lbl_footer_status.setText(f"{cur}/{total} {name}")
        self.lbl_overlay_status.setText(f"正在识别并生成 mask：{name}")

    def _on_failed(self, err: str) -> None:
        self._preview_after_plan = False
        self._pending_plan = None
        self._hide_overlay()
        if hasattr(self, "log_panel"):
            self.log_panel.append(f"[ERROR] {err}")
        QMessageBox.critical(self, "失败", err)
        self._set_busy(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._preview_after_plan = False
        self._pending_plan = None
        if self.worker is not None:
            self._retire_thread(self.worker)
            self.worker = None
        if self._boot is not None:
            self._retire_thread(self._boot)
            self._boot = None
        if self._name_suggest is not None:
            self._retire_thread(self._name_suggest)
            self._name_suggest = None
        try:
            opt = getattr(self.orch.backend, "options", None) or worker_options(self.backends_config)
            stop_owned_worker(opt, log=lambda m: self.log_panel.append(m) if hasattr(self, "log_panel") else None)
        except Exception:
            pass
        super().closeEvent(event)
