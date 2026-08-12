"""MainWindow：布局组装 + signal/slot 接线。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from backend import (
    AssistantSummaryWorker,
    PostTrainWorker,
    TrainWorker,
    resolve_post_train_context,
)
from data_quality_panel import DataQualityPanel
from dialogs.post_train import PostTrainDialog
from dialogs.templates import (
    ClickableFrame,
    QuickTemplatesDialog,
    SaveTemplateDialog,
)
from llm_settings_dialog import SettingsDialog
from suggestions import SmartSuggestionEngine as SuggestionEngine
from widgets.log_panel import CollapsibleLogPanel
from widgets.ui_scale import UIScale


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoTrain 4.0")
        self.setMinimumSize(780, 600)
        self.resize(UIScale.REF_W, UIScale.REF_H)
        self.worker: Optional[TrainWorker] = None
        self.post_worker: Optional[PostTrainWorker] = None
        self._summary_worker: Optional[AssistantSummaryWorker] = None
        self._ui_scale = UIScale()
        self._last_compact = False
        self._last_post_train_action: str = "report"
        self._last_task_type_hint: str = ""
        # 子进程若已通过 CHAT_RESPONSE 写过助手框，训练收尾摘要改为追加
        self._chat_from_subprocess: bool = False

        self.suggestion_engine: Optional[SuggestionEngine] = None
        # settings 尚未加载时先占位；_load_settings 后会 _apply_llm_settings_to_env + 重建

        self._ime_composing = False
        self._suggestion_debounce = QTimer(self)
        self._suggestion_debounce.setSingleShot(True)
        self._suggestion_debounce.timeout.connect(self._do_update_suggestions)
        self._scale_debounce = QTimer(self)
        self._scale_debounce.setSingleShot(True)
        self._scale_debounce.timeout.connect(self._apply_responsive_scale)

        # 空输入时点击「刷新建议」会轮换这几组默认模板
        self._default_suggestion_sets = [
            ["训练 OCR 模型", "训练 CLS 模型", "训练 DET 模型", "训练 DETR 模型", "训练 ISEG 模型"],
            ["使用随机旋转增强", "训练完成后自动测试", "对比 Baseline 模型", "导出 ONNX", "Epoch 设置为 20"],
            ["不使用数据增强", "生成推理报告", "批量训练多个模型", "设置 batch_size=8", "训练完成后关闭浏览器"],
        ]
        self._current_suggestion_set_index = 0

        # 快捷模板：默认 + 用户自定义（持久化到 JSON），最多 8 个
        self.MAX_QUICK_TEMPLATES = 8
        self._quick_templates_file = Path(__file__).parent / "quick_templates.json"
        self._quick_templates = self._load_quick_templates()

        # 系统设置持久化
        self._settings_file = Path(__file__).parent / "settings.json"
        self._settings = self._load_settings()
        # 若刚从旧扁平字段迁移出 profiles，立刻写回，避免下次再猜
        try:
            if self._settings.get("llm_profiles"):
                self._settings_file.write_text(
                    json.dumps(self._settings, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass
        # 设置优先：同步到环境 + 重建本进程 LLM，训练子进程也会再注入一份
        self._apply_llm_settings_to_env()
        self._rebuild_suggestion_llm()

        self._setup_ui()
        self._apply_responsive_scale(force=True)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # 拖拽缩放时防抖，避免每像素重算样式
        self._scale_debounce.start(40)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_responsive_scale(force=True)

    def _apply_responsive_scale(self, force: bool = False) -> None:
        """窗口尺寸变化时，等比例缩放字体、间距、固定控件尺寸。"""
        changed = self._ui_scale.update_from_size(self.width(), self.height())
        compact = self.width() < 1080
        if not force and not changed and compact == self._last_compact:
            return
        self._last_compact = compact
        self._apply_layout_metrics(compact)
        self._apply_styles()
        if hasattr(self, "log_panel"):
            self.log_panel.apply_ui_scale(self._ui_scale)
        if hasattr(self, "chat_response_box"):
            self._apply_chat_box_style()
        if hasattr(self, "data_quality_panel"):
            self.data_quality_panel.apply_ui_scale(self._ui_scale)
        # 模板卡片内部 spacing 也随缩放
        self._refresh_template_item_metrics()

    def _apply_layout_metrics(self, compact: bool) -> None:
        s = self._ui_scale
        gap = s.px(16 if compact else 22, 10)
        side = s.px(200 if compact else 250, 160)

        if hasattr(self, "_sidebar"):
            self._sidebar.setFixedWidth(side)
            self._sidebar_layout.setSpacing(s.px(18 if compact else 24, 10))
            self._sidebar_layout.setContentsMargins(
                s.px(14 if compact else 18, 8),
                s.px(18 if compact else 24, 10),
                s.px(14 if compact else 18, 8),
                s.px(18 if compact else 24, 10),
            )
        if hasattr(self, "_brand_mark"):
            mark = s.px(34 if compact else 38, 28)
            self._brand_mark.setFixedSize(mark, mark)
        if hasattr(self, "_nav_layout"):
            self._nav_layout.setSpacing(s.px(6 if compact else 8, 4))
        if hasattr(self, "_status_layout"):
            self._status_layout.setSpacing(s.px(6 if compact else 8, 4))

        if hasattr(self, "_main_vlayout"):
            self._main_vlayout.setSpacing(gap)
            self._main_vlayout.setContentsMargins(
                s.px(18 if compact else 28, 12),
                s.px(16 if compact else 24, 10),
                s.px(20 if compact else 34, 12),
                s.px(20 if compact else 32, 12),
            )
        if hasattr(self, "_content_layout"):
            self._content_layout.setSpacing(gap)
        if hasattr(self, "_topbar_layout"):
            self._topbar_layout.setSpacing(s.px(10 if compact else 12, 6))
        if hasattr(self, "_left_col_layout"):
            self._left_col_layout.setSpacing(gap)
        if hasattr(self, "_right_col_layout"):
            self._right_col_layout.setSpacing(gap)
        if hasattr(self, "_prompt_card_layout"):
            self._prompt_card_layout.setSpacing(s.px(12 if compact else 16, 8))
        if hasattr(self, "prompt_input"):
            self.prompt_input.setMaximumHeight(s.px(100 if compact else 125, 72))
            self.prompt_input.setMinimumHeight(s.px(72 if compact else 90, 56))
        if hasattr(self, "suggestions_container"):
            self.suggestions_container.setMinimumHeight(s.px(30 if compact else 36, 24))
        if hasattr(self, "suggestions_layout"):
            self.suggestions_layout.setSpacing(s.px(8 if compact else 10, 4))
        if hasattr(self, "_sugg_header_layout"):
            self._sugg_header_layout.setSpacing(s.px(6 if compact else 8, 4))
        if hasattr(self, "_btn_row_layout"):
            self._btn_row_layout.setSpacing(s.px(8 if compact else 12, 6))
        if hasattr(self, "_metrics_layout"):
            self._metrics_layout.setSpacing(s.px(8 if compact else 12, 6))
        for layout in getattr(self, "_metric_item_layouts", []):
            layout.setSpacing(s.px(4 if compact else 6, 2))
        if hasattr(self, "_tpl_grid_layout") and self._tpl_grid_layout is not None:
            self._tpl_grid_layout.setSpacing(s.px(8 if compact else 10, 4))
        if hasattr(self, "chat_response_box"):
            self.chat_response_box.setMinimumHeight(s.px(120 if compact else 150, 90))

    def _refresh_template_item_metrics(self) -> None:
        if not hasattr(self, "quick_templates_grid"):
            return
        s = self._ui_scale
        layout = self.quick_templates_grid.layout()
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            inner = widget.layout()
            if inner is not None:
                inner.setSpacing(s.px(4, 2))
                inner.setContentsMargins(
                    s.px(8, 4),
                    s.px(8, 4),
                    s.px(8, 4),
                    s.px(8, 4),
                )

    def _apply_chat_box_style(self) -> None:
        s = self._ui_scale
        self.chat_response_box.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background: #f7f5ff;
                border: 1px solid #e6e1ff;
                border-radius: {s.qss_px(14, 8)};
                padding: {s.qss_px(14, 8)};
                font-size: {s.font(15)}px;
                color: #4c1d95;
                line-height: 1.7;
            }}
            """
        )

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        self._root_layout = root

        # ---------------- 左侧边栏 ----------------
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # ---------------- 主内容区 ----------------
        main_area = QWidget()
        main_area.setObjectName("mainArea")
        main_vlayout = QVBoxLayout(main_area)
        main_vlayout.setSpacing(22)
        main_vlayout.setContentsMargins(28, 24, 34, 32)
        self._main_vlayout = main_vlayout

        # 顶栏
        topbar = self._build_topbar()
        main_vlayout.addWidget(topbar)

        # 两列内容
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setSpacing(22)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setAlignment(Qt.AlignTop)
        self._content_layout = content_layout

        left_col = self._build_left_column()
        # 左栏（训练需求 + 执行控制台）略窄于原先 2:1，给右侧模板/任务更多空间
        content_layout.addWidget(left_col, 3)

        right_col = self._build_right_column()
        content_layout.addWidget(right_col, 2)

        main_vlayout.addWidget(content, 1)
        root.addWidget(main_area, 1)

        # 初始化建议气泡
        self._do_update_suggestions()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        self._sidebar = sidebar
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(24)
        layout.setContentsMargins(18, 24, 18, 24)
        self._sidebar_layout = layout

        # 品牌
        brand = QHBoxLayout()
        brand_mark = QLabel("AT")
        brand_mark.setObjectName("brandMark")
        brand_mark.setAlignment(Qt.AlignCenter)
        brand_mark.setFixedSize(38, 38)
        self._brand_mark = brand_mark
        brand.addWidget(brand_mark)

        brand_text = QVBoxLayout()
        brand_title = QLabel("AutoTrain 4.0")
        brand_title.setObjectName("brandTitle")
        brand_sub = QLabel("DLCV AI Automation")
        brand_sub.setObjectName("brandSub")
        brand_text.addWidget(brand_title)
        brand_text.addWidget(brand_sub)
        brand.addLayout(brand_text)
        brand.addStretch()
        layout.addLayout(brand)

        # 导航
        nav = QVBoxLayout()
        nav.setSpacing(8)
        self._nav_layout = nav
        nav_items = [
            ("⌂", "工作台", True, None),
            ("⚙", "系统设置", False, self._open_settings),
        ]
        for icon, name, active, handler in nav_items:
            btn = QPushButton(f"{icon}  {name}")
            btn.setObjectName("navBtnActive" if active else "navBtn")
            btn.setCursor(Qt.PointingHandCursor)
            if handler is not None:
                btn.clicked.connect(handler)
            nav.addWidget(btn)
        layout.addLayout(nav)

        layout.addStretch()

        # 状态卡片
        status_card = QFrame()
        status_card.setObjectName("sideCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setSpacing(8)
        self._status_layout = status_layout
        status_title = QLabel("平台状态")
        status_title.setObjectName("sideCardTitle")
        status_desc = QLabel("Backend 9887 与 Frontend 8843 均已连接。")
        status_desc.setObjectName("sideCardDesc")
        status_desc.setWordWrap(True)
        status_badge = QLabel("运行正常")
        status_badge.setObjectName("sideCardBadge")
        status_layout.addWidget(status_title)
        status_layout.addWidget(status_desc)
        status_layout.addWidget(status_badge)
        layout.addWidget(status_card)

        return sidebar

    def _build_topbar(self) -> QWidget:
        topbar = QWidget()
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setSpacing(12)
        topbar_layout.setContentsMargins(0, 0, 0, 0)
        self._topbar_layout = topbar_layout

        title_col = QVBoxLayout()
        title = QLabel("自动化训练工作台")
        title.setObjectName("pageTitle")
        subtitle = QLabel("用自然语言创建、监控并管理你的视觉训练任务。")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        topbar_layout.addLayout(title_col, 1)

        topbar_layout.addStretch()

        doc_btn = QPushButton("查看文档")
        doc_btn.setObjectName("ghostBtn")
        doc_btn.setCursor(Qt.PointingHandCursor)
        doc_btn.clicked.connect(self._open_readme)
        topbar_layout.addWidget(doc_btn)

        return topbar

    def _build_left_column(self) -> QWidget:
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setSpacing(22)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setAlignment(Qt.AlignTop)
        self._left_col_layout = col_layout

        # 训练需求卡片
        prompt_card = QFrame()
        prompt_card.setObjectName("card")
        card_layout = QVBoxLayout(prompt_card)
        card_layout.setSpacing(16)
        self._prompt_card_layout = card_layout

        header = QHBoxLayout()
        header_title = QLabel("训练需求")
        header_title.setObjectName("cardTitle")
        header.addWidget(header_title)
        header.addStretch()
        tag = QLabel("AI Intent")
        tag.setObjectName("tag")
        header.addWidget(tag)
        card_layout.addLayout(header)

        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText(
            "例如：训练一个 cls 模型来辨别王不留行和白芷"
        )
        self.prompt_input.setMaximumHeight(125)
        self.prompt_input.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.prompt_input.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.prompt_input.setInputMethodHints(Qt.ImhNone)
        self.prompt_input.textChanged.connect(self._on_input_text_changed)
        self.prompt_input.installEventFilter(self)
        card_layout.addWidget(self.prompt_input)

        sugg_header = QHBoxLayout()
        sugg_header.setSpacing(8)
        self._sugg_header_layout = sugg_header
        sugg_title = QLabel("猜你想说")
        sugg_title.setObjectName("sectionTitle")
        sugg_header.addWidget(sugg_title)
        sugg_header.addStretch()
        refresh_btn = QPushButton("刷新建议")
        refresh_btn.setObjectName("tinyGhostBtn")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_suggestions)
        sugg_header.addWidget(refresh_btn)
        card_layout.addLayout(sugg_header)

        self.suggestions_container = QWidget()
        self.suggestions_container.setObjectName("suggestionsContainer")
        self.suggestions_container.setMinimumHeight(36)
        self.suggestions_layout = QHBoxLayout(self.suggestions_container)
        self.suggestions_layout.setSpacing(10)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestions_layout.setAlignment(Qt.AlignLeft)
        card_layout.addWidget(self.suggestions_container)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._btn_row_layout = btn_row
        self.start_btn = QPushButton("▶ 提交需求")
        self.start_btn.setObjectName("runBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_run)

        self.clear_btn = QPushButton("清空任务")
        self.clear_btn.setObjectName("softBtn")
        self.clear_btn.clicked.connect(self._clear_logs)

        save_tpl_btn = QPushButton("保存为模板")
        save_tpl_btn.setObjectName("secondaryBtn")
        save_tpl_btn.setCursor(Qt.PointingHandCursor)
        save_tpl_btn.clicked.connect(self._save_current_as_template)

        self.close_browser_btn = QPushButton("关闭浏览器")
        self.close_browser_btn.setObjectName("closeBrowserBtn")
        self.close_browser_btn.setEnabled(False)
        self.close_browser_btn.setCursor(Qt.PointingHandCursor)
        self.close_browser_btn.clicked.connect(self._close_browser)

        btn_row.addWidget(self.start_btn, 2)
        btn_row.addWidget(self.clear_btn, 1)
        btn_row.addWidget(save_tpl_btn, 1)
        btn_row.addWidget(self.close_browser_btn, 1)
        card_layout.addLayout(btn_row)

        # 指标概览
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self._metrics_layout = metrics
        self._metric_item_layouts: list = []
        self.metric_labels = {}
        for key, label in [
            ("task_type", "任务类型"),
            ("dataset", "匹配数据集"),
            ("estimate", "预计耗时"),
        ]:
            m = QFrame()
            m.setObjectName("metric")
            mv = QVBoxLayout(m)
            mv.setSpacing(6)
            self._metric_item_layouts.append(mv)
            ms = QLabel(label)
            ms.setObjectName("metricLabel")
            mv.addWidget(ms)
            mv_val = QLabel("-")
            mv_val.setObjectName("metricValue")
            mv_val.setWordWrap(True)
            self.metric_labels[key] = mv_val
            mv.addWidget(mv_val)
            metrics.addWidget(m)
        card_layout.addLayout(metrics)

        col_layout.addWidget(prompt_card)

        # 运行日志卡片
        log_card = QFrame()
        log_card.setObjectName("card")
        log_layout = QVBoxLayout(log_card)
        log_header = QHBoxLayout()
        log_title = QLabel("执行控制台")
        log_title.setObjectName("cardTitle")
        log_header.addWidget(log_title)
        log_header.addStretch()
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusBadge")
        log_header.addWidget(self.status_label)
        log_layout.addLayout(log_header)

        self.log_panel = CollapsibleLogPanel()
        self.log_panel.setObjectName("logPanel")
        self.log_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_panel)
        col_layout.addWidget(log_card, 1)

        return col

    def _build_right_column(self) -> QWidget:
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setSpacing(22)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setAlignment(Qt.AlignTop)
        self._right_col_layout = col_layout

        # 快捷模板
        tpl_card = QFrame()
        tpl_card.setObjectName("card")
        tpl_layout = QVBoxLayout(tpl_card)
        tpl_header = QHBoxLayout()
        tpl_title = QLabel("快捷模板")
        tpl_title.setObjectName("cardTitle")
        tpl_header.addWidget(tpl_title)
        tpl_header.addStretch()
        manage_btn = QPushButton("管理")
        manage_btn.setObjectName("tinyGhostBtn")
        manage_btn.setCursor(Qt.PointingHandCursor)
        manage_btn.clicked.connect(self._open_template_manager)
        tpl_header.addWidget(manage_btn)
        tpl_tag = QLabel("效率工具")
        tpl_tag.setObjectName("tag")
        tpl_header.addWidget(tpl_tag)
        tpl_layout.addLayout(tpl_header)

        self.quick_templates_grid = QWidget()
        self._tpl_grid_layout = None
        self._render_quick_templates()
        tpl_layout.addWidget(self.quick_templates_grid)
        col_layout.addWidget(tpl_card)

        # 数据质检（独立模块，不介入训练子进程）
        self.data_quality_panel = DataQualityPanel()
        self.data_quality_panel.summary_ready.connect(self._on_quality_summary_ready)
        self.data_quality_panel.summary_failed.connect(self._on_quality_summary_failed)
        col_layout.addWidget(self.data_quality_panel)

        # 助手回复卡片（永久显示在右侧）
        self.chat_response_card = QFrame()
        self.chat_response_card.setObjectName("card")
        chat_layout = QVBoxLayout(self.chat_response_card)
        chat_header = QHBoxLayout()
        chat_title = QLabel("💬 助手回复")
        chat_title.setObjectName("cardTitle")
        chat_header.addWidget(chat_title)
        chat_header.addStretch()
        chat_tag = QLabel("通用对话")
        chat_tag.setObjectName("tag")
        chat_header.addWidget(chat_tag)
        chat_layout.addLayout(chat_header)

        self.chat_response_box = QPlainTextEdit()
        self.chat_response_box.setReadOnly(True)
        self.chat_response_box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.chat_response_box.setPlaceholderText("暂无回复，输入闲聊或问题后将在此显示助手回答。")
        self._apply_chat_box_style()
        chat_layout.addWidget(self.chat_response_box)
        col_layout.addWidget(self.chat_response_card)

        return col

    def _render_quick_templates(self) -> None:
        """根据 self._quick_templates 重新渲染快捷模板网格。"""
        # 如果已有布局，清空旧子控件，复用该布局
        existing_layout = self.quick_templates_grid.layout()
        if existing_layout is not None:
            while existing_layout.count():
                item = existing_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            grid_layout = existing_layout
        else:
            grid_layout = QGridLayout(self.quick_templates_grid)
            grid_layout.setSpacing(10)
            grid_layout.setContentsMargins(0, 0, 0, 0)
        self._tpl_grid_layout = grid_layout
        s = getattr(self, "_ui_scale", UIScale())

        for i, (title, desc, prompt) in enumerate(self._quick_templates[: self.MAX_QUICK_TEMPLATES]):
            item = ClickableFrame(i)
            item.setObjectName("quickItem")
            iv = QVBoxLayout(item)
            iv.setSpacing(s.px(4, 2))
            iv.setContentsMargins(s.px(8, 4), s.px(8, 4), s.px(8, 4), s.px(8, 4))
            it = QLabel(title)
            it.setObjectName("quickTitle")
            it.setWordWrap(True)
            idesc = QLabel(desc)
            idesc.setObjectName("quickDesc")
            idesc.setWordWrap(True)
            iv.addWidget(it)
            iv.addWidget(idesc)
            item.clicked.connect(self._apply_quick_template)
            grid_layout.addWidget(item, i // 2, i % 2)

    def _load_quick_templates(self) -> list:
        """从 JSON 加载快捷模板，没有则返回默认模板。"""
        defaults = [
            ("OCR 基础训练", "标准参数", "训练一个 OCR 模型，使用标准参数"),
            ("DET 快速实验", "低 Epoch", "训练一个 DET 目标检测模型，epoch 设为 5"),
            ("Baseline 对比", "自动评估", "训练一个 CLS 分类模型并对比 baseline"),
            ("批量任务", "多模型队列", "依次训练 OCR、CLS、DET 三个模型"),
        ]
        if not self._quick_templates_file.exists():
            return defaults
        try:
            data = json.loads(self._quick_templates_file.read_text(encoding="utf-8"))
            loaded = [
                (item.get("title", ""), item.get("desc", ""), item.get("prompt", ""))
                for item in data
                if item.get("title") and item.get("prompt")
            ]
            return loaded if loaded else defaults
        except Exception:
            return defaults

    def _save_quick_templates(self) -> None:
        """把当前模板列表持久化到 JSON。"""
        try:
            data = [
                {"title": t, "desc": d, "prompt": p}
                for t, d, p in self._quick_templates
            ]
            self._quick_templates_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存模板：{e}")

    def _load_settings(self) -> dict:
        """从 JSON 加载系统设置；自动迁移旧版扁平 LLM 字段为 profiles。"""
        from autotrain4.llm import migrate_settings

        defaults = {
            "backend_url": "http://127.0.0.1:9887",
            "frontend_url": "http://localhost:8843",
            "platform_exe": r"C:\dlcv\Lib\site-packages\dlcv_train\DLCV AI Training Platform-win_x64.exe",
            "cdp_port": "9222",
            "adapter": "playwright",
            "llm_base_url": "",
            "llm_api_key": "",
            "llm_model": "",
            "llm_protocol": "auto",
            "llm_profiles": [],
            "llm_active_id": "",
            "assistant_summary": "1",
            "assistant_summary_llm": "0",
        }

        # 无文件时可用环境变量预填一条（随后 migrate 成 profile）
        env_llm = {
            "llm_base_url": os.environ.get("AUTOTRAIN_LLM_BASE_URL", ""),
            "llm_api_key": os.environ.get("AUTOTRAIN_LLM_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY", ""),
            "llm_model": os.environ.get("AUTOTRAIN_LLM_MODEL", "")
            or os.environ.get("DEEPSEEK_MODEL", ""),
            "llm_protocol": os.environ.get("AUTOTRAIN_LLM_PROTOCOL", ""),
        }
        if (
            not env_llm["llm_base_url"]
            and env_llm["llm_api_key"]
            and os.environ.get("DEEPSEEK_API_KEY")
            and not os.environ.get("AUTOTRAIN_LLM_API_KEY")
        ):
            env_llm["llm_base_url"] = "https://api.deepseek.com"
            env_llm["llm_protocol"] = env_llm["llm_protocol"] or "openai"
        defaults.update({k: v for k, v in env_llm.items() if v})

        data = dict(defaults)
        if self._settings_file.exists():
            try:
                loaded = json.loads(self._settings_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = {**defaults, **loaded}
            except Exception:
                pass
        return migrate_settings(data)

    def _apply_llm_settings_to_env(self) -> None:
        """把 active LLM profile 同步到当前进程环境。"""
        from autotrain4.llm import apply_settings_to_environ

        apply_settings_to_environ(self._settings)

    def _rebuild_suggestion_llm(self) -> None:
        """按当前 active profile 重建输入建议 LLM。"""
        from autotrain4.llm import LLMClient

        client = LLMClient.from_settings(self._settings)
        self.suggestion_engine = SuggestionEngine(llm_client=client)
        self.suggestion_engine.on_llm_result(self._on_llm_suggestions)

    def _save_settings(self) -> None:
        """持久化设置并立即生效到本进程 LLM。"""
        from autotrain4.llm import migrate_settings

        try:
            self._settings = migrate_settings(self._settings)
            self._settings_file.write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._apply_llm_settings_to_env()
            self._rebuild_suggestion_llm()
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存设置：{e}")

    def _open_settings(self) -> None:
        """打开系统设置弹窗。"""
        from autotrain4.llm import get_active_profile

        dialog = SettingsDialog(self._settings, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._settings = dialog.get_settings()
            self._save_settings()
            active = get_active_profile(self._settings)
            if active:
                detail = (
                    f"已切换到「{active.name}」\n"
                    f"模型: {active.model or '(空)'}\n"
                    f"URL: {active.base_url or '(空)'}"
                )
            else:
                detail = "未配置 LLM（训练意图解析将不可用）。"
            QMessageBox.information(self, "设置已保存", detail + "\n训练与助手将使用此配置。")

    def _apply_quick_template(self, index: int) -> None:
        """点击快捷模板时，把对应 prompt 填入输入框。"""
        if 0 <= index < len(self._quick_templates):
            _, _, prompt = self._quick_templates[index]
            self.prompt_input.setPlainText(prompt)
            self._update_metrics_from_prompt()
            self._do_update_suggestions()

    def _save_current_as_template(self) -> None:
        """把当前输入框内容保存为新的快捷模板，弹出弹窗让用户自定义标题。"""
        text = self.prompt_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请先输入训练需求再保存为模板")
            return

        if len(self._quick_templates) >= self.MAX_QUICK_TEMPLATES:
            QMessageBox.warning(
                self,
                "模板数量已达上限",
                f"最多保存 {self.MAX_QUICK_TEMPLATES} 个模板，请通过「管理」删除不再使用的模板。",
            )
            return

        suggested_title = text[:12] + ("..." if len(text) > 12 else "")
        dialog = SaveTemplateDialog(text, suggested_title, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        template = dialog.get_template()
        if template is None:
            QMessageBox.information(self, "提示", "标题和 Prompt 不能为空")
            return

        self._quick_templates.append(template)
        self._save_quick_templates()
        self._render_quick_templates()
        QMessageBox.information(self, "保存成功", f"已保存模板：{template[0]}")

    def _open_template_manager(self) -> None:
        """打开快捷模板管理弹窗，支持查看和删除。"""
        dialog = QuickTemplatesDialog(self._quick_templates, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._quick_templates = dialog.get_templates()
            self._save_quick_templates()
            self._render_quick_templates()

    def _open_readme(self) -> None:
        """读取项目 README.md，转换为 HTML 后在弹窗中展示。"""
        from dialogs.documentation import open_readme

        open_readme(self)

    def _apply_styles(self) -> None:
        """应用 Web 前端风格主题；字号/圆角/内边距随窗口等比例缩放。"""
        from styles import apply_main_styles

        apply_main_styles(self)

    def _on_input_text_changed(self) -> None:
        """输入框内容变化时触发防抖。中文 IME 组字期间不刷新，英文输入实时刷新。"""
        self._suggestion_debounce.stop()
        self._update_metrics_from_prompt()
        if self._ime_composing:
            return
        self._suggestion_debounce.start(400)

    def _update_metrics_from_prompt(self) -> None:
        from metrics import update_metrics_from_prompt

        update_metrics_from_prompt(self)

    def _update_metrics_from_log(self, line: str) -> None:
        from metrics import update_metrics_from_log

        update_metrics_from_log(self, line)

    def _estimate_total_seconds(self) -> int:
        from metrics import estimate_total_seconds

        return estimate_total_seconds(self)

    def eventFilter(self, obj, event):
        """监听 IME 组字事件，确保在组字期间不读取/刷新输入框。"""
        if obj is self.prompt_input and event.type() == QEvent.InputMethod:
            preedit = event.preeditString()
            commit = event.commitString()

            if preedit:
                self._ime_composing = True
                self._suggestion_debounce.stop()
            else:
                was_composing = self._ime_composing
                self._ime_composing = False
                if commit or was_composing:
                    QTimer.singleShot(50, self._do_update_suggestions)

        elif obj is self.prompt_input and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self._suggestion_debounce.stop()
                self._do_update_suggestions()
        # 必须交给父类默认处理，否则 IME 组字/候选窗口会失效
        return super().eventFilter(obj, event)

    def _do_update_suggestions(self) -> None:
        """Refresh suggestion chips based on input text."""
        if self._ime_composing:
            return
        if self.suggestion_engine is None:
            return
        text = self.prompt_input.toPlainText()
        suggestions = self.suggestion_engine.suggest(text)
        self._render_suggestions(suggestions)

    def _refresh_suggestions(self) -> None:
        """点击「刷新建议」：空输入时轮换默认模板，有输入时清缓存重新请求。"""
        if self._ime_composing:
            return
        text = self.prompt_input.toPlainText().strip()
        if not text:
            self._current_suggestion_set_index = (self._current_suggestion_set_index + 1) % len(self._default_suggestion_sets)
            self._render_suggestions(self._default_suggestion_sets[self._current_suggestion_set_index])
        else:
            if self.suggestion_engine is None:
                return
            # 清掉 LLM 缓存，强制重新生成建议
            self.suggestion_engine._cache.clear()
            self._do_update_suggestions()

    def _on_llm_suggestions(self, text: str, suggestions: list) -> None:
        """LLM 异步建议到达时，与本地建议混合后渲染。"""
        current = self.prompt_input.toPlainText()
        if current.strip().lower() != text.strip().lower():
            return
        if self.suggestion_engine is None:
            return

        def _blend_and_render():
            if self.suggestion_engine is None:
                return
            local = self.suggestion_engine.suggest(current)
            blended: list = []
            seen: set = set()
            for s in suggestions + local:
                key = s.strip()
                if key and key not in seen:
                    seen.add(key)
                    blended.append(key)
            self._render_suggestions(blended[:5])

        QTimer.singleShot(0, _blend_and_render)

    def _render_suggestions(self, suggestions: list) -> None:
        """Render suggestion chip buttons inline below input."""
        while self.suggestions_layout.count():
            item = self.suggestions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        cleaned: list = []
        for phrase in suggestions[:5]:
            phrase = phrase.strip().replace("\n", " ").replace("\r", "")
            phrase = "".join(ch for ch in phrase if ch.isprintable() or ch in " ")
            phrase = phrase.strip()[:24]
            if phrase:
                cleaned.append(phrase)

        if not cleaned:
            self.suggestions_container.setVisible(False)
            return

        self.suggestions_container.setVisible(True)
        for phrase in cleaned:
            btn = QPushButton(phrase)
            btn.setObjectName("suggestionChip")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, p=phrase: self._apply_suggestion(p))
            self.suggestions_layout.addWidget(btn)

        self.suggestions_layout.addStretch()

    def _apply_suggestion(self, suggestion: str) -> None:
        """Append suggestion to input and refresh."""
        self._suggestion_debounce.stop()
        current = self.prompt_input.toPlainText()
        new_text = self.suggestion_engine.take_suggestion(current, suggestion)
        self.prompt_input.setPlainText(new_text)
        cursor = self.prompt_input.textCursor()
        cursor.movePosition(cursor.End)
        self.prompt_input.setTextCursor(cursor)
        self.prompt_input.setFocus()
        self._do_update_suggestions()

    def _start_run(self) -> None:
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请输入训练需求")
            return

        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.post_worker and self.post_worker.isRunning():
            self.post_worker.stop()

        self.log_panel.clear()
        self.log_panel.append(f"[新运行] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.chat_response_box.clear()
        self._chat_from_subprocess = False
        self.start_btn.setEnabled(False)
        self.close_browser_btn.setEnabled(False)
        self._set_status("运行中", "running")

        if self._summary_worker and self._summary_worker.isRunning():
            self._summary_worker.wait(200)

        self.worker = TrainWorker(prompt, self._settings, self)
        self.worker.log_received.connect(self._append_log)
        self.worker.chat_response.connect(self._show_chat_response)
        self.worker.status_changed.connect(self._set_status)
        self.worker.finished_signal.connect(self._on_train_finished)
        self.worker.prompt_for_close.connect(self._enable_close_browser)
        self.worker.process_started.connect(self._enable_close_browser)
        self.worker.start()

        # 先把指标从当前 prompt 里解析出来，再清空输入框
        self._update_metrics_from_prompt()
        self.prompt_input.clear()
        self.prompt_input.setFocus()

    def _enable_close_browser(self) -> None:
        adapter = self._settings.get("adapter", "playwright").lower()
        self.close_browser_btn.setText("关闭应用" if adapter == "webview2_cdp" else "关闭浏览器")
        self.close_browser_btn.setEnabled(True)

    def _close_browser(self) -> None:
        """关闭保持打开的应用：网页端按 Enter，壳端直接结束 EXE 进程。"""
        adapter = self._settings.get("adapter", "playwright").lower()
        if adapter == "webview2_cdp":
            self._close_shell_app()
        elif self.worker:
            self.worker.send_enter()
            self._append_log("[操作] 已请求关闭浏览器，等待后台进程退出...")
            self._set_status("关闭中", "running")
        self.close_browser_btn.setEnabled(False)

    def _close_shell_app(self) -> None:
        """终止壳端 EXE 进程，并通过关闭信号让训练子进程干净退出。"""
        exe_path = self._settings.get("platform_exe", "")
        killed = []

        # 1) 结束壳端 EXE 进程
        if exe_path:
            exe_name = Path(exe_path).name.lower()
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    proc_name = (proc.info.get("name") or "").lower()
                    proc_exe = (proc.info.get("exe") or "").lower()
                    if exe_name in proc_name or exe_path.lower() in proc_exe:
                        proc.terminate()
                        killed.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        # 2) 等待 graceful terminate，超时则强制 kill
        gone, alive = psutil.wait_procs(killed, timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed:
            self._append_log(f"[操作] 已关闭壳端应用，PID: {', '.join(str(p.pid) for p in killed)}")
        else:
            self._append_log("[操作] 未找到运行中的壳端应用")

        # 3) 写关闭信号文件，让 auto_train2.0 干净退出，从而触发 post-train 弹窗
        if self.worker:
            self.worker.send_enter()
            self._append_log("[操作] 已发送关闭信号，等待后台进程退出...")

        self._set_status("关闭中", "running")

    def _clear_logs(self) -> None:
        self.log_panel.clear()
        self._set_status("就绪", "")
        self.close_browser_btn.setEnabled(False)
        self.chat_response_box.clear()
        for key in self.metric_labels:
            self.metric_labels[key].setText("-")
        self._update_metrics_from_prompt()
        if hasattr(self, "data_quality_panel"):
            self.data_quality_panel.reset_view()

    def _append_log(self, line: str) -> None:
        self.log_panel.append(line)
        self._update_metrics_from_log(line)

    def _current_task_type_hint(self) -> str:
        """从指标栏读取当前任务类型（若有）。"""
        try:
            text = self.metric_labels.get("task_type")
            if text is None:
                return self._last_task_type_hint or ""
            val = text.text().strip()
            if val and val != "-":
                self._last_task_type_hint = val
                return val
        except Exception:
            pass
        return self._last_task_type_hint or ""

    def _show_chat_response(self, answer: str, *, from_subprocess: bool = True) -> None:
        """显示助手回复；子进程 CHAT_RESPONSE 会覆盖当前内容。"""
        text = (answer or "").strip()
        if not text:
            return
        self.chat_response_box.setPlainText(text)
        self.chat_response_card.setVisible(True)
        if from_subprocess:
            self._chat_from_subprocess = True
        # 滚到顶部方便先看结论
        cursor = self.chat_response_box.textCursor()
        cursor.movePosition(cursor.Start)
        self.chat_response_box.setTextCursor(cursor)

    def _append_chat_response(self, answer: str) -> None:
        """在已有助手回复后追加一段（例如 chat 后再附训练收尾摘要）。"""
        text = (answer or "").strip()
        if not text:
            return
        prev = self.chat_response_box.toPlainText().strip()
        if prev:
            self.chat_response_box.setPlainText(prev + "\n\n" + "—" * 12 + "\n\n" + text)
        else:
            self.chat_response_box.setPlainText(text)
        self.chat_response_card.setVisible(True)

    def _assistant_summary_enabled(self) -> bool:
        return str(self._settings.get("assistant_summary", "1")).strip() not in (
            "0",
            "false",
            "False",
            "",
        )

    def _assistant_summary_use_llm(self) -> bool:
        return str(self._settings.get("assistant_summary_llm", "0")).strip() in (
            "1",
            "true",
            "True",
        )

    def _request_assistant_summary(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        append: bool = False,
        placeholder: Optional[str] = None,
    ) -> None:
        """异步生成摘要并写入助手框。"""
        if not self._assistant_summary_enabled():
            return

        use_llm = self._assistant_summary_use_llm()
        if placeholder:
            if append and self.chat_response_box.toPlainText().strip():
                self._append_chat_response(placeholder)
            else:
                self._show_chat_response(placeholder, from_subprocess=False)
        elif use_llm and not append:
            # 仅 LLM 可能较慢时给一点提示；模板路径通常立刻返回
            pass

        if self._summary_worker and self._summary_worker.isRunning():
            # 不强制杀掉上一轮；新结果到达时覆盖/追加即可
            try:
                self._summary_worker.finished_ok.disconnect()
                self._summary_worker.finished_err.disconnect()
            except Exception:
                pass

        worker = AssistantSummaryWorker(
            kind,
            payload,
            use_llm=use_llm,
            llm_settings=self._settings,
            parent=self,
        )
        self._summary_worker = worker

        def _ok(text: str, *, _append: bool = append) -> None:
            body = (text or "").strip()
            if not body:
                return
            # 若 placeholder 已写入，用最终文本替换末段更干净：统一整框策略
            if _append and self._chat_from_subprocess:
                # 子进程已有完整对话时追加
                prev = self.chat_response_box.toPlainText().strip()
                # 去掉可能的「生成中」占位尾部
                if prev.endswith("正在生成总结…") or prev.endswith("正在生成助手总结…"):
                    # 简单处理：整段重写为 原文(去占位) + 摘要
                    base = prev
                    for marker in ("正在生成总结…", "正在生成助手总结…"):
                        if marker in base:
                            base = base.split(marker)[0].rstrip("\n— \t")
                    if base:
                        self.chat_response_box.setPlainText(
                            base + "\n\n" + "—" * 12 + "\n\n" + body
                        )
                    else:
                        self._show_chat_response(body, from_subprocess=False)
                else:
                    self._append_chat_response(body)
            else:
                self._show_chat_response(body, from_subprocess=False)
            self._append_log("[助手] 已根据结果生成总结")

        def _err(msg: str) -> None:
            self._append_log(f"[助手] 总结生成失败: {msg}")

        worker.finished_ok.connect(_ok)
        worker.finished_err.connect(_err)
        worker.start()

    def _on_quality_summary_ready(self, report: dict) -> None:
        self._request_assistant_summary(
            "quality",
            {"report": report or {}},
            append=False,
            placeholder="数据质检完成，正在生成助手总结…" if self._assistant_summary_use_llm() else None,
        )

    def _on_quality_summary_failed(self, err: str) -> None:
        if not self._assistant_summary_enabled():
            return
        head = (err or "").strip().splitlines()[0] if err else "未知错误"
        if len(head) > 200:
            head = head[:197] + "…"
        self._show_chat_response(
            "【数据质检失败】\n"
            f"· 原因：{head}\n"
            "· 建议：检查目录是否存在、任务类型是否匹配后重试。",
            from_subprocess=False,
        )

    def _set_status(self, text: str, style_class: str) -> None:
        """更新执行控制台状态 badge。"""
        name_map = {
            "running": "statusBadgeRunning",
            "success": "statusBadgeSuccess",
            "error": "statusBadgeError",
            "": "statusBadge",
        }
        obj_name = name_map.get(style_class, "statusBadge")
        lbl = self.status_label
        lbl.setText(text)
        lbl.setObjectName(obj_name)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)

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
            self._request_assistant_summary(
                "train",
                {
                    "success": False,
                    "training_completed": False,
                    "elapsed_sec": elapsed,
                    "run_dir": run_dir or "",
                    "exit_code": code,
                    "task_type": self._current_task_type_hint() or None,
                },
                append=self._chat_from_subprocess,
            )
            return

        # 成功结束：仅「真正完成模型训练」才弹 post-train 选择窗
        # 测试最新模型 / 闲聊 / 仅生成配置 等：只更新状态，不弹窗
        if training_completed:
            self._append_log(f"[完成] 训练成功，耗时 {elapsed} 秒")
            self._set_status("训练成功", "success")
            self._show_run_dir_popup(run_dir, success=True)
            self._request_assistant_summary(
                "train",
                {
                    "success": True,
                    "training_completed": True,
                    "elapsed_sec": elapsed,
                    "run_dir": run_dir or "",
                    "exit_code": code,
                    "task_type": self._current_task_type_hint() or None,
                },
                append=self._chat_from_subprocess,
            )
            self._offer_post_train_actions()
        else:
            self._append_log(f"[完成] 任务结束，耗时 {elapsed} 秒（非训练流程，跳过后续选择）")
            self.start_btn.setEnabled(True)
            self._set_status("成功", "success")
            self._show_run_dir_popup(run_dir, success=True)
            # 子进程若已推送完整 chat 回复，则不再用模板盖掉
            if not self._chat_from_subprocess:
                self._request_assistant_summary(
                    "train",
                    {
                        "success": True,
                        "training_completed": False,
                        "elapsed_sec": elapsed,
                        "run_dir": run_dir or "",
                        "exit_code": code,
                        "task_type": self._current_task_type_hint() or None,
                    },
                    append=False,
                )

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
        self._last_post_train_action = action or "report"
        if context.get("task_type"):
            self._last_task_type_hint = str(context.get("task_type"))

        self.post_worker = PostTrainWorker(action, context, parent=self)
        self.post_worker.log_received.connect(self._append_log)
        self.post_worker.status_changed.connect(self._set_status)
        self.post_worker.finished_signal.connect(self._on_post_train_finished)
        self.post_worker.start()

    def _on_post_train_finished(
        self,
        success: bool,
        message: str,
        action: str = "",
        report_obj: object = None,
    ) -> None:
        """评估/对比结束后更新状态、写助手总结，并提示报告路径。"""
        self.start_btn.setEnabled(True)
        action_name = action or self._last_post_train_action or "report"
        report_dict: Dict[str, Any] = report_obj if isinstance(report_obj, dict) else {}
        elapsed = report_dict.get("_elapsed_sec")

        if success:
            self._set_status("成功", "success")
            self._append_log(f"[训练后] 报告路径: {message}")
            self._request_assistant_summary(
                "eval",
                {
                    "success": True,
                    "report": report_dict or None,
                    "report_path": message,
                    "action": action_name,
                    "task_type": self._current_task_type_hint()
                    or (report_dict.get("task_type") if report_dict else None),
                    "elapsed_sec": elapsed,
                },
                append=True,
                placeholder="评估完成，正在生成助手总结…" if self._assistant_summary_use_llm() else None,
            )
            tip = QMessageBox(self)
            tip.setWindowTitle("评估完成")
            tip.setIcon(QMessageBox.Information)
            tip.setText("报告已保存")
            tip.setInformativeText(message)
            tip.setStandardButtons(QMessageBox.Ok)
            tip.exec_()
        else:
            self._set_status("失败", "error")
            self._request_assistant_summary(
                "eval",
                {
                    "success": False,
                    "error_message": message,
                    "action": action_name,
                    "task_type": self._current_task_type_hint() or None,
                },
                append=True,
            )
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
        if self._summary_worker and self._summary_worker.isRunning():
            try:
                self._summary_worker.wait(500)
            except Exception:
                pass
        event.accept()
