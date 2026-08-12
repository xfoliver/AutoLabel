"""LLM 多 Profile 设置 UI（从主窗口拆出，保持 main 精简）。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from autotrain4.llm import (
    LLMClient,
    LLMProfile,
    delete_profile,
    get_active_profile,
    list_presets,
    list_profiles,
    migrate_settings,
    new_profile_id,
    normalize_provider,
    set_active_profile,
    upsert_profile,
)
from autotrain4.llm.presets import get_preset
from autotrain4.llm.profiles import resolve_profile_protocol


# ---------------------------------------------------------------------------
# 统一字号 / 间距 token（避免各处 12/13/14 混用）
# ---------------------------------------------------------------------------
_FS_TITLE = 20          # 弹窗主标题
_FS_SECTION = 16        # 分区标题
_FS_BODY = 15           # 输入框、下拉、勾选、按钮正文
_FS_LABEL = 14          # 字段标签
_FS_HINT = 13           # 辅助说明 / 状态
_FS_BTN = 14            # 按钮文字

_COLOR_TITLE = "#192033"
_COLOR_LABEL = "#5c6578"
_COLOR_HINT = "#7b8499"
_COLOR_OK = "#1e8e3e"
_COLOR_ERR = "#c0392b"
_COLOR_BORDER = "#e2e6ef"
_COLOR_BG_SOFT = "#f6f7fb"
_COLOR_PRIMARY_START = "#6c5ce7"
_COLOR_PRIMARY_END = "#8b7cf6"

_DIALOG_QSS = f"""
QDialog {{
    background: #ffffff;
    color: {_COLOR_TITLE};
    font-size: {_FS_BODY}px;
}}
QLabel {{
    color: {_COLOR_TITLE};
    font-size: {_FS_BODY}px;
}}
QCheckBox {{
    color: {_COLOR_TITLE};
    font-size: {_FS_BODY}px;
    spacing: 10px;
    padding: 4px 0;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
}}
QGroupBox {{
    font-size: {_FS_SECTION}px;
    font-weight: 600;
    color: {_COLOR_TITLE};
    border: 1px solid {_COLOR_BORDER};
    border-radius: 12px;
    margin-top: 14px;
    padding: 18px 14px 12px 14px;
    background: {_COLOR_BG_SOFT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {_COLOR_TITLE};
}}
"""

_LINE_EDIT = f"""
QLineEdit {{
    border: 1px solid {_COLOR_BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    font-size: {_FS_BODY}px;
    min-height: 22px;
    background: #ffffff;
    color: {_COLOR_TITLE};
    selection-background-color: #dcd6ff;
}}
QLineEdit:focus {{
    border: 1.5px solid #b8afff;
    background: #fefeff;
}}
QLineEdit:disabled {{
    background: {_COLOR_BG_SOFT};
    color: {_COLOR_HINT};
}}
"""

_COMBO = f"""
QComboBox {{
    border: 1px solid {_COLOR_BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    font-size: {_FS_BODY}px;
    min-height: 22px;
    background: #ffffff;
    color: {_COLOR_TITLE};
}}
QComboBox:focus, QComboBox:on {{
    border: 1.5px solid #b8afff;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {_COLOR_BORDER};
    border-radius: 8px;
    background: #ffffff;
    selection-background-color: #ece8ff;
    selection-color: {_COLOR_TITLE};
    font-size: {_FS_BODY}px;
    padding: 4px;
    outline: none;
}}
"""

_BTN_SECONDARY = f"""
QPushButton {{
    background: #eef0f7;
    color: #4d5566;
    border: none;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: {_FS_BTN}px;
    font-weight: 500;
    min-height: 20px;
}}
QPushButton:hover {{
    background: #e4e7f2;
}}
QPushButton:pressed {{
    background: #d8dce9;
}}
QPushButton:disabled {{
    color: #a0a7b8;
    background: #f2f3f8;
}}
"""

_BTN_PRIMARY = f"""
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {_COLOR_PRIMARY_START}, stop:1 {_COLOR_PRIMARY_END});
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: {_FS_BTN}px;
    font-weight: 600;
    min-height: 20px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #7a6cf0, stop:1 #9a8cf8);
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #5b4dd4, stop:1 #7a6ce0);
}}
"""

_BTN_DANGER = f"""
QPushButton {{
    background: #fdecee;
    color: #c0392b;
    border: none;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: {_FS_BTN}px;
    font-weight: 500;
    min-height: 20px;
}}
QPushButton:hover {{
    background: #f8d7da;
}}
"""

_LABEL_FIELD = f"color: {_COLOR_LABEL}; font-size: {_FS_LABEL}px; font-weight: 500;"
_LABEL_TITLE = f"color: {_COLOR_TITLE}; font-size: {_FS_TITLE}px; font-weight: 700;"
_LABEL_SECTION = (
    f"color: {_COLOR_TITLE}; font-size: {_FS_SECTION}px; font-weight: 700; "
    f"padding-top: 4px;"
)
_HINT = f"color: {_COLOR_HINT}; font-size: {_FS_HINT}px;"
_HINT_OK = f"color: {_COLOR_OK}; font-size: {_FS_HINT}px;"
_HINT_ERR = f"color: {_COLOR_ERR}; font-size: {_FS_HINT}px;"

_SECTION_CARD = f"""
QFrame#settingsSection {{
    background: {_COLOR_BG_SOFT};
    border: 1px solid {_COLOR_BORDER};
    border-radius: 14px;
}}
"""


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_LABEL_FIELD)
    return lbl


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_LABEL_SECTION)
    return lbl


def _make_section_card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("settingsSection")
    frame.setStyleSheet(_SECTION_CARD)
    frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    return frame


class ProfileEditDialog(QDialog):
    """添加 / 编辑单条 LLM Profile。"""

    def __init__(
        self,
        profile: Optional[LLMProfile] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑 LLM" if profile else "添加 LLM")
        # 独立对话框窗口；关闭时不得触发 QApplication 退出（Windows 上常见连带关主窗）
        self.setWindowFlags(
            Qt.Dialog
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowSystemMenuHint
        )
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)
        self.setStyleSheet(_DIALOG_QSS)
        self._profile = profile or LLMProfile(
            id=new_profile_id(),
            name="",
            provider="custom",
            protocol="auto",
        )
        self._is_new = profile is None
        self._setup_ui()
        self._load_into_form()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 22, 24, 22)

        title = QLabel("编辑 LLM 配置" if not self._is_new else "添加 LLM 配置")
        title.setStyleSheet(_LABEL_TITLE)
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(14)
        form.setHorizontalSpacing(18)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如 My Grok / 公司 DeepSeek")
        self.name_edit.setStyleSheet(_LINE_EDIT)
        form.addRow(_field_label("配置名称"), self.name_edit)

        self.provider_combo = QComboBox()
        self.provider_combo.setStyleSheet(_COMBO)
        for preset in list_presets():
            self.provider_combo.addItem(str(preset["name"]), preset["id"])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow(_field_label("服务商"), self.provider_combo)

        self.base_edit = QLineEdit()
        self.base_edit.setStyleSheet(_LINE_EDIT)
        form.addRow(_field_label("Base URL"), self.base_edit)

        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-...")
        self.key_edit.setStyleSheet(_LINE_EDIT)
        key_row.addWidget(self.key_edit, 1)
        self.toggle_key_btn = QPushButton("显示")
        self.toggle_key_btn.setCheckable(True)
        self.toggle_key_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_key_btn.setStyleSheet(_BTN_SECONDARY)
        self.toggle_key_btn.setFixedWidth(72)
        self.toggle_key_btn.clicked.connect(self._toggle_key)
        key_row.addWidget(self.toggle_key_btn)
        form.addRow(_field_label("API Key"), key_row)

        self.model_edit = QLineEdit()
        self.model_edit.setStyleSheet(_LINE_EDIT)
        form.addRow(_field_label("Model"), self.model_edit)

        layout.addLayout(form)

        # 高级
        self.advanced_box = QGroupBox("高级设置")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(False)
        adv = QFormLayout(self.advanced_box)
        adv.setSpacing(12)
        adv.setHorizontalSpacing(18)
        adv.setContentsMargins(8, 12, 8, 8)
        self.protocol_combo = QComboBox()
        self.protocol_combo.setStyleSheet(_COMBO)
        for val, label in (
            ("auto", "自动探测（推荐）"),
            ("openai", "OpenAI 兼容 Chat Completions"),
            ("anthropic", "Anthropic Messages"),
        ):
            self.protocol_combo.addItem(label, val)
        adv.addRow(_field_label("API 协议"), self.protocol_combo)
        self.timeout_edit = QLineEdit("60")
        self.timeout_edit.setStyleSheet(_LINE_EDIT)
        adv.addRow(_field_label("超时(秒)"), self.timeout_edit)
        layout.addWidget(self.advanced_box)

        self.test_status = QLabel("")
        self.test_status.setWordWrap(True)
        self.test_status.setStyleSheet(_HINT)
        layout.addWidget(self.test_status)

        layout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.setCursor(Qt.PointingHandCursor)
        self.test_btn.setStyleSheet(_BTN_SECONDARY)
        self.test_btn.clicked.connect(self._test_connection)
        btn_row.addWidget(self.test_btn)
        btn_row.addStretch()
        cancel = QPushButton("取消")
        cancel.setCursor(Qt.PointingHandCursor)
        cancel.setStyleSheet(_BTN_SECONDARY)
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存")
        save.setCursor(Qt.PointingHandCursor)
        save.setStyleSheet(_BTN_PRIMARY)
        save.setMinimumWidth(96)
        save.clicked.connect(self._on_save)
        btn_row.addWidget(cancel)
        btn_row.addWidget(save)
        layout.addLayout(btn_row)

    def _load_into_form(self) -> None:
        p = self._profile
        self.name_edit.setText(p.name)
        idx = self.provider_combo.findData(normalize_provider(p.provider))
        if idx < 0:
            idx = self.provider_combo.findData("custom")
        self.provider_combo.blockSignals(True)
        self.provider_combo.setCurrentIndex(max(0, idx))
        self.provider_combo.blockSignals(False)
        self.base_edit.setText(p.base_url)
        self.key_edit.setText(p.api_key)
        self.model_edit.setText(p.model)
        pidx = self.protocol_combo.findData(p.protocol or "auto")
        self.protocol_combo.setCurrentIndex(max(0, pidx))
        self.timeout_edit.setText(str(p.timeout or 60))
        preset = get_preset(p.provider)
        self.base_edit.setPlaceholderText(str(preset.get("base_url_hint") or ""))
        self.model_edit.setPlaceholderText(str(preset.get("model_hint") or ""))
        # 新配置且选了非 custom：可预填
        if self._is_new and not p.base_url:
            self._apply_provider_defaults(fill_name=True)

    def _on_provider_changed(self) -> None:
        self._apply_provider_defaults(fill_name=not bool(self.name_edit.text().strip()))

    def _apply_provider_defaults(self, *, fill_name: bool) -> None:
        pid = str(self.provider_combo.currentData() or "custom")
        preset = get_preset(pid)
        if not self.base_edit.text().strip():
            self.base_edit.setText(str(preset.get("default_base_url") or ""))
        if not self.model_edit.text().strip():
            self.model_edit.setText(str(preset.get("default_model") or ""))
        # 协议：高级未展开时跟随预设
        pidx = self.protocol_combo.findData(str(preset.get("default_protocol") or "auto"))
        if pidx >= 0 and not self.advanced_box.isChecked():
            self.protocol_combo.setCurrentIndex(pidx)
        self.base_edit.setPlaceholderText(str(preset.get("base_url_hint") or ""))
        self.model_edit.setPlaceholderText(str(preset.get("model_hint") or ""))
        if fill_name and not self.name_edit.text().strip():
            self.name_edit.setText(str(preset.get("name") or "LLM"))

    def _toggle_key(self) -> None:
        if self.toggle_key_btn.isChecked():
            self.key_edit.setEchoMode(QLineEdit.Normal)
            self.toggle_key_btn.setText("隐藏")
        else:
            self.key_edit.setEchoMode(QLineEdit.Password)
            self.toggle_key_btn.setText("显示")

    def _build_profile(self) -> Optional[LLMProfile]:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "校验", "请填写配置名称")
            return None
        try:
            timeout = int(self.timeout_edit.text().strip() or "60")
        except ValueError:
            timeout = 60
        return LLMProfile(
            id=self._profile.id,
            name=name,
            provider=str(self.provider_combo.currentData() or "custom"),
            base_url=self.base_edit.text().strip(),
            api_key=self.key_edit.text().strip(),
            model=self.model_edit.text().strip(),
            protocol=str(self.protocol_combo.currentData() or "auto"),
            timeout=max(5, timeout),
            enabled=True,
        )

    def _test_connection(self) -> None:
        profile = self._build_profile()
        if profile is None:
            return
        if not profile.base_url or not profile.api_key:
            self.test_status.setText("请先填写 Base URL 与 API Key")
            self.test_status.setStyleSheet(_HINT_ERR)
            return
        self.test_status.setText("测试中…")
        self.test_status.setStyleSheet(_HINT)
        QApplication = __import__("PyQt5.QtWidgets", fromlist=["QApplication"]).QApplication
        QApplication.processEvents()
        client = LLMClient(
            base_url=profile.base_url,
            api_key=profile.api_key,
            model=profile.model,
            protocol=resolve_profile_protocol(profile),
            timeout=min(profile.timeout, 30),
        )
        result = client.test_connection()
        if result.get("ok"):
            proto = result.get("protocol") or client.active_protocol
            proto_label = {
                "openai": "OpenAI Chat Completions",
                "anthropic": "Anthropic Messages",
                "auto": "自动",
            }.get(str(proto), str(proto))
            self.test_status.setText(
                f"✓ 连接成功\n模型：{profile.model or '(空)'}\n协议：{proto_label}"
            )
            self.test_status.setStyleSheet(_HINT_OK)
            # 若用户是 auto，把探测结果写回高级协议（不强制）
            if profile.protocol == "auto" and proto in ("openai", "anthropic"):
                idx = self.protocol_combo.findData(proto)
                if idx >= 0:
                    self.protocol_combo.setCurrentIndex(idx)
        else:
            self.test_status.setText(f"✗ {result.get('message') or '连接失败'}")
            self.test_status.setStyleSheet(_HINT_ERR)

    def _on_save(self) -> None:
        profile = self._build_profile()
        if profile is None:
            return
        if not profile.base_url or not profile.api_key:
            if (
                QMessageBox.question(
                    self,
                    "配置不完整",
                    "Base URL 或 API Key 为空，仍要保存吗？",
                )
                != QMessageBox.Yes
            ):
                return
        self._profile = profile
        self.accept()

    def get_profile(self) -> LLMProfile:
        return self._profile


class SettingsDialog(QDialog):
    """系统设置：平台参数 + LLM Profile 切换。"""

    def __init__(self, settings: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        # 必须是独立 Dialog 顶层窗，且关闭时不 quit 整个应用。
        # Windows + 父级 QMainWindow 时，默认 WA_QuitOnClose 可能导致点 X 连带退出前端。
        self.setWindowFlags(
            Qt.Dialog
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowSystemMenuHint
        )
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(640)
        self.resize(600, 720)
        self.setStyleSheet(_DIALOG_QSS)
        self._settings = migrate_settings(copy.deepcopy(settings))
        self.fields: Dict[str, QLineEdit] = {}
        self._setup_ui()
        self._reload_profile_combo()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # 可滚动内容区，避免小屏裁切
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 10px; background: transparent; }"
            "QScrollBar::handle:vertical { background: #d0d5e2; border-radius: 5px; min-height: 32px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        content = QWidget()
        content.setStyleSheet("background: #ffffff;")
        layout = QVBoxLayout(content)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 24, 28, 16)

        title = QLabel("⚙  系统设置")
        title.setStyleSheet(_LABEL_TITLE)
        layout.addWidget(title)

        subtitle = QLabel("配置平台连接、LLM 服务与助手总结行为")
        subtitle.setStyleSheet(_HINT)
        layout.addWidget(subtitle)

        # —— 平台连接 ——
        layout.addWidget(_section_title("平台连接"))
        platform_card = _make_section_card()
        plat = QVBoxLayout(platform_card)
        plat.setSpacing(12)
        plat.setContentsMargins(16, 16, 16, 16)

        for key, label, placeholder in (
            ("backend_url", "推理后端地址", "http://127.0.0.1:9887"),
            ("frontend_url", "前端地址", "http://localhost:8843"),
            (
                "platform_exe",
                "壳端 EXE 路径",
                r"C:\dlcv\Lib\site-packages\dlcv_train\DLCV AI Training Platform-win_x64.exe",
            ),
            ("cdp_port", "CDP 调试端口", "9222"),
        ):
            plat.addWidget(_field_label(label))
            w = QLineEdit(str(self._settings.get(key, "")))
            w.setPlaceholderText(placeholder)
            w.setStyleSheet(_LINE_EDIT)
            self.fields[key] = w
            plat.addWidget(w)

        plat.addWidget(_field_label("默认 Adapter"))
        self.adapter_combo = QComboBox()
        self.adapter_combo.setStyleSheet(_COMBO)
        self.adapter_combo.addItems(["playwright", "webview2_cdp", "stub", "skill", "http"])
        self.adapter_combo.setCurrentText(self._settings.get("adapter", "playwright"))
        plat.addWidget(self.adapter_combo)
        layout.addWidget(platform_card)

        # —— LLM ——
        layout.addWidget(_section_title("🤖  LLM 服务"))
        llm_card = _make_section_card()
        llm = QVBoxLayout(llm_card)
        llm.setSpacing(12)
        llm.setContentsMargins(16, 16, 16, 16)

        llm.addWidget(_field_label("当前配置"))
        row = QHBoxLayout()
        row.setSpacing(8)
        self.profile_combo = QComboBox()
        self.profile_combo.setStyleSheet(_COMBO)
        self.profile_combo.setMinimumWidth(220)
        self.profile_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.profile_combo, 1)
        self.add_btn = QPushButton("+ 添加")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.setStyleSheet(_BTN_SECONDARY)
        self.add_btn.clicked.connect(self._add_profile)
        row.addWidget(self.add_btn)
        self.edit_btn = QPushButton("编辑")
        self.edit_btn.setCursor(Qt.PointingHandCursor)
        self.edit_btn.setStyleSheet(_BTN_SECONDARY)
        self.edit_btn.clicked.connect(self._edit_profile)
        row.addWidget(self.edit_btn)
        self.del_btn = QPushButton("删除")
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.setStyleSheet(_BTN_DANGER)
        self.del_btn.clicked.connect(self._delete_profile)
        row.addWidget(self.del_btn)
        llm.addLayout(row)

        self.profile_summary = QLabel("")
        self.profile_summary.setWordWrap(True)
        self.profile_summary.setStyleSheet(
            f"""
            color: {_COLOR_HINT};
            font-size: {_FS_HINT}px;
            background: #ffffff;
            border: 1px solid {_COLOR_BORDER};
            border-radius: 10px;
            padding: 12px 14px;
            line-height: 1.45;
            """
        )
        llm.addWidget(self.profile_summary)

        test_row = QHBoxLayout()
        test_row.setSpacing(10)
        self.test_active_btn = QPushButton("测试当前连接")
        self.test_active_btn.setCursor(Qt.PointingHandCursor)
        self.test_active_btn.setStyleSheet(_BTN_SECONDARY)
        self.test_active_btn.clicked.connect(self._test_active)
        test_row.addWidget(self.test_active_btn)
        test_row.addStretch()
        llm.addLayout(test_row)

        self.test_active_status = QLabel("")
        self.test_active_status.setWordWrap(True)
        self.test_active_status.setStyleSheet(_HINT)
        llm.addWidget(self.test_active_status)

        self.profile_combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(llm_card)

        # —— 助手总结 ——
        layout.addWidget(_section_title("💬  助手总结"))
        sum_card = _make_section_card()
        sum_lay = QVBoxLayout(sum_card)
        sum_lay.setSpacing(8)
        sum_lay.setContentsMargins(16, 14, 16, 14)

        self.summary_enable_cb = QCheckBox("训练 / 质检 / 评估结束后自动写入助手回复")
        self.summary_enable_cb.setChecked(
            str(self._settings.get("assistant_summary", "1")).strip()
            not in ("0", "false", "False")
        )
        sum_lay.addWidget(self.summary_enable_cb)

        self.summary_llm_cb = QCheckBox("使用 LLM 润色摘要（失败则回退模板）")
        self.summary_llm_cb.setChecked(
            str(self._settings.get("assistant_summary_llm", "0")).strip()
            in ("1", "true", "True")
        )
        sum_lay.addWidget(self.summary_llm_cb)
        layout.addWidget(sum_card)

        layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # 底部固定操作栏
        footer = QFrame()
        footer.setStyleSheet(
            f"""
            QFrame {{
                background: #fafbfe;
                border-top: 1px solid {_COLOR_BORDER};
            }}
            """
        )
        btn_row = QHBoxLayout(footer)
        btn_row.setContentsMargins(28, 14, 28, 16)
        btn_row.setSpacing(10)
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(_BTN_SECONDARY)
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("保存")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet(_BTN_PRIMARY)
        save_btn.setMinimumWidth(108)
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        root.addWidget(footer)

    def _profiles(self) -> List[LLMProfile]:
        return list_profiles(self._settings)

    def _reload_profile_combo(self) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        profiles = self._profiles()
        active = str(self._settings.get("llm_active_id") or "")
        sel = 0
        for i, p in enumerate(profiles):
            self.profile_combo.addItem(p.display_label(), p.id)
            if p.id == active:
                sel = i
        if not profiles:
            self.profile_combo.addItem("（未配置，请添加）", "")
        self.profile_combo.setCurrentIndex(sel)
        self.profile_combo.blockSignals(False)
        self._update_summary()

    def _update_summary(self) -> None:
        p = get_active_profile(self._settings)
        if not p:
            self.profile_summary.setText("尚未配置 LLM。点击「+ 添加」创建连接。")
            return
        proto = resolve_profile_protocol(p)
        self.profile_summary.setText(
            f"{p.name}\n"
            f"URL: {p.base_url or '(空)'} · 模型: {p.model or '(空)'}\n"
            f"服务商: {get_preset(p.provider).get('name', p.provider)} · 协议: {proto}"
        )

    def _on_combo_changed(self) -> None:
        pid = str(self.profile_combo.currentData() or "")
        if not pid:
            return
        try:
            self._settings = set_active_profile(self._settings, pid)
        except ValueError:
            return
        self._update_summary()
        self.test_active_status.setText("")

    def _add_profile(self) -> None:
        dlg = ProfileEditDialog(parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        profile = dlg.get_profile()
        self._settings = upsert_profile(self._settings, profile, make_active=True)
        self._reload_profile_combo()

    def _edit_profile(self) -> None:
        p = get_active_profile(self._settings)
        if not p:
            self._add_profile()
            return
        dlg = ProfileEditDialog(profile=p, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._settings = upsert_profile(self._settings, dlg.get_profile(), make_active=True)
        self._reload_profile_combo()

    def _delete_profile(self) -> None:
        p = get_active_profile(self._settings)
        if not p:
            return
        if (
            QMessageBox.question(self, "删除", f"确定删除「{p.name}」？")
            != QMessageBox.Yes
        ):
            return
        self._settings = delete_profile(self._settings, p.id)
        self._reload_profile_combo()

    def _test_active(self) -> None:
        p = get_active_profile(self._settings)
        if not p:
            self.test_active_status.setText("没有可测试的配置")
            self.test_active_status.setStyleSheet(_HINT_ERR)
            return
        self.test_active_status.setText("测试中…")
        self.test_active_status.setStyleSheet(_HINT)
        from PyQt5.QtWidgets import QApplication

        QApplication.processEvents()
        client = LLMClient(
            base_url=p.base_url,
            api_key=p.api_key,
            model=p.model,
            protocol=resolve_profile_protocol(p),
            timeout=min(p.timeout, 30),
        )
        result = client.test_connection()
        if result.get("ok"):
            proto = result.get("protocol")
            self.test_active_status.setStyleSheet(_HINT_OK)
            self.test_active_status.setText(
                f"✓ 连接成功 · 模型 {p.model} · 协议 {proto}"
            )
        else:
            self.test_active_status.setStyleSheet(_HINT_ERR)
            self.test_active_status.setText(f"✗ {result.get('message')}")

    def get_settings(self) -> dict:
        result = migrate_settings(dict(self._settings))
        for key, widget in self.fields.items():
            result[key] = widget.text().strip()
        result["adapter"] = self.adapter_combo.currentText()
        result["assistant_summary"] = "1" if self.summary_enable_cb.isChecked() else "0"
        result["assistant_summary_llm"] = "1" if self.summary_llm_cb.isChecked() else "0"
        # 确保 active 与 combo 一致
        pid = str(self.profile_combo.currentData() or "")
        if pid:
            try:
                result = set_active_profile(result, pid)
            except ValueError:
                pass
        return migrate_settings(result)
