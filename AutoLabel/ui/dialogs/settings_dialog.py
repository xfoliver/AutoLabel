"""Settings: LLM (reuse autotrain4 profiles fields) + Vision backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.styles import apply_window_styles
from ui.widgets.ui_scale import UIScale


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: Dict[str, Any],
        backends_config: Dict[str, Any],
        settings_path: Path,
        backends_path: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("AutoLabel 设置")
        self.setMinimumSize(420, 360)
        self.resize(560, 480)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.settings = dict(settings or {})
        self.backends_config = dict(backends_config or {})
        self.settings_path = Path(settings_path)
        self.backends_path = Path(backends_path)
        self._ui_scale = getattr(parent, "_ui_scale", None) or UIScale()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(8)

        layout.addWidget(QLabel("LLM（意图解析，与 Vision 独立）"))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.ed_base = QLineEdit(str(self.settings.get("llm_base_url") or ""))
        self.ed_key = QLineEdit(str(self.settings.get("llm_api_key") or ""))
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_model = QLineEdit(str(self.settings.get("llm_model") or ""))
        self.ed_proto = QLineEdit(str(self.settings.get("llm_protocol") or "auto"))
        form.addRow("Base URL", self.ed_base)
        form.addRow("API Key", self.ed_key)
        form.addRow("Model", self.ed_model)
        form.addRow("Protocol", self.ed_proto)
        layout.addLayout(form)

        layout.addWidget(QLabel("Vision Backend（标注推理）"))
        self.cmb_backend = QComboBox()
        self.cmb_backend.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        backends: List[dict] = list(self.backends_config.get("backends") or [])
        active = self.backends_config.get("active_backend")
        for b in backends:
            self.cmb_backend.addItem(f"{b.get('id')} ({b.get('type')})", b.get("id"))
        idx = self.cmb_backend.findData(active)
        if idx >= 0:
            self.cmb_backend.setCurrentIndex(idx)
        self.cmb_backend.currentIndexChanged.connect(self._on_backend_changed)
        layout.addWidget(self.cmb_backend)

        self.ed_vision_url = QLineEdit()
        self.ed_vision_model = QLineEdit()
        self.ed_vision_key = QLineEdit()
        self.ed_vision_key.setEchoMode(QLineEdit.Password)
        self.ed_vision_size = QLineEdit()
        self.ed_dlcv_model = QLineEdit()
        form2 = QFormLayout()
        form2.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form2.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form2.addRow("Vision base_url", self.ed_vision_url)
        form2.addRow("Vision model", self.ed_vision_model)
        form2.addRow("Vision API Key", self.ed_vision_key)
        form2.addRow("Image size（gpt）", self.ed_vision_size)
        form2.addRow("DLCV model_path", self.ed_dlcv_model)
        layout.addLayout(form2)
        self._fill_backend_fields()

        hint = QLabel(
            "LLM 与 Vision 配置独立可替换。gpt_image_mask 走 /v1/images/edits，"
            "返回彩色 Mask 后本地拆实例；无 model confidence 时结果进抽检。"
            "本地 settings.json / backends.json 已 gitignore，勿提交密钥。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(scroll, 1)
        outer.addWidget(buttons)
        apply_window_styles(self)

    def _backend_entry(self, backend_id: Optional[str] = None) -> Optional[dict]:
        bid = backend_id if backend_id is not None else self.cmb_backend.currentData()
        for b in self.backends_config.get("backends") or []:
            if isinstance(b, dict) and b.get("id") == bid:
                return b
        return None

    def _on_backend_changed(self, _idx: int = 0):
        self._fill_backend_fields()

    def _fill_backend_fields(self):
        self.ed_vision_url.clear()
        self.ed_vision_model.clear()
        self.ed_vision_key.clear()
        self.ed_vision_size.clear()
        self.ed_dlcv_model.clear()

        for b in self.backends_config.get("backends") or []:
            if b.get("id") == "dlcv_9890":
                opt = b.get("options") or {}
                self.ed_dlcv_model.setText(str(opt.get("model_path") or ""))

        entry = self._backend_entry()
        if not entry:
            return
        opt = entry.get("options") or {}
        btype = str(entry.get("type") or "")
        if btype in ("http_api", "openai_image_edits") or entry.get("id") in ("http_seg", "gpt_image_mask"):
            self.ed_vision_url.setText(str(opt.get("base_url") or ""))
            self.ed_vision_model.setText(str(opt.get("model") or ""))
            self.ed_vision_key.setText(str(opt.get("api_key") or ""))
            if btype == "openai_image_edits" or entry.get("id") == "gpt_image_mask":
                self.ed_vision_size.setText(str(opt.get("size") or "1024x1024"))

    def _save(self):
        self.settings["llm_base_url"] = self.ed_base.text().strip()
        self.settings["llm_api_key"] = self.ed_key.text().strip()
        self.settings["llm_model"] = self.ed_model.text().strip()
        self.settings["llm_protocol"] = self.ed_proto.text().strip() or "auto"

        active = self.cmb_backend.currentData()
        self.backends_config["active_backend"] = active

        for b in self.backends_config.get("backends") or []:
            if isinstance(b, dict) and b.get("id") == "dlcv_9890":
                b.setdefault("options", {})
                b["options"]["model_path"] = self.ed_dlcv_model.text().strip()

        entry = self._backend_entry(active)
        if entry is not None:
            btype = str(entry.get("type") or "")
            if btype in ("http_api", "openai_image_edits") or active in ("http_seg", "gpt_image_mask"):
                entry.setdefault("options", {})
                entry["options"]["base_url"] = self.ed_vision_url.text().strip()
                entry["options"]["model"] = self.ed_vision_model.text().strip()
                key = self.ed_vision_key.text().strip()
                if key:
                    entry["options"]["api_key"] = key
                if btype == "openai_image_edits" or active == "gpt_image_mask":
                    size = self.ed_vision_size.text().strip()
                    if size:
                        entry["options"]["size"] = size

        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(
                json.dumps(self.settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.backends_path.write_text(
                json.dumps(self.backends_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.accept()
