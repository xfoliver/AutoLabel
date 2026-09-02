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
        self.ed_sam3_python = QLineEdit()
        self.ed_sam3_host = QLineEdit()
        self.ed_sam3_port = QLineEdit()
        self.cmb_intent = QComboBox()
        for mode, label in (("auto", "auto（短词跳过 LLM）"), ("simple", "simple（不调用 LLM）"), ("llm", "llm（必须解析）")):
            self.cmb_intent.addItem(label, mode)
        idx_mode = self.cmb_intent.findData(str(self.settings.get("intent_mode") or "auto"))
        if idx_mode >= 0:
            self.cmb_intent.setCurrentIndex(idx_mode)
        form.addRow("Intent 模式", self.cmb_intent)
        form2 = QFormLayout()
        form2.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form2.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form2.addRow("Vision base_url", self.ed_vision_url)
        form2.addRow("Vision model", self.ed_vision_model)
        form2.addRow("Vision API Key", self.ed_vision_key)
        form2.addRow("Image size（gpt）", self.ed_vision_size)
        form2.addRow("DLCV model_path", self.ed_dlcv_model)
        form2.addRow("SAM3 python", self.ed_sam3_python)
        form2.addRow("SAM3 host", self.ed_sam3_host)
        form2.addRow("SAM3 port", self.ed_sam3_port)
        layout.addLayout(form2)
        self._fill_backend_fields()

        hint = QLabel(
            "LLM 只解析意图（label + model_prompt）。Vision 默认 sam3_local："
            "AutoLabel 经 HTTP 调用独立 SAM3 Worker（sam3 Conda Python），"
            "不在本进程 import torch/sam3。可切回 gpt_image_mask 做 A/B。"
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
        self.ed_sam3_python.clear()
        self.ed_sam3_host.clear()
        self.ed_sam3_port.clear()

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
        if btype == "sam3_local" or entry.get("id") == "sam3_local":
            self.ed_sam3_python.setText(str(opt.get("python_path") or r"C:\dlcv\envs\sam3\python.exe"))
            self.ed_sam3_host.setText(str(opt.get("host") or "127.0.0.1"))
            self.ed_sam3_port.setText(str(opt.get("port") or 9892))

    def _save(self):
        self.settings["llm_base_url"] = self.ed_base.text().strip()
        self.settings["llm_api_key"] = self.ed_key.text().strip()
        self.settings["llm_model"] = self.ed_model.text().strip()
        self.settings["llm_protocol"] = self.ed_proto.text().strip() or "auto"
        mode = self.cmb_intent.currentData()
        if mode:
            self.settings["intent_mode"] = str(mode)

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
            if btype == "sam3_local" or active == "sam3_local":
                entry.setdefault("options", {})
                py = self.ed_sam3_python.text().strip()
                if py:
                    entry["options"]["python_path"] = py
                host = self.ed_sam3_host.text().strip()
                if host:
                    entry["options"]["host"] = host
                port = self.ed_sam3_port.text().strip()
                if port:
                    try:
                        entry["options"]["port"] = int(port)
                    except ValueError:
                        entry["options"]["port"] = port
                entry["options"]["base_url"] = (
                    f"http://{entry['options'].get('host') or '127.0.0.1'}:"
                    f"{entry['options'].get('port') or 9892}"
                )

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
