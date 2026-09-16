"""Settings: SAM3.1 is built-in; extra annotation engines are user-added APIs."""

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
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from autolabel.engines import (
    API_ENGINE_TYPE,
    BACKEND_CHAT_IMAGE,
    ENGINE_CHAT_IMAGE,
    ENGINE_SAM31,
    api_engine_configured,
    engine_explanation,
    is_api_engine_entry,
    list_user_engine_choices,
    normalize_engine,
    slug_engine_id,
)
from autolabel.paths import is_frozen, resolve_python_path
from ui.styles import apply_window_styles
from ui.widgets.ui_scale import UIScale


def _entry(cfg: Dict[str, Any], backend_id: str) -> Optional[dict]:
    want = str(backend_id or "").strip()
    if not want:
        return None
    want_l = want.lower()
    for b in cfg.get("backends") or []:
        if isinstance(b, dict) and str(b.get("id") or "") in {want, want_l}:
            return b
        if isinstance(b, dict) and str(b.get("id") or "").lower() == want_l:
            return b
    return None


def _ensure_entry(cfg: Dict[str, Any], backend_id: str, type_name: str) -> dict:
    found = _entry(cfg, backend_id)
    if found is not None:
        found.setdefault("options", {})
        return found
    item = {"id": backend_id, "type": type_name, "options": {}}
    cfg.setdefault("backends", []).append(item)
    return item


def _configure_form(form: QFormLayout) -> None:
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    form.setContentsMargins(8, 10, 8, 6)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(10)


def _default_api_options() -> Dict[str, Any]:
    return {
        "base_url": "",
        "endpoint": "/images/edits",
        "model": "",
        "api_key_env": "AUTOLABEL_VISION_KEY",
        "size": "1024x1024",
        "quality": "low",
        "upload_format": "jpeg",
        "jpeg_quality": 85,
        "downscale_upload": True,
        "timeout": 180,
        "http_proxy": "",
        "min_area": 64,
        "quantize": 8,
        "capabilities": {
            "instance_mask": True,
            "text_guided_segmentation": True,
            "confidence": False,
        },
        "epsilon_ratio": 0.0005,
        "mask_close_ksize": 0,
        "mask_open_ksize": 0,
        "grabcut": True,
        "grabcut_iters": 3,
        "grabcut_pad_px": 8,
        "grabcut_sure_fg_erode": 3,
        "grabcut_sure_bg_dilate": 8,
        "merge_bbox_iou": 0.2,
        "merge_gap_px": 5,
        "merge_fragments": True,
        "merge_iou": 0.15,
        "min_relative_area": 0.12,
    }


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
        self.resize(560, 560)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.settings = dict(settings or {})
        self.backends_config = dict(backends_config or {})
        self.backends_config.setdefault("backends", [])
        self.settings_path = Path(settings_path)
        self.backends_path = Path(backends_path)
        self._ui_scale = getattr(parent, "_ui_scale", None) or UIScale()
        self._loading_engine = False
        self._seed_legacy_vision()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(12)

        layout.addWidget(QLabel("标注引擎"))
        engine_row = QHBoxLayout()
        engine_row.setSpacing(8)
        self.cmb_engine = QComboBox()
        self.cmb_engine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        engine_row.addWidget(self.cmb_engine, 1)
        self.btn_add_engine = QPushButton("添加")
        self.btn_add_engine.setObjectName("Secondary")
        self.btn_add_engine.setCursor(Qt.PointingHandCursor)
        self.btn_add_engine.clicked.connect(self._add_api_engine)
        self.btn_del_engine = QPushButton("删除")
        self.btn_del_engine.setObjectName("Secondary")
        self.btn_del_engine.setCursor(Qt.PointingHandCursor)
        self.btn_del_engine.clicked.connect(self._delete_api_engine)
        engine_row.addWidget(self.btn_add_engine, 0)
        engine_row.addWidget(self.btn_del_engine, 0)
        layout.addLayout(engine_row)
        self.lbl_engine_hint = QLabel("")
        self.lbl_engine_hint.setObjectName("Hint")
        self.lbl_engine_hint.setWordWrap(True)
        layout.addWidget(self.lbl_engine_hint)

        sam3 = (
            _entry(self.backends_config, "sam3_1_local") or _entry(self.backends_config, "sam3_local") or {}
        ).get("options") or {}

        self.vision_box = QGroupBox("视觉识别模型")
        vision_form = QFormLayout(self.vision_box)
        _configure_form(vision_form)
        self.ed_vision_name = QLineEdit()
        self.ed_vision_name.setPlaceholderText("显示名称，例如 gpt-image-2")
        self.ed_vision_url = QLineEdit()
        self.ed_vision_url.setPlaceholderText("例如 https://api.example.com/v1")
        self.ed_vision_model = QLineEdit()
        self.ed_vision_model.setPlaceholderText("例如 gpt-image-2")
        self.ed_vision_key = QLineEdit()
        self.ed_vision_key.setEchoMode(QLineEdit.Password)
        self.ed_vision_key.setPlaceholderText("粘贴 API Key，仅保存在本机")
        self.ed_vision_size = QLineEdit("1024x1024")
        vision_form.addRow("名称", self.ed_vision_name)
        vision_form.addRow("Base URL", self.ed_vision_url)
        vision_form.addRow("Model", self.ed_vision_model)
        vision_form.addRow("API Key", self.ed_vision_key)
        vision_form.addRow("Size", self.ed_vision_size)
        layout.addWidget(self.vision_box)

        self.llm_box = QGroupBox("意图解析 LLM（可选，短类别名不需要）")
        form = QFormLayout(self.llm_box)
        _configure_form(form)
        self.ed_base = QLineEdit(str(self.settings.get("llm_base_url") or ""))
        self.ed_base.setPlaceholderText("留空则不调用 LLM")
        self.ed_key = QLineEdit(str(self.settings.get("llm_api_key") or ""))
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_key.setPlaceholderText("自行填写，安装包不含 Key")
        self.ed_model = QLineEdit(str(self.settings.get("llm_model") or ""))
        self.ed_proto = QLineEdit(str(self.settings.get("llm_protocol") or "openai"))
        self.cmb_intent = QComboBox()
        for mode, label in (("auto", "auto（短词跳过 LLM）"), ("simple", "simple（不调用 LLM）"), ("llm", "llm（必须解析）")):
            self.cmb_intent.addItem(label, mode)
        idx_mode = self.cmb_intent.findData(str(self.settings.get("intent_mode") or "auto"))
        if idx_mode >= 0:
            self.cmb_intent.setCurrentIndex(idx_mode)
        form.addRow("Base URL", self.ed_base)
        form.addRow("API Key", self.ed_key)
        form.addRow("Model", self.ed_model)
        form.addRow("Protocol", self.ed_proto)
        form.addRow("Intent 模式", self.cmb_intent)
        layout.addWidget(self.llm_box)

        self.sam3_box = QGroupBox("SAM3.1 Worker（开发选项）")
        form2 = QFormLayout(self.sam3_box)
        _configure_form(form2)
        self.ed_sam3_python = QLineEdit(str(sam3.get("python_path") or resolve_python_path()))
        self.ed_sam3_host = QLineEdit(str(sam3.get("host") or "127.0.0.1"))
        self.ed_sam3_port = QLineEdit(str(sam3.get("port") or 9892))
        form2.addRow("SAM3.1 python", self.ed_sam3_python)
        form2.addRow("SAM3.1 host", self.ed_sam3_host)
        form2.addRow("SAM3.1 port", self.ed_sam3_port)
        self.sam3_box.setVisible(not is_frozen())
        layout.addWidget(self.sam3_box)

        hint = QLabel(
            "SAM3.1 随软件内置，不能删除。其它标注引擎用 Base URL / API Key / 模型自行添加。"
            "「AI 识别名称」使用同一套视觉识别凭证看图，不生成 mask。"
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

        self._reload_engine_combo(
            normalize_engine(
                self.backends_config.get("annotation_engine"),
                self.backends_config,
            )
        )
        self.cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        self._on_engine_changed()

    def _seed_legacy_vision(self) -> None:
        """Lift leftover settings.vision_* onto gpt_image_mask so it shows as an engine."""
        gpt = _entry(self.backends_config, BACKEND_CHAT_IMAGE)
        url = str(self.settings.get("vision_base_url") or "").strip()
        model = str(self.settings.get("vision_model") or "").strip()
        key = str(self.settings.get("vision_api_key") or "").strip()
        if gpt is None and not (url or model or key):
            return
        if not (url or model or key) and gpt is not None and not api_engine_configured(gpt):
            return
        if gpt is None:
            gpt = _ensure_entry(self.backends_config, BACKEND_CHAT_IMAGE, API_ENGINE_TYPE)
            gpt["options"] = _default_api_options()
        opt = gpt.setdefault("options", {})
        if url and not str(opt.get("base_url") or "").strip():
            opt["base_url"] = url
        if model and not str(opt.get("model") or "").strip():
            opt["model"] = model
        if key and not str(opt.get("api_key") or "").strip():
            opt["api_key"] = key
        if api_engine_configured(gpt) and not str(gpt.get("label") or "").strip():
            gpt["label"] = str(opt.get("model") or "").strip() or ENGINE_CHAT_IMAGE

    def _current_engine_id(self) -> str:
        return str(self.cmb_engine.currentData() or ENGINE_SAM31)

    def _reload_engine_combo(self, select: Optional[str] = None) -> None:
        self._loading_engine = True
        current = select or self._current_engine_id()
        self.cmb_engine.blockSignals(True)
        self.cmb_engine.clear()
        for eid, elabel in list_user_engine_choices(self.backends_config):
            self.cmb_engine.addItem(elabel, eid)
        idx = self.cmb_engine.findData(current)
        if idx < 0:
            idx = self.cmb_engine.findData(ENGINE_SAM31)
        if idx >= 0:
            self.cmb_engine.setCurrentIndex(idx)
        self.cmb_engine.blockSignals(False)
        self._loading_engine = False

    def _is_api_engine(self, engine: Optional[str] = None) -> bool:
        eid = str(engine or self._current_engine_id())
        if eid == ENGINE_SAM31:
            return False
        return is_api_engine_entry(_entry(self.backends_config, eid)) or eid in {
            BACKEND_CHAT_IMAGE,
            ENGINE_CHAT_IMAGE,
        }

    def _flush_vision_fields(self, engine: Optional[str] = None) -> None:
        eid = str(engine or "")
        if not eid or eid == ENGINE_SAM31:
            self.settings["vision_base_url"] = self.ed_vision_url.text().strip()
            self.settings["vision_model"] = self.ed_vision_model.text().strip()
            self.settings["vision_api_key"] = self.ed_vision_key.text().strip()
            return
        entry = _ensure_entry(self.backends_config, eid, API_ENGINE_TYPE)
        opt = entry.setdefault("options", {})
        if not opt:
            opt.update(_default_api_options())
            entry["options"] = opt
        name = self.ed_vision_name.text().strip()
        if name:
            entry["label"] = name
        opt["base_url"] = self.ed_vision_url.text().strip()
        opt["model"] = self.ed_vision_model.text().strip()
        vkey = self.ed_vision_key.text().strip()
        if vkey:
            opt["api_key"] = vkey
        else:
            opt.pop("api_key", None)
        size = self.ed_vision_size.text().strip()
        if size:
            opt["size"] = size
        opt["endpoint"] = opt.get("endpoint") or "/images/edits"
        if eid in {BACKEND_CHAT_IMAGE, ENGINE_CHAT_IMAGE}:
            self.settings["vision_base_url"] = opt.get("base_url") or ""
            self.settings["vision_model"] = opt.get("model") or ""
            self.settings["vision_api_key"] = opt.get("api_key") or ""

    def _load_vision_fields(self, engine: str) -> None:
        if engine == ENGINE_SAM31:
            gpt = _entry(self.backends_config, BACKEND_CHAT_IMAGE) or {}
            opt = gpt.get("options") or {}
            self.ed_vision_name.setText(str(gpt.get("label") or "AI 识别名称"))
            self.ed_vision_url.setText(str(self.settings.get("vision_base_url") or opt.get("base_url") or ""))
            self.ed_vision_model.setText(str(self.settings.get("vision_model") or opt.get("model") or ""))
            self.ed_vision_key.setText(str(self.settings.get("vision_api_key") or opt.get("api_key") or ""))
            self.ed_vision_size.setText(str(opt.get("size") or "1024x1024"))
            self.ed_vision_name.setEnabled(False)
            self.ed_vision_size.setEnabled(False)
            return
        entry = _entry(self.backends_config, engine) or {}
        opt = entry.get("options") or {}
        self.ed_vision_name.setText(str(entry.get("label") or opt.get("model") or engine))
        self.ed_vision_url.setText(str(opt.get("base_url") or ""))
        self.ed_vision_model.setText(str(opt.get("model") or ""))
        self.ed_vision_key.setText(str(opt.get("api_key") or ""))
        self.ed_vision_size.setText(str(opt.get("size") or "1024x1024"))
        self.ed_vision_name.setEnabled(True)
        self.ed_vision_size.setEnabled(True)

    def _on_engine_changed(self, _idx: int = 0) -> None:
        if self._loading_engine:
            return
        engine = self._current_engine_id()
        prev = getattr(self, "_last_engine", None)
        if prev and prev != engine:
            self._flush_vision_fields(prev)
            if prev != ENGINE_SAM31:
                self._reload_engine_combo(engine)
        self._last_engine = engine
        self.lbl_engine_hint.setText(engine_explanation(engine, self.backends_config))
        self.btn_del_engine.setEnabled(engine != ENGINE_SAM31)
        self._load_vision_fields(engine)

    def _existing_ids(self) -> List[str]:
        ids: List[str] = []
        for item in self.backends_config.get("backends") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        return ids

    def _add_api_engine(self) -> None:
        name, ok = QInputDialog.getText(self, "添加视觉识别模型", "显示名称：")
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            QMessageBox.information(self, "添加失败", "请填写显示名称。")
            return
        self._flush_vision_fields(self._current_engine_id())
        eid = slug_engine_id(name, self._existing_ids())
        entry = _ensure_entry(self.backends_config, eid, API_ENGINE_TYPE)
        entry["label"] = name
        opt = dict(_default_api_options())
        opt["model"] = name
        entry["options"] = opt
        self._reload_engine_combo(eid)
        self._last_engine = eid
        self._load_vision_fields(eid)
        self.lbl_engine_hint.setText(engine_explanation(eid, self.backends_config))
        self.btn_del_engine.setEnabled(True)
        self.ed_vision_url.setFocus()

    def _delete_api_engine(self) -> None:
        engine = self._current_engine_id()
        if engine == ENGINE_SAM31:
            return
        if (
            QMessageBox.question(
                self,
                "删除引擎",
                f"确定删除「{self.cmb_engine.currentText()}」？",
            )
            != QMessageBox.Yes
        ):
            return
        self.backends_config["backends"] = [
            b
            for b in (self.backends_config.get("backends") or [])
            if not (isinstance(b, dict) and str(b.get("id") or "") == engine)
        ]
        if self.backends_config.get("annotation_engine") == engine:
            self.backends_config["annotation_engine"] = ENGINE_SAM31
        self._last_engine = ENGINE_SAM31
        self._reload_engine_combo(ENGINE_SAM31)
        self._on_engine_changed()

    def _save(self):
        self.settings["llm_base_url"] = self.ed_base.text().strip()
        self.settings["llm_api_key"] = self.ed_key.text().strip()
        self.settings["llm_model"] = self.ed_model.text().strip()
        self.settings["llm_protocol"] = self.ed_proto.text().strip() or "auto"
        mode = self.cmb_intent.currentData()
        if mode:
            self.settings["intent_mode"] = str(mode)
        self.settings.pop("show_dev_backends", None)
        self.settings.pop("vision_chat_model", None)

        engine = normalize_engine(self._current_engine_id(), self.backends_config)
        self._flush_vision_fields(engine)
        if engine != ENGINE_SAM31 and not self.ed_vision_url.text().strip():
            QMessageBox.warning(self, "保存失败", "API 引擎需要填写 Base URL。")
            return
        if engine != ENGINE_SAM31 and not self.ed_vision_model.text().strip():
            QMessageBox.warning(self, "保存失败", "API 引擎需要填写 Model。")
            return

        self.backends_config["annotation_engine"] = engine
        self.backends_config.pop("vision", None)
        self.backends_config.pop("active_backend", None)

        sam31 = _ensure_entry(self.backends_config, "sam3_1_local", "sam3_local")
        py = self.ed_sam3_python.text().strip()
        if py:
            sam31["options"]["python_path"] = py
        host = self.ed_sam3_host.text().strip() or "127.0.0.1"
        sam31["options"]["host"] = host
        port_txt = self.ed_sam3_port.text().strip() or "9892"
        try:
            port = int(port_txt)
        except ValueError:
            port = 9892
        sam31["options"]["port"] = port
        sam31["options"]["base_url"] = f"http://{host}:{port}"
        sam31["options"]["sam3_version"] = "sam3.1"

        keep: List[dict] = []
        seen = set()
        for b in self.backends_config.get("backends") or []:
            if not isinstance(b, dict):
                continue
            eid = str(b.get("id") or "")
            if eid in seen:
                continue
            etype = str(b.get("type") or "")
            if eid == "sam3_1_local" or eid == "mock" or etype == "mock":
                keep.append(b)
                seen.add(eid)
                continue
            if eid == "sam3_local":
                continue
            if is_api_engine_entry(b):
                if not api_engine_configured(b) and eid != engine:
                    continue
                keep.append(b)
                seen.add(eid)
        self.backends_config["backends"] = keep

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
