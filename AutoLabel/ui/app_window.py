"""AutoLabel main window."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QBoxLayout,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autolabel.inference.base import build_backend_from_config, load_backends_config
from autolabel.orchestrator import AutoLabelOrchestrator
from ui.backend_worker import LabelWorker
from ui.dialogs.plan_preview import PlanPreviewDialog
from ui.dialogs.preview_gallery import PreviewGalleryDialog
from ui.dialogs.settings_dialog import SettingsDialog
from ui.styles import apply_window_styles
from ui.widgets.log_panel import LogPanel
from ui.widgets.ui_scale import UIScale

AUTO_LABEL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUTO_LABEL_ROOT.parent


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoLabel — 自动标注 (ISEG MVP)")
        self.setMinimumSize(640, 480)
        self.resize(UIScale.REF_W, UIScale.REF_H)
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

        self._build_ui()
        self._refresh_backend_chip()
        self._apply_responsive_scale(force=True)

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

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self._root_layout = root

        top = QHBoxLayout()
        self._top_layout = top
        title = QLabel("AutoLabel")
        title.setObjectName("Title")
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.lbl_backend = QLabel("")
        self.lbl_backend.setObjectName("Hint")
        self.lbl_backend.setWordWrap(True)
        self.lbl_backend.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        btn_settings = QPushButton("设置")
        btn_settings.clicked.connect(self.open_settings)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.lbl_backend, 1)
        top.addWidget(btn_settings)
        root.addLayout(top)

        # Dataset + manual fields
        box_ds = QGroupBox("数据集 / 目标（MVP：实例分割）")
        self._box_ds = box_ds
        form = QFormLayout(box_ds)
        self._ds_form = form
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        path_row = QHBoxLayout()
        self._path_row = path_row
        self.ed_dataset = QLineEdit()
        self.ed_dataset.setMinimumWidth(80)
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self.browse_dataset)
        path_row.addWidget(self.ed_dataset, 1)
        path_row.addWidget(btn_browse)
        form.addRow("数据集目录", path_row)
        self.ed_label = QLineEdit("object")
        self.ed_desc = QLineEdit()
        self.ed_exclude = QLineEdit()
        self.sp_preview = QSpinBox()
        self.sp_preview.setRange(1, 20)
        self.sp_preview.setValue(5)
        form.addRow("类别名 label", self.ed_label)
        form.addRow("描述 description", self.ed_desc)
        form.addRow("排除 exclude(,)", self.ed_exclude)
        form.addRow("试标张数", self.sp_preview)
        task = QLabel("任务类型：实例分割（ISEG）— 本版固定")
        task.setObjectName("Hint")
        task.setWordWrap(True)
        form.addRow(task)
        box_ds.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        root.addWidget(box_ds)

        # NL
        box_nl = QGroupBox("自然语言（可选，优先于手工字段）")
        self._box_nl = box_nl
        nl_layout = QVBoxLayout(box_nl)
        self._nl_layout = nl_layout
        self.ed_prompt = QTextEdit()
        self.ed_prompt.setPlaceholderText(
            "例：把该目录里每一颗王不留行分别标出来；杂质碎屑不要；低置信度留给我检查。"
        )
        self.ed_prompt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        nl_layout.addWidget(self.ed_prompt)
        box_nl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        root.addWidget(box_nl)

        # Actions
        actions = QHBoxLayout()
        self._actions_layout = actions
        self.btn_plan = QPushButton("解析并预览计划")
        self.btn_plan.setObjectName("Primary")
        self.btn_plan.clicked.connect(self.on_plan)
        self.btn_commit = QPushButton("Commit 到数据集")
        self.btn_commit.clicked.connect(self.on_commit)
        self.btn_commit.setEnabled(False)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        actions.addWidget(self.btn_plan)
        actions.addWidget(self.btn_commit)
        actions.addWidget(self.btn_stop)
        actions.addStretch(1)
        root.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.log_panel = LogPanel()
        root.addWidget(self.log_panel, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪 — 默认 MockBackend，可不接 GPU")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._scale_debounce.start(40)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_responsive_scale(force=True)

    def _apply_responsive_scale(self, force: bool = False) -> None:
        """窗口尺寸变化时，等比例缩放字体、间距、固定控件尺寸。"""
        changed = self._ui_scale.update_from_size(self.width(), self.height())
        compact = self.width() < 820
        if not force and not changed and compact == self._last_compact:
            return
        self._last_compact = compact
        self._apply_layout_metrics(compact)
        apply_window_styles(self)
        if hasattr(self, "log_panel"):
            self.log_panel.apply_ui_scale(self._ui_scale)

    def _apply_layout_metrics(self, compact: bool) -> None:
        s = self._ui_scale
        gap = s.px(6 if compact else 8, 4)
        pad = s.px(8 if compact else 10, 6)

        if hasattr(self, "_root_layout"):
            self._root_layout.setSpacing(gap)
            self._root_layout.setContentsMargins(pad, pad, pad, pad)
        if hasattr(self, "_top_layout"):
            self._top_layout.setSpacing(s.px(6 if compact else 8, 4))
        if hasattr(self, "_ds_form"):
            self._ds_form.setHorizontalSpacing(s.px(8 if compact else 10, 4))
            self._ds_form.setVerticalSpacing(s.px(4 if compact else 6, 2))
            self._ds_form.setRowWrapPolicy(
                QFormLayout.WrapLongRows if compact else QFormLayout.DontWrapRows
            )
            self._ds_form.setContentsMargins(
                s.px(6 if compact else 8, 4),
                s.px(8 if compact else 10, 6),
                s.px(6 if compact else 8, 4),
                s.px(6 if compact else 8, 4),
            )
        if hasattr(self, "_path_row"):
            self._path_row.setSpacing(s.px(4 if compact else 6, 2))
        if hasattr(self, "_nl_layout"):
            self._nl_layout.setContentsMargins(
                s.px(6 if compact else 8, 4),
                s.px(8 if compact else 10, 6),
                s.px(6 if compact else 8, 4),
                s.px(6 if compact else 8, 4),
            )
        if hasattr(self, "ed_prompt"):
            self.ed_prompt.setMinimumHeight(s.px(48 if compact else 56, 40))
            self.ed_prompt.setMaximumHeight(s.px(72 if compact else 88, 56))
        if hasattr(self, "_actions_layout"):
            self._actions_layout.setSpacing(s.px(6 if compact else 8, 4))
            self._actions_layout.setDirection(
                QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
            )
            btn_policy = QSizePolicy.Expanding if compact else QSizePolicy.Preferred
            for btn in (self.btn_plan, self.btn_commit, self.btn_stop):
                btn.setSizePolicy(btn_policy, QSizePolicy.Fixed)
        if hasattr(self, "progress"):
            self.progress.setMinimumHeight(s.px(12 if compact else 14, 8))

    def _refresh_backend_chip(self) -> None:
        active = self.backends_config.get("active_backend")
        caps = self.orch.backend.capabilities.to_dict()
        self.lbl_backend.setText(f"Backend: {active} | instance_mask={caps.get('instance_mask')}")

    def browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.ed_dataset.setText(path)

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
            self.log_panel.append("设置已保存并重新加载 backend")

    def _prepare_intent(self) -> bool:
        dataset = self.ed_dataset.text().strip()
        if not dataset:
            QMessageBox.warning(self, "缺少路径", "请先选择数据集目录")
            return False
        prompt = self.ed_prompt.toPlainText().strip()
        try:
            if prompt:
                self.orch.parse_prompt(prompt, dataset_path=dataset)
                if self.orch.state.intent:
                    self.orch.state.intent.preview_count = int(self.sp_preview.value())
            else:
                label = self.ed_label.text().strip()
                if not label:
                    QMessageBox.warning(
                        self,
                        "缺少类别名",
                        "请填写短类别名（如「手机」），或改用上方自然语言意图",
                    )
                    return False
                excludes = [x.strip() for x in self.ed_exclude.text().split(",") if x.strip()]
                self.orch.make_manual_intent(
                    label=label,
                    description=self.ed_desc.text().strip(),
                    dataset_path=dataset,
                    exclude=excludes,
                    preview_count=int(self.sp_preview.value()),
                )
        except Exception as e:
            QMessageBox.warning(self, "意图解析失败", str(e))
            return False
        return True

    def _set_busy(self, busy: bool) -> None:
        self.btn_plan.setEnabled(not busy)
        self.btn_commit.setEnabled(not busy and bool(self.orch.state.mapping))
        self.btn_stop.setEnabled(busy)

    def on_plan(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self._prepare_intent():
            return
        self.log_panel.append("—— 构建 Plan ——")
        self._set_busy(True)
        self.worker = LabelWorker("plan", self.orch, dataset_path=self.ed_dataset.text().strip())
        self.worker.log_received.connect(self.log_panel.append)
        self.worker.status_changed.connect(lambda s: self.status.showMessage(s))
        self.worker.plan_ready.connect(self._on_plan_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_plan_ready(self, plan) -> None:
        dlg = PlanPreviewDialog(plan, self)
        if dlg.exec_() != PlanPreviewDialog.Accepted:
            return
        if dlg.choice == "edit":
            self.status.showMessage("请修改意图后重试")
            return
        if dlg.choice == "preview":
            self._start_preview()

    def _start_preview(self) -> None:
        self.log_panel.append("—— 试标 ——")
        self._set_busy(True)
        self.progress.setValue(0)
        self.worker = LabelWorker("preview", self.orch)
        self.worker.log_received.connect(self.log_panel.append)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_preview_ready(self, results) -> None:
        dlg = PreviewGalleryDialog(results, self)
        if dlg.exec_() != PreviewGalleryDialog.Accepted:
            return
        if dlg.choice == "retry":
            self.status.showMessage("请修改描述或设置中的 backend 后重试")
            return
        if dlg.choice == "batch":
            self._start_batch()

    def _start_batch(self) -> None:
        self.log_panel.append("—— 全量标注 ——")
        self._set_busy(True)
        self.progress.setValue(0)
        self.worker = LabelWorker("batch", self.orch)
        self.worker.log_received.connect(self.log_panel.append)
        self.worker.progress.connect(self._on_progress)
        self.worker.batch_finished.connect(self._on_batch_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_batch_finished(self, report) -> None:
        self.btn_commit.setEnabled(True)
        QMessageBox.information(
            self,
            "全量完成",
            f"处理 {report.get('processed')} 张\n"
            f"buckets={report.get('buckets')}\n"
            f"work: {report.get('run_dir')}\n\n"
            "确认无误后点击「Commit 到数据集」。",
        )

    def on_commit(self) -> None:
        if not self.orch.state.mapping:
            QMessageBox.warning(self, "无法 Commit", "请先完成全量标注")
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
        self.worker.log_received.connect(self.log_panel.append)
        self.worker.commit_finished.connect(self._on_commit_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _on_commit_finished(self, summary) -> None:
        QMessageBox.information(
            self,
            "Commit 完成",
            f"写入 {summary.get('committed_count')}，跳过 {summary.get('skipped_count')}，"
            f"错误 {summary.get('error_count')}",
        )

    def on_stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_panel.append("已请求停止…")

    def _on_progress(self, cur: int, total: int, path: str) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(cur)
        self.status.showMessage(f"{cur}/{total} {Path(path).name}")

    def _on_failed(self, err: str) -> None:
        self.log_panel.append(f"[ERROR] {err}")
        QMessageBox.critical(self, "失败", err)
        self._set_busy(False)
