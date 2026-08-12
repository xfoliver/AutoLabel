"""
工作台「数据质检」独立模块。

- 不依赖、不修改 auto_train4.0 训练流水线
- 仅调用 autotrain4.data_quality.check_dataset
- UI 自包含：DataQualityPanel；后台扫描：DataQualityWorker
- MainWindow 只需 addWidget + 可选同步 task/dataset 指标
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# 保证能 import 项目根下的 autotrain4
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 下拉：显示名 -> check_dataset 任务别名
from autotrain4.task_types import (
    guess_task_type_from_text,
    task_dropdown_options,
    to_dropdown_value,
)

# 下拉：显示名 -> check_dataset 任务别名（单源 task_types）
TASK_OPTIONS: List[Tuple[str, str]] = list(task_dropdown_options())


def normalize_ui_task(value: Optional[str]) -> Optional[str]:
    """把 UI/日志里的任务字符串规范成 TASK_OPTIONS 的 value。"""
    return to_dropdown_value(value)


def guess_task_from_path(path: str) -> Optional[str]:
    if not path:
        return None
    canon = guess_task_type_from_text(path)
    return to_dropdown_value(canon) if canon else None

def _default_dataset_root() -> str:
    cand = _PROJECT_ROOT / "AI训练平台测试"
    return str(cand) if cand.is_dir() else str(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Worker（不碰训练子进程）
# ---------------------------------------------------------------------------


class DataQualityWorker(QThread):
    """后台跑 check_dataset，结果用信号回传，失败不抛到 UI 线程。"""

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)  # report.to_dict()
    finished_err = pyqtSignal(str)

    def __init__(
        self,
        folder: str,
        task: str,
        *,
        resolve_subdir: bool = True,
        decode: str = "sample",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.folder = folder
        self.task = task
        self.resolve_subdir = resolve_subdir
        self.decode = decode

    def run(self) -> None:  # noqa: N802
        try:
            from autotrain4.data_quality import QualityPolicy, check_dataset

            def _prog(msg: str) -> None:
                self.progress.emit(str(msg))

            policy = QualityPolicy(image_decode=self.decode or "sample")
            report = check_dataset(
                self.folder,
                self.task,
                policy=policy,
                resolve_train_subdir=self.resolve_subdir,
                progress_cb=_prog,
            )
            self.finished_ok.emit(report.to_dict())
        except Exception as e:
            self.finished_err.emit(f"{e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class DataQualityPanel(QFrame):
    """
    右侧栏「数据质检」卡片。

    对外接口（MainWindow 用）：
      - set_task_hint(task_str)
      - set_dataset_hint(path_str)
      - apply_ui_scale(scale)   # scale 需有 px/font/qss_px
      - reset_view()
      - 信号 summary_ready(str) / summary_failed(str)：质检结束摘要（由 MainWindow 写入助手框）
    """

    # 质检完成后把结构化报告交给 MainWindow 生成助手总结
    summary_ready = pyqtSignal(dict)
    summary_failed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._worker: Optional[DataQualityWorker] = None
        self._last_report: Optional[Dict[str, Any]] = None
        self._ui_scale = None  # 由 MainWindow 注入 UIScale
        self._build()

    # ---- public API -------------------------------------------------------

    def set_task_hint(self, task: Optional[str]) -> None:
        """根据工作台指标/日志同步任务下拉（不强制覆盖用户手选时可再扩展）。"""
        val = normalize_ui_task(task)
        if not val:
            return
        idx = self.task_combo.findData(val)
        if idx >= 0:
            self.task_combo.setCurrentIndex(idx)

    def set_dataset_hint(self, path: Optional[str]) -> None:
        if not path:
            return
        text = str(path).strip()
        if not text or text == "-":
            return
        # 指标里可能是相对名，尽量拼到测试根下
        p = Path(text)
        if not p.is_absolute():
            root = _PROJECT_ROOT / "AI训练平台测试" / text
            if root.exists():
                p = root
            else:
                alt = _PROJECT_ROOT / text
                if alt.exists():
                    p = alt
        if p.exists():
            self.folder_edit.setText(str(p.resolve()))
            guessed = guess_task_from_path(str(p))
            if guessed:
                self.set_task_hint(guessed)

    def reset_view(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._last_report = None
        self.summary_lbl.setText("尚未检查")
        self.summary_lbl.setObjectName("dqSummaryIdle")
        self._repolish(self.summary_lbl)
        self.detail_box.clear()
        self.detail_box.setPlaceholderText(
            "选择数据集目录与任务类型后点击「开始质检」。"
            "分类任务按文件夹类别统计，不检查图↔JSON 配对。"
        )
        self.status_badge.setText("空闲")
        self.status_badge.setObjectName("statusBadge")
        self._repolish(self.status_badge)

    def apply_ui_scale(self, scale: Any) -> None:
        """跟随主窗口 UIScale 刷新本卡片内部尺寸。"""
        self._ui_scale = scale
        if scale is None:
            return
        try:
            compact_h = scale.px(110, 80)
            self.detail_box.setMinimumHeight(compact_h)
            self.detail_box.setMaximumHeight(scale.px(160, 110))
            self.folder_edit.setMinimumHeight(scale.px(32, 26))
            self.task_combo.setMinimumHeight(scale.px(32, 26))
            lay = self.layout()
            if lay is not None:
                lay.setSpacing(scale.px(10, 6))
        except Exception:
            pass
        self._apply_local_styles()

    # ---- build ------------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        title = QLabel("数据质检")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch()
        tag = QLabel("训练就绪")
        tag.setObjectName("tag")
        header.addWidget(tag)
        self.status_badge = QLabel("空闲")
        self.status_badge.setObjectName("statusBadge")
        header.addWidget(self.status_badge)
        root.addLayout(header)

        hint = QLabel("独立检查训练集质量，不启动训练、不改配置。")
        hint.setObjectName("dqHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # 路径
        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        self.folder_edit = QLineEdit()
        self.folder_edit.setObjectName("dqLineEdit")
        self.folder_edit.setPlaceholderText("数据集目录（项目根或训练集均可）")
        self.folder_edit.setText(_default_dataset_root())
        path_row.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("浏览")
        browse_btn.setObjectName("tinyGhostBtn")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        root.addLayout(path_row)

        # 任务 + 操作
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self.task_combo = QComboBox()
        self.task_combo.setObjectName("dqCombo")
        for label, val in TASK_OPTIONS:
            self.task_combo.addItem(label, val)
        ctrl.addWidget(self.task_combo, 1)

        self.run_btn = QPushButton("开始质检")
        self.run_btn.setObjectName("softBtn")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self._start_check)
        ctrl.addWidget(self.run_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setObjectName("tinyGhostBtn")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.reset_view)
        ctrl.addWidget(self.clear_btn)
        root.addLayout(ctrl)

        self.summary_lbl = QLabel("尚未检查")
        self.summary_lbl.setObjectName("dqSummaryIdle")
        self.summary_lbl.setWordWrap(True)
        root.addWidget(self.summary_lbl)

        self.detail_box = QPlainTextEdit()
        self.detail_box.setObjectName("dqDetail")
        self.detail_box.setReadOnly(True)
        self.detail_box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.detail_box.setPlaceholderText(
            "选择数据集目录与任务类型后点击「开始质检」。"
            "分类任务按文件夹类别统计，不检查图↔JSON 配对。"
        )
        self.detail_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.detail_box.setMinimumHeight(110)
        self.detail_box.setMaximumHeight(160)
        root.addWidget(self.detail_box)

        self._apply_local_styles()

    def _apply_local_styles(self) -> None:
        """仅本卡片局部样式，避免污染全局 QSS 语义。"""
        s = self._ui_scale
        if s is not None:
            f = s.font
            p = s.qss_px
        else:
            f = lambda x, m=8: max(m, int(x))  # noqa: E731
            p = lambda x, m=0: f"{max(m, int(x))}px"  # noqa: E731

        self.setStyleSheet(
            f"""
            QLabel#dqHint {{
                font-size: {f(12)}px;
                color: #7b8499;
            }}
            QLabel#dqSummaryIdle {{
                font-size: {f(13)}px;
                font-weight: 600;
                color: #5e6678;
                background: #f8f9fc;
                border: 1px solid #e8ebf2;
                border-radius: {p(12, 8)};
                padding: {p(10, 6)} {p(12, 8)};
            }}
            QLabel#dqSummaryPass {{
                font-size: {f(13)}px;
                font-weight: 600;
                color: #0f7a4c;
                background: #e9f8f2;
                border: 1px solid #c6ecd9;
                border-radius: {p(12, 8)};
                padding: {p(10, 6)} {p(12, 8)};
            }}
            QLabel#dqSummaryFail {{
                font-size: {f(13)}px;
                font-weight: 600;
                color: #b42318;
                background: #fef3f2;
                border: 1px solid #fecdca;
                border-radius: {p(12, 8)};
                padding: {p(10, 6)} {p(12, 8)};
            }}
            QLineEdit#dqLineEdit {{
                background: #fbfcff;
                border: 1px solid #e5e8f0;
                border-radius: {p(12, 8)};
                padding: {p(8, 5)} {p(10, 6)};
                font-size: {f(13)}px;
                color: #192033;
            }}
            QLineEdit#dqLineEdit:focus {{
                border: 1px solid #b8afff;
            }}
            QComboBox#dqCombo {{
                background: #fbfcff;
                border: 1px solid #e5e8f0;
                border-radius: {p(12, 8)};
                padding: {p(6, 4)} {p(10, 6)};
                font-size: {f(13)}px;
                color: #192033;
            }}
            QPlainTextEdit#dqDetail {{
                background: #fbfcff;
                border: 1px solid #e8ebf2;
                border-radius: {p(14, 8)};
                padding: {p(10, 6)};
                font-size: {f(12)}px;
                font-family: Consolas, "SF Mono", "Microsoft YaHei UI", monospace;
                color: #3d4454;
            }}
            """
        )

    # ---- actions ----------------------------------------------------------

    def _browse(self) -> None:
        initial = self.folder_edit.text().strip() or _default_dataset_root()
        if not Path(initial).is_dir():
            initial = _default_dataset_root()
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录", initial)
        if not path:
            return
        self.folder_edit.setText(path)
        guessed = guess_task_from_path(path)
        if guessed:
            self.set_task_hint(guessed)

    def _current_task(self) -> str:
        data = self.task_combo.currentData()
        return str(data) if data else "cls"

    def _start_check(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "提示", "正在质检中，请稍候…")
            return

        folder = self.folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "提示", "请先选择数据集目录")
            return
        if not Path(folder).is_dir():
            QMessageBox.warning(self, "提示", f"目录不存在：\n{folder}")
            return

        task = self._current_task()
        self.run_btn.setEnabled(False)
        self._set_badge("检查中", "running")
        self.summary_lbl.setText("正在扫描…")
        self.summary_lbl.setObjectName("dqSummaryIdle")
        self._repolish(self.summary_lbl)
        self.detail_box.setPlainText("扫描中，请稍候…")

        self._worker = DataQualityWorker(
            folder,
            task,
            resolve_subdir=True,
            decode="sample",
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        # 只更新摘要行，避免刷屏
        short = msg if len(msg) < 80 else msg[:77] + "…"
        self.summary_lbl.setText(short)

    def _on_error(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self._set_badge("失败", "error")
        self.summary_lbl.setText("质检失败")
        self.summary_lbl.setObjectName("dqSummaryFail")
        self._repolish(self.summary_lbl)
        self.detail_box.setPlainText(err)
        head = err.split("\n", 1)[0]
        QMessageBox.warning(self, "质检失败", head)
        try:
            self.summary_failed.emit(err)
        except Exception:
            pass

    def _on_done(self, report: dict) -> None:
        self.run_btn.setEnabled(True)
        self._last_report = report
        passed = bool(report.get("pass"))
        grade = report.get("grade", "?")
        score = report.get("score", 0.0)
        summary = report.get("summary") or {}
        n_img = summary.get("n_images", 0)
        n_err = summary.get("blocking_issues", 0)
        n_warn = summary.get("warnings", 0)
        task = report.get("task_type") or ""
        label_mode = summary.get("label_mode") or (
            "folder" if task == "分类" else "json"
        )
        is_cls = label_mode == "folder" or task == "分类"
        pair = summary.get("pair_rate")

        if is_cls:
            pair_txt = "文件夹分类"
        elif isinstance(pair, (int, float)):
            pair_txt = f"配对 {pair:.0%}"
        else:
            pair_txt = "配对 N/A"

        verdict = "PASS" if passed else "FAIL"
        self.summary_lbl.setText(
            f"{verdict} · {grade} · {score:.2f}  |  {task or '-'}  |  "
            f"图 {n_img} · {pair_txt}  |  错 {n_err} 警 {n_warn}"
        )
        self.summary_lbl.setObjectName("dqSummaryPass" if passed else "dqSummaryFail")
        self._repolish(self.summary_lbl)
        self._set_badge("通过" if passed else "未通过", "success" if passed else "error")
        self.detail_box.setPlainText(self._format_detail(report))
        try:
            self.summary_ready.emit(report)
        except Exception:
            pass

    @staticmethod
    def _format_detail(report: dict) -> str:
        lines: List[str] = []
        lines.append(f"目录: {report.get('dataset_dir', '')}")
        lines.append(f"任务: {report.get('task_type', '')}")
        summary = report.get("summary") or {}
        stats = report.get("stats") or {}
        if summary.get("label_mode") == "folder":
            lines.append("标注: 文件夹分类（不检查 JSON 配对）")
        else:
            pr = summary.get("pair_rate")
            pr_txt = f"{pr:.1%}" if isinstance(pr, (int, float)) else "N/A"
            lines.append(
                f"JSON: {summary.get('n_jsons', 0)}  配对率: {pr_txt}"
            )
        hist = report.get("label_histogram") or {}
        if hist:
            top = list(hist.items())[:8]
            lines.append("类别: " + ", ".join(f"{k}={v}" for k, v in top))
            if len(hist) > 8:
                lines.append(f"      …共 {len(hist)} 类")
        issues = report.get("issues") or []
        if not issues:
            lines.append("")
            lines.append("未发现问题。")
        else:
            lines.append("")
            lines.append("问题列表:")
            for iss in issues:
                sev = str(iss.get("severity", "")).upper()
                code = iss.get("code", "")
                msg = iss.get("message", "")
                lines.append(f"  [{sev}] {code}: {msg}")
                sug = iss.get("suggestion") or ""
                if sug:
                    lines.append(f"         建议: {sug}")
                samples = iss.get("samples") or []
                if samples:
                    preview = ", ".join(str(x) for x in samples[:4])
                    more = f" …(+{len(samples)-4})" if len(samples) > 4 else ""
                    lines.append(f"         样例: {preview}{more}")
        if stats.get("resolved_from"):
            lines.append("")
            lines.append(f"解析自: {stats.get('resolved_from')}")
        return "\n".join(lines)

    def _set_badge(self, text: str, style: str) -> None:
        name_map = {
            "running": "statusBadgeRunning",
            "success": "statusBadgeSuccess",
            "error": "statusBadgeError",
            "": "statusBadge",
        }
        self.status_badge.setText(text)
        self.status_badge.setObjectName(name_map.get(style, "statusBadge"))
        self._repolish(self.status_badge)

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
