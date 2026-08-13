"""提交历史对话框与主界面最近列表卡片。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
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

import submission_history as sh


_STATUS_LABEL = {
    "running": "运行中",
    "done": "成功",
    "error": "失败",
}


def _status_object_name(status: str) -> str:
    mapping = {
        "running": "historyStatusRunning",
        "done": "historyStatusDone",
        "error": "historyStatusError",
    }
    return mapping.get(status, "historyStatusDone")


def _format_time(iso: str) -> str:
    text = (iso or "").strip()
    if not text:
        return "-"
    # 2026-08-13T10:15:30 -> 08-13 10:15
    try:
        if "T" in text:
            date, time_part = text.split("T", 1)
            md = date[5:] if len(date) >= 10 else date
            hm = time_part[:5]
            return f"{md} {hm}"
    except Exception:
        pass
    return text[:16]


def _format_elapsed(seconds) -> str:
    if seconds is None:
        return ""
    try:
        val = float(seconds)
    except (TypeError, ValueError):
        return ""
    if val < 60:
        return f"{val:.0f}s"
    if val < 3600:
        return f"{val / 60:.1f}min"
    return f"{val / 3600:.1f}h"


def _prompt_preview(prompt: str, limit: int = 72) -> str:
    text = " ".join((prompt or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _run_dir_name(run_dir: str) -> str:
    if not run_dir:
        return ""
    return Path(run_dir).name or run_dir


def open_run_dir(run_dir: str, parent: Optional[QWidget] = None) -> bool:
    """在资源管理器中打开 run_dir。"""
    path = Path(run_dir) if run_dir else None
    if path is None or not path.exists():
        if parent is not None:
            QMessageBox.information(parent, "提示", "运行目录不存在或尚未生成。")
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


class _HistoryRow(QFrame):
    """单条历史行。"""

    refill = pyqtSignal(str)
    resubmit = pyqtSignal(str)
    deleted = pyqtSignal(str)
    open_dir = pyqtSignal(str)
    activated = pyqtSignal(str)  # 双击回填

    def __init__(self, entry: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.entry = dict(entry)
        self.setObjectName("historyItem")
        self.setCursor(Qt.PointingHandCursor)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        top = QHBoxLayout()
        top.setSpacing(8)

        status = str(self.entry.get("status") or "done")
        badge = QLabel(_STATUS_LABEL.get(status, status))
        badge.setObjectName(_status_object_name(status))
        top.addWidget(badge)

        task = (self.entry.get("task_type_hint") or "").strip()
        if task:
            task_lbl = QLabel(task)
            task_lbl.setObjectName("historyMeta")
            top.addWidget(task_lbl)

        time_lbl = QLabel(_format_time(str(self.entry.get("submitted_at") or "")))
        time_lbl.setObjectName("historyMeta")
        top.addWidget(time_lbl)

        elapsed = _format_elapsed(self.entry.get("elapsed_s"))
        if elapsed:
            el_lbl = QLabel(elapsed)
            el_lbl.setObjectName("historyMeta")
            top.addWidget(el_lbl)

        top.addStretch()
        layout.addLayout(top)

        prompt_lbl = QLabel(_prompt_preview(str(self.entry.get("prompt") or "")))
        prompt_lbl.setObjectName("historyPrompt")
        prompt_lbl.setWordWrap(True)
        prompt_lbl.setToolTip(str(self.entry.get("prompt") or ""))
        layout.addWidget(prompt_lbl)

        run_dir = str(self.entry.get("run_dir") or "")
        if run_dir:
            dir_lbl = QLabel(_run_dir_name(run_dir))
            dir_lbl.setObjectName("historyMeta")
            dir_lbl.setToolTip(run_dir)
            layout.addWidget(dir_lbl)

        err = str(self.entry.get("error_summary") or "").strip()
        if err and status == "error":
            err_lbl = QLabel(err)
            err_lbl.setObjectName("historyError")
            err_lbl.setWordWrap(True)
            layout.addWidget(err_lbl)

        actions = QHBoxLayout()
        actions.setSpacing(6)

        refill_btn = QPushButton("回填")
        refill_btn.setObjectName("tinyGhostBtn")
        refill_btn.setCursor(Qt.PointingHandCursor)
        refill_btn.clicked.connect(lambda: self.refill.emit(str(self.entry.get("prompt") or "")))

        resubmit_btn = QPushButton("再次提交")
        resubmit_btn.setObjectName("tinyGhostBtn")
        resubmit_btn.setCursor(Qt.PointingHandCursor)
        resubmit_btn.clicked.connect(lambda: self.resubmit.emit(str(self.entry.get("prompt") or "")))

        open_btn = QPushButton("打开目录")
        open_btn.setObjectName("tinyGhostBtn")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setEnabled(bool(run_dir and Path(run_dir).exists()))
        open_btn.clicked.connect(lambda: self.open_dir.emit(run_dir))

        del_btn = QPushButton("删除")
        del_btn.setObjectName("tinyDangerBtn")
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.clicked.connect(self._on_delete)

        actions.addWidget(refill_btn)
        actions.addWidget(resubmit_btn)
        actions.addWidget(open_btn)
        actions.addStretch()
        actions.addWidget(del_btn)
        layout.addLayout(actions)

    def _on_delete(self) -> None:
        reply = QMessageBox.question(
            self,
            "删除历史",
            "确定删除这条提交记录？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.deleted.emit(str(self.entry.get("id") or ""))

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.activated.emit(str(self.entry.get("prompt") or ""))
        super().mouseDoubleClickEvent(event)


class RecentSubmissionsCard(QFrame):
    """主界面右侧：最近 N 条提交历史。"""

    open_all_requested = pyqtSignal()
    refill_requested = pyqtSignal(str)
    resubmit_requested = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        limit: int = 5,
        history_path: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._limit = limit
        self._history_path = history_path
        self._list_layout: Optional[QVBoxLayout] = None
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("提交历史")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch()
        view_btn = QPushButton("查看全部")
        view_btn.setObjectName("tinyGhostBtn")
        view_btn.setCursor(Qt.PointingHandCursor)
        view_btn.clicked.connect(self.open_all_requested.emit)
        header.addWidget(view_btn)
        tag = QLabel("最近")
        tag.setObjectName("tag")
        header.addWidget(tag)
        layout.addLayout(header)

        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setSpacing(8)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setAlignment(Qt.AlignTop)
        layout.addWidget(self._list_host)

    def refresh(self) -> None:
        assert self._list_layout is not None
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        entries = sh.load_history(self._history_path)[: self._limit]
        if not entries:
            empty = QLabel("暂无提交记录")
            empty.setObjectName("historyEmpty")
            empty.setAlignment(Qt.AlignCenter)
            self._list_layout.addWidget(empty)
            return

        for entry in entries:
            row = self._make_compact_row(entry)
            self._list_layout.addWidget(row)

    def _make_compact_row(self, entry: dict) -> QFrame:
        frame = QFrame()
        frame.setObjectName("historyItem")
        frame.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(frame)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 8, 10, 8)

        top = QHBoxLayout()
        status = str(entry.get("status") or "done")
        badge = QLabel(_STATUS_LABEL.get(status, status))
        badge.setObjectName(_status_object_name(status))
        top.addWidget(badge)
        task = (entry.get("task_type_hint") or "").strip()
        if task:
            t = QLabel(task)
            t.setObjectName("historyMeta")
            top.addWidget(t)
        top.addStretch()
        time_lbl = QLabel(_format_time(str(entry.get("submitted_at") or "")))
        time_lbl.setObjectName("historyMeta")
        top.addWidget(time_lbl)
        layout.addLayout(top)

        prompt = str(entry.get("prompt") or "")
        p = QLabel(_prompt_preview(prompt, 56))
        p.setObjectName("historyPrompt")
        p.setWordWrap(True)
        p.setToolTip(prompt)
        layout.addWidget(p)

        actions = QHBoxLayout()
        refill_btn = QPushButton("回填")
        refill_btn.setObjectName("tinyGhostBtn")
        refill_btn.setCursor(Qt.PointingHandCursor)
        refill_btn.clicked.connect(lambda _=False, text=prompt: self.refill_requested.emit(text))
        resubmit_btn = QPushButton("再提")
        resubmit_btn.setObjectName("tinyGhostBtn")
        resubmit_btn.setCursor(Qt.PointingHandCursor)
        resubmit_btn.clicked.connect(lambda _=False, text=prompt: self.resubmit_requested.emit(text))
        actions.addWidget(refill_btn)
        actions.addWidget(resubmit_btn)
        actions.addStretch()
        layout.addLayout(actions)

        # 双击整行回填
        def _dbl(_event, text=prompt):
            self.refill_requested.emit(text)

        frame.mouseDoubleClickEvent = _dbl  # type: ignore[method-assign]
        return frame


class SubmissionHistoryDialog(QDialog):
    """完整提交历史管理对话框。"""

    refill_requested = pyqtSignal(str)
    resubmit_requested = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        history_path: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("提交历史")
        self.setMinimumSize(560, 520)
        self.resize(640, 600)
        self._history_path = history_path
        self._entries: List[dict] = []
        self._setup_ui()
        self.reload()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("训练需求提交历史")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #192033;")
        layout.addWidget(title)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索 prompt / 任务类型 / 目录…")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._render_list)
        toolbar.addWidget(self.search_input, 1)

        self.status_filter = QComboBox()
        self.status_filter.addItem("全部", "all")
        self.status_filter.addItem("运行中", "running")
        self.status_filter.addItem("成功", "done")
        self.status_filter.addItem("失败", "error")
        self.status_filter.currentIndexChanged.connect(self._render_list)
        toolbar.addWidget(self.status_filter)

        clear_btn = QPushButton("清空历史")
        clear_btn.setObjectName("tinyDangerBtn")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_all)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setSpacing(10)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        self._list_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._list_host)
        layout.addWidget(scroll, 1)

        close_btn = QPushButton("关闭")
        close_btn.setObjectName("secondaryBtn")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, Qt.AlignRight)

    def reload(self) -> None:
        self._entries = sh.load_history(self._history_path)
        self._render_list()

    def _render_list(self) -> None:
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        query = self.search_input.text()
        status = self.status_filter.currentData()
        filtered = sh.filter_submissions(self._entries, query=query, status=status)

        if not filtered:
            empty = QLabel("暂无匹配的提交记录")
            empty.setObjectName("historyEmpty")
            empty.setAlignment(Qt.AlignCenter)
            empty.setMinimumHeight(120)
            self._list_layout.addWidget(empty)
            self._list_layout.addStretch()
            return

        for entry in filtered:
            row = _HistoryRow(entry)
            row.refill.connect(self._on_refill)
            row.resubmit.connect(self._on_resubmit)
            row.deleted.connect(self._on_delete)
            row.open_dir.connect(self._on_open_dir)
            row.activated.connect(self._on_refill_and_close)
            self._list_layout.addWidget(row)
        self._list_layout.addStretch()

    def _on_refill(self, prompt: str) -> None:
        if prompt:
            self.refill_requested.emit(prompt)

    def _on_refill_and_close(self, prompt: str) -> None:
        if prompt:
            self.refill_requested.emit(prompt)
            self.accept()

    def _on_resubmit(self, prompt: str) -> None:
        if prompt:
            self.resubmit_requested.emit(prompt)
            self.accept()

    def _on_open_dir(self, run_dir: str) -> None:
        open_run_dir(run_dir, parent=self)

    def _on_delete(self, entry_id: str) -> None:
        if sh.delete_submission(entry_id, path=self._history_path):
            self.reload()

    def _clear_all(self) -> None:
        if not self._entries:
            return
        reply = QMessageBox.question(
            self,
            "清空历史",
            f"确定清空全部 {len(self._entries)} 条提交记录？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        sh.clear_history(path=self._history_path)
        self.reload()
