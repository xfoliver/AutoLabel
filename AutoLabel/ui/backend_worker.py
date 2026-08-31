"""Background LabelWorker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from autolabel.orchestrator import AutoLabelOrchestrator


class LabelWorker(QThread):
    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    plan_ready = pyqtSignal(object)
    preview_ready = pyqtSignal(object)
    batch_finished = pyqtSignal(object)
    commit_finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        action: str,
        orch: AutoLabelOrchestrator,
        parent=None,
        **kwargs,
    ):
        super().__init__(parent)
        self.action = action
        self.orch = orch
        self.kwargs = kwargs
        self._stop = False
        self.orch._log = lambda m: self.log_received.emit(str(m))

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if self.action == "plan":
                self.status_changed.emit("building_plan")
                plan = self.orch.build_plan(dataset_path=self.kwargs.get("dataset_path"))
                self.plan_ready.emit(plan)
                self.status_changed.emit("plan_ready")
            elif self.action == "preview":
                self.status_changed.emit("previewing")
                results = self.orch.run_preview(
                    progress=lambda i, t, p: self.progress.emit(i, t, p)
                )
                self.preview_ready.emit(results)
                self.status_changed.emit("preview_ready")
            elif self.action == "batch":
                self.status_changed.emit("batching")
                self.orch.confirm_preview()
                report = self.orch.run_batch(
                    progress=lambda i, t, p: self.progress.emit(i, t, p),
                    stop_flag=lambda: self._stop,
                )
                self.batch_finished.emit(report)
                self.status_changed.emit("batch_done")
            elif self.action == "commit":
                self.status_changed.emit("committing")
                summary = self.orch.commit()
                self.commit_finished.emit(summary)
                self.status_changed.emit("commit_done")
            else:
                self.failed.emit(f"unknown action: {self.action}")
        except Exception as e:
            self.failed.emit(str(e))
            self.status_changed.emit("error")
