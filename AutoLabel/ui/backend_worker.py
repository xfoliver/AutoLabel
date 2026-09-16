"""Background LabelWorker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from autolabel.name_suggest import suggest_names_from_image
from autolabel.orchestrator import AutoLabelOrchestrator


def _qt_alive(obj) -> bool:
    """False when the C++ QObject was already deleteLater'd."""
    if obj is None:
        return False
    try:
        from PyQt5 import sip

        return not sip.isdeleted(obj)
    except Exception:
        try:
            obj.thread()
            return True
        except RuntimeError:
            return False


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
        self.orch._log = self._forward_log

    def _forward_log(self, msg: str) -> None:
        try:
            self._emit("log_received", str(msg))
        except RuntimeError:
            return

    def _emit(self, name: str, *args) -> None:
        try:
            if not _qt_alive(self):
                return
            getattr(self, name).emit(*args)
        except RuntimeError:
            return

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if self.action == "plan":
                if "prompt" in self.kwargs or "short_label" in self.kwargs:
                    self._emit("status_changed", "parsing_intent")
                    self.orch.apply_ui_intent(
                        prompt=str(self.kwargs.get("prompt") or ""),
                        dataset_path=str(self.kwargs.get("dataset_path") or ""),
                        short_label=str(self.kwargs.get("short_label") or ""),
                        preview_count=int(self.kwargs.get("preview_count") or 5),
                    )
                self._emit("status_changed", "building_plan")
                plan = self.orch.build_plan(dataset_path=self.kwargs.get("dataset_path"))
                self._emit("plan_ready", plan)
                self._emit("status_changed", "plan_ready")
            elif self.action == "preview":
                self._emit("status_changed", "previewing")
                results = self.orch.run_preview(
                    progress=lambda i, t, p: self._emit("progress", i, t, p),
                    stop_flag=lambda: self._stop,
                )
                self._emit("preview_ready", results)
                self._emit("status_changed", "preview_ready")
            elif self.action == "batch":
                self._emit("status_changed", "batching")
                self.orch.confirm_preview()
                report = self.orch.run_batch(
                    progress=lambda i, t, p: self._emit("progress", i, t, p),
                    stop_flag=lambda: self._stop,
                )
                self._emit("batch_finished", report)
                self._emit("status_changed", "batch_done")
            elif self.action == "commit":
                self._emit("status_changed", "committing")
                summary = self.orch.commit()
                self._emit("commit_finished", summary)
                self._emit("status_changed", "commit_done")
            else:
                self._emit("failed", f"unknown action: {self.action}")
        except RuntimeError as e:
            if "has been deleted" in str(e):
                return
            self._emit("failed", str(e))
            self._emit("status_changed", "error")
        except Exception as e:
            self._emit("failed", str(e))
            self._emit("status_changed", "error")


class NameSuggestWorker(QThread):
    """Vision name-suggest only. Never starts SAM3 / plan / preview."""

    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        image_paths,
        settings: Dict[str, Any],
        backends_config: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        if isinstance(image_paths, (str, Path)):
            self.image_paths = [str(image_paths)]
        else:
            self.image_paths = [str(p) for p in image_paths]
        self.settings = settings
        self.backends_config = backends_config

    def run(self) -> None:
        try:
            cands = suggest_names_from_image(
                self.image_paths,
                settings=self.settings,
                backends_config=self.backends_config,
            )
            self.finished_ok.emit([c.to_dict() for c in cands])
        except RuntimeError as e:
            if "has been deleted" in str(e):
                return
            try:
                self.failed.emit(str(e))
            except RuntimeError:
                return
        except Exception as e:
            try:
                self.failed.emit(str(e))
            except RuntimeError:
                return
