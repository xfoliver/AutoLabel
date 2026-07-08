"""
AutoTrain 4.0 PyQt 前端的后端 Worker。

前后端分离：本文件只负责执行 auto_train4.0.py 子进程并通过信号把日志发回 UI，
不处理任何界面逻辑。
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUTO_TRAIN = PROJECT_ROOT / "auto_train4.0.py"


class TrainWorker(QThread):
    """后台工作线程：直接运行 auto_train4.0.py 并通过信号实时返回日志。"""

    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)  # status_text, style_class
    finished_signal = pyqtSignal(int, float)  # return_code, elapsed_seconds

    def __init__(self, prompt: str, parent: Optional["QObject"] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self.prompt = prompt
        self._running = True

    def run(self) -> None:
        start_time = time.time()
        self.status_changed.emit("运行中", "running")
        self.log_received.emit(f"[开始] 提示词: {self.prompt}")

        cmd = [sys.executable, str(AUTO_TRAIN), "--prompt", self.prompt]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            for line in process.stdout:
                if not self._running:
                    process.terminate()
                    break
                line = line.rstrip("\n")
                if line:
                    self.log_received.emit(line)

            process.wait()
        except FileNotFoundError:
            self.log_received.emit(f"[错误] 找不到文件: {AUTO_TRAIN}")
            self.status_changed.emit("出错", "error")
            self.finished_signal.emit(-1, round(time.time() - start_time, 2))
            return
        except Exception as e:
            self.log_received.emit(f"[错误] {e}")
            self.status_changed.emit("出错", "error")
            self.finished_signal.emit(-1, round(time.time() - start_time, 2))
            return

        elapsed = round(time.time() - start_time, 2)
        self.finished_signal.emit(process.returncode, elapsed)

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
