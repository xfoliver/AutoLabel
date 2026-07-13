"""AutoTrain 4.0 PyQt 前端的后端 Worker。

前后端分离：本文件只负责执行 auto_train4.0.py 子进程并通过信号把日志发回 UI，
不处理任何界面逻辑。
"""

import os
import subprocess
import sys
import tempfile
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
    prompt_for_close = pyqtSignal()  # 训练完成、等待关闭浏览器时触发
    process_started = pyqtSignal()  # auto_train4.0 子进程已启动
    post_train_prompt = pyqtSignal(str)  # 训练后 LLM 询问用户后续需求

    def __init__(self, prompt: str, parent: Optional["QObject"] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self.prompt = prompt
        self._running = True
        self.process: Optional[subprocess.Popen] = None
        self._close_prompted = False
        # 关闭浏览器信号文件：即使嵌套 stdin 管道在 Windows 下失效，也能通过文件通知 2.0 退出
        self._close_signal_path = Path(tempfile.gettempdir()) / f"autotrain_close_{os.getpid()}_{id(self)}.signal"

    def run(self) -> None:
        start_time = time.time()
        self.status_changed.emit("运行中", "running")
        self.log_received.emit(f"[开始] 提示词: {self.prompt}")

        # 清理可能残留的信号文件
        try:
            if self._close_signal_path.exists():
                self._close_signal_path.unlink()
        except Exception:
            pass

        cmd = [
            sys.executable,
            str(AUTO_TRAIN),
            "--prompt",
            self.prompt,
            "--no-post-train",
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["AUTOTRAIN_CLOSE_SIGNAL"] = str(self._close_signal_path)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            self.process_started.emit()

            for line in self.process.stdout:
                if not self._running:
                    self.process.terminate()
                    break
                line = line.rstrip("\n")
                if line:
                    self.log_received.emit(line)
                    if not self._close_prompted and "训练已完成，浏览器窗口将保持打开" in line:
                        self._close_prompted = True
                        self.prompt_for_close.emit()
                    # 捕获训练后 LLM 询问，弹窗让用户输入
                    if line.startswith("[POST_TRAIN_PROMPT] "):
                        question = line[len("[POST_TRAIN_PROMPT] "):]
                        self.post_train_prompt.emit(question)

            self.process.wait()
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
        finally:
            self._cleanup_signal_file()

        elapsed = round(time.time() - start_time, 2)
        self.finished_signal.emit(self.process.returncode, elapsed)

    def send_enter(self) -> None:
        """向子进程发送关闭浏览器指令（stdin 回车 + 信号文件双通道）。"""
        # 1. 信号文件：auto_train2.0 主线程会轮询 AUTOTRAIN_CLOSE_SIGNAL
        try:
            self._close_signal_path.parent.mkdir(parents=True, exist_ok=True)
            self._close_signal_path.write_text("close", encoding="utf-8")
            self.log_received.emit(f"[操作] 已写入关闭信号文件: {self._close_signal_path.name}")
        except Exception as e:
            self.log_received.emit(f"[错误] 写入关闭信号文件失败: {e}")

        # 2. stdin 回车：兼容命令行直接跑 2.0 的场景
        if self.process and self.process.poll() is None and self.process.stdin:
            try:
                self.process.stdin.write("\n")
                self.process.stdin.flush()
                self.log_received.emit("[操作] 已发送关闭浏览器指令（模拟回车）")
            except Exception as e:
                self.log_received.emit(f"[错误] 发送关闭指令失败: {e}")
        else:
            self.log_received.emit("[调试] 子进程 stdin 不可用，已依赖信号文件关闭")

    def send_input(self, text: str) -> None:
        """向子进程发送任意输入（用于训练后 LLM 对话）。"""
        if self.process and self.process.poll() is None and self.process.stdin:
            try:
                self.process.stdin.write(text + "\n")
                self.process.stdin.flush()
                self.log_received.emit(f"[操作] 已发送输入: {text}")
            except Exception as e:
                self.log_received.emit(f"[错误] 发送输入失败: {e}")
        else:
            self.log_received.emit("[调试] 无法发送输入：子进程已结束或未接管 stdin")

    def stop(self) -> None:
        self._running = False
        # 尽量先发关闭信号，让 2.0 正常收尾
        try:
            self._close_signal_path.write_text("close", encoding="utf-8")
        except Exception:
            pass
        if self.process and self.process.poll() is None:
            try:
                if self.process.stdin:
                    self.process.stdin.write("\n")
                    self.process.stdin.flush()
            except Exception:
                pass
            self.process.terminate()
        self.wait(3000)
        self._cleanup_signal_file()

    def _cleanup_signal_file(self) -> None:
        try:
            if self._close_signal_path.exists():
                self._close_signal_path.unlink()
        except Exception:
            pass
