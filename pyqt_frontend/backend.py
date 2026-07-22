"""AutoTrain 4.0 PyQt 前端的后端 Worker。

前后端分离：
- TrainWorker：执行 auto_train4.0.py 子进程并通过信号把日志发回 UI
- PostTrainWorker：训练成功后在桌面端直接跑 inference report / baseline 对比
  （不走 4.0 CLI 的 LLM + input，避免嵌套 stdin 卡死）
"""

from __future__ import annotations

# 优先从项目根目录 .env 加载环境变量
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal

# 保证能 import 项目根下的 autotrain4
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if load_dotenv is not None:
    _dotenv_path = PROJECT_ROOT / ".env"
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=_dotenv_path, override=True)

from autotrain4.common import (  # noqa: E402
    baseline_dir,
    find_baseline_model,
    find_latest_config,
    find_latest_dvp,
    resolve_path,
)
from autotrain4.output_protocol import decode_chat_response  # noqa: E402

AUTO_TRAIN = PROJECT_ROOT / "auto_train4.0.py"
DEFAULT_INFERENCE_URL = "http://127.0.0.1:9890"


def resolve_post_train_context() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解析训练后评估所需路径。

    Returns:
        (context_dict, error_message)
        context 含: config_path, task_type, dataset_dir, work_dir, new_dvp, baseline_dvp
    """
    cfg_path = find_latest_config()
    if not cfg_path:
        return None, "未找到 config/auto_latest_*.json，无法做训练后评估"

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        return None, f"读取配置失败: {e}"

    project = cfg.get("project", {}) or {}
    task_type = project.get("task_type") or ""
    dataset_dir = project.get("dataset_dir")
    work_dir = project.get("work_dir")

    if not dataset_dir or not work_dir:
        return None, "配置缺少 project.dataset_dir 或 project.work_dir"

    dataset_path = resolve_path(dataset_dir)
    work_path = resolve_path(work_dir)

    if not dataset_path.exists():
        return None, f"测试集目录不存在: {dataset_path}"
    if not work_path.exists():
        return None, f"工作目录不存在: {work_path}"

    new_dvp = find_latest_dvp(work_path)
    if not new_dvp:
        return None, f"工作目录下未找到 .dvp 模型: {work_path}"

    baseline = find_baseline_model(task_type) if task_type else None

    return {
        "config_path": str(cfg_path),
        "task_type": task_type,
        "dataset_dir": str(dataset_path),
        "work_dir": str(work_path),
        "run_dir": str(work_path),
        "new_dvp": str(new_dvp),
        "baseline_dvp": str(baseline) if baseline else None,
    }, None


class TrainWorker(QThread):
    """后台工作线程：运行 auto_train4.0.py 并实时返回日志。"""

    log_received = pyqtSignal(str)
    chat_response = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)
    # return_code, elapsed_seconds, training_completed, run_dir
    finished_signal = pyqtSignal(int, float, bool, str)
    prompt_for_close = pyqtSignal()
    process_started = pyqtSignal()

    _TRAIN_MARKERS = (
        "========== AUTO TRAIN 4.0 -> 2.0 ==========",
        "========== AUTO TRAIN 2.0 (WebBridge) START ==========",
        "========== AUTO TRAIN 2.0 (WebBridge) DONE ==========",
        "训练已完成，浏览器窗口将保持打开",
    )
    _NON_TRAIN_MARKERS = (
        "========== AUTO TRAIN 4.0 测试最新模型 ==========",
        "========== AUTO TRAIN 4.0 通用对话 ==========",
        "--generate-config-only",
        "--dry-run: 已生成 .dva",
    )

    def __init__(
        self,
        prompt: str,
        settings: Optional[Dict[str, str]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.prompt = prompt
        self.settings = settings or {}
        self.llm_settings = {
            k: v for k, v in (settings or {}).items()
            if k.startswith("llm_")
        }
        self._running = True
        self.process: Optional[subprocess.Popen] = None
        self._close_prompted = False
        self._saw_training = False
        self._saw_non_train = False
        self._close_signal_path = (
            Path(tempfile.gettempdir())
            / f"autotrain_close_{os.getpid()}_{id(self)}.signal"
        )
        self.run_dir: Optional[Path] = None
        self.train_log_path: Optional[Path] = None
        self.stderr_log_path: Optional[Path] = None
        self.exception_path: Optional[Path] = None

    def run(self) -> None:
        start_time = time.time()
        self.status_changed.emit("运行中", "running")
        self._emit_and_log(f"[开始] 提示词: {self.prompt}")
        self._cleanup_signal_file()

        cmd = [
            sys.executable,
            str(AUTO_TRAIN),
            "--prompt",
            self.prompt,
            "--no-post-train",
        ]

        adapter = self.settings.get("adapter", "playwright")
        if adapter:
            cmd.extend(["--adapter", adapter])
        if adapter == "webview2_cdp":
            cdp_port = self.settings.get("cdp_port", "9222")
            platform_exe = self.settings.get("platform_exe", "")
            cmd.extend(["--cdp-port", str(cdp_port)])
            if platform_exe:
                cmd.extend(["--platform-exe", platform_exe])
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["AUTOTRAIN_CLOSE_SIGNAL"] = str(self._close_signal_path)

        # 注入用户在 UI 设置的 LLM 配置（如果填写了）
        if self.llm_settings.get("llm_base_url"):
            env["AUTOTRAIN_LLM_BASE_URL"] = self.llm_settings["llm_base_url"]
        if self.llm_settings.get("llm_api_key"):
            env["AUTOTRAIN_LLM_API_KEY"] = self.llm_settings["llm_api_key"]
        if self.llm_settings.get("llm_model"):
            env["AUTOTRAIN_LLM_MODEL"] = self.llm_settings["llm_model"]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            self.process_started.emit()

            import threading

            def _read_stdout() -> None:
                for line in self.process.stdout:  # type: ignore[union-attr]
                    if not self._running:
                        break
                    self._handle_output_line(line, is_stderr=False)

            def _read_stderr() -> None:
                for line in self.process.stderr:  # type: ignore[union-attr]
                    if not self._running:
                        break
                    self._handle_output_line(line, is_stderr=True)

            t_out = threading.Thread(target=_read_stdout, daemon=True)
            t_err = threading.Thread(target=_read_stderr, daemon=True)
            t_out.start()
            t_err.start()

            self.process.wait()
            t_out.join(timeout=2)
            t_err.join(timeout=2)
        except FileNotFoundError:
            self._emit_and_log(f"[错误] 找不到文件: {AUTO_TRAIN}")
            self.status_changed.emit("出错", "error")
            self.finished_signal.emit(-1, round(time.time() - start_time, 2), False, "")
            return
        except Exception as e:
            self._emit_and_log(f"[错误] {e}")
            self.status_changed.emit("出错", "error")
            self.finished_signal.emit(-1, round(time.time() - start_time, 2), False, "")
            return
        finally:
            self._cleanup_signal_file()

        elapsed = round(time.time() - start_time, 2)
        code = self.process.returncode if self.process else -1
        training_completed = bool(code == 0 and self._saw_training and not self._saw_non_train)

        if code != 0 and self.run_dir is not None:
            self._save_exception(
                exit_code=code,
                message=f"auto_train4.0.py 子进程退出码非零: {code}",
            )

        run_dir_str = str(self.run_dir) if self.run_dir else ""
        self.finished_signal.emit(code, elapsed, training_completed, run_dir_str)

    def _handle_output_line(self, line: str, is_stderr: bool = False) -> None:
        line = line.rstrip("\n")
        if not line:
            return

        chat_response = decode_chat_response(line)
        if chat_response is not None:
            self.chat_response.emit(chat_response)
            return

        # 捕获 [RUN_DIR] 标记，初始化本地日志文件
        if line.startswith("[RUN_DIR] "):
            run_dir = Path(line[len("[RUN_DIR] "):].strip())
            if run_dir.exists():
                self._init_run_logs(run_dir)

        self.log_received.emit(line)
        self._observe_log_line(line)
        self._write_log(line, is_stderr=is_stderr)

        if not self._close_prompted and "训练已完成，浏览器窗口将保持打开" in line:
            self._close_prompted = True
            self.prompt_for_close.emit()

    def _emit_and_log(self, line: str) -> None:
        self.log_received.emit(line)
        self._write_log(line, is_stderr=False)

    def _init_run_logs(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.train_log_path = run_dir / "train.log"
        self.stderr_log_path = run_dir / "stderr.log"
        self.exception_path = run_dir / "exception.json"

    def _write_log(self, line: str, is_stderr: bool = False) -> None:
        if self.train_log_path:
            try:
                with open(self.train_log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        if is_stderr and self.stderr_log_path:
            try:
                with open(self.stderr_log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _save_exception(self, exit_code: int, message: str) -> None:
        if not self.exception_path:
            return
        try:
            data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "exit_code": exit_code,
                "exception_type": "SubprocessError",
                "message": message,
            }
            self.exception_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.exception_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _observe_log_line(self, line: str) -> None:
        for marker in self._TRAIN_MARKERS:
            if marker in line:
                self._saw_training = True
                break
        for marker in self._NON_TRAIN_MARKERS:
            if marker in line:
                self._saw_non_train = True
                break

    def send_enter(self) -> None:
        try:
            self._close_signal_path.parent.mkdir(parents=True, exist_ok=True)
            self._close_signal_path.write_text("close", encoding="utf-8")
            self.log_received.emit(
                f"[操作] 已写入关闭信号文件: {self._close_signal_path.name}"
            )
        except Exception as e:
            self.log_received.emit(f"[错误] 写入关闭信号文件失败: {e}")

        if self.process and self.process.poll() is None and self.process.stdin:
            try:
                self.process.stdin.write("\n")
                self.process.stdin.flush()
                self.log_received.emit("[操作] 已发送关闭浏览器指令（模拟回车）")
            except Exception as e:
                self.log_received.emit(f"[错误] 发送关闭指令失败: {e}")

    def stop(self) -> None:
        self._running = False
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


class PostTrainWorker(QThread):
    """训练成功后：生成 inference report 或与 baseline 对比。"""

    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        action: str,
        context: Dict[str, Any],
        inference_url: str = DEFAULT_INFERENCE_URL,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.action = action
        self.context = context
        self.inference_url = inference_url
        self._running = True

    def run(self) -> None:
        from autotrain4.evaluator import InferenceEvaluator
        from autotrain4.inference_client import InferenceClient

        start = time.time()
        self.status_changed.emit("评估中", "running")

        new_dvp = Path(self.context["new_dvp"])
        image_dir = Path(self.context["dataset_dir"])
        baseline = self.context.get("baseline_dvp")
        task_type = self.context.get("task_type") or "model"

        compare = self.action == "compare"
        if compare and not baseline:
            self.log_received.emit(
                f"[警告] 未找到任务「{task_type}」对应的 baseline 模型"
                f"（目录: {baseline_dir()}），已改为仅生成 inference report"
            )
            compare = False

        run_dir = Path(self.context.get("run_dir") or PROJECT_ROOT)
        report_name = (
            "baseline_compare_report.json"
            if compare
            else "inference_report.json"
        )
        report_path = run_dir / report_name

        self.log_received.emit("========== 训练后评估 ==========")
        self.log_received.emit(f"动作: {'与 baseline 对比' if compare else '生成 inference report'}")
        self.log_received.emit(f"新模型: {new_dvp}")
        self.log_received.emit(f"测试图片: {image_dir}")
        if compare:
            self.log_received.emit(f"基准模型: {baseline}")
        self.log_received.emit(f"推理服务: {self.inference_url}")

        if not new_dvp.exists():
            self.log_received.emit(f"[错误] 模型文件不存在: {new_dvp}")
            self.status_changed.emit("失败", "error")
            self.finished_signal.emit(False, f"模型文件不存在: {new_dvp}")
            return

        try:
            def _on_progress(current: int, total: int, image_name: str, model_name: str) -> None:
                if total <= 0:
                    return
                pct = int(current * 100 / total)
                bar_len = 20
                filled = int(bar_len * current / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                self.log_received.emit(
                    f"[进度] [{bar}] {current}/{total} ({pct}%)  "
                    f"模型={model_name}  当前={image_name}"
                )

            client = InferenceClient(base_url=self.inference_url, progress=False)
            evaluator = InferenceEvaluator(client=client)
            report = evaluator.evaluate(
                new_model_path=new_dvp,
                image_dir=image_dir,
                baseline_model_path=baseline if compare else None,
                on_progress=_on_progress,
            )
            report.save(report_path)

            summary = report.to_dict().get("summary", {})
            total = summary.get("total_images", 0)
            success = summary.get("new_success_count", 0)
            failed = total - success
            elapsed = round(time.time() - start, 2)
            avg = round(elapsed / total, 3) if total > 0 else 0.0
            self.log_received.emit(
                f"[汇总] 总图={total} 成功={success} 失败={failed} 平均={avg}s/张"
            )
            if compare:
                self.log_received.emit(
                    f"[汇总] baseline成功={summary.get('baseline_success_count')} "
                    f"一致={summary.get('match_count')} 不一致={summary.get('mismatch_count')}"
                )
            self.log_received.emit(f"[完成] 报告已保存: {report_path} （总耗时 {elapsed}s）")
            self.status_changed.emit("成功", "success")
            self.finished_signal.emit(True, str(report_path))
        except Exception as e:
            err_msg = str(e)
            self.log_received.emit(f"[错误] 训练后评估失败: {err_msg}")
            if "未 ready" in err_msg or "无法连接" in err_msg or "Connection" in err_msg:
                self.log_received.emit(
                    "[提示] 请先启动 DLCV Test（推理后端，默认 http://127.0.0.1:9890）。"
                    "训练平台 Backend(9887) 不能代替推理服务。"
                )
            self.status_changed.emit("失败", "error")
            self.finished_signal.emit(False, err_msg)

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
