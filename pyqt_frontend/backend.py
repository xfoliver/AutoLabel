"""AutoTrain 4.0 PyQt 前端的后端 Worker。

前后端分离：
- TrainWorker：执行 auto_train4.0.py 子进程并通过信号把日志发回 UI
- PostTrainWorker：训练成功后在桌面端直接跑 inference report / baseline 对比
  （不走 4.0 CLI 的 LLM + input，避免嵌套 stdin 卡死）
- AssistantSummaryWorker：训练 / 质检 / 评估结束后生成助手摘要（模板 + 可选 LLM）
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

from autotrain4.output_protocol import (  # noqa: E402
    decode_at4_event,
    decode_chat_response,
)
from autotrain4.post_train import (  # noqa: E402
    resolve_post_train_context as _resolve_post_train_context,
    run_post_train_action,
)

AUTO_TRAIN = PROJECT_ROOT / "auto_train4.0.py"
try:
    from autotrain4.constants import DEFAULT_INFERENCE_URL
except ImportError:  # pragma: no cover
    DEFAULT_INFERENCE_URL = "http://127.0.0.1:9890"


def resolve_post_train_context() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解析训练后评估所需路径（委托 autotrain4.post_train，保持 UI 调用签名）。"""
    return _resolve_post_train_context()


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
        # 完整 settings（含 llm_profiles）；子进程通过 env 桥接注入
        self.llm_settings = dict(settings or {})
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
        from autotrain4.llm import inject_llm_env_for_subprocess, runtime_from_settings

        env = inject_llm_env_for_subprocess(os.environ, self.llm_settings)
        env["PYTHONIOENCODING"] = "utf-8"
        env["AUTOTRAIN_CLOSE_SIGNAL"] = str(self._close_signal_path)

        rt = runtime_from_settings(self.llm_settings)
        self._emit_and_log(
            f"[LLM] profile={rt.profile_name or '(default)'} "
            f"base={rt.base_url or '(空)'} model={rt.model or '(空)'} "
            f"protocol={rt.protocol or 'auto'}"
        )

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

        # 优先解析结构化 __AT4__ 事件；旧字符串 marker 作 fallback
        at4 = decode_at4_event(line)
        if at4 is not None:
            self._handle_at4_event(at4, line=line, is_stderr=is_stderr)
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

    def _handle_at4_event(
        self, event: Dict[str, Any], *, line: str, is_stderr: bool
    ) -> None:
        """处理 autotrain4 结构化 IPC 行。"""
        etype = str(event.get("type") or "")
        if etype == "run_dir":
            path_raw = event.get("path") or ""
            run_dir = Path(str(path_raw).strip())
            if run_dir.exists():
                self._init_run_logs(run_dir)
        elif etype == "stage" and str(event.get("name") or "") == "train_done":
            if not self._close_prompted:
                # train_done 仅作阶段信号；关闭确认仍由文案/旧 marker 触发
                pass
        # 结构化行也进日志，便于排查
        self.log_received.emit(line)
        self._observe_log_line(line)
        self._write_log(line, is_stderr=is_stderr)

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


class AssistantSummaryWorker(QThread):
    """后台生成助手摘要（模板 + 可选 LLM 润色），避免卡住 UI。"""

    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        use_llm: bool = False,
        llm_settings: Optional[Dict[str, str]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self.payload = payload or {}
        self.use_llm = bool(use_llm)
        self.llm_settings = llm_settings or {}

    def run(self) -> None:  # noqa: N802
        try:
            from autotrain4.result_summarizer import (
                summarize_eval_with_optional_llm,
                summarize_quality_with_optional_llm,
                summarize_train_with_optional_llm,
            )

            llm = None
            if self.use_llm:
                llm = self._make_llm()

            kind = self.kind
            p = self.payload
            if kind == "quality":
                text = summarize_quality_with_optional_llm(
                    p.get("report") or {},
                    use_llm=self.use_llm,
                    llm_client=llm,
                )
            elif kind == "train":
                text = summarize_train_with_optional_llm(
                    use_llm=self.use_llm,
                    llm_client=llm,
                    success=bool(p.get("success")),
                    training_completed=bool(p.get("training_completed")),
                    elapsed_sec=p.get("elapsed_sec"),
                    run_dir=p.get("run_dir"),
                    exit_code=p.get("exit_code"),
                    task_type=p.get("task_type"),
                    extra_note=p.get("extra_note"),
                )
            elif kind == "eval":
                text = summarize_eval_with_optional_llm(
                    p.get("report"),
                    use_llm=self.use_llm,
                    llm_client=llm,
                    success=bool(p.get("success", True)),
                    report_path=p.get("report_path"),
                    error_message=p.get("error_message"),
                    action=p.get("action"),
                    task_type=p.get("task_type"),
                    elapsed_sec=p.get("elapsed_sec"),
                )
            else:
                self.finished_err.emit(f"未知摘要类型: {kind}")
                return

            self.finished_ok.emit(text or "")
        except Exception as e:
            self.finished_err.emit(str(e))

    def _make_llm(self) -> Any:
        from autotrain4.llm import LLMClient

        if self.llm_settings:
            return LLMClient.from_settings(self.llm_settings, timeout=45)
        return LLMClient.from_environ(timeout=45)


class PostTrainWorker(QThread):
    """训练成功后：生成 inference report 或与 baseline 对比。"""

    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)
    # success, message(path|error), action, report_dict_or_empty
    finished_signal = pyqtSignal(bool, str, str, object)

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
        self.status_changed.emit("评估中", "running")

        new_dvp = Path(self.context["new_dvp"])
        image_dir = Path(self.context.get("image_dir") or self.context["dataset_dir"])
        baseline = self.context.get("baseline_dvp")
        want_compare = self.action == "compare"

        self.log_received.emit("========== 训练后评估 ==========")
        self.log_received.emit(
            f"动作: {'与 baseline 对比' if want_compare else '生成 inference report'}"
        )
        self.log_received.emit(f"新模型: {new_dvp}")
        self.log_received.emit(f"测试图片: {image_dir}")
        if want_compare and baseline:
            self.log_received.emit(f"基准模型: {baseline}")
        self.log_received.emit(f"推理服务: {self.inference_url}")

        action_name = "compare" if want_compare else "report"

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

            result = run_post_train_action(
                self.action,
                self.context,
                inference_url=self.inference_url,
                progress=False,
                on_progress=_on_progress,
            )
            for w in result.get("warnings") or []:
                self.log_received.emit(f"[警告] {w}")

            action_name = result.get("action") or action_name
            report_path = result.get("report_path") or ""
            report_dict = result.get("report") or {}
            summary = result.get("summary") or {}
            total = summary.get("total_images", 0)
            success = summary.get("new_success_count", 0)
            failed = total - success
            elapsed = float(result.get("elapsed_sec") or 0.0)
            avg = round(elapsed / total, 3) if total > 0 else 0.0
            self.log_received.emit(
                f"[汇总] 总图={total} 成功={success} 失败={failed} 平均={avg}s/张"
            )
            if result.get("compare"):
                self.log_received.emit(
                    f"[汇总] baseline成功={summary.get('baseline_success_count')} "
                    f"一致={summary.get('match_count')} 不一致={summary.get('mismatch_count')}"
                )
            self.log_received.emit(f"[完成] 报告已保存: {report_path} （总耗时 {elapsed}s）")
            self.status_changed.emit("成功", "success")
            self.finished_signal.emit(True, str(report_path), action_name, report_dict)
        except Exception as e:
            err_msg = str(e)
            self.log_received.emit(f"[错误] 训练后评估失败: {err_msg}")
            if "未 ready" in err_msg or "无法连接" in err_msg or "Connection" in err_msg:
                self.log_received.emit(
                    "[提示] 请先启动 DLCV Test（推理后端，默认 http://127.0.0.1:9890）。"
                    "训练平台 Backend(9887) 不能代替推理服务。"
                )
            self.status_changed.emit("失败", "error")
            self.finished_signal.emit(False, err_msg, action_name, {})

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
