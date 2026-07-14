"""AutoTrain 4.0 PyQt 前端的后端 Worker。

前后端分离：
- TrainWorker：执行 auto_train4.0.py 子进程并通过信号把日志发回 UI
- PostTrainWorker：训练成功后在桌面端直接跑 inference report / baseline 对比
  （不走 4.0 CLI 的 LLM + input，避免嵌套 stdin 卡死）
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUTO_TRAIN = PROJECT_ROOT / "auto_train4.0.py"
CONFIG_DIR = PROJECT_ROOT / "config"
BASELINE_DIR = PROJECT_ROOT / "Baseline Models"
DEFAULT_INFERENCE_URL = "http://127.0.0.1:9890"

# 与 autotrain4.common.BASELINE_PREFIXES 保持一致
BASELINE_PREFIXES = {
    "OCR": "OCR",
    "分类": "CLS",
    "目标检测": "DET",
    "DETR": "DETR",
    "实例分割": "ISEG",
    "语义分割": "SSEG",
    "旋转框检测": "ROTATE",
}


# ---------------------------------------------------------------------------
# 路径解析工具（桌面端训练后评估用，不依赖 4.0 CLI 交互）
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_config(task_type: str = "") -> Optional[Path]:
    """查找最新的 auto_latest_*.json。"""
    if not CONFIG_DIR.exists():
        return None
    candidates = list(CONFIG_DIR.glob("auto_latest_*.json"))
    if not candidates:
        return None
    if task_type:
        safe = re.sub(r"[^\w\-]+", "_", task_type)
        matched = [p for p in candidates if safe in p.name]
        if matched:
            return max(matched, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_dvp(directory: Path) -> Optional[Path]:
    """目录下最新的 .dvp。"""
    if not directory.exists():
        return None
    dvps = list(directory.glob("*.dvp"))
    if not dvps:
        return None
    return max(dvps, key=lambda p: p.stat().st_mtime)


def find_baseline_model(task_type: str) -> Optional[Path]:
    """按任务类型在 Baseline Models 下找最新 baseline。"""
    if not BASELINE_DIR.exists():
        return None
    prefix = BASELINE_PREFIXES.get(task_type)
    if not prefix:
        return None
    candidates = [
        p
        for p in BASELINE_DIR.glob("*.dvp")
        if p.name.upper().startswith(prefix.upper() + "-")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_post_train_context() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解析训练后评估所需路径。

    Returns:
        (context_dict, error_message)
        context 含: config_path, task_type, dataset_dir, work_dir, new_dvp, baseline_dvp
    """
    cfg_path = find_latest_config()
    if not cfg_path:
        return None, "未找到 config/auto_latest_*.json，无法做训练后评估"

    try:
        cfg = _load_json(cfg_path)
    except Exception as e:
        return None, f"读取配置失败: {e}"

    project = cfg.get("project", {}) or {}
    task_type = project.get("task_type") or ""
    dataset_dir = project.get("dataset_dir")
    work_dir = project.get("work_dir")

    if not dataset_dir or not work_dir:
        return None, "配置缺少 project.dataset_dir 或 project.work_dir"

    dataset_path = Path(dataset_dir)
    work_path = Path(work_dir)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    if not work_path.is_absolute():
        work_path = (PROJECT_ROOT / work_path).resolve()

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
        "new_dvp": str(new_dvp),
        "baseline_dvp": str(baseline) if baseline else None,
    }, None


# ---------------------------------------------------------------------------
# 训练 Worker
# ---------------------------------------------------------------------------
class TrainWorker(QThread):
    """后台工作线程：运行 auto_train4.0.py 并实时返回日志。

    固定使用 --no-post-train：训练后评估改由桌面端 PostTrainWorker 负责，
    避免 4.0 CLI 里 LLM + input() 经嵌套管道卡死。
    """

    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)  # status_text, style_class
    # return_code, elapsed_seconds, training_completed
    # training_completed=True 仅当真正跑过 2.0 可视化训练时为真；
    # 测试最新模型 / 闲聊 / 仅生成配置 等成功结束时为 False，UI 不应弹 post-train 窗。
    finished_signal = pyqtSignal(int, float, bool)
    prompt_for_close = pyqtSignal()  # 训练完成、等待关闭浏览器时触发
    process_started = pyqtSignal()  # auto_train4.0 子进程已启动

    # 日志特征：真正进入训练执行引擎
    _TRAIN_MARKERS = (
        "========== AUTO TRAIN 4.0 -> 2.0 ==========",
        "========== AUTO TRAIN 2.0 (WebBridge) START ==========",
        "========== AUTO TRAIN 2.0 (WebBridge) DONE ==========",
        "训练已完成，浏览器窗口将保持打开",
    )
    # 日志特征：测试 / 评估 / 对话，不应弹训练后选择窗
    _NON_TRAIN_MARKERS = (
        "========== AUTO TRAIN 4.0 测试最新模型 ==========",
        "========== AUTO TRAIN 4.0 通用对话 ==========",
        "--generate-config-only",
        "--dry-run: 已生成 .dva",
    )

    def __init__(self, prompt: str, parent: Optional["QObject"] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self.prompt = prompt
        self._running = True
        self.process: Optional[subprocess.Popen] = None
        self._close_prompted = False
        self._saw_training = False
        self._saw_non_train = False
        self._close_signal_path = (
            Path(tempfile.gettempdir())
            / f"autotrain_close_{os.getpid()}_{id(self)}.signal"
        )

    def run(self) -> None:
        start_time = time.time()
        self.status_changed.emit("运行中", "running")
        self.log_received.emit(f"[开始] 提示词: {self.prompt}")

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
                    self._observe_log_line(line)
                    if (
                        not self._close_prompted
                        and "训练已完成，浏览器窗口将保持打开" in line
                    ):
                        self._close_prompted = True
                        self.prompt_for_close.emit()

            self.process.wait()
        except FileNotFoundError:
            self.log_received.emit(f"[错误] 找不到文件: {AUTO_TRAIN}")
            self.status_changed.emit("出错", "error")
            self.finished_signal.emit(-1, round(time.time() - start_time, 2), False)
            return
        except Exception as e:
            self.log_received.emit(f"[错误] {e}")
            self.status_changed.emit("出错", "error")
            self.finished_signal.emit(-1, round(time.time() - start_time, 2), False)
            return
        finally:
            self._cleanup_signal_file()

        elapsed = round(time.time() - start_time, 2)
        code = self.process.returncode if self.process else -1
        # 只有真正跑过训练流程才允许 UI 弹 post-train 窗；测试/对话即使成功也不弹
        training_completed = bool(code == 0 and self._saw_training and not self._saw_non_train)
        self.finished_signal.emit(code, elapsed, training_completed)

    def _observe_log_line(self, line: str) -> None:
        """根据 4.0/2.0 日志判断本次是训练还是测试/闲聊。"""
        for marker in self._TRAIN_MARKERS:
            if marker in line:
                self._saw_training = True
                break
        for marker in self._NON_TRAIN_MARKERS:
            if marker in line:
                self._saw_non_train = True
                break

    def send_enter(self) -> None:
        """关闭浏览器：信号文件 + stdin 回车双通道。"""
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
        else:
            self.log_received.emit("[调试] 子进程 stdin 不可用，已依赖信号文件关闭")

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


# ---------------------------------------------------------------------------
# 训练后评估 Worker（桌面端直接调用推理，不走 LLM/input）
# ---------------------------------------------------------------------------
class PostTrainWorker(QThread):
    """训练成功后：生成 inference report 或与 baseline 对比。"""

    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(
        self,
        action: str,
        context: Dict[str, Any],
        inference_url: str = DEFAULT_INFERENCE_URL,
        parent: Optional["QObject"] = None,  # type: ignore[name-defined]
    ) -> None:
        super().__init__(parent)
        self.action = action  # "report" | "compare"
        self.context = context
        self.inference_url = inference_url
        self._running = True

    def run(self) -> None:
        # 保证能 import 项目根下的 autotrain4
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        start = time.time()
        self.status_changed.emit("评估中", "running")

        new_dvp = Path(self.context["new_dvp"])
        image_dir = Path(self.context["dataset_dir"])
        baseline = self.context.get("baseline_dvp")
        task_type = self.context.get("task_type") or "model"

        compare = self.action == "compare"
        if compare and not baseline:
            msg = (
                f"未找到任务「{task_type}」对应的 baseline 模型"
                f"（目录: {BASELINE_DIR}），已改为仅生成 inference report"
            )
            self.log_received.emit(f"[警告] {msg}")
            compare = False

        report_name = (
            f"inference_report_{new_dvp.stem}.json"
            if not compare
            else f"compare_report_{new_dvp.stem}.json"
        )
        report_path = PROJECT_ROOT / report_name

        self.log_received.emit("========== 训练后评估 ==========")
        self.log_received.emit(f"动作: {'与 baseline 对比' if compare else '生成 inference report'}")
        self.log_received.emit(f"新模型: {new_dvp}")
        self.log_received.emit(f"测试图片: {image_dir}")
        if compare:
            self.log_received.emit(f"基准模型: {baseline}")
        self.log_received.emit(f"推理服务: {self.inference_url}")

        # 启动前先探测推理服务，避免只报模糊的「加载模型失败」
        try:
            import requests as _requests

            probe = _requests.get(self.inference_url.rstrip("/") + "/", timeout=3)
            self.log_received.emit(
                f"[检查] 推理服务可达: {self.inference_url} (HTTP {probe.status_code})"
            )
        except Exception as probe_err:
            self.log_received.emit(
                f"[错误] 推理服务不可达: {self.inference_url} — {probe_err}"
            )
            self.log_received.emit(
                "[提示] 请先启动 DLCV Test（推理后端，默认 http://127.0.0.1:9890）。"
                "训练平台 Backend(9887) 不能代替推理服务。"
            )
            self.status_changed.emit("失败", "error")
            self.finished_signal.emit(
                False,
                f"推理服务未启动或端口不对: {self.inference_url}",
            )
            return

        if not new_dvp.exists():
            self.log_received.emit(f"[错误] 模型文件不存在: {new_dvp}")
            self.status_changed.emit("失败", "error")
            self.finished_signal.emit(False, f"模型文件不存在: {new_dvp}")
            return

        try:
            from autotrain4.evaluator import InferenceEvaluator
            from autotrain4.inference_client import InferenceClient

            def _on_progress(current: int, total: int, image_name: str, model_name: str) -> None:
                if total <= 0:
                    return
                pct = int(current * 100 / total)
                # 条形进度，便于在运行日志里扫一眼
                bar_len = 20
                filled = int(bar_len * current / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                self.log_received.emit(
                    f"[进度] [{bar}] {current}/{total} ({pct}%)  "
                    f"模型={model_name}  当前={image_name}"
                )

            # progress=False：不用 tqdm（它只打到终端，PyQt 日志区看不到）
            # on_progress：把百分比写进运行日志
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
            elapsed = round(time.time() - start, 2)
            self.log_received.emit(
                f"[汇总] 总图={summary.get('total_images')} "
                f"新模型成功={summary.get('new_success_count')} "
                f"一致={summary.get('match_count')} "
                f"不一致={summary.get('mismatch_count')}"
            )
            self.log_received.emit(f"[完成] 报告已保存: {report_path} （耗时 {elapsed}s）")
            self.status_changed.emit("成功", "success")
            self.finished_signal.emit(True, str(report_path))
        except Exception as e:
            self.log_received.emit(f"[错误] 训练后评估失败: {e}")
            self.status_changed.emit("失败", "error")
            self.finished_signal.emit(False, str(e))

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
