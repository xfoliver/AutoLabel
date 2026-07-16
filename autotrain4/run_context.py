"""
单次训练运行的上下文与归档管理。

为每一次训练生成唯一的 run_id 与对应的 run_dir，
集中保存 config、日志、runtime_info、exception、截图、录屏、报告等。
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_run_id(folder_name: str) -> str:
    """生成 run_id：数据集文件夹名 + 时间戳。

    示例：
        2-CLS-20260713_164352
        6-SSEG-20260715_130333
    """
    safe = "".join(c for c in folder_name.strip() if c.isalnum() or c in "-_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe}-{timestamp}"


def _collect_runtime_info() -> Dict[str, Any]:
    """收集当前运行环境信息。"""
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid(),
    }

    try:
        import psutil

        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
        }
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except Exception:
        pass

    # GPU / CUDA
    cuda_info: Dict[str, Any] = {"available": False}
    try:
        import torch

        cuda_info["available"] = torch.cuda.is_available()
        if cuda_info["available"]:
            cuda_info["version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["devices"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    info["cuda"] = cuda_info

    # 平台版本 / backend_url 等由调用方补充
    return info


class RunContext:
    """单次训练的运行上下文。"""

    def __init__(
        self,
        dataset_dir: Path,
        project_dir: Optional[Path] = None,
        folder_name: Optional[str] = None,
        run_id: Optional[str] = None,
        work_root: Optional[Path] = None,
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        # 运行归档目录放在项目文件夹下（如 2-CLS/work/），而不是训练集子目录下，
        # 这样 dva/dvp、录屏、日志更容易找到，也与旧版行为一致。
        self.project_dir = Path(project_dir).resolve() if project_dir else self.dataset_dir
        self.folder_name = folder_name or self.project_dir.name
        self.run_id = run_id or generate_run_id(self.folder_name)

        # run_dir: <project_dir>/work/<run_id>/
        work_root = Path(work_root) if work_root else self.project_dir / "work"
        self.run_dir = (work_root / self.run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.run_dir / "config.json"
        self.train_log_path = self.run_dir / "train.log"
        self.stdout_log_path = self.run_dir / "stdout.log"
        self.stderr_log_path = self.run_dir / "stderr.log"
        self.runtime_info_path = self.run_dir / "runtime_info.json"
        self.exception_path = self.run_dir / "exception.json"
        self.screenshots_dir = self.run_dir / "screenshots"
        self.screen_recording_path = self.run_dir / "screen_recording.mp4"
        self.inference_report_path = self.run_dir / "inference_report.json"
        self.baseline_compare_report_path = self.run_dir / "baseline_compare_report.json"

    def __repr__(self) -> str:
        return f"RunContext(run_id={self.run_id}, run_dir={self.run_dir})"

    def save_runtime_info(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        """保存运行环境信息。"""
        info = _collect_runtime_info()
        info["run_id"] = self.run_id
        info["run_dir"] = str(self.run_dir)
        info["dataset_dir"] = str(self.dataset_dir)
        if extra:
            info.update(extra)
        return self._write_json(self.runtime_info_path, info)

    def save_config(self, config: Dict[str, Any]) -> Path:
        """保存本次训练配置。"""
        cfg = dict(config)
        cfg["run_id"] = self.run_id
        cfg["run_dir"] = str(self.run_dir)
        return self._write_json(self.config_path, cfg)

    def save_exception(
        self,
        exc: Optional[BaseException] = None,
        exit_code: Optional[int] = None,
        message: Optional[str] = None,
    ) -> Path:
        """保存异常/失败信息。"""
        data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "exit_code": exit_code,
            "message": message or (str(exc) if exc else ""),
        }
        if exc is not None:
            data["exception_type"] = type(exc).__name__
            data["traceback"] = traceback.format_exception(
                type(exc), exc, exc.__traceback__
            )
        else:
            data["exception_type"] = None
            data["traceback"] = []
        return self._write_json(self.exception_path, data)

    def ensure_screenshots_dir(self) -> Path:
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        return self.screenshots_dir

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def as_config_fragment(self) -> Dict[str, Any]:
        """返回需要写入 project 配置的字段。"""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "work_dir": str(self.run_dir),
            "original_dataset_dir": str(self.dataset_dir),
        }
