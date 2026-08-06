"""
DLCV AI 训练平台自动化脚本（方案 B - Kimi WebBridge 版）

功能：
1. 读取 JSON 配置文件中的项目与训练参数。
2. 生成/更新 .dva 项目文件。
3. 启动 AI 训练平台桌面应用（或复用已有实例）。
4. 等待后端/前端服务就绪。
5. 通过后端 API 确保项目存在、写入超参数。
6. 通过 Kimi WebBridge 让 Agent 在浏览器中完成训练参数设置、
   保存并点击开始训练。
7. 监控训练状态直到完成/失败/中止。

使用方法：
    # 使用测试桩跑通流程（不连接真实 WebBridge）
    python auto_train2.0.py --config config/train_automation_config_2.0.json --dry-run

    # 连接真实 Kimi WebBridge HTTP 服务
    python auto_train2.0.py --config config/train_automation_config_2.0.json

依赖：
    - Python >= 3.8
    - pyyaml
    - requests
    - psutil
    - kimi_webbridge（本项目内包）
"""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import psutil
import requests
import yaml

from kimi_webbridge import WebBridgeError, create_webbridge

# 在 Windows Git Bash / CMD / PowerShell 中保持中文日志不乱码
if sys.platform == "win32" and sys.stdout.encoding not in ("utf-8", "UTF-8"):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AutoTrainError(Exception):
    """自动化过程中的自定义异常。"""
    pass


class DLCVAutoTrainer:
    """DLCV AI 训练平台自动化控制器（Kimi WebBridge 版）。"""

    PLATFORM_BACKEND_EXE = "DLCV AI Training Backend.exe"

    def __init__(
        self,
        config_path: str,
        force_new: bool = False,
        dry_run: bool = False,
        reuse: bool = False,
        keep_browser_open: bool = True,
    ):
        self.config_path = Path(config_path).resolve()
        self.config: Dict[str, Any] = {}
        self.platform_proc: Optional[subprocess.Popen] = None
        self.force_new = force_new
        self.dry_run = dry_run
        self.reuse = reuse
        self.keep_browser_open = keep_browser_open
        self.backend_url: str = "http://127.0.0.1:9887"
        self.frontend_url: str = "http://localhost:8842"
        self.project_id: Optional[str] = None
        self.training_id: Optional[str] = None
        # 一旦通过 WebBridge 或 API 成功拿到 training_id，禁止再二次 start
        self.training_launch_attempted: bool = False
        self.training_failure_detail: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 配置加载与校验
    # ------------------------------------------------------------------
    def load_config(self) -> None:
        """加载 JSON 配置文件，支持环境变量扩展。"""
        if not self.config_path.exists():
            raise AutoTrainError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        logger.info("已加载配置文件: %s", self.config_path)

        self._expand_env_vars(self.config)
        self._validate_config()
        self._log_effective_config()

        platform_cfg = self.config.get("platform", {})
        self.backend_url = platform_cfg.get("backend_url", self.backend_url).rstrip("/")
        self.frontend_url = self.config.get("webbridge", {}).get(
            "frontend_url", self.config.get("ui", {}).get("url", "http://localhost:8842")
        ).rstrip("/")

    def _log_effective_config(self) -> None:
        project = self.config.get("project", {})
        training = self.config.get("training", {})
        logger.info("生效配置 -> 项目名: %s, 任务类型: %s, 数据集: %s, 工作目录: %s",
                    project.get("name"), project.get("task_type"),
                    project.get("dataset_dir"), project.get("work_dir"))
        logger.info("生效配置 -> 超参数: %s", training.get("hyperparameters", {}))
        if self.force_new:
            logger.info("已启用 --force-new: 将强制生成新的 project_id")
        if self.dry_run:
            logger.info("已启用 --dry-run: 仅生成 .dva 并打印生效配置")
        if self.reuse:
            logger.info("已启用 --reuse: 将复用已有平台/后端实例")

    def _expand_env_vars(self, obj: Any) -> None:
        """递归展开字符串中的 Windows 环境变量（%VAR%）。"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    obj[key] = os.path.expandvars(value)
                elif isinstance(value, (dict, list)):
                    self._expand_env_vars(value)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                if isinstance(value, str):
                    obj[idx] = os.path.expandvars(value)
                elif isinstance(value, (dict, list)):
                    self._expand_env_vars(value)

    def _validate_config(self) -> None:
        """校验配置中必填项。"""
        required_top = ["platform", "project", "training"]
        for key in required_top:
            if key not in self.config:
                raise AutoTrainError(f"配置缺少必填项: {key}")

        platform = self.config["platform"]
        if "executable_path" not in platform:
            raise AutoTrainError("platform.executable_path 未配置")

        project = self.config["project"]
        for key in ["name", "task_type", "dataset_dir", "work_dir"]:
            if key not in project:
                raise AutoTrainError(f"project.{key} 未配置")

        training = self.config["training"]
        if "hyperparameters" not in training:
            logger.warning("training.hyperparameters 未配置，将使用后端默认参数")
            training["hyperparameters"] = {}

    # ------------------------------------------------------------------
    # 路径与文件工具
    # ------------------------------------------------------------------
    @staticmethod
    def _to_windows_path(p: str) -> str:
        """统一使用 Windows 反斜杠路径风格。"""
        return str(Path(p).resolve()).replace("/", "\\")

    def _prepare_project_dirs(self) -> None:
        """确保数据集目录与工作目录存在。"""
        dataset_dir = Path(self.config["project"]["dataset_dir"])
        work_dir = Path(self.config["project"]["work_dir"])
        if not dataset_dir.exists():
            raise AutoTrainError(f"数据集目录不存在: {dataset_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("数据集目录: %s", dataset_dir)
        logger.info("工作目录: %s", work_dir)

    def _get_dva_path(self) -> Path:
        """获取/生成 .dva 文件路径。"""
        project_cfg = self.config["project"]
        if "dva_file" in project_cfg and project_cfg["dva_file"]:
            return Path(os.path.expandvars(project_cfg["dva_file"])).resolve()
        dataset_dir = Path(project_cfg["dataset_dir"]).resolve()
        return dataset_dir.parent / f"{project_cfg['name']}.dva"

    # ------------------------------------------------------------------
    # 数据集类别自动检测
    # ------------------------------------------------------------------
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

    def _is_image_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self._IMAGE_EXTS

    def _find_train_root(self, dataset_dir: Path) -> Path:
        children = [p for p in dataset_dir.iterdir() if p.is_dir()]
        if not children:
            return dataset_dir
        split_names = {"train", "val", "valid", "validation", "test"}
        if all(p.name.lower() in split_names for p in children) and len(children) <= 4:
            train_dir = dataset_dir / "train"
            if train_dir.exists():
                return train_dir
        return dataset_dir

    def _auto_detect_categories(self, dataset_dir: Path, task_type: str) -> Tuple[Dict[str, int], Dict[str, str]]:
        root = self._find_train_root(dataset_dir)
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if not subdirs:
            return {}, {}

        category_dist: Dict[str, int] = {}
        category_map: Dict[str, str] = {}
        for subdir in subdirs:
            if subdir.name.lower() in {"annotations", "labels", "image", "images", "ok", "ng"}:
                continue
            image_count = sum(1 for f in subdir.iterdir() if self._is_image_file(f))
            if image_count > 0:
                category_dist[subdir.name] = image_count
                category_map[subdir.name] = subdir.name
        return category_dist, category_map

    def _resolve_dataset_categories(self, dataset_cfg: Dict[str, Any], dataset_dir: Path, task_type: str) -> Tuple[Dict[str, int], Dict[str, str]]:
        explicit_dist = dataset_cfg.get("category_dist")
        explicit_map = dataset_cfg.get("category_map")
        auto_detect = dataset_cfg.get("auto_detect", True)

        if (explicit_dist and explicit_map) and not auto_detect:
            return explicit_dist, explicit_map
        if explicit_dist and explicit_map:
            return explicit_dist, explicit_map
        if auto_detect:
            return self._auto_detect_categories(dataset_dir, task_type)
        return {}, {}

    # ------------------------------------------------------------------
    # DVA 文件生成/更新
    # ------------------------------------------------------------------
    def generate_dva(self) -> Path:
        """根据配置生成或更新 .dva 项目文件。"""
        dva_path = self._get_dva_path()
        project_cfg = self.config["project"]
        training_cfg = self.config["training"]
        dataset_cfg = self.config.get("dataset", {})

        existing: Dict[str, Any] = {}
        if dva_path.exists():
            try:
                with open(dva_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
                logger.info("已存在 .dva 文件，将基于现有文件更新: %s", dva_path)
            except Exception as e:
                logger.warning("读取现有 .dva 文件失败: %s，将重新生成", e)
                existing = {}

        if self.force_new or self.dry_run:
            project_id = str(uuid.uuid1())
            if self.force_new:
                logger.info("--force-new: 生成新的 project_id: %s", project_id)
        else:
            project_id = existing.get("project_id") or str(uuid.uuid1())

        self.project_id = project_id

        dataset_dir = Path(project_cfg["dataset_dir"]).resolve()
        category_dist, category_map = self._resolve_dataset_categories(
            dataset_cfg, dataset_dir, project_cfg["task_type"]
        )

        dva_data = {
            "project_id": project_id,
            "project_name": project_cfg["name"],
            "task_type": project_cfg["task_type"],
            "dataset_dir": self._to_windows_path(project_cfg["dataset_dir"]),
            "work_dir": self._to_windows_path(project_cfg["work_dir"]),
            "last_time_dva_path": self._to_windows_path(str(dva_path)),
            "train_platform_param": training_cfg.get("hyperparameters", {}),
            "test_platform_param": existing.get("test_platform_param", {}),
            "category_dist": category_dist,
            "category_map": category_map,
            "proj_desc": project_cfg.get("description", existing.get("proj_desc", "")),
        }

        dva_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dva_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dva_data, f, allow_unicode=True, sort_keys=False)
        logger.info("已生成/更新 .dva 文件: %s", dva_path)
        return dva_path

    # ------------------------------------------------------------------
    # 配置标准化
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """标准化 config：处理 Infinity/NaN，统一超参命名。"""
        def _walk(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_walk(v) for v in obj]
            if isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return None
            return obj

        cfg = _walk(cfg)
        hp = cfg.get("training", {}).get("hyperparameters", {})
        if hp and isinstance(hp, dict):
            if "train_ratio" in hp and "train_set_val" not in hp:
                hp["train_set_val"] = int(round(float(hp.pop("train_ratio")) * 100))
            if "val_ratio" in hp and "val_set_val" not in hp:
                hp["val_set_val"] = int(round(float(hp.pop("val_ratio")) * 100))
            if "img_scale_width" in hp and "img_scale_height" in hp and "img_scale" not in hp:
                hp["img_scale"] = {
                    "width": int(hp.pop("img_scale_width")),
                    "height": int(hp.pop("img_scale_height")),
                }
        return cfg

    # ------------------------------------------------------------------
    # 进程与端口管理
    # ------------------------------------------------------------------
    @staticmethod
    def _get_processes_by_port(port: int) -> List[psutil.Process]:
        procs: List[psutil.Process] = []
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port and conn.pid:
                    try:
                        procs.append(psutil.Process(conn.pid))
                    except psutil.NoSuchProcess:
                        pass
        except Exception as e:
            logger.debug("获取端口 %s 占用进程失败: %s", port, e)
        return procs

    @staticmethod
    def _get_processes_by_name(name: str) -> List[psutil.Process]:
        matched: List[psutil.Process] = []
        name_lower = name.lower()
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                proc_name = (proc.info.get("name") or "").lower()
                proc_exe = (proc.info.get("exe") or "").lower()
                if name_lower in proc_name or name_lower in proc_exe:
                    matched.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return matched

    def _get_target_processes(self) -> List[psutil.Process]:
        """返回需要清理的平台/后端/端口占用进程。注意：不杀前端 dev server。"""
        # 只清理后端端口，前端 8842 可能是独立的 node dev server，不应误杀
        backend_port = self._extract_port(self.backend_url)

        exe_path = Path(os.path.expandvars(self.config["platform"]["executable_path"])).resolve()
        platform_exe_name = exe_path.name
        backend_exe_name = self.PLATFORM_BACKEND_EXE

        targets: Dict[int, psutil.Process] = {}
        for name in (platform_exe_name, backend_exe_name):
            for proc in self._get_processes_by_name(name):
                targets[proc.pid] = proc
        if backend_port:
            for proc in self._get_processes_by_port(backend_port):
                targets[proc.pid] = proc
        return list(targets.values())

    def kill_conflicts(self) -> None:
        """清理平台、后端以及端口占用进程，确保干净启动。"""
        targets = self._get_target_processes()
        if not targets:
            logger.info("未发现冲突进程")
            return

        killed_pids: set = set()
        # 先并行 terminate 所有目标及其子进程，再统一等待，避免串行 5s 超时叠加
        all_procs: List[psutil.Process] = []
        for proc in targets:
            if proc.pid in killed_pids:
                continue
            try:
                children = proc.children(recursive=True)
                for child in children:
                    if child.pid not in killed_pids:
                        all_procs.append(child)
                all_procs.append(proc)
            except psutil.NoSuchProcess:
                killed_pids.add(proc.pid)

        # 统一发 terminate
        for p in all_procs:
            if p.pid in killed_pids:
                continue
            try:
                logger.info("终止进程: %s (PID %s)", p.name(), p.pid)
                p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                killed_pids.add(p.pid)
            except Exception as e:
                logger.debug("终止进程 %s 失败: %s", p.pid, e)

        # 统一等 1s，还活着的强制 kill
        _, alive = psutil.wait_procs(all_procs, timeout=1)
        for p in alive:
            if p.pid in killed_pids:
                continue
            try:
                logger.warning("强制结束进程: %s (PID %s)", p.name(), p.pid)
                p.kill()
                killed_pids.add(p.pid)
            except Exception:
                pass

        # 把已消失的也记为已处理
        for p in all_procs:
            try:
                if not p.is_running():
                    killed_pids.add(p.pid)
            except Exception:
                killed_pids.add(p.pid)

        if killed_pids:
            logger.info("已清理 %s 个冲突进程，等待端口释放...", len(killed_pids))
            time.sleep(1)

    @staticmethod
    def _extract_port(url: str) -> Optional[int]:
        match = re.search(r"://[^/]+:(\d+)", url)
        if match:
            return int(match.group(1))
        return None

    # ------------------------------------------------------------------
    # 启动平台
    # ------------------------------------------------------------------
    def launch_platform(self, dva_path: Path, extra_env: Optional[Dict[str, str]] = None) -> None:
        """启动 AI 训练平台桌面应用，并传入 .dva 文件。"""
        exe_path = Path(os.path.expandvars(self.config["platform"]["executable_path"])).resolve()
        if not exe_path.exists():
            raise AutoTrainError(f"可执行文件不存在: {exe_path}")

        cmd = [str(exe_path), str(dva_path)]
        cwd = exe_path.parent

        logger.info("启动 AI 训练平台: %s", exe_path)
        logger.info("传入 .dva 参数: %s", dva_path)

        log_dir = Path(self.config.get("project", {}).get("work_dir", "./")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.platform_output_log = log_dir / "platform_output.log"
        self.platform_output_log.write_text("", encoding="utf-8")  # 清空/创建
        logger.info("平台控制台输出将保存到: %s", self.platform_output_log)

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
            for k, v in extra_env.items():
                logger.info("注入环境变量: %s=%s", k, v)

        try:
            self.platform_output_fh = open(self.platform_output_log, "a", encoding="utf-8", errors="replace")
            self.platform_proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                shell=False,
                stdout=self.platform_output_fh,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
        except Exception as e:
            if hasattr(self, "platform_output_fh"):
                self.platform_output_fh.close()
            raise AutoTrainError(f"启动平台失败: {e}")
        logger.info("平台进程 PID: %s", self.platform_proc.pid)

    # ------------------------------------------------------------------
    # 服务就绪探测
    # ------------------------------------------------------------------
    def is_backend_ready(self, timeout: float = 2.0) -> bool:
        """检查后端真正可用：能返回项目列表才算初始化完成。"""
        try:
            resp = requests.get(f"{self.backend_url}/projects/get_all/", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def is_frontend_ready(self, timeout: float = 2.0) -> bool:
        try:
            resp = requests.get(self.frontend_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _log_backend_diagnosis(self) -> None:
        """打印后端进程/端口诊断信息。"""
        backend_procs = self._get_processes_by_name(self.PLATFORM_BACKEND_EXE)
        backend_port = self._extract_port(self.backend_url)
        port_procs = self._get_processes_by_port(backend_port) if backend_port else []
        logger.info("后端进程 %s 数量: %s", self.PLATFORM_BACKEND_EXE, len(backend_procs))
        for proc in backend_procs:
            logger.info("  - PID %s, status=%s", proc.pid, proc.status())
        logger.info("端口 %s 占用进程数量: %s", backend_port, len(port_procs))
        for proc in port_procs:
            logger.info("  - PID %s, name=%s", proc.pid, proc.name())

    def _dump_platform_output_tail(self, n_lines: int = 30) -> None:
        """把平台进程控制台输出的最后 N 行打印到日志，便于诊断后端启动失败。"""
        if not hasattr(self, "platform_output_log") or not self.platform_output_log.exists():
            return
        try:
            text = self.platform_output_log.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if not lines:
                return
            logger.info("===== 平台控制台输出最后 %s 行 =====", min(n_lines, len(lines)))
            for line in lines[-n_lines:]:
                logger.info("[平台输出] %s", line)
            logger.info("===== 平台控制台输出结束 =====")
        except Exception as e:
            logger.warning("读取平台控制台输出失败: %s", e)

    def wait_backend_ready(self) -> None:
        timeout = int(self.config["platform"].get("backend_ready_timeout", 300))
        interval = float(self.config["platform"].get("backend_ready_interval", 2.0))
        logger.info("等待后端服务就绪: %s (超时 %ss)", self.backend_url, timeout)
        start = time.time()
        last_log = start
        while time.time() - start < timeout:
            if self.is_backend_ready(timeout=2.0):
                logger.info("后端服务已就绪")
                return
            now = time.time()
            if now - last_log >= 10:
                logger.info("后端仍未就绪，已等待 %.0fs...", now - start)
                self._log_backend_diagnosis()
                last_log = now
            time.sleep(interval)
        self._log_backend_diagnosis()
        self._dump_platform_output_tail()
        raise AutoTrainError(f"后端服务在 {timeout}s 内未就绪: {self.backend_url}")

    def is_cdp_ready(self, cdp_port: int, timeout: float = 2.0) -> bool:
        """检查 WebView2 CDP 调试端口是否就绪。"""
        import urllib.request
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{cdp_port}/json/version", timeout=timeout
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def wait_cdp_ready(self, cdp_port: int) -> None:
        """轮询等待 WebView2 CDP 端口就绪。"""
        cfg = self.config.get("webbridge", {})
        timeout = int(cfg.get("frontend_ready_timeout", 60))
        interval = float(cfg.get("frontend_ready_interval", 1.0))
        logger.info("等待 WebView2 CDP 端口就绪: 127.0.0.1:%s (超时 %ss)", cdp_port, timeout)
        start = time.time()
        while time.time() - start < timeout:
            if self.is_cdp_ready(cdp_port, timeout=2.0):
                logger.info("CDP 端口已就绪")
                return
            time.sleep(interval)
        raise AutoTrainError(f"WebView2 CDP 端口在 {timeout}s 内未就绪: 127.0.0.1:{cdp_port}")

    def wait_frontend_ready(self) -> None:
        """轮询等待前端 dev server 或 WebView2 CDP 端口就绪。"""
        cfg = self.config.get("webbridge", {})
        adapter = cfg.get("adapter", "playwright").lower()
        if adapter == "webview2_cdp":
            cdp_port = int(cfg.get("cdp_port", 9222))
            self.wait_cdp_ready(cdp_port)
            return

        timeout = int(cfg.get("frontend_ready_timeout", 60))
        interval = float(cfg.get("frontend_ready_interval", 1.0))
        logger.info("等待前端服务就绪: %s (超时 %ss)", self.frontend_url, timeout)
        start = time.time()
        while time.time() - start < timeout:
            if self.is_frontend_ready(timeout=2.0):
                logger.info("前端服务已就绪")
                return
            time.sleep(interval)
        raise AutoTrainError(f"前端服务在 {timeout}s 内未就绪: {self.frontend_url}")

    # ------------------------------------------------------------------
    # 后端 API 封装
    # ------------------------------------------------------------------
    def _api_get(self, path: str, params: Optional[Dict] = None, timeout: int = 30, retries: int = 3) -> Dict:
        url = f"{self.backend_url}{path}"
        last_err = None
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                if attempt < retries - 1:
                    logger.warning("API GET 失败，%ss 后重试 (%d/%d): %s", 2 ** attempt, attempt + 1, retries, e)
                    time.sleep(2 ** attempt)
        raise AutoTrainError(f"API 请求失败 [{url}]: {last_err}")

    def _api_post(self, path: str, params: Optional[Dict] = None, json_data: Any = None, timeout: int = 30, retries: int = 3) -> Dict:
        url = f"{self.backend_url}{path}"
        last_err = None
        for attempt in range(retries):
            try:
                resp = requests.post(url, params=params, json=json_data, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                if attempt < retries - 1:
                    logger.warning("API POST 失败，%ss 后重试 (%d/%d): %s", 2 ** attempt, attempt + 1, retries, e)
                    time.sleep(2 ** attempt)
        raise AutoTrainError(f"API 请求失败 [{url}]: {last_err}")

    def _api_put(self, path: str, json_data: Any, timeout: int = 30) -> Dict:
        url = f"{self.backend_url}{path}"
        try:
            resp = requests.put(url, json=json_data, headers={"Content-Type": "application/json"}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise AutoTrainError(f"API 请求失败 [{url}]: {e}")

    def ensure_project(self, dva_path: Path) -> str:
        """确保项目已创建并返回 project_id。"""
        project_cfg = self.config["project"]
        with open(dva_path, "r", encoding="utf-8") as f:
            dva_data = yaml.safe_load(f) or {}
        existing_project_id = dva_data.get("project_id")

        if self.force_new:
            existing_project_id = None
            logger.info("--force-new: 忽略已有 project_id，准备创建新项目")

        if existing_project_id:
            try:
                info = self._api_get("/db/project/", {"project_id": existing_project_id})
                if info.get("project_id") == existing_project_id:
                    logger.info("项目已存在: %s (%s)", info.get("project_name"), existing_project_id)
                    self.project_id = existing_project_id
                    return self.project_id
            except AutoTrainError:
                pass

        params = {
            "project_name": project_cfg["name"],
            "task_type": project_cfg["task_type"],
            "dataset_path": self._to_windows_path(project_cfg["dataset_dir"]),
            "work_dir": self._to_windows_path(project_cfg["work_dir"]),
        }
        payload = {"project_id": existing_project_id} if existing_project_id else {}
        logger.info("创建项目: %s", project_cfg["name"])
        result = self._api_post("/projects/create/", params=params, json_data=payload)
        if result.get("code") != 0:
            raise AutoTrainError(f"创建项目失败: {result.get('message')}")

        self.project_id = result.get("project_id")
        logger.info("项目创建成功，project_id: %s", self.project_id)

        dva_data["project_id"] = self.project_id
        dva_data["last_time_dva_path"] = self._to_windows_path(str(dva_path))
        with open(dva_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dva_data, f, allow_unicode=True, sort_keys=False)
        return self.project_id

    def update_hyperparameters(self) -> None:
        """将配置中的超参数写入后端。"""
        if not self.project_id:
            raise AutoTrainError("project_id 为空，无法更新超参数")
        hyper_params = self.config["training"].get("hyperparameters", {})
        if not hyper_params:
            logger.info("未配置超参数，跳过写入")
            return
        logger.info("更新超参数: %s", hyper_params)
        result = self._api_put(f"/db/projects/{self.project_id}", {"hyperparameters": hyper_params})
        if result.get("code") != 0:
            raise AutoTrainError(f"更新超参数失败: {result.get('message')}")
        logger.info("超参数更新成功")

    def start_training_api(self) -> str:
        """直接调用后端接口开始训练。"""
        if not self.project_id:
            raise AutoTrainError("project_id 为空，无法开始训练")
        if self.training_id:
            logger.warning(
                "已存在 training_id=%s，跳过重复 API 启动，避免二次拉起训练",
                self.training_id,
            )
            return self.training_id
        if self.training_launch_attempted:
            raise AutoTrainError(
                "本次运行已尝试启动过训练且未拿到可用 training_id，拒绝再次 API 启动，"
                "请检查后端 session / platform_output.log"
            )
        hyper_params = self.config["training"].get("hyperparameters", {})
        logger.info("通过后端 API 开始训练项目: %s", self.project_id)
        self.training_launch_attempted = True
        result = self._api_post("/training/start/", params={"project_id": self.project_id}, json_data=hyper_params)
        if result.get("code") != 0:
            raise AutoTrainError(f"开始训练失败: {result.get('message')}")
        self.training_id = result.get("training_id")
        if result.get("queued"):
            logger.info("训练已加入排队，位置: %s, training_id: %s", result.get("queue_position"), self.training_id)
        else:
            logger.info("训练已启动，training_id: %s", self.training_id)
        return self.training_id

    def get_project_training_history(self, project_id: str) -> List[Dict[str, Any]]:
        """获取指定项目的所有训练会话历史，按创建时间倒序排列。"""
        try:
            resp = self._api_get("/training/get_project_training_history/", {"project_id": project_id})
            if isinstance(resp, list):
                sessions = sorted(
                    resp,
                    key=lambda s: s.get("create_time") or 0,
                    reverse=True,
                )
                return sessions
        except Exception as e:
            logger.warning("获取项目训练历史失败: %s", e)
        return []

    def discover_training_id(self, project_id: Optional[str] = None) -> Optional[str]:
        """发现当前/最新的 training_id。优先从项目历史取最新一条，fallback 到状态接口。"""
        pid = project_id or self.project_id
        if pid:
            try:
                sessions = self.get_project_training_history(pid)
                if sessions:
                    tid = sessions[0].get("training_id")
                    if tid:
                        logger.info("从项目历史发现最新 training_id: %s (status=%s)",
                                    tid, sessions[0].get("status"))
                        return tid
            except Exception as e:
                logger.warning("从项目历史发现 training_id 失败: %s", e)

        try:
            status = self._api_get("/training/status/")
            tid = status.get("training_id")
            if tid:
                logger.info("从后端状态接口发现 training_id: %s", tid)
                return tid
        except Exception as e:
            logger.warning("从状态接口发现 training_id 失败: %s", e)
        return None

    def _wait_for_new_training_id(
        self,
        previous_id: Optional[str],
        timeout: int = 60,
        interval: float = 2.0,
    ) -> Optional[str]:
        """轮询等待出现一条与 previous_id 不同的最新 training_id。"""
        logger.info("等待新的 training_id 出现 (previous=%s, timeout=%ss)", previous_id or "无", timeout)
        start = time.time()
        while time.time() - start < timeout:
            tid = self.discover_training_id(self.project_id)
            if tid and tid != previous_id:
                logger.info("发现新的 training_id: %s", tid)
                return tid
            time.sleep(interval)
        logger.warning("超时未检测到新的 training_id")
        return None

    def _fetch_session(self, training_id: str) -> Optional[Dict[str, Any]]:
        """拉取完整 training session 记录。"""
        try:
            session = self._api_get(f"/db/training_sessions/{training_id}")
            if isinstance(session, dict):
                return session
        except Exception as e:
            logger.warning("获取 training session 失败 (%s): %s", training_id, e)
        return None

    def _fetch_session_status(self, training_id: str) -> Optional[str]:
        session = self._fetch_session(training_id)
        if not session:
            return None
        return session.get("status")

    def _extract_session_error_text(self, session: Optional[Dict[str, Any]]) -> str:
        """从 session 字典中尽量抽出可读错误信息。"""
        if not session:
            return ""
        chunks: List[str] = []
        for key in (
            "error",
            "error_message",
            "err_msg",
            "message",
            "fail_reason",
            "failure_reason",
            "exception",
            "traceback",
            "stack",
            "detail",
            "details",
            "log",
            "logs",
            "remark",
            "note",
        ):
            val = session.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (dict, list)):
                try:
                    text = json.dumps(val, ensure_ascii=False)
                except Exception:
                    text = str(val)
            else:
                text = str(val).strip()
            if text and text not in chunks:
                chunks.append(text)
        return " | ".join(chunks)

    def _work_dir(self) -> Optional[Path]:
        raw = (self.config.get("project") or {}).get("work_dir") or (
            self.config.get("project") or {}
        ).get("run_dir")
        if not raw:
            return None
        try:
            path = Path(raw)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            return None

    def capture_training_diagnostics(
        self,
        training_id: Optional[str] = None,
        reason: str = "",
    ) -> Optional[Path]:
        """把后端 session / 全局 status / 平台输出尾部落到 work_dir，便于失败复盘。"""
        tid = training_id or self.training_id
        work_dir = self._work_dir()
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": reason,
            "project_id": self.project_id,
            "training_id": tid,
        }

        session = self._fetch_session(tid) if tid else None
        if session is not None:
            payload["session"] = session
            payload["session_status"] = session.get("status")
            payload["session_progress"] = session.get("progress")
            payload["session_error"] = self._extract_session_error_text(session)

        try:
            payload["training_status"] = self._api_get("/training/status/")
        except Exception as e:
            payload["training_status_error"] = str(e)

        if self.project_id:
            try:
                payload["project_history"] = self.get_project_training_history(self.project_id)[:5]
            except Exception as e:
                payload["project_history_error"] = str(e)

        # 平台控制台尾部
        if hasattr(self, "platform_output_log") and self.platform_output_log.exists():
            try:
                text = self.platform_output_log.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                payload["platform_output_tail"] = lines[-80:]
                payload["platform_output_log"] = str(self.platform_output_log)
            except Exception as e:
                payload["platform_output_error"] = str(e)

        self.training_failure_detail = payload
        err_text = payload.get("session_error") or ""
        if err_text:
            logger.error("训练引擎错误摘要: %s", err_text[:2000])
        elif payload.get("session_status"):
            logger.error(
                "训练 session 状态=%s progress=%s（后端未返回明确 error 字段）",
                payload.get("session_status"),
                payload.get("session_progress"),
            )

        if work_dir is None:
            logger.warning("无 work_dir，训练诊断信息仅保留在内存/日志中")
            return None

        out_path = work_dir / "training_failure.json"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            logger.info("训练失败诊断已写入: %s", out_path)
        except Exception as e:
            logger.warning("写入 training_failure.json 失败: %s", e)
            return None

        # 同步追加一份人类可读摘要到 train 侧日志
        self._dump_platform_output_tail(n_lines=50)
        return out_path

    def _raise_training_failed(self, status: str, training_id: Optional[str] = None) -> None:
        tid = training_id or self.training_id
        self.capture_training_diagnostics(tid, reason=f"terminal_status={status}")
        detail = ""
        if self.training_failure_detail:
            detail = self.training_failure_detail.get("session_error") or ""
        msg = f"训练未成功完成，状态: {status}"
        if tid:
            msg += f", training_id={tid}"
        if detail:
            msg += f", error={detail[:500]}"
        raise AutoTrainError(msg)

    TERMINAL_STATUSES = {"已完成", "训练失败", "训练中止", "未知错误"}
    STARTED_STATUSES = {"已启动", "进行中"}

    def monitor_training(self) -> None:
        """轮询训练状态直到完成/失败/中止。"""
        if not self.training_id:
            logger.warning("没有 training_id，跳过监控")
            return

        raw_timeout = int(self.config["training"].get("monitor_timeout", 0))
        timeout = raw_timeout if raw_timeout > 0 else 3600
        interval = float(self.config["training"].get("monitor_interval", 10.0))
        logger.info("开始监控训练状态 (training_id=%s, timeout=%ss)", self.training_id, timeout)
        start = time.time()
        has_started = False
        missing_session_count = 0
        MAX_MISSING_SESSION = 6

        while True:
            self._keep_browser_active()
            session_status = self._fetch_session_status(self.training_id)

            if session_status in self.TERMINAL_STATUSES:
                logger.info("训练结束，最终状态: %s", session_status)
                if session_status != "已完成":
                    self._raise_training_failed(session_status or "未知")
                return

            if session_status in self.STARTED_STATUSES:
                has_started = True
                missing_session_count = 0

            status = self._api_get("/training/status/")
            current_training_id = status.get("training_id")

            if current_training_id == self.training_id:
                has_started = True
                missing_session_count = 0

            if not session_status:
                missing_session_count += 1
                logger.warning("未查询到 training_id=%s 的 session (第 %s 次)",
                               self.training_id, missing_session_count)
            elif current_training_id != self.training_id:
                logger.info("训练 %s 尚未开始，当前运行: %s, 队列长度: %s",
                            self.training_id, current_training_id, status.get("queue_size", 0))
            else:
                progress_info = ""
                try:
                    session = self._fetch_session(self.training_id) or {}
                    progress = session.get("progress")
                    if progress is not None:
                        progress_info = f", 进度: {progress:.1%}, 状态: {session_status}"
                    else:
                        progress_info = f", 状态: {session_status}"
                except Exception:
                    progress_info = f", 状态: {session_status}"
                logger.info("训练状态: %s%s", status.get("message", ""), progress_info)

            # 已经启动过，但现在没有运行中的训练任务 -> 训练结束/失败/中止
            if has_started and not status.get("project_id") and not current_training_id:
                final_status = self._fetch_session_status(self.training_id)
                logger.info("没有运行中的训练任务，最终状态: %s", final_status or "未知")
                if final_status == "排队中":
                    logger.info("训练仍在队列中，将继续后台运行")
                    return
                if final_status != "已完成":
                    self._raise_training_failed(final_status or "未知")
                return

            # session 连续多次查不到，且当前也没有运行中的任务 -> 大概率 ID 错了或已被清理
            if missing_session_count >= MAX_MISSING_SESSION and not current_training_id:
                final_status = self._fetch_session_status(self.training_id)
                if final_status == "排队中":
                    logger.info("训练仍在队列中，将继续后台运行")
                    return
                self.capture_training_diagnostics(
                    self.training_id,
                    reason="missing_session",
                )
                raise AutoTrainError(
                    f"连续 {MAX_MISSING_SESSION} 次未查询到 training_id={self.training_id}，"
                    "可能拿到的不是本次训练 ID 或训练已被后端清理"
                )

            if time.time() - start > timeout:
                final_status = self._fetch_session_status(self.training_id)
                if final_status == "排队中":
                    logger.info("监控超时但训练仍在队列中，将继续后台运行")
                    return
                self.capture_training_diagnostics(self.training_id, reason="monitor_timeout")
                raise AutoTrainError(f"监控超时 ({timeout}s)")
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Kimi WebBridge 自动化
    # ------------------------------------------------------------------
    def _build_training_task_prompt(self, hp: Dict[str, Any]) -> str:
        """把超参配置转换成给 Kimi Agent 的自然语言任务描述。"""
        project_name = self.config["project"]["name"]
        task_type = self.config["project"]["task_type"]

        lines = [
            f"请在当前 DLCV AI 训练平台的「训练参数」页面完成以下操作（项目：{project_name}，任务类型：{task_type}）：",
            "",
            "1. 如果页面还在加载，请等待加载完成。",
            "2. 展开所有折叠面板（数据参数、训练参数、评估参数等），确保所有字段可见。",
            "3. 设置每个字段前，先检查当前值是否与目标值一致；如果已经一致，则不要修改，避免触发不必要的保存。",
        ]

        idx = 4
        field_desc = {
            "train_set_val": ("训练集比例", "%"),
            "val_set_val": ("验证集比例", "%"),
            "copy_times": ("数据倍增", ""),
            "max_epochs": ("训练代数", ""),
            "batch_size": ("批量大小", ""),
            "lr": ("学习率", ""),
            "weight_decay": ("权重衰减", ""),
            "val_interval": ("评估间隔", ""),
            "distill_ratio": ("蒸馏比例", ""),
            "mask_factor": ("Mask放大系数", ""),
            "random_seed": ("随机种子", ""),
            "train_num_workers": ("训练数据读取线程数", ""),
            "val_num_workers": ("验证数据读取线程数", ""),
        }

        for key, (label, unit) in field_desc.items():
            if key in hp:
                value = hp[key]
                unit_str = f" {unit}" if unit else ""
                lines.append(f"{idx}. 设置「{label}」为 {value}{unit_str}。")
                idx += 1

        bool_desc = {
            "enable_fp16": "启用AMP混合精度",
            "save_error_data": "保存过漏检数据",
            "keep_consistency": "保持一致性",
            "auto_deploy": "自动加速模型(.dvt)",
            "use_ok_images": "使用OK图训练",
            "use_thread_dataloader": "使用线程读取数据",
        }
        for key, label in bool_desc.items():
            if key in hp:
                action = "勾选" if hp[key] else "取消勾选"
                lines.append(f"{idx}. {action}「{label}」。")
                idx += 1

        if "img_scale" in hp:
            img = hp["img_scale"]
            lines.append(f"{idx}. 设置图片缩放尺寸为宽 {img.get('width')}，高 {img.get('height')}。")
            idx += 1

        lines.extend([
            "",
            f"{idx}. 点击「保存」按钮。",
            f"{idx + 1}. 点击「开始训练」按钮。",
            f"{idx + 2}. 如果出现确认弹窗，请点击「确定」。",
            f"{idx + 3}. 点击完成后，关闭页面上出现的「保存成功」「开始训练」等成功提示/Toast 弹窗。",
            "",
            "注意：",
            "- 请优先通过输入框、滑块、复选框等标准控件操作。",
            "- 设置完成后请确认「开始训练」按钮从 loading 状态恢复或页面跳转。",
            "- 如果某个字段当前值已经与目标值相同，请不要点击/修改它，避免重复触发保存。",
        ])
        return "\n".join(lines)

    def _run_webbridge_in_thread(self) -> None:
        """在独立线程中运行 Playwright 自动化（包含录屏），实现线程隔离。

        Playwright 的启动/操作/关闭必须在同一线程完成。
        训练完成后的「关闭浏览器」等待也放在该线程内，用 stdin + 信号文件双通道，
        避免 Windows 嵌套子进程下 input() 收不到 PyQt 管道回车、UI 卡在「运行中」。
        """
        import threading

        exception = None

        def target():
            nonlocal exception
            try:
                self.run_webbridge_automation()
            except Exception as e:
                exception = e

        thread = threading.Thread(target=target, name="PlaywrightWebBridgeWorker")
        thread.start()
        thread.join()

        if exception:
            raise exception

    def _wait_for_browser_close_signal(self) -> None:
        """等待用户关闭浏览器：stdin 回车 或 关闭信号文件。

        PyQt 前端会同时写 stdin 回车 + 创建信号文件，双通道保证嵌套子进程
        在 Windows 下也能可靠退出，从而让前端状态从「运行中」恢复。
        """
        import threading

        signal_path = os.environ.get("AUTOTRAIN_CLOSE_SIGNAL", "").strip()
        signal_file = Path(signal_path) if signal_path else None

        # 若用户在进入等待前已点击「关闭浏览器」，信号文件已存在，直接返回
        if signal_file and signal_file.exists():
            logger.info("检测到已有关闭信号文件，直接关闭浏览器")
            return

        done = threading.Event()

        def _watch_stdin() -> None:
            try:
                # 放到线程里读 stdin，以便同时轮询信号文件
                sys.stdin.readline()
            except Exception:
                pass
            finally:
                done.set()

        def _watch_signal_file() -> None:
            if not signal_file:
                return
            while not done.is_set():
                try:
                    if signal_file.exists():
                        done.set()
                        return
                except Exception:
                    pass
                time.sleep(0.2)

        t_stdin = threading.Thread(target=_watch_stdin, name="WaitBrowserCloseStdin", daemon=True)
        t_file = threading.Thread(target=_watch_signal_file, name="WaitBrowserCloseFile", daemon=True)
        t_stdin.start()
        t_file.start()
        done.wait()
        logger.info("收到关闭浏览器信号，继续收尾...")

    def _build_webbridge_config(self) -> dict:
        """构造传给 WebBridge 适配器的配置（含 platform，供壳端录屏定位窗口）。"""
        wb_cfg = dict(self.config.get("webbridge", {}) or {})
        platform_cfg = self.config.get("platform") or {}
        if platform_cfg:
            wb_cfg["platform"] = dict(platform_cfg)
        return wb_cfg

    def _start_shell_recording_early(self) -> None:
        """
        在 ensure_project / CDP connect 之前启动壳端窗口录屏，
        确保从项目信息页出现起就有画面，而不是等到 connect 之后。
        """
        wb_cfg = self._build_webbridge_config()
        if not wb_cfg.get("enabled", True):
            return
        if not wb_cfg.get("record_video_dir"):
            return
        adapter = str(wb_cfg.get("adapter", "playwright")).lower()
        if adapter != "webview2_cdp":
            return
        if getattr(self, "_bridge", None) is not None:
            return
        try:
            bridge = create_webbridge(wb_cfg)
            if hasattr(bridge, "start_recording"):
                ok = bridge.start_recording(timeout=25.0)
                if ok:
                    logger.info("已提前启动壳端录屏（覆盖项目信息页之前）: %s", getattr(bridge, "video_path", None))
                else:
                    logger.warning("提前启动壳端录屏未成功，将在 connect 时重试")
            self._bridge = bridge
        except Exception as e:
            logger.warning("提前启动壳端录屏失败，将在 connect 时重试: %s", e)
            self._bridge = None

    def run_webbridge_automation(self) -> None:
        """通过 Kimi WebBridge 完成 UI 自动化。"""
        if not self.project_id:
            raise AutoTrainError("project_id 为空，无法执行 WebBridge 自动化")

        wb_cfg = self._build_webbridge_config()
        if not wb_cfg.get("enabled", True):
            logger.info("webbridge.enabled=false，跳过 WebBridge 自动化")
            return

        project_url = f"{self.frontend_url}/#/proj_panel/{self.project_id}"
        param_url = f"{self.frontend_url}/#/proj_panel/{self.project_id}/parameter_edit"
        hp = self.config["training"].get("hyperparameters", {})
        task_prompt = self._build_training_task_prompt(hp)
        ui_flow = self.config.get("ui_flow", {})
        monitor = self.config["training"].get("monitor_until_complete", True)

        bridge = None

        try:
            # 复用提前启动录屏时创建的 bridge（已在录制）；否则新建
            bridge = getattr(self, "_bridge", None)
            if bridge is None:
                bridge = create_webbridge(wb_cfg)
                # 在 CDP attach 前再尝试一次录屏，保证从项目信息页开始就有画面
                if hasattr(bridge, "start_recording"):
                    try:
                        bridge.start_recording(timeout=20.0)
                    except Exception as e:
                        logger.warning("connect 前启动录屏失败: %s", e)

            bridge.connect()
            self._bridge = bridge
            logger.info("连接 Kimi WebBridge (adapter=%s)", wb_cfg.get("adapter", "playwright"))

            # webview2_cdp 模式下，壳端自带前端服务器（如 localhost:13222），
            # 不要再往 localhost:8842 跳，否则会把壳端前端顶掉、后端随之断开。
            already_on_project_page = False
            if wb_cfg.get("adapter", "playwright").lower() == "webview2_cdp":
                page = getattr(bridge, "_page", None)
                if page:
                    actual_url = page.url or ""
                    parsed = urlparse(actual_url)
                    actual_base = f"{parsed.scheme}://{parsed.netloc}/#/"
                    if actual_base != self.frontend_url:
                        logger.info("检测到壳端实际前端 URL: %s，替换配置的 frontend_url", actual_base)
                        self.frontend_url = actual_base
                        project_url = f"{self.frontend_url}proj_panel/{self.project_id}"
                        param_url = f"{self.frontend_url}proj_panel/{self.project_id}/parameter_edit"
                    # 仅当已在「项目信息/编辑」页时才跳过导航。
                    # train_detail / parameter_edit 等子页没有侧边栏，必须回到项目页。
                    path = (parsed.fragment or "").lower()
                    on_subpage = any(
                        token in path
                        for token in (
                            "train_detail",
                            "parameter_edit",
                            "data_analysis",
                            "data_augmentation",
                            "augmentation",
                        )
                    )
                    if (
                        self.project_id
                        and self.project_id in actual_url
                        and not on_subpage
                    ):
                        logger.info("壳端已处于项目页面: %s，跳过导航", actual_url)
                        already_on_project_page = True
                    elif on_subpage and self.project_id in actual_url:
                        logger.info(
                            "壳端当前在子页面 (%s)，将导航回项目信息页",
                            actual_url,
                        )

            # 如果 Playwright 适配器支持新流程，从项目信息页按菜单一步步走
            if hasattr(bridge, "run_training_workflow"):
                if not already_on_project_page:
                    logger.info("导航至项目信息页: %s", project_url)
                    bridge.navigate(project_url)

                # 记录点击「开始训练」前的最新 training_id，便于之后识别本次新建的 session
                previous_id = self.discover_training_id(self.project_id)
                logger.info("点击开始训练前最新 training_id: %s", previous_id or "无")

                result = bridge.run_training_workflow(ui_flow, hp)
            else:
                # 兜底：旧版直接跳转到参数页
                if not already_on_project_page:
                    logger.info("导航至训练参数页: %s", param_url)
                    bridge.navigate(param_url)

                previous_id = self.discover_training_id(self.project_id)
                logger.info("点击开始训练前最新 training_id: %s", previous_id or "无")

                if hasattr(bridge, "apply_hyperparameters"):
                    result = bridge.apply_hyperparameters(hp)
                else:
                    result = bridge.execute_task(task_prompt)

            if not result.success:
                raise AutoTrainError(f"WebBridge 任务执行失败: {result.message}")

            logger.info("WebBridge 任务执行结果: %s", result.message)

            # 等待并识别本次新创建的 training_id
            # WebBridge 点击后可能异步建 session：先等 new id，再查是否已有运行中/历史 id
            self.training_id = self._wait_for_new_training_id(previous_id, timeout=60)
            if not self.training_id:
                logger.warning("未能在 60s 内发现新的 training_id，尝试兜底查询")
                self.training_id = self.discover_training_id(self.project_id)

            if self.training_id:
                # UI 路径已观察到 training session，禁止外层再 API start
                self.training_launch_attempted = True
                logger.info("WebBridge 路径已关联 training_id=%s", self.training_id)
            else:
                # 仅当完全没有 session 时才允许 API 兜底启动一次
                logger.warning("WebBridge 未返回 training_id，尝试使用后端 API 兜底启动训练")
                self.start_training_api()

            if monitor:
                self.monitor_training()
        except WebBridgeError as e:
            logger.error("Kimi WebBridge 调用失败: %s", e)
            self._save_error_screenshot(bridge, "webbridge_error")
            raise AutoTrainError(f"WebBridge 调用失败: {e}")
        except AutoTrainError:
            # 训练失败类错误：尽量落盘诊断后再抛
            if self.training_id and not self.training_failure_detail:
                try:
                    self.capture_training_diagnostics(self.training_id, reason="webbridge_autotrain_error")
                except Exception:
                    pass
            self._save_error_screenshot(bridge, "automation_error")
            raise
        except Exception as e:
            logger.exception("WebBridge 自动化发生未预期错误: %s", e)
            if self.training_id:
                try:
                    self.capture_training_diagnostics(self.training_id, reason="webbridge_unexpected_error")
                except Exception:
                    pass
            self._save_error_screenshot(bridge, "automation_error")
            raise
        finally:
            if bridge is not None:
                try:
                    bridge.close()
                except Exception as e:
                    logger.warning("关闭浏览器时出错: %s", e)
                try:
                    self._rename_playwright_video(bridge)
                except Exception as e:
                    logger.warning("保存录屏时出错: %s", e)
                self._bridge = None

    def _rename_playwright_video(self, bridge) -> None:
        """将 Playwright 生成的 <hash>.webm 重命名为 run_dir/screen_recording.mp4。"""
        video_path = getattr(bridge, "video_path", None)
        if not video_path:
            return
        src = Path(video_path)
        if not src.exists():
            return
        wb_cfg = self.config.get("webbridge", {})
        record_dir = wb_cfg.get("record_video_dir")
        if not record_dir:
            return
        try:
            target_dir = Path(record_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            # 统一命名为 screen_recording.mp4，方便 run_dir 归档
            dst = target_dir / "screen_recording.mp4"
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            logger.info("录屏文件已保存: %s", dst)
        except Exception as e:
            logger.warning("重命名 Playwright 录屏文件失败: %s", e)

    def _save_error_screenshot(self, bridge, name: str = "error") -> None:
        """异常发生时自动截取当前页面，保存到 screenshot_dir。"""
        if bridge is None:
            return
        if not hasattr(bridge, "screenshot"):
            return
        try:
            result = bridge.screenshot(name)
            if result.success and result.screenshot_path:
                logger.info("错误现场截图已保存: %s", result.screenshot_path)
        except Exception as e:
            logger.warning("保存错误现场截图失败: %s", e)

    def _keep_browser_active(self) -> None:
        """定期调用，防止 Chromium 因窗口不在前台而节流渲染/定时器，导致录屏画面卡住。"""
        try:
            bridge = getattr(self, "_bridge", None)
            if not bridge:
                return
            page = getattr(bridge, "_page", None)
            if not page or page.is_closed():
                return
            page.bring_to_front()
            # 触发一次轻量 JS 执行，让浏览器认为页面仍在活跃交互
            page.evaluate("() => document.title")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self) -> None:
        """执行完整自动化流程。"""
        self.load_config()
        self._prepare_project_dirs()

        self.config = self.normalize_config(self.config)
        dva_path = self.generate_dva()

        if self.dry_run:
            logger.info("--dry-run: 已生成 .dva 文件，跳过启动平台与训练")
            logger.info("--dry-run: .dva 路径: %s", dva_path)
            logger.info("--dry-run: project_id: %s", self.project_id)
            return

        wb_cfg = self.config.get("webbridge", {})
        wb_enabled = wb_cfg.get("enabled", True)
        adapter = wb_cfg.get("adapter", "playwright").lower()
        is_webview2 = adapter == "webview2_cdp"

        # 壳端模式：准备干净的 WebView2 用户数据目录，避免缓存/状态导致白屏
        webview2_user_data = None
        if is_webview2:
            work_dir = Path(self.config.get("project", {}).get("work_dir", "./")).resolve()
            webview2_user_data = work_dir / "webview2_user_data"
            webview2_user_data.mkdir(parents=True, exist_ok=True)

        logger.info("========== AUTO TRAIN 2.0 (WebBridge) START ==========")
        logger.info("WebBridge adapter: %s", adapter)

        # 进程管理
        backend_already_ready = self.is_backend_ready()
        cdp_already_ready = False
        if is_webview2:
            cdp_port = int(wb_cfg.get("cdp_port", 9222))
            cdp_already_ready = self.is_cdp_ready(cdp_port, timeout=2.0)

        if self.reuse and backend_already_ready and (not is_webview2 or cdp_already_ready):
            if is_webview2:
                logger.info("--reuse: 检测到后端与 CDP 端口均已运行，将复用现有实例")
            else:
                logger.info("--reuse: 检测到后端已运行，将复用现有实例")
            self.platform_proc = None
        else:
            if self.reuse:
                if backend_already_ready and is_webview2 and not cdp_already_ready:
                    logger.info("--reuse: 后端已运行但 CDP 端口未就绪，执行干净重启")
                else:
                    logger.info("--reuse: 后端未运行，执行干净启动")
            self.kill_conflicts()
            extra_env = None
            if is_webview2:
                extra_env = {
                    "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS": (
                        f"--remote-debugging-port={cdp_port} "
                        "--remote-allow-origins=* "
                        "--disable-gpu "
                        "--disable-software-rasterizer "
                        "--disable-features=CalculateNativeWinOcclusion "
                        "--disable-backgrounding-occluded-windows "
                        "--disable-renderer-backgrounding "
                        "--disable-background-timer-throttling"
                    ),
                }
                if webview2_user_data:
                    extra_env["WEBVIEW2_USER_DATA_FOLDER"] = str(webview2_user_data)
            self.launch_platform(dva_path, extra_env=extra_env)
            logger.info("已启动客户端 exe")

        # 等待服务就绪
        self.wait_backend_ready()
        logger.info("backend ready")

        if wb_enabled:
            self.wait_frontend_ready()
            logger.info("frontend ready")

        # webview2：在 ensure_project 之前启动壳端录屏，覆盖「项目信息页」出现过程
        if wb_enabled and is_webview2:
            try:
                self._start_shell_recording_early()
            except Exception as e:
                logger.warning("提前录屏启动异常（忽略，后续 connect 会重试）: %s", e)

        # 确保项目与超参
        self.ensure_project(dva_path)
        self.update_hyperparameters()

        # WebBridge 自动化 或 API 兜底
        # 把 Playwright 相关操作放到独立线程，避免和主线程/UI 线程竞争导致录屏失败
        training_launched = False
        if wb_enabled:
            try:
                self._run_webbridge_in_thread()
                training_launched = bool(self.training_id)
            except Exception as e:
                logger.error("WebBridge 自动化失败: %s", e)
                # 关键：已经有 training_id / 已尝试启动过，则禁止再 API 二次拉起
                if self.training_id:
                    training_launched = True
                    logger.error(
                        "WebBridge 路径已产生 training_id=%s，不再 API 兜底二次启动；改为采集诊断",
                        self.training_id,
                    )
                    try:
                        if not self.training_failure_detail:
                            self.capture_training_diagnostics(
                                self.training_id,
                                reason="webbridge_failed_after_launch",
                            )
                    except Exception:
                        pass
                    # 若错误本身已是 AutoTrainError（含训练失败），直接抛出
                    if isinstance(e, AutoTrainError):
                        raise
                    raise AutoTrainError(
                        f"训练已启动但 WebBridge/监控失败: {e}; "
                        f"training_id={self.training_id}"
                    ) from e
                if self.training_launch_attempted:
                    logger.error("本次已尝试启动训练，跳过重复 API 兜底")
                    try:
                        self.capture_training_diagnostics(reason="launch_attempted_no_retry")
                    except Exception:
                        pass
                    raise
                logger.info("尝试使用后端 API 兜底启动训练（仅一次）")
                try:
                    self.start_training_api()
                    training_launched = True
                    if self.config["training"].get("monitor_until_complete", True):
                        try:
                            self.monitor_training()
                        except AutoTrainError as me:
                            # 明确失败状态要向上抛；仅排队/超时类可降级
                            msg = str(me)
                            if "未成功完成" in msg or "训练失败" in msg or "训练中止" in msg:
                                raise
                            logger.warning("训练监控未等到最终状态: %s", me)
                except Exception as api_e:
                    logger.error("后端 API 兜底也失败: %s", api_e)
                    try:
                        self.capture_training_diagnostics(reason="api_fallback_failed")
                    except Exception:
                        pass
                    raise
        else:
            self.start_training_api()
            training_launched = True
            if self.config["training"].get("monitor_until_complete", True):
                try:
                    self.monitor_training()
                except AutoTrainError as me:
                    msg = str(me)
                    if "未成功完成" in msg or "训练失败" in msg or "训练中止" in msg:
                        raise
                    logger.warning("训练监控未等到最终状态: %s", me)

        if training_launched:
            # 注意：training_launched 只表示「训练被拉起」，不代表「训练成功完成」。
            # 若最终状态是训练失败，不能再标成成功，否则上层多任务流水线会误判。
            final_ok = True
            try:
                if self.training_id:
                    st = self._fetch_session_status(self.training_id)
                    if st and st != "已完成" and st in getattr(self, "TERMINAL_STATUSES", set()):
                        final_ok = False
                        logger.error(
                            "训练最终状态非成功: training_id=%s status=%s",
                            self.training_id,
                            st,
                        )
                        try:
                            self.capture_training_diagnostics(
                                self.training_id,
                                reason=f"final_status={st}",
                            )
                        except Exception:
                            pass
            except Exception:
                pass

            if final_ok:
                logger.info("训练已成功启动（training_id=%s），任务视为成功", self.training_id or "未知")
            else:
                logger.error(
                    "训练已结束但未成功（training_id=%s），任务视为失败",
                    self.training_id or "未知",
                )
                detail = ""
                if self.training_failure_detail:
                    detail = self.training_failure_detail.get("session_error") or ""
                msg = f"训练未成功完成，training_id={self.training_id}"
                if detail:
                    msg += f", error={detail[:500]}"
                raise AutoTrainError(msg)

        # 训练结束后若配置为保持窗口打开，统一等待关闭信号（覆盖 WebBridge 成功和 API 兜底两条路径）
        # 注意：失败路径已在上面 raise，不会走到这里的 keep_open
        keep_open = wb_cfg.get("keep_browser_open", self.keep_browser_open)
        if keep_open:
            logger.info("训练已完成，应用窗口将保持打开，点击「关闭应用」或按 Enter 关闭...")
            print("训练已完成，点击「关闭应用」或按 Enter 关闭窗口...", flush=True)
            try:
                self._wait_for_browser_close_signal()
            except (EOFError, KeyboardInterrupt):
                logger.info("收到中断，准备关闭应用")

        logger.info("========== AUTO TRAIN 2.0 (WebBridge) DONE ==========")

    def shutdown(self) -> None:
        """关闭已启动的平台进程，并确保提前启动的录屏被收尾。"""
        # WebBridge 失败走 API 兜底时，录屏 bridge 可能仍挂在 self._bridge 上，这里统一收尾
        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            try:
                if hasattr(bridge, "close"):
                    bridge.close()
            except Exception as e:
                logger.warning("shutdown 关闭 bridge 失败: %s", e)
            try:
                self._rename_playwright_video(bridge)
            except Exception as e:
                logger.warning("shutdown 保存录屏失败: %s", e)
            self._bridge = None

        if self.reuse:
            logger.info("--reuse: 不关闭复用的平台进程")
            return
        if self.platform_proc and self.platform_proc.poll() is None:
            logger.info("关闭平台进程 PID: %s", self.platform_proc.pid)
            self.platform_proc.terminate()
            try:
                self.platform_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.platform_proc.kill()
        if hasattr(self, "platform_output_fh") and self.platform_output_fh:
            try:
                self.platform_output_fh.flush()
                self.platform_output_fh.close()
            except Exception:
                pass
            self.platform_output_fh = None


def main() -> int:
    parser = argparse.ArgumentParser(description="DLCV AI 训练平台自动化脚本（Kimi WebBridge 版）")
    parser.add_argument(
        "--config",
        default="config/train_automation_config_2.0.json",
        help="配置文件路径",
    )
    parser.add_argument("--force-new", action="store_true", help="强制创建新项目")
    parser.add_argument("--dry-run", action="store_true", help="仅生成 .dva，不启动平台")
    parser.add_argument("--keep-alive", action="store_true", help="训练结束后保持平台运行")
    parser.add_argument(
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持浏览器窗口打开，按 Enter 关闭（默认开启）",
    )
    parser.add_argument("--reuse", action="store_true", help="复用已运行的平台/后端实例")
    args = parser.parse_args()

    trainer = DLCVAutoTrainer(
        args.config,
        force_new=args.force_new,
        dry_run=args.dry_run,
        reuse=args.reuse,
        keep_browser_open=args.keep_browser_open,
    )
    try:
        trainer.run()
        return 0
    except AutoTrainError as e:
        logger.error("自动化流程失败: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    finally:
        if not args.keep_alive:
            trainer.shutdown()


if __name__ == "__main__":
    sys.exit(main())
