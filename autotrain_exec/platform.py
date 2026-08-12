"""platform — part of DLCVAutoTrainer (autotrain_exec)."""
from __future__ import annotations

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

try:
    from autotrain4.constants import DEFAULT_BACKEND_URL, DEFAULT_FRONTEND_URL
    from autotrain4.dataset_fs import (
        IMAGE_EXTENSIONS as _AT4_IMAGE_EXTENSIONS,
        detect_categories as _at4_detect_categories,
        find_train_split_root as _at4_find_train_split_root,
        is_image_file as _at4_is_image_file,
    )
except ImportError:  # pragma: no cover
    DEFAULT_BACKEND_URL = "http://127.0.0.1:9887"
    DEFAULT_FRONTEND_URL = "http://localhost:8843"
    _AT4_IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"
    }
    _at4_detect_categories = None
    _at4_find_train_split_root = None
    _at4_is_image_file = None

from autotrain_exec.errors import AutoTrainError

logger = logging.getLogger(__name__)


class PlatformMixin:
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
