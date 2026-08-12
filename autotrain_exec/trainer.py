"""DLCVAutoTrainer — orchestration skeleton for auto_train 2.0 execution kernel."""
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


from autotrain_exec.backend_api import BackendApiMixin
from autotrain_exec.config import ConfigMixin
from autotrain_exec.dva import DvaMixin
from autotrain_exec.monitor import MonitorMixin
from autotrain_exec.platform import PlatformMixin
from autotrain_exec.webbridge_runner import WebBridgeRunnerMixin


class DLCVAutoTrainer(
    ConfigMixin,
    DvaMixin,
    PlatformMixin,
    BackendApiMixin,
    MonitorMixin,
    WebBridgeRunnerMixin,
):
    """DLCV AI 训练平台自动化控制器（Kimi WebBridge 版）。"""

    PLATFORM_BACKEND_EXE = "DLCV AI Training Backend.exe"
    _IMAGE_EXTS = set(_AT4_IMAGE_EXTENSIONS)

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
        self.backend_url: str = DEFAULT_BACKEND_URL
        self.frontend_url: str = DEFAULT_FRONTEND_URL
        self.project_id: Optional[str] = None
        self.training_id: Optional[str] = None
        # 一旦通过 WebBridge 或 API 成功拿到 training_id，禁止再二次 start
        self.training_launch_attempted: bool = False
        self.training_failure_detail: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 配置加载与校验
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



