"""webbridge_runner — part of DLCVAutoTrainer (autotrain_exec)."""
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


class WebBridgeRunnerMixin:
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

