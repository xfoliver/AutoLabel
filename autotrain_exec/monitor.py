"""monitor — training status polling and diagnostics."""
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


class MonitorMixin:
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
