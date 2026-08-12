"""backend_api — part of DLCVAutoTrainer (autotrain_exec)."""
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


class BackendApiMixin:
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

