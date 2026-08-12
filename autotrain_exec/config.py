"""config — part of DLCVAutoTrainer (autotrain_exec)."""
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


class ConfigMixin:
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
            "frontend_url", self.config.get("ui", {}).get("url", DEFAULT_FRONTEND_URL)
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

