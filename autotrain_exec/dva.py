"""dva — part of DLCVAutoTrainer (autotrain_exec)."""
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


class DvaMixin:
    def _is_image_file(self, path: Path) -> bool:
        if _at4_is_image_file is not None:
            return _at4_is_image_file(path)
        return path.is_file() and path.suffix.lower() in self._IMAGE_EXTS


    def _find_train_root(self, dataset_dir: Path) -> Path:
        if _at4_find_train_split_root is not None:
            return _at4_find_train_split_root(dataset_dir)
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
        if _at4_detect_categories is not None:
            dist, cmap = _at4_detect_categories(
                dataset_dir, with_counts=True, use_train_split=True
            )
            return dict(dist), dict(cmap)
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

