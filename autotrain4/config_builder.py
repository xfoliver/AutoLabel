"""
配置生成器。

根据解析出的意图和数据集匹配结果，加载对应任务模板，
生成最终传递给 auto_train2.0.py 的 JSON 配置文件。
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .common import DEFAULT_WEBBRIDGE, TASK_TEMPLATES
from .nlu import TrainIntent
from .resolver import DatasetMatch

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """训练配置生成器。"""

    def __init__(
        self,
        intent: TrainIntent,
        match: DatasetMatch,
        project_root: Optional[Path] = None,
        adapter: Optional[str] = None,
        full_train: bool = False,
    ):
        self.intent = intent
        self.match = match
        self.project_root = project_root or Path(__file__).parent.parent.resolve()
        self.adapter = adapter
        self.full_train = full_train

    def build(self) -> Dict[str, Any]:
        """生成完整配置。"""
        template = self._load_template()
        cfg = json.loads(json.dumps(template))  # 深拷贝

        # project 区块
        project = cfg.setdefault("project", {})
        project["name"] = self._build_project_name()
        # DETR 模板底层 task_type 是 目标检测，保留模板值；其余用意图解析值
        if "task_type" not in project or not project["task_type"]:
            project["task_type"] = self.intent.task_type
        project["dataset_dir"] = str(self.match.dataset_dir.resolve())
        project["work_dir"] = str(self._build_work_dir().resolve())
        project["dva_file"] = str(self._build_dva_path().resolve())
        project.setdefault("description", f"AutoTrain 4.0 {self.intent.task_type} 项目")

        # platform 区块
        platform = cfg.setdefault("platform", {})
        detected_exe = self._detect_platform_exe(platform.get("executable_path"))
        if detected_exe:
            platform["executable_path"] = detected_exe
        platform["backend_ready_timeout"] = 300
        platform["backend_ready_interval"] = 2.0

        # webbridge 区块
        if "webbridge" not in cfg:
            cfg["webbridge"] = dict(DEFAULT_WEBBRIDGE)
        if self.adapter:
            cfg["webbridge"]["adapter"] = self.adapter

        # dataset 区块：保留 auto_detect，同时可记录匹配到的类别
        dataset_cfg = cfg.setdefault("dataset", {})
        dataset_cfg["auto_detect"] = True
        if self.match.categories:
            dataset_cfg["resolved_categories"] = self.match.categories
            dataset_cfg["matched_user_categories"] = self.match.matched_user_categories

        # training 区块：根据数据量做轻量调整
        training = cfg.setdefault("training", {})
        self._tune_hyperparameters(training)

        # 全量训练模式：训练集拉到 100%，验证集保持模板默认值（如 20%）
        if self.full_train:
            hp = training.setdefault("hyperparameters", {})
            hp["train_set_val"] = 100
            logger.info("启用全量训练模式: train_set_val=100")

        return cfg

    def save(self, config_path: Optional[Path] = None) -> Path:
        """保存配置并返回路径。"""
        cfg = self.build()
        if config_path is None:
            config_dir = self.project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            safe_task = re.sub(r"[^\w\-]+", "_", self.intent.task_type)
            config_path = config_dir / f"auto_latest_{safe_task}.json"
        else:
            config_path = Path(config_path).resolve()

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info("已生成配置文件: %s", config_path)
        return config_path

    def _load_template(self) -> Dict[str, Any]:
        """加载任务类型对应的模板配置。"""
        template_file = TASK_TEMPLATES.get(self.intent.task_type)
        if not template_file:
            raise ValueError(f"未找到任务类型 {self.intent.task_type} 的模板配置")

        template_path = self.project_root / template_file
        if not template_path.exists():
            raise ValueError(f"模板配置文件不存在: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_project_name(self) -> str:
        """生成项目名。"""
        folder_name = self._sanitize_name(self.match.folder.name)
        cats = "_".join(self._sanitize_name(c) for c in self.intent.categories[:3])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if cats:
            return f"{folder_name}-{cats}-{timestamp}"
        return f"{folder_name}-{timestamp}"

    def _build_work_dir(self) -> Path:
        """生成工作目录。"""
        return self.match.folder / "work"

    def _build_dva_path(self) -> Path:
        """生成 .dva 文件路径。"""
        return self.match.folder / f"{self._build_project_name()}.dva"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """把字符串转成合法的文件/项目名。"""
        name = name.strip()
        name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
        name = re.sub(r"\s+", "_", name)
        return name or "Project"

    @staticmethod
    def _detect_platform_exe(template_path: Optional[str] = None) -> Optional[str]:
        """自动探测平台可执行文件路径。"""
        env_exe = os.environ.get("DLCV_TRAIN_EXE")
        if env_exe:
            expanded = os.path.expandvars(env_exe)
            if Path(expanded).exists():
                return expanded

        if template_path:
            expanded = os.path.expandvars(template_path)
            if Path(expanded).exists():
                return expanded

        base_dirs = [
            "C:/dlcv/Lib/site-packages/dlcv_train",
            "C:/Program Files/dlcv_train",
            "C:/Program Files (x86)/dlcv_train",
        ]
        names = [
            "DLCV AI Training Platform-win_x64.exe",
            "DLCV AI Training Platform-win_64.exe",
            "DLCV AI Training Platform.exe",
        ]
        for base in base_dirs:
            for name in names:
                candidate = Path(base) / name
                if candidate.exists():
                    return str(candidate)
        return None

    def _tune_hyperparameters(self, training: Dict[str, Any]) -> None:
        """根据数据集规模轻量调整超参。"""
        hp = training.setdefault("hyperparameters", {})
        total = self.match.total_images
        num_classes = len(self.match.categories)

        # 小数据集：更多 epoch，更小 batch
        if total > 0 and total < 500:
            hp.setdefault("max_epochs", 50)
            hp.setdefault("batch_size", 4)
        elif total < 2000:
            hp.setdefault("max_epochs", 30)
            hp.setdefault("batch_size", 8)
        else:
            hp.setdefault("max_epochs", 12)
            hp.setdefault("batch_size", 16)

        # 类别数影响学习率（类别多可稍大一点）
        if num_classes > 10:
            hp.setdefault("lr", 0.005)
        else:
            hp.setdefault("lr", 0.001)

        # 其他默认保持不变
        hp.setdefault("train_set_val", 80)
        hp.setdefault("val_set_val", 20)
        hp.setdefault("random_seed", 42)
