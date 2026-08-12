"""
DLCV AutoTrain 4.0 领域库（编排 / 质检 / post-train / NLU / LLM）。

公开 API 见 ``__all__``。常用可选入口：

- 编排：``AutoTrain4Orchestrator``（CLI 薄入口仍是 ``auto_train4.0.py``）
- 训练后：``resolve_post_train_context`` / ``run_post_train_action``
- 任务类型：``normalize_task_type`` / ``to_ui_code``（权威在 ``task_types``）
- LLM：``from autotrain4.llm import LLMClient``（勿再依赖 ``llm_client`` 模块路径）
"""

from .common import TASK_ALIASES, TASK_TEMPLATES, TASK_FOLDER_KEYWORDS, DATASET_CANDIDATES
from .config_builder import ConfigBuilder
from .constants import DEFAULT_BACKEND_URL, DEFAULT_FRONTEND_URL, DEFAULT_INFERENCE_URL
from .dataset_fs import IMAGE_EXTENSIONS, detect_categories, find_dataset_dir
from .evaluator import InferenceEvaluator
from .inference_client import InferenceClient
from .nlu import IntentParser, TrainIntent
from .orchestrator import AutoTrain4Error, AutoTrain4Orchestrator
from .post_train import (
    resolve_post_train_context,
    run_baseline_compare,
    run_inference_report,
    run_post_train_action,
)
from .resolver import DatasetResolver, DatasetMatch
from .task_types import (
    BASELINE_PREFIXES,
    TASK_UI_CODES,
    normalize_task_type,
    to_ui_code,
    from_ui_code,
)

__all__ = [
    # task / dataset constants
    "TASK_ALIASES",
    "TASK_TEMPLATES",
    "TASK_FOLDER_KEYWORDS",
    "DATASET_CANDIDATES",
    "IMAGE_EXTENSIONS",
    "BASELINE_PREFIXES",
    "TASK_UI_CODES",
    "DEFAULT_FRONTEND_URL",
    "DEFAULT_BACKEND_URL",
    "DEFAULT_INFERENCE_URL",
    "normalize_task_type",
    "to_ui_code",
    "from_ui_code",
    "find_dataset_dir",
    "detect_categories",
    # pipeline building blocks
    "ConfigBuilder",
    "InferenceClient",
    "InferenceEvaluator",
    "IntentParser",
    "TrainIntent",
    "DatasetResolver",
    "DatasetMatch",
    # orchestration
    "AutoTrain4Error",
    "AutoTrain4Orchestrator",
    # post-train
    "resolve_post_train_context",
    "run_inference_report",
    "run_baseline_compare",
    "run_post_train_action",
]
