"""
DLCV AI 训练平台自动化脚本（方案 B - Kimi WebBridge 版）

薄入口：argparse / main → autotrain_exec

使用方法：
    python auto_train2.0.py --config config/train_automation_config_2.0.json --dry-run
    python auto_train2.0.py --config config/train_automation_config_2.0.json
"""

from __future__ import annotations

import sys

from autotrain_exec.cli import main
from autotrain_exec.errors import AutoTrainError
from autotrain_exec.trainer import DLCVAutoTrainer

# Backward-compatible re-exports
__all__ = ["AutoTrainError", "DLCVAutoTrainer", "main"]


if __name__ == "__main__":
    sys.exit(main())
