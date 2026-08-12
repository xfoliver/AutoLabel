"""CLI for auto_train 2.0 execution kernel."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autotrain4.bootstrap import bootstrap
from autotrain_exec.errors import AutoTrainError
from autotrain_exec.trainer import DLCVAutoTrainer

bootstrap(Path(__file__).resolve().parent.parent)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DLCV AI 训练平台自动化脚本（Kimi WebBridge 版）"
    )
    parser.add_argument(
        "--config",
        default="config/train_automation_config_2.0.json",
        help="配置文件路径",
    )
    parser.add_argument("--force-new", action="store_true", help="强制创建新项目")
    parser.add_argument("--dry-run", action="store_true", help="仅生成 .dva，不启动平台")
    parser.add_argument("--keep-alive", action="store_true", help="训练结束后保持平台运行")
    parser.add_argument(
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持浏览器窗口打开，按 Enter 关闭（默认开启）",
    )
    parser.add_argument("--reuse", action="store_true", help="复用已运行的平台/后端实例")
    args = parser.parse_args()

    trainer = DLCVAutoTrainer(
        args.config,
        force_new=args.force_new,
        dry_run=args.dry_run,
        reuse=args.reuse,
        keep_browser_open=args.keep_browser_open,
    )
    try:
        trainer.run()
        return 0
    except AutoTrainError as e:
        logger.error("自动化流程失败: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    finally:
        if not args.keep_alive:
            trainer.shutdown()


if __name__ == "__main__":
    sys.exit(main())
