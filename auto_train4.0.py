"""
DLCV AI 训练平台自动化入口 4.0（自然语言 Agent 层）

一句话用法：
    python auto_train4.0.py --prompt "训练一个 cls 模型来辨别郁金香和菊花" --auto

功能：
1. 解析用户自然语言意图（任务类型、目标类别、数据集提示）。
2. 语义匹配扫描目录下的数据集文件夹。
3. 自动生成训练配置并调用 auto_train2.0.py 完成可视化训练。

保留前端 WebBridge 自动化，无需 API-first。
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from autotrain4.config_builder import ConfigBuilder
from autotrain4.llm_client import LLMClient
from autotrain4.nlu import IntentParser
from autotrain4.resolver import DatasetResolver

# 在 Windows Git Bash / CMD / PowerShell 中保持中文日志不乱码
if sys.platform == "win32" and sys.stdout.encoding not in ("utf-8", "UTF-8"):
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AutoTrain4Error(Exception):
    """4.0 自动化流程异常。"""

    pass


class AutoTrain4Orchestrator:
    """AutoTrain 4.0 主控制器。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.scan_dir = self._resolve_path(args.scan_dir or "AI训练平台测试")
        self.llm = self._build_llm_client()
        self.intent: Optional[Any] = None
        self.match: Optional[Any] = None
        self.config_path: Optional[Path] = None

    @staticmethod
    def _resolve_path(raw: str) -> Path:
        """解析相对路径，支持 %VAR% 环境变量。"""
        expanded = os.path.expandvars(raw)
        p = Path(expanded)
        if not p.is_absolute():
            p = (Path(__file__).parent / p).resolve()
        return p

    @staticmethod
    def _build_llm_client() -> LLMClient:
        """根据命令行/环境变量构建 LLM 客户端。"""
        return LLMClient()

    def run(self) -> int:
        """执行完整 4.0 流程。"""
        logger.info("========== AUTO TRAIN 4.0 START ==========")
        logger.info("用户输入: %s", self.args.prompt)

        # 1. 解析意图
        parser = IntentParser(self.llm)
        self.intent = parser.parse(self.args.prompt)
        logger.info("解析意图: %s", self.intent.to_dict())

        # 2. 语义匹配数据集（或被 --folder 覆盖）
        self.match = self._resolve_match()
        logger.info("匹配数据集: %s", self.match.to_dict())

        # 3. 生成配置
        builder = ConfigBuilder(
            self.intent,
            self.match,
            project_root=self.project_root,
            adapter=self.args.adapter,
        )
        self.config_path = builder.save(self.args.config_out)

        if self.args.generate_config_only:
            logger.info("--generate-config-only: 已生成配置，不启动训练")
            logger.info("配置文件: %s", self.config_path)
            return 0

        # 4. 调用 auto_train2.0.py
        return self._run_auto_train2()

    def _resolve_match(self):
        """解析数据集匹配结果。"""
        resolver = DatasetResolver(self.scan_dir, llm_client=self.llm)
        if self.args.folder:
            folder = self.scan_dir / self.args.folder
            if not folder.exists():
                raise AutoTrain4Error(f"指定文件夹不存在: {folder}")
            return resolver._score_folder(folder, self.intent)
        return resolver.resolve(
            self.intent,
            auto_threshold=self.args.auto_threshold,
        )

    def _run_auto_train2(self) -> int:
        """调用 auto_train2.0.py 执行训练。"""
        cmd = [
            sys.executable,
            str(self.project_root / "auto_train2.0.py"),
            "--config",
            str(self.config_path),
        ]
        if self.args.force_new:
            cmd.append("--force-new")
        if self.args.dry_run:
            cmd.append("--dry-run")
        if self.args.keep_alive:
            cmd.append("--keep-alive")
        if self.args.reuse:
            cmd.append("--reuse")

        logger.info("========== AUTO TRAIN 4.0 -> 2.0 ==========")
        logger.info("启动子进程: %s", " ".join(cmd))

        result = subprocess.run(cmd, check=False)
        return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DLCV AI 训练平台自动化入口 4.0：自然语言一句话启动训练"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="自然语言训练指令，例如：训练一个 cls 模型来辨别郁金香和菊花",
    )
    parser.add_argument(
        "--scan-dir",
        default="AI训练平台测试",
        help="数据集根目录，默认: AI训练平台测试",
    )
    parser.add_argument(
        "--folder",
        help="直接指定数据集文件夹名（跳过语义匹配）",
    )
    parser.add_argument(
        "--config-out",
        help="生成的配置文件保存路径",
    )
    parser.add_argument(
        "--adapter",
        choices=["playwright", "skill", "http", "stub"],
        help="覆盖 WebBridge adapter 类型",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.55,
        help="自动执行的数据集匹配置信度阈值（0-1），默认 0.55",
    )
    parser.add_argument(
        "--generate-config-only",
        action="store_true",
        help="仅生成配置文件，不调用 auto_train2.0.py",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="强制创建新项目（传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成 .dva，不启动平台（传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--keep-alive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持平台运行（默认开启，传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用已运行的平台/后端实例（默认开启，传给 auto_train2.0.py）",
    )
    args = parser.parse_args()

    try:
        orchestrator = AutoTrain4Orchestrator(args)
        return orchestrator.run()
    except AutoTrain4Error as e:
        logger.error("AutoTrain 4.0 流程失败: %s", e)
        return 1
    except Exception as e:
        logger.exception("AutoTrain 4.0 发生未预期错误: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
