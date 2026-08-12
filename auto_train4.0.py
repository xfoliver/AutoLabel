"""
DLCV AI 训练平台自动化入口 4.0（自然语言 Agent 层）

一句话用法：
    python auto_train4.0.py --prompt "训练一个 cls 模型来辨别郁金香和菊花"

薄入口：bootstrap + argparse → autotrain4.orchestrator
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from autotrain4.bootstrap import bootstrap
from autotrain4.constants import DEFAULT_INFERENCE_URL
from autotrain4.orchestrator import (
    AutoTrain4Error,
    AutoTrain4Orchestrator,
    save_exception_fallback,
)

# Process bootstrap before logging config
bootstrap(Path(__file__).parent.resolve())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Backward-compatible re-exports for tests / external imports
__all__ = [
    "AutoTrain4Error",
    "AutoTrain4Orchestrator",
    "main",
    "save_exception_fallback",
]


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
        action="append",
        default=None,
        help=(
            "数据集扫描根目录，可重复指定以启用多根浅扫；"
            "也可在单个值内用 ; 或 | 分隔多个根。"
            "默认: AI训练平台测试。"
            "示例: --scan-dir Z:\\data --scan-dir Y:\\data"
            " 或 --scan-dir \"Z:\\data;Y:\\data;C:\\data\""
        ),
    )
    parser.add_argument(
        "--folder",
        help="直接指定数据集目录：可为绝对路径，或任一 --scan-dir 下的文件夹名（跳过语义匹配）",
    )
    parser.add_argument(
        "--scan-depth",
        type=int,
        default=4,
        help="自然语言模糊定位时，相对锚点的最大搜索深度（默认 4）",
    )
    parser.add_argument(
        "--no-nl-locate",
        action="store_true",
        help="禁用自然语言锚点深搜，仅使用显式路径 / --folder / 多根浅扫",
    )
    parser.add_argument(
        "--output-dir",
        help="模型与 work 归档根目录；指定后 run 落在 <output-dir>/work/<run_id>/，"
             "不再默认写到数据集项目目录下",
    )
    parser.add_argument(
        "--config-out",
        help="生成的配置文件保存路径",
    )
    parser.add_argument(
        "--adapter",
        choices=["playwright", "webview2_cdp", "skill", "http", "stub"],
        help="覆盖 WebBridge adapter 类型",
    )
    parser.add_argument(
        "--platform-exe",
        help="壳端 EXE 路径（webview2_cdp 模式使用）",
    )
    parser.add_argument(
        "--cdp-port",
        type=int,
        default=9222,
        help="WebView2 CDP 调试端口（默认 9222）",
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
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持浏览器窗口打开，按 Enter 关闭（默认开启，传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 Playwright 内置录屏（默认开启）",
    )
    parser.add_argument(
        "--record-dir",
        default="./record",
        help="Playwright 录屏保存目录（默认 ./record）",
    )
    parser.add_argument(
        "--post-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后进入交互式后续操作询问（默认开启；--no-post-train 直接退出）",
    )
    parser.add_argument(
        "--reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用已运行的平台/后端实例（默认开启，传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="训练集拉到 100%%（train_set_val=100，验证集保持模板默认值），用于可复现性验证",
    )
    parser.add_argument(
        "--baseline-model",
        help="基准模型 .dvp 路径，用于与新模型推理结果对比",
    )
    parser.add_argument(
        "--inference-api-url",
        default=DEFAULT_INFERENCE_URL,
        help=f"DLCV 推理接口地址，默认 {DEFAULT_INFERENCE_URL}",
    )
    parser.add_argument(
        "--inference-report-path",
        help="推理对比报告保存路径，默认 inference_report.json",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭推理进度条显示",
    )
    parser.add_argument(
        "--test-latest",
        action="store_true",
        help="跳过训练，直接测试最近一次训练生成的模型（用训练集作为测试集）",
    )
    parser.add_argument(
        "--latest-config",
        help="指定要回看的配置文件路径（配合 --test-latest 使用，默认自动找 config/auto_latest_*.json）",
    )
    parser.add_argument(
        "--chat-history",
        default="./record/chat_history.json",
        help="通用对话聊天记录保存路径，默认 ./record/chat_history.json",
    )
    parser.add_argument(
        "--clear-chat-history",
        action="store_true",
        help="启动前清空历史聊天记录",
    )
    args = parser.parse_args()

    orchestrator: Optional[AutoTrain4Orchestrator] = None
    try:
        orchestrator = AutoTrain4Orchestrator(args)
        return orchestrator.run()
    except AutoTrain4Error as e:
        logger.error("AutoTrain 4.0 流程失败: %s", e)
        save_exception_fallback(orchestrator, e, exit_code=1)
        return 1
    except Exception as e:
        logger.exception("AutoTrain 4.0 发生未预期错误: %s", e)
        save_exception_fallback(orchestrator, e, exit_code=1)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        save_exception_fallback(
            orchestrator,
            message="用户通过 Ctrl+C 中断训练",
            exit_code=130,
        )
        return 130




if __name__ == "__main__":
    sys.exit(main())
