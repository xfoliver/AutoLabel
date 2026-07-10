"""
DLCV AI 训练平台自动化入口 4.0（自然语言 Agent 层）

一句话用法：
    python auto_train4.0.py --prompt "训练一个 cls 模型来辨别郁金香和菊花"

功能：
1. 解析用户自然语言意图（任务类型、目标类别、数据集提示）。
2. 语义匹配扫描目录下的数据集文件夹。
3. 自动生成训练配置并调用 auto_train2.0.py 完成可视化训练。

保留前端 WebBridge 自动化，无需 API-first。
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from autotrain4.common import BASELINE_PREFIXES
from autotrain4.config_builder import ConfigBuilder
from autotrain4.evaluator import InferenceEvaluator
from autotrain4.inference_client import InferenceClient
from autotrain4.judge import RequirementJudge
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
        self.chat_history_path = self._resolve_path(args.chat_history or "./record/chat_history.json")
        if getattr(args, "clear_chat_history", False):
            self._clear_chat_history()
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

        # 2. 测试模式：直接对最新模型 + 训练集跑推理
        if self.args.test_latest or self.intent.mode == "test":
            return self._run_test_latest()

        # 3. 非训练意图：进入通用对话兜底
        if not self.intent.needs_training:
            return self._run_general_chat()

        # 4. 语义匹配数据集（或被 --folder 覆盖）
        self.match = self._resolve_match()
        logger.info("匹配数据集: %s", self.match.to_dict())

        # 4. 第三层：DeepSeek 需求分析与判断
        judge = RequirementJudge()
        judge_result = judge.judge(self.intent, self.match)
        logger.info("DeepSeek 判断: %s", judge_result.to_dict())

        # 5. 生成配置
        builder = ConfigBuilder(
            self.intent,
            self.match,
            project_root=self.project_root,
            adapter=self.args.adapter,
            full_train=self.args.full_train,
            adjustments=judge_result.adjustments,
            llm_client=self.llm,
        )
        self.config_path = builder.save(self.args.config_out)

        if self.args.generate_config_only:
            logger.info("--generate-config-only: 已生成配置，不启动训练")
            logger.info("配置文件: %s", self.config_path)
            return 0

        # 5. 调用 auto_train2.0.py
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
        if self.args.keep_browser_open is False:
            cmd.append("--no-keep-browser-open")
        if self.args.record is False:
            cmd.append("--no-record")
        if self.args.reuse:
            cmd.append("--reuse")

        logger.info("========== AUTO TRAIN 4.0 -> 2.0 ==========")
        logger.info("启动子进程: %s", " ".join(cmd))

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode

        # 训练成功，执行推理与基准对比
        self._run_inference_and_compare()
        return 0

    def _run_test_latest(self) -> int:
        """对最近一次训练的模型，用训练集作为测试集跑推理。"""
        logger.info("========== AUTO TRAIN 4.0 测试最新模型 ==========")

        cfg_path = self._find_latest_config()
        if not cfg_path:
            logger.error("未找到任何历史训练配置文件（config/auto_latest_*.json），无法测试")
            return 1

        logger.info("使用配置: %s", cfg_path)
        cfg = self._load_json_config(cfg_path)

        project = cfg.get("project", {})
        dataset_dir = project.get("dataset_dir")
        work_dir = project.get("work_dir")
        task_type = project.get("task_type") or self.intent.task_type

        if not dataset_dir or not work_dir:
            logger.error("配置文件缺少 project.dataset_dir 或 project.work_dir")
            return 1

        dataset_dir = self._resolve_path(dataset_dir)
        work_dir = self._resolve_path(work_dir)

        if not dataset_dir.exists():
            logger.error("训练集目录不存在: %s", dataset_dir)
            return 1

        # 查找工作目录下最新的 .dvp 模型
        new_dvp = self._find_latest_dvp_in_dir(work_dir)
        if not new_dvp:
            logger.error("未在 work_dir 找到 .dvp 模型文件: %s", work_dir)
            return 1

        logger.info("任务类型: %s", task_type)
        logger.info("新模型: %s", new_dvp)
        logger.info("测试图片目录（复用训练集）: %s", dataset_dir)

        self.config_path = cfg_path
        baseline_model = self._effective_baseline_model(task_type)
        self._run_inference_and_compare_with_paths(
            new_dvp=new_dvp,
            image_dir=dataset_dir,
            baseline_model_path=baseline_model,
        )
        return 0

    def _load_chat_history(self) -> List[Dict[str, str]]:
        """加载历史聊天记录。"""
        if not self.chat_history_path.exists():
            return []
        try:
            with open(self.chat_history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning("读取聊天记录失败: %s", e)
        return []

    def _save_chat_history(self, messages: List[Dict[str, str]]) -> None:
        """保存聊天记录，最多保留最近 50 条。"""
        try:
            self.chat_history_path.parent.mkdir(parents=True, exist_ok=True)
            # 保留最近 50 条，避免上下文过长
            trimmed = messages[-50:]
            with open(self.chat_history_path, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存聊天记录失败: %s", e)

    def _clear_chat_history(self) -> None:
        """清空聊天记录文件。"""
        try:
            if self.chat_history_path.exists():
                self.chat_history_path.unlink()
                logger.info("已清空聊天记录: %s", self.chat_history_path)
        except Exception as e:
            logger.warning("清空聊天记录失败: %s", e)

    def _run_general_chat(self) -> int:
        """通用对话兜底：当用户输入不是训练/测试意图时，用 LLM 做友好回复并记住历史。"""
        logger.info("========== AUTO TRAIN 4.0 通用对话 ==========")
        logger.info("用户输入与训练/测试无关，进入通用对话模式")

        system_msg = (
            "你是 DLCV AI 训练平台的智能助手。用户当前输入看起来不是想启动训练或测试任务，"
            "请用友好、简洁的中文回答用户的问题。如果用户是在闲聊，可以正常回应；"
            "如果用户其实想问训练相关的问题，也可以简要介绍本平台的功能。"
        )

        messages = self._load_chat_history()
        # 确保 system 消息在最前
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_msg})
        # 更新 system 消息内容（如果后续有调整）
        messages[0] = {"role": "system", "content": system_msg}

        messages.append({"role": "user", "content": self.args.prompt})

        try:
            answer = self.llm.chat(messages)
            if answer:
                logger.info("[助手回复] %s", answer)
                messages.append({"role": "assistant", "content": answer})
                self._save_chat_history(messages)
            else:
                logger.info("[助手回复] 抱歉，我没有理解你的问题，可以再说一次吗？")
        except Exception as e:
            logger.error("通用对话调用 LLM 失败: %s", e)
            return 1
        return 0

    def _find_latest_config(self) -> Optional[Path]:
        """查找最新的 auto_train4 配置文件。"""
        if self.args.latest_config:
            p = self._resolve_path(self.args.latest_config)
            if p.exists():
                return p
            logger.warning("--latest-config 指定文件不存在: %s", p)

        config_dir = self.project_root / "config"
        if not config_dir.exists():
            return None

        candidates = list(config_dir.glob("auto_latest_*.json"))
        if not candidates:
            return None

        # 如果有意图任务类型，优先匹配对应任务
        task_type = self.intent.task_type if self.intent else ""
        if task_type:
            safe_task = re.sub(r"[^\w\-]+", "_", task_type)
            matched = [p for p in candidates if safe_task in p.name]
            if matched:
                return max(matched, key=lambda p: p.stat().st_mtime)

        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _find_latest_dvp_in_dir(self, directory: Path) -> Optional[Path]:
        """查找目录下最新的 .dvp 文件。"""
        dvps = list(directory.glob("*.dvp"))
        if not dvps:
            return None
        return max(dvps, key=lambda p: p.stat().st_mtime)

    def _resolve_baseline_model(
        self,
        task_type: str,
        hint: Optional[str] = None,
    ) -> Optional[Path]:
        """根据任务类型自动定位 Baseline Models 目录下的基准模型。"""
        baseline_dir = self.project_root / "Baseline Models"
        if not baseline_dir.exists():
            return None

        prefix = BASELINE_PREFIXES.get(task_type)
        if not prefix:
            return None

        candidates = [
            p for p in baseline_dir.glob("*.dvp")
            if p.name.upper().startswith(prefix.upper() + "-")
        ]
        if hint:
            hint_lower = hint.lower()
            candidates = [p for p in candidates if hint_lower in p.name.lower()]

        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _effective_baseline_model(self, task_type: str) -> Optional[str]:
        """综合命令行参数、意图中的路径、自动定位，得到最终 baseline 模型路径。"""
        # 1. 命令行参数优先级最高
        if self.args.baseline_model:
            return self.args.baseline_model

        # 2. 意图中显式指定的路径
        intent_baseline = self.intent.baseline_model if self.intent else None
        if intent_baseline:
            path = self._resolve_path(intent_baseline)
            if path.exists():
                return str(path)
            # 也尝试相对于项目根目录
            rel_path = self.project_root / intent_baseline
            if rel_path.exists():
                return str(rel_path)

        # 3. 用户表达了对比意愿，自动定位 baseline
        wants_baseline = self.intent.wants_baseline if self.intent else False
        if wants_baseline:
            resolved = self._resolve_baseline_model(task_type, hint=intent_baseline)
            if resolved:
                logger.info("自动定位到基准模型: %s", resolved)
                return str(resolved)
            logger.warning("用户希望对比 baseline，但未找到任务类型 %s 对应的基准模型", task_type)

        return None

    def _load_json_config(self, path: Path) -> Dict[str, Any]:
        """加载 JSON 配置文件。"""
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run_inference_and_compare_with_paths(
        self,
        new_dvp: Path,
        image_dir: Path,
        baseline_model_path: Optional[str] = None,
    ) -> None:
        """使用指定路径执行推理评估。"""
        client = InferenceClient(
            base_url=self.args.inference_api_url,
            progress=not self.args.no_progress,
        )
        evaluator = InferenceEvaluator(client=client)

        try:
            report = evaluator.evaluate(
                new_model_path=new_dvp,
                image_dir=image_dir,
                baseline_model_path=baseline_model_path,
            )

            report_path = self._resolve_path(
                self.args.inference_report_path
                or f"test_report_{new_dvp.stem}.json"
            )
            report.save(report_path)
            logger.info("推理评估完成，报告: %s", report_path)
        except Exception as e:
            logger.exception("推理评估失败: %s", e)

    def _run_inference_and_compare(self) -> None:
        """训练完成后，对新模型推理并与基准模型对比。"""
        if self.args.skip_inference or not self.args.test_image_dir:
            logger.info("跳过推理对比（未指定测试图片目录或启用 --skip-inference）")
            return

        test_image_dir = self._resolve_path(self.args.test_image_dir)
        if not test_image_dir.exists():
            logger.warning("测试图片目录不存在，跳过推理: %s", test_image_dir)
            return

        new_dvp = self._find_new_dvp()
        if not new_dvp:
            logger.warning("未找到新生成的 .dvp 模型文件，跳过推理")
            return

        task_type = self.config.get("project", {}).get("task_type") or (
            self.intent.task_type if self.intent else ""
        )
        baseline_model = self._effective_baseline_model(task_type)

        logger.info("========== AUTO TRAIN 4.0 推理评估 ==========")
        logger.info("新模型: %s", new_dvp)
        logger.info("测试图片目录: %s", test_image_dir)
        if baseline_model:
            logger.info("基准模型: %s", baseline_model)

        client = InferenceClient(
            base_url=self.args.inference_api_url,
            progress=not self.args.no_progress,
        )
        evaluator = InferenceEvaluator(client=client)

        try:
            report = evaluator.evaluate(
                new_model_path=new_dvp,
                image_dir=test_image_dir,
                baseline_model_path=baseline_model,
            )

            report_path = self._resolve_path(self.args.inference_report_path or "inference_report.json")
            report.save(report_path)
            logger.info("推理评估完成，报告: %s", report_path)
        except Exception as e:
            logger.exception("推理评估失败: %s", e)

    def _find_new_dvp(self) -> Optional[Path]:
        """查找本次训练生成的 .dvp 模型文件。"""
        if not self.config_path:
            return None

        # 1. 优先在 work_dir 下找最新的 .dvp
        cfg = self._load_generated_config()
        work_dir_str = cfg.get("project", {}).get("work_dir")
        if work_dir_str:
            work_dir = self._resolve_path(work_dir_str)
            if work_dir.exists():
                dvps = list(work_dir.glob("*.dvp"))
                if dvps:
                    return max(dvps, key=lambda p: p.stat().st_mtime)

        # 2. 在 .dva 同目录找同名 .dvp
        dva_path = self.config_path.with_suffix(".dva")
        if dva_path.exists():
            dvp_path = dva_path.with_suffix(".dvp")
            if dvp_path.exists():
                return dvp_path

        # 3. 在 config 路径同目录下找同名 .dvp
        dvp_path = self.config_path.with_suffix(".dvp")
        if dvp_path.exists():
            return dvp_path

        return None

    def _load_generated_config(self) -> Dict[str, Any]:
        """加载生成的 JSON 配置文件。"""
        import json

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


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
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持浏览器窗口打开，按 Enter 关闭（默认开启，传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="录制前端自动化流程并保存到 record 目录（默认开启，传给 auto_train2.0.py）",
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
        "--test-image-dir",
        help="训练完成后用于推理测试的图片目录",
    )
    parser.add_argument(
        "--baseline-model",
        help="基准模型 .dvp 路径，用于与新模型推理结果对比",
    )
    parser.add_argument(
        "--inference-api-url",
        default="http://127.0.0.1:9890",
        help="DLCV 推理接口地址，默认 http://127.0.0.1:9890",
    )
    parser.add_argument(
        "--inference-report-path",
        help="推理对比报告保存路径，默认 inference_report.json",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="跳过训练后的推理对比",
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
