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

import os
from pathlib import Path

# 优先从项目根目录 .env 加载环境变量，避免在代码里硬编码密钥
# python-dotenv 未安装时静默跳过
try:
    from dotenv import load_dotenv

    _dotenv_path = Path(__file__).parent / ".env"
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=_dotenv_path, override=True)
except Exception:
    pass

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from autotrain4.common import (
    find_baseline_model,
    find_latest_config,
    find_latest_dvp,
    resolve_path,
)
from autotrain4.config_builder import ConfigBuilder
from autotrain4.evaluator import InferenceEvaluator
from autotrain4.inference_client import InferenceClient
from autotrain4.judge import RequirementJudge
from autotrain4.llm_client import LLMClient
from autotrain4.nlu import IntentParser
from autotrain4.resolver import DatasetResolver
from autotrain4.run_context import RunContext

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
        self.scan_dir = resolve_path(args.scan_dir or "AI训练平台测试")
        self.chat_history_path = resolve_path(args.chat_history or "./record/chat_history.json")
        if getattr(args, "clear_chat_history", False):
            self._clear_chat_history()
        self.llm = self._build_llm_client()
        self.intent: Optional[Any] = None
        self.match: Optional[Any] = None
        self.config_path: Optional[Path] = None
        self.run_ctx: Optional[RunContext] = None

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

        # 5. 创建本次 run 的归档上下文
        self.run_ctx = RunContext(
            dataset_dir=self.match.dataset_dir,
            project_dir=self.match.folder,
            folder_name=self.match.folder.name,
        )
        self.run_ctx.save_runtime_info({
            "backend_url": self.args.inference_api_url,
            "frontend_url": self.config.get("webbridge", {}).get("frontend_url") if hasattr(self, "config") else None,
        })
        logger.info("本次训练归档目录: %s", self.run_ctx.run_dir)
        print(f"[RUN_DIR] {self.run_ctx.run_dir}", flush=True)

        # 6. 第三层：DeepSeek 需求分析与判断
        judge = RequirementJudge()
        judge_result = judge.judge(self.intent, self.match)
        logger.info("DeepSeek 判断: %s", judge_result.to_dict())

        # 7. 生成配置
        builder = ConfigBuilder(
            self.intent,
            self.match,
            project_root=self.project_root,
            adapter=self.args.adapter,
            full_train=self.args.full_train,
            adjustments=judge_result.adjustments,
            llm_client=self.llm,
            record=self.args.record,
            record_dir=self.args.record_dir,
            run_context=self.run_ctx,
        )
        self.config_path = builder.save(self.args.config_out)
        self.run_ctx.save_config(builder.build())

        if self.args.generate_config_only:
            logger.info("--generate-config-only: 已生成配置，不启动训练")
            logger.info("配置文件: %s", self.config_path)
            return 0

        # 8. 调用 auto_train2.0.py
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
        if self.args.reuse:
            cmd.append("--reuse")

        logger.info("========== AUTO TRAIN 4.0 -> 2.0 ==========")
        logger.info("启动子进程: %s", " ".join(cmd))

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            if getattr(self, "run_ctx", None):
                self.run_ctx.save_exception(
                    message=f"auto_train2.0.py 子进程退出码非零: {result.returncode}",
                    exit_code=result.returncode,
                )
            return result.returncode

        # 训练成功后，通过 LLM 询问用户是否生成 inference report / 与 baseline 对比
        if self.args.post_train:
            self._interactive_post_train()
        else:
            logger.info("--no-post-train 已设置，跳过训练后交互")
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

        dataset_dir = resolve_path(dataset_dir)
        work_dir = resolve_path(work_dir)

        if not dataset_dir.exists():
            logger.error("训练集目录不存在: %s", dataset_dir)
            return 1

        # 查找工作目录下最新的 .dvp 模型
        new_dvp = find_latest_dvp(work_dir)
        if not new_dvp:
            logger.error("未在 work_dir 找到 .dvp 模型文件: %s", work_dir)
            return 1

        # 为本次测试创建独立的 run 目录
        self.run_ctx = RunContext(
            dataset_dir=dataset_dir,
            project_dir=work_dir.parent,
            folder_name=f"{work_dir.parent.name}-test",
        )
        self.run_ctx.save_runtime_info()
        logger.info("本次测试归档目录: %s", self.run_ctx.run_dir)
        print(f"[RUN_DIR] {self.run_ctx.run_dir}", flush=True)

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
            p = resolve_path(self.args.latest_config)
            if p.exists():
                return p
            logger.warning("--latest-config 指定文件不存在: %s", p)

        task_type = self.intent.task_type if self.intent else ""
        return find_latest_config(task_type)

    def _effective_baseline_model(self, task_type: str) -> Optional[str]:
        """综合命令行参数、意图中的路径、自动定位，得到最终 baseline 模型路径。"""
        if self.args.baseline_model:
            return self.args.baseline_model

        intent_baseline = self.intent.baseline_model if self.intent else None
        if intent_baseline:
            path = resolve_path(intent_baseline)
            if path.exists():
                return str(path)
            rel_path = self.project_root / intent_baseline
            if rel_path.exists():
                return str(rel_path)

        wants_baseline = self.intent.wants_baseline if self.intent else False
        if wants_baseline:
            resolved = find_baseline_model(task_type)
            if resolved:
                logger.info("自动定位到基准模型: %s", resolved)
                return str(resolved)
            logger.warning("用户希望对比 baseline，但未找到任务类型 %s 对应的基准模型", task_type)

        return None

    @staticmethod
    def _load_json_config(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run_inference_and_compare_with_paths(
        self,
        new_dvp: Path,
        image_dir: Path,
        baseline_model_path: Optional[str] = None,
        report_path: Optional[Path] = None,
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

            if report_path is None:
                if self.args.inference_report_path:
                    report_path = resolve_path(self.args.inference_report_path)
                elif getattr(self, "run_ctx", None):
                    if baseline_model_path:
                        report_path = self.run_ctx.baseline_compare_report_path
                    else:
                        report_path = self.run_ctx.inference_report_path
                else:
                    report_path = resolve_path(f"test_report_{new_dvp.stem}.json")
            report.save(report_path)
            summary = report.to_dict().get("summary", {})
            logger.info("========== 推理评估完成 ==========")
            logger.info("报告保存路径: %s", report_path)
            logger.info("总图片数: %s", summary.get("total_images"))
            logger.info("成功: %s, 失败: %s", summary.get("new_success_count"), summary.get("total_images", 0) - summary.get("new_success_count", 0))
            if baseline_model_path:
                logger.info("baseline 成功: %s", summary.get("baseline_success_count"))
                logger.info("一致: %s, 不一致: %s", summary.get("match_count"), summary.get("mismatch_count"))
        except Exception as e:
            logger.exception("推理评估失败: %s", e)

    def _interactive_post_train(self) -> None:
        """训练成功后，通过 LLM 询问用户是否生成 inference report 或与 baseline 对比。"""
        logger.info("========== 训练完成，进入后续操作询问 ==========")

        if not self.llm or not self.llm.available:
            logger.info("LLM 不可用，跳过训练后交互")
            return

        cfg = self._load_generated_config() if self.config_path else {}
        project = cfg.get("project", {})
        task_type = project.get("task_type") or (
            self.intent.task_type if self.intent else "unknown"
        )
        work_dir = project.get("work_dir")
        dataset_dir = project.get("dataset_dir")
        baseline_model = self._effective_baseline_model(task_type)
        new_dvp = self._find_new_dvp()

        context_lines = [
            "训练已完成，上下文如下：",
            f"- 任务类型: {task_type}",
            f"- 工作目录: {work_dir or '未知'}",
            f"- 数据集目录: {dataset_dir or '未知'}",
            f"- 新模型: {new_dvp or '未找到'}",
        ]
        if baseline_model:
            context_lines.append(f"- 可用基准模型: {baseline_model}")
        else:
            context_lines.append("- 可用基准模型: 无")
        context = "\n".join(context_lines)

        # 1. LLM 生成询问语句
        ask_messages = [
            {
                "role": "system",
                "content": (
                    "你是 DLCV 训练助手。训练刚刚完成，请用一句简洁友好的中文询问用户接下来想做什么。"
                    "可选项包括：生成 inference report、与基准模型对比、结束。"
                    "请只输出要问用户的问题，不要输出其他内容。"
                ),
            },
            {"role": "user", "content": context},
        ]
        try:
            question = self.llm.chat(ask_messages, temperature=0.7) or "训练已完成，您希望接下来做什么？"
        except Exception as e:
            logger.warning("LLM 生成询问失败: %s", e)
            question = "训练已完成，您希望接下来做什么？（生成 inference report / 与基准模型对比 / 结束）"

        # 输出带标记的问题，方便 PyQt 前端捕获并弹窗
        print(f"[POST_TRAIN_PROMPT] {question.strip()}")

        # 2. 读取用户输入
        try:
            user_input = input("[POST_TRAIN_INPUT] ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ""

        if not user_input or user_input.lower() in ("结束", "exit", "no", "不用", "否", "跳过"):
            print("[POST_TRAIN_RESULT] 用户未要求后续操作，流程结束")
            return

        # 3. LLM 解析用户意图
        parse_messages = [
            {
                "role": "system",
                "content": (
                    "你是任务解析助手。训练已完成，请根据上下文解析用户后续需求。"
                    "必须返回 JSON，格式如下（不要 Markdown 代码块）：\n"
                    "{\n"
                    '  "action": "inference_report|compare_baseline|chat|exit",\n'
                    '  "test_image_dir": "测试图片目录，未指定则默认使用训练集目录",\n'
                    '  "baseline_model": "基准模型路径，未指定则使用上下文中的可用基准模型",\n'
                    '  "report_path": "报告保存路径，默认 inference_report.json"\n'
                    "}\n"
                    "action 说明：\n"
                    "- inference_report：仅对新模型生成 inference report\n"
                    "- compare_baseline：对新模型和基准模型做对比报告\n"
                    "- chat：闲聊/不需要执行操作\n"
                    "- exit：用户明确结束"
                ),
            },
            {"role": "user", "content": f"{context}\n\n用户需求: {user_input}"},
        ]
        try:
            parse_resp = self.llm.chat(parse_messages, temperature=0.0) or ""
            decision = json.loads(parse_resp.strip())
        except Exception as e:
            logger.warning("解析训练后需求失败: %s", e)
            print(f"[POST_TRAIN_RESULT] 无法解析需求，不执行后续操作")
            return

        action = decision.get("action", "exit")
        if action == "exit":
            print("[POST_TRAIN_RESULT] 用户结束，不执行后续操作")
            return
        if action == "chat":
            print(f"[POST_TRAIN_RESULT] {decision.get('reply', '收到')}")
            return

        # 4. 执行动作
        if not new_dvp:
            print("[POST_TRAIN_RESULT] 未找到新模型，无法生成报告")
            return

        test_dir_str = decision.get("test_image_dir") or dataset_dir
        if not test_dir_str:
            print("[POST_TRAIN_RESULT] 未指定测试图片目录，无法生成报告")
            return
        test_dir = resolve_path(test_dir_str)
        if not test_dir.exists():
            print(f"[POST_TRAIN_RESULT] 测试图片目录不存在: {test_dir}")
            return

        baseline = decision.get("baseline_model") or baseline_model

        # 默认报告路径落到 run_dir
        if decision.get("report_path"):
            report_path = resolve_path(decision["report_path"])
        elif getattr(self, "run_ctx", None):
            report_path = (
                self.run_ctx.baseline_compare_report_path
                if baseline
                else self.run_ctx.inference_report_path
            )
        else:
            report_path = resolve_path("inference_report.json")

        logger.info("========== AUTO TRAIN 4.0 训练后推理评估 ==========")
        logger.info("新模型: %s", new_dvp)
        logger.info("测试图片目录: %s", test_dir)
        if baseline:
            logger.info("基准模型: %s", baseline)

        self._run_inference_and_compare_with_paths(
            new_dvp=new_dvp,
            image_dir=test_dir,
            baseline_model_path=baseline,
            report_path=report_path,
        )
        print(f"[POST_TRAIN_RESULT] 报告已保存: {report_path}")

    def _find_new_dvp(self) -> Optional[Path]:
        """查找本次训练生成的 .dvp 模型文件。"""
        if not self.config_path:
            return None

        cfg = self._load_generated_config()
        work_dir_str = cfg.get("project", {}).get("work_dir")
        if work_dir_str:
            work_dir = resolve_path(work_dir_str)
            if work_dir.exists():
                latest = find_latest_dvp(work_dir)
                if latest:
                    return latest

        # fallback：在 .dva / config 同目录找同名 .dvp
        for base in (self.config_path.with_suffix(".dva"), self.config_path):
            dvp_path = base.with_suffix(".dvp")
            if dvp_path.exists():
                return dvp_path

        return None

    def _load_generated_config(self) -> Dict[str, Any]:
        """加载生成的 JSON 配置文件。"""
        if not self.config_path:
            return {}
        return self._load_json_config(self.config_path)


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
        default="http://127.0.0.1:9890",
        help="DLCV 推理接口地址，默认 http://127.0.0.1:9890",
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
        _save_exception_fallback(orchestrator, e, exit_code=1)
        return 1
    except Exception as e:
        logger.exception("AutoTrain 4.0 发生未预期错误: %s", e)
        _save_exception_fallback(orchestrator, e, exit_code=1)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        _save_exception_fallback(
            orchestrator,
            message="用户通过 Ctrl+C 中断训练",
            exit_code=130,
        )
        return 130


def _save_exception_fallback(
    orchestrator: Optional[AutoTrain4Orchestrator],
    exc: Optional[BaseException] = None,
    message: Optional[str] = None,
    exit_code: int = 1,
) -> None:
    """保存异常到 run_ctx 目录；若 run_ctx 尚未创建，则落到 failed_runs/<timestamp>/。"""
    if orchestrator is not None and getattr(orchestrator, "run_ctx", None):
        if exc is not None:
            orchestrator.run_ctx.save_exception(exc=exc, exit_code=exit_code)
        else:
            orchestrator.run_ctx.save_exception(message=message, exit_code=exit_code)
        return

    # 早期失败兜底
    try:
        failed_dir = Path(__file__).parent / "failed_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(dataset_dir=failed_dir, folder_name="early-failure")
        if exc is not None:
            ctx.save_exception(exc=exc, exit_code=exit_code)
        else:
            ctx.save_exception(message=message, exit_code=exit_code)
        logger.info("早期失败异常已保存到: %s", ctx.exception_path)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
