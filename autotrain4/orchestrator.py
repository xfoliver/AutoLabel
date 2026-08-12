"""AutoTrain 4.0 orchestration (NLU → resolve → judge → config → auto_train2.0).

CLI entry remains ``auto_train4.0.py`` (argparse + main). Import this module for
in-process tests without spinning up the UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from autotrain4.common import (
    extract_filesystem_paths,
    find_baseline_model,
    find_latest_config,
    find_latest_dvp,
    normalize_scan_dirs,
    resolve_output_dir,
    resolve_path,
)
from autotrain4.config_builder import ConfigBuilder
from autotrain4.constants import DEFAULT_INFERENCE_URL
from autotrain4.judge import RequirementJudge
from autotrain4.llm import LLMClient
from autotrain4.nlu import IntentParser, TrainIntent
from autotrain4.output_protocol import (
    emit_run_dir,
    emit_stage,
    encode_chat_response,
)
from autotrain4.post_train import (
    context_from_paths,
    resolve_post_train_context,
    run_post_train_action,
)
from autotrain4.resolver import DatasetResolver
from autotrain4.run_context import RunContext, cleanup_run_artifacts

logger = logging.getLogger(__name__)


class AutoTrain4Error(Exception):
    """4.0 自动化流程异常。"""

    pass


class AutoTrain4Orchestrator:
    """AutoTrain 4.0 主控制器。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).resolve().parent.parent
        # 多根浅扫：--scan-dir 可重复，或单参数内用 ; | 分隔
        self.scan_dirs = normalize_scan_dirs(
            getattr(args, "scan_dir", None),
            default="AI训练平台测试",
        )
        self.scan_dir = self.scan_dirs[0]  # 兼容旧逻辑（如最近 run 查找的默认根）
        self.chat_history_path = resolve_path(args.chat_history or "./record/chat_history.json")
        if getattr(args, "clear_chat_history", False):
            self._clear_chat_history()
        self.llm = self._build_llm_client()
        self.intents: List[TrainIntent] = []
        self.intent: Optional[TrainIntent] = None  # 当前/最后一个任务意图
        self.match: Optional[Any] = None
        self.config_path: Optional[Path] = None
        self.run_ctx: Optional[RunContext] = None

    @staticmethod
    def _build_llm_client() -> LLMClient:
        """根据环境变量构建 LLM 客户端（UI 注入 AUTOTRAIN_LLM_*）。"""
        client = LLMClient.from_environ()
        logger.info(
            "LLM 配置: base_url=%s model=%s protocol=%s profile=%s available=%s",
            client.base_url,
            client.model,
            client.active_protocol,
            getattr(client, "profile_name", "") or "-",
            client.available,
        )
        return client

    def run(self) -> int:
        """执行完整 4.0 流程，支持单任务或多任务顺序执行。"""
        logger.info("========== AUTO TRAIN 4.0 START ==========")
        logger.info("用户输入: %s", self.args.prompt)

        # 1. 解析意图（可能返回多个任务）
        parser = IntentParser(self.llm)
        self.intents = parser.parse(self.args.prompt)
        logger.info("解析到 %d 个任务", len(self.intents))
        for idx, intent in enumerate(self.intents, 1):
            logger.info("任务 %d/%d: %s", idx, len(self.intents), intent.to_dict())

        # 2. 单任务特殊模式：测试 / 定位数据集 / 通用对话
        if len(self.intents) == 1:
            self.intent = self.intents[0]
            if self.args.test_latest or self.intent.mode == "test":
                return self._run_test_latest()
            if self.intent.mode == "locate":
                return self._run_locate_dataset(self.intent)
            if self.intent.mode == "chat" or not self.intent.needs_training:
                return self._run_general_chat()

        # 3. 顺序执行每个意图；locate 只扫盘回报，train 走完整训练链
        total = len(self.intents)
        failed_tasks: List[str] = []
        for idx, intent in enumerate(self.intents, 1):
            self.intent = intent
            label = intent.task_type or intent.mode or f"task-{idx}"
            print(f"[TASK {idx}/{total}] {label} ({intent.mode})", flush=True)
            logger.info(
                "========== 开始执行任务 %d/%d: %s mode=%s ==========",
                idx,
                total,
                intent.task_type,
                intent.mode,
            )
            try:
                if intent.mode == "locate":
                    ret = self._run_locate_dataset(intent, task_index=idx, task_total=total)
                elif intent.mode == "test":
                    ret = self._run_test_latest()
                elif intent.mode == "chat" or not intent.needs_training:
                    ret = self._run_general_chat()
                else:
                    ret = self._run_single_training(intent, task_index=idx, task_total=total)
                if ret != 0:
                    logger.error("任务 %d/%d 执行失败，继续执行后续任务", idx, total)
                    failed_tasks.append(label)
                else:
                    logger.info("任务 %d/%d 执行成功，继续执行后续任务", idx, total)
            except Exception as e:
                logger.exception("任务 %d/%d 发生异常，继续执行后续任务: %s", idx, total, e)
                failed_tasks.append(label)
                if self.run_ctx is not None:
                    self.run_ctx.save_exception(exc=e)

        if failed_tasks:
            logger.error("========== 部分任务执行失败 ==========")
            logger.error("失败任务: %s", failed_tasks)
            print(f"[WARNING] 以下任务失败: {', '.join(failed_tasks)}", flush=True)
            return 1

        logger.info("========== 所有任务执行完成 ==========")
        return 0

    def _run_single_training(
        self,
        intent: TrainIntent,
        task_index: int = 1,
        task_total: int = 1,
    ) -> int:
        """执行单个训练任务。"""
        self.intent = intent

        # 4. 语义匹配数据集（或被 --folder 覆盖；多任务模式下 --folder 仅作用于第一个任务）
        use_folder = self.args.folder if task_index == 1 else None
        self.match = self._resolve_match(intent, folder_name=use_folder)
        logger.info("任务 %d/%d 匹配数据集: %s", task_index, task_total, self.match.to_dict())

        # 5. 创建本次 run 的归档上下文
        # 模型输出目录优先级：CLI --output-dir > 意图 output_dir > 提示词正则
        output_dir = self._resolve_output_dir(intent)
        self.run_ctx = RunContext(
            dataset_dir=self.match.dataset_dir,
            project_dir=self.match.folder,
            folder_name=self.match.folder.name,
            output_dir=output_dir,
        )
        self.run_ctx.save_runtime_info({
            "backend_url": self.args.inference_api_url,
            "frontend_url": self.config.get("webbridge", {}).get("frontend_url") if hasattr(self, "config") else None,
            "task_index": task_index,
            "task_total": task_total,
            "output_dir": str(output_dir) if output_dir else None,
        })
        if output_dir:
            logger.info(
                "任务 %d/%d 使用自定义模型输出目录: %s -> run_dir=%s",
                task_index,
                task_total,
                output_dir,
                self.run_ctx.run_dir,
            )
        logger.info("任务 %d/%d 归档目录: %s", task_index, task_total, self.run_ctx.run_dir)
        # 旧 marker + 结构化协议并行一轮，TrainWorker 优先 __AT4__
        print(f"[RUN_DIR] {self.run_ctx.run_dir}", flush=True)
        print(emit_run_dir(self.run_ctx.run_dir), flush=True)
        if output_dir:
            print(f"[OUTPUT_DIR] {output_dir}", flush=True)

        # 6. 第三层：需求分析与判断（复用 UI/环境配置的 LLM，禁止默认 DeepSeek）
        judge = RequirementJudge(llm_client=self.llm)
        task_context = self._build_task_context(intent)
        judge_result = judge.judge(intent, self.match, task_context=task_context)
        logger.info(
            "任务 %d/%d LLM 判断 (protocol=%s model=%s): %s",
            task_index,
            task_total,
            getattr(self.llm, "active_protocol", "?"),
            getattr(self.llm, "model", "?"),
            judge_result.to_dict(),
        )

        # 7. 生成配置
        builder = ConfigBuilder(
            intent,
            self.match,
            project_root=self.project_root,
            adapter=self.args.adapter,
            cdp_port=self.args.cdp_port,
            platform_exe=self.args.platform_exe,
            full_train=self.args.full_train,
            adjustments=judge_result.adjustments,
            llm_client=self.llm,
            record=self.args.record,
            record_dir=self.args.record_dir,
            run_context=self.run_ctx,
        )
        config = builder.build()
        self.config_path = builder.save(self.args.config_out, config=config)
        self.run_ctx.save_config(config)

        if self.args.generate_config_only:
            logger.info("任务 %d/%d --generate-config-only: 已生成配置，不启动训练", task_index, task_total)
            logger.info("配置文件: %s", self.config_path)
            return 0

        # 8. 调用 auto_train2.0.py
        return self._run_auto_train2(task_index=task_index, task_total=task_total)

    @staticmethod
    def _build_task_context(intent: TrainIntent) -> str:
        """为单个任务生成简洁描述，避免多任务串扰；按 mode 措辞，禁止一律写成「训练」。"""
        mode = (intent.mode or "train").strip().lower()
        task = intent.task_type or "未知任务"

        if mode == "locate":
            parts = [f"仅定位 {task} 数据集（不要训练、不要创建项目）"]
        elif mode == "test":
            parts = [f"测试/评估 {task} 模型"]
        elif mode == "chat":
            parts = ["通用对话，无训练动作"]
        elif mode == "dry_run":
            parts = [f"仅生成 {task} 训练配置（dry_run，不启动训练）"]
        else:
            parts = [f"训练一个 {task} 模型"]

        if intent.categories:
            parts.append(f"类别：{', '.join(intent.categories)}")
        if getattr(intent, "dataset_path", ""):
            parts.append(f"训练集路径：{intent.dataset_path}")
        elif intent.dataset_hint:
            parts.append(f"数据集提示：{intent.dataset_hint}")
        if getattr(intent, "output_dir", "") and mode not in {"locate", "chat"}:
            parts.append(f"模型输出目录：{intent.output_dir}")
        if mode in {"train", "dry_run"}:
            if intent.data_augmentation:
                parts.append(f"启用数据增强：{', '.join(intent.data_augmentation)}")
            else:
                parts.append("不启用数据增强")
        return "，".join(parts)

    def _resolve_output_dir(self, intent: TrainIntent) -> Optional[Path]:
        """解析模型输出根目录。

        优先级：
        1. CLI ``--output-dir``
        2. 意图中的 ``output_dir``（由 LLM 填写）
        """
        candidates = [
            getattr(self.args, "output_dir", None),
            getattr(intent, "output_dir", None),
        ]

        for raw in candidates:
            path = resolve_output_dir(raw)
            if path is None:
                continue
            # 避免把输出目录指到训练集自身，导致和数据集目录缠在一起
            try:
                if self.match and path.resolve() == Path(self.match.dataset_dir).resolve():
                    logger.warning(
                        "模型输出目录与训练集相同（%s），已忽略，回退默认 project/work",
                        path,
                    )
                    continue
            except Exception:
                pass
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AutoTrain4Error(f"无法创建模型输出目录: {path} ({exc})") from exc
            if not getattr(intent, "output_dir", ""):
                intent.output_dir = str(path)
            return path
        return None

    def _resolve_match(
        self,
        intent: TrainIntent,
        folder_name: Optional[str] = None,
    ):
        """解析数据集匹配结果。

        优先级：
        1. CLI ``--folder``（支持绝对路径或任一 scan_dir 下相对名）
        2. 意图中的 ``dataset_path`` / 路径型 ``dataset_hint``（来自 LLM 或提示词正则）
        3. 回退到 scan_dirs 多根语义扫描
        """
        enable_nl = not bool(getattr(self.args, "no_nl_locate", False))
        scan_depth = int(getattr(self.args, "scan_depth", 4) or 4)
        resolver = DatasetResolver(
            scan_dirs=self.scan_dirs,
            llm_client=self.llm,
            enable_nl_locate=enable_nl,
            scan_depth=scan_depth,
        )
        if folder_name:
            try:
                return resolver.resolve_folder_arg(folder_name, intent)
            except ValueError as exc:
                raise AutoTrain4Error(str(exc)) from exc

        # 仅补充用户原句中的绝对路径字面量（不做语义关键词推断）
        if not getattr(intent, "dataset_path", "") and intent.raw_prompt:
            for guessed in extract_filesystem_paths(intent.raw_prompt):
                try:
                    p = Path(guessed)
                    if p.exists() and p.is_dir():
                        intent.dataset_path = guessed
                        logger.info("从提示词路径字面量回填数据集路径: %s", guessed)
                        break
                except OSError:
                    continue

        try:
            return resolver.resolve(
                intent,
                auto_threshold=self.args.auto_threshold,
                task_context=self._build_task_context(intent),
            )
        except ValueError as exc:
            raise AutoTrain4Error(str(exc)) from exc

    def _run_auto_train2(
        self,
        task_index: int = 1,
        task_total: int = 1,
    ) -> int:
        """调用 auto_train2.0.py 执行训练。

        多任务批量时，中间任务不保持平台或等待用户交互；最后一个任务
        保留用户指定的 keep-alive / keep-browser-open 行为。
        """
        cmd = self._build_auto_train2_command(task_index, task_total)
        is_batch = task_total > 1
        is_last = task_index >= task_total
        keep_alive = "--keep-alive" in cmd
        keep_browser_open = "--keep-browser-open" in cmd
        reuse = "--reuse" in cmd

        if is_batch:
            logger.info(
                "批量任务 %d/%d 策略: force-new=on, keep-alive=%s, keep-browser-open=%s, reuse=%s",
                task_index,
                task_total,
                keep_alive,
                keep_browser_open,
                reuse,
            )

        logger.info("========== AUTO TRAIN 4.0 -> 2.0 (任务 %d/%d) ==========", task_index, task_total)
        logger.info("启动子进程: %s", " ".join(cmd))

        # 每个任务用独立的关闭信号文件，避免上一个任务残留的 close signal
        # 让下一个任务的 keep-open 等待立刻返回、或误杀流程
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # 批量任务间强制清理可能残留的关闭信号
        if is_batch:
            env.pop("AUTOTRAIN_CLOSE_SIGNAL", None)

        result = subprocess.run(cmd, check=False, env=env)

        cleanup = cleanup_run_artifacts(
            self.run_ctx.run_dir,
            success=result.returncode == 0,
        )
        logger.info(
            "任务 %d/%d 归档清理完成: effective_success=%s, removed=%s",
            task_index,
            task_total,
            cleanup["effective_success"],
            cleanup["removed"],
        )
        for warning in cleanup["warnings"]:
            logger.warning("任务 %d/%d 归档清理提示: %s", task_index, task_total, warning)

        if result.returncode != 0:
            if getattr(self, "run_ctx", None):
                self.run_ctx.save_exception(
                    message=f"auto_train2.0.py 子进程退出码非零: {result.returncode}",
                    exit_code=result.returncode,
                )
            return result.returncode

        # 训练成功后，通过 LLM 询问用户是否生成 inference report / 与 baseline 对比
        # 批量中间任务跳过交互，避免卡住后续任务
        if self.args.post_train and (is_last or not is_batch):
            self._interactive_post_train()
        else:
            if is_batch and not is_last:
                logger.info("批量任务 %d/%d 非末任务，跳过训练后交互", task_index, task_total)
            else:
                logger.info("--no-post-train 已设置，跳过训练后交互")
        return 0

    def _build_auto_train2_command(self, task_index: int, task_total: int) -> List[str]:
        """构造 auto_train2.0.py 命令；只传递子脚本实际支持的参数。"""
        is_batch = task_total > 1
        is_last = task_index >= task_total
        cmd = [
            sys.executable,
            str(self.project_root / "auto_train2.0.py"),
            "--config",
            str(self.config_path),
        ]
        if self.args.force_new or is_batch:
            cmd.append("--force-new")
        if self.args.dry_run:
            cmd.append("--dry-run")

        # auto_train2.0.py 中 keep-alive/reuse 是默认关闭的 store_true 参数，
        # 因此关闭时不要传入不存在的 --no-keep-alive / --no-reuse。
        if bool(self.args.keep_alive) and (is_last or not is_batch):
            cmd.append("--keep-alive")
        if bool(self.args.keep_browser_open) and (is_last or not is_batch):
            cmd.append("--keep-browser-open")
        else:
            cmd.append("--no-keep-browser-open")
        if bool(self.args.reuse) and not is_batch:
            cmd.append("--reuse")
        return cmd

    def _run_test_latest(self) -> int:
        """对最近一次训练的模型，用训练集作为测试集跑推理。"""
        logger.info("========== AUTO TRAIN 4.0 测试最新模型 ==========")

        task_hint = self.intent.task_type if self.intent else ""

        # 先解析 config/work/dvp（与 UI 同一套规则）；baseline 仍走 CLI 的 _effective_*
        ctx, err = resolve_post_train_context(
            latest_config=self.args.latest_config or None,
            task_type=task_hint or "",
            find_baseline=False,
        )
        if err or not ctx:
            logger.error("%s", err or "无法解析训练后上下文")
            return 1

        effective_baseline = self._effective_baseline_model(ctx.get("task_type") or task_hint)
        ctx["baseline_dvp"] = effective_baseline

        dataset_dir = resolve_path(ctx["dataset_dir"])
        work_dir = resolve_path(ctx["work_dir"])
        new_dvp = Path(ctx["new_dvp"])
        task_type = ctx.get("task_type") or task_hint

        # 为本次测试创建独立的 run 目录
        self.run_ctx = RunContext(
            dataset_dir=dataset_dir,
            project_dir=work_dir.parent,
            folder_name=f"{work_dir.parent.name}-test",
        )
        self.run_ctx.save_runtime_info()
        logger.info("本次测试归档目录: %s", self.run_ctx.run_dir)
        print(f"[RUN_DIR] {self.run_ctx.run_dir}", flush=True)
        print(emit_run_dir(self.run_ctx.run_dir), flush=True)

        ctx["run_dir"] = str(self.run_ctx.run_dir)
        ctx["image_dir"] = str(dataset_dir)
        self.config_path = Path(ctx["config_path"])

        logger.info("使用配置: %s", self.config_path)
        logger.info("任务类型: %s", task_type)
        logger.info("新模型: %s", new_dvp)
        logger.info("测试图片目录（复用训练集）: %s", dataset_dir)

        # test-latest：有 baseline 则对比，否则仅 report
        action = "compare" if ctx.get("baseline_dvp") else "report"
        try:
            result = run_post_train_action(
                action,
                ctx,
                inference_url=self.args.inference_api_url,
                progress=not self.args.no_progress,
                report_path=self.args.inference_report_path or None,
            )
            for w in result.get("warnings") or []:
                logger.warning("%s", w)
            logger.info("报告已保存: %s", result.get("report_path"))
        except Exception as e:
            logger.exception("推理评估失败: %s", e)
            return 1
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

    def _run_locate_dataset(
        self,
        intent: TrainIntent,
        task_index: int = 1,
        task_total: int = 1,
    ) -> int:
        """仅定位数据集：匹配路径后回报用户，不生成训练配置、不调用 auto_train2。"""
        logger.info(
            "========== AUTO TRAIN 4.0 定位数据集 (%d/%d) ==========",
            task_index,
            task_total,
        )
        self.intent = intent
        use_folder = self.args.folder if task_index == 1 else None
        try:
            match = self._resolve_match(intent, folder_name=use_folder)
        except Exception as exc:
            logger.error("定位数据集失败: %s", exc)
            msg = (
                f"未能定位到数据集。\n"
                f"任务类型：{intent.task_type or '（未指定）'}\n"
                f"线索：{intent.dataset_hint or intent.dataset_path or '（无）'}\n"
                f"原因：{exc}"
            )
            print(encode_chat_response(msg), flush=True)
            return 1

        self.match = match
        logger.info("定位结果: %s", match.to_dict())

        lines = [
            "已定位到数据集（仅查找，未启动训练）。",
            f"- 任务类型：{intent.task_type or match.task_type or '（未指定）'}",
            f"- 项目目录：{match.folder}",
            f"- 训练集目录：{match.dataset_dir}",
            f"- 图片数量：{match.total_images}",
        ]
        if match.categories:
            lines.append(f"- 检测到的类别：{', '.join(match.categories)}")
        if match.reasoning:
            lines.append(f"- 匹配说明：{match.reasoning}")
        if intent.dataset_hint:
            lines.append(f"- 用户线索：{intent.dataset_hint}")
        lines.append("如需训练，请明确说「训练」并指定任务类型。")
        answer = "\n".join(lines)
        print(f"[DATASET_DIR] {match.dataset_dir}", flush=True)
        print(f"[PROJECT_DIR] {match.folder}", flush=True)
        print(encode_chat_response(answer), flush=True)
        return 0

    def _run_general_chat(self) -> int:
        """通用对话兜底：当用户输入不是训练/测试/定位意图时，用 LLM 做友好回复并记住历史。"""
        logger.info("========== AUTO TRAIN 4.0 通用对话 ==========")
        logger.info("用户输入进入通用对话模式")

        system_msg = (
            "你是 AutoTrain 4.0 训练自动化系统的智能助手。"
            "请根据提供的上下文，友好、简洁地回答用户的问题；不要编造不存在的信息。"
            "若用户其实是在找数据集，可提示他们使用「定位/查找某某训练集」的说法。"
        )

        # 查找最近的几个 run_dir，作为真实上下文
        recent_runs = self._find_recent_run_dirs(max_count=3)
        context_lines = [
            "本项目的训练文件结构如下：",
            "每次训练会在 <项目文件夹>/work/<任务类型-时间戳>/ 目录下生成一个 run 目录。",
            "例如：AI训练平台测试/1-OCR/work/1-OCR-20260716_150004/",
            "成功 run 主要保留 config.json、runtime_info.json、train.log 和 *.dva/*.dvp。",
            "失败 run 还会保留 exception.json、stderr.log、截图、录屏和最后一个检查点用于复现。",
            "",
            "最近的训练运行目录：",
        ]
        if recent_runs:
            for run_dir in recent_runs:
                context_lines.append(f"- {run_dir}")
                for name in ("screen_recording.mp4", "config.json"):
                    path = run_dir / name
                    marker = "✓" if path.exists() else "✗"
                    context_lines.append(f"  {marker} {name}")
        else:
            context_lines.append("- 暂无训练记录")
        run_context = "\n".join(context_lines)

        messages = self._load_chat_history()
        # 确保 system 消息在最前
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_msg})
        # 更新 system 消息内容（如果后续有调整）
        messages[0] = {"role": "system", "content": system_msg}

        user_content = f"{self.args.prompt}\n\n{run_context}"
        messages.append({"role": "user", "content": user_content})

        try:
            answer = self.llm.chat(messages)
            if answer:
                messages.append({"role": "assistant", "content": answer})
                self._save_chat_history(messages)
                logger.info("通用对话回复已生成（%d 字符）", len(answer))
                print(encode_chat_response(answer), flush=True)
            else:
                fallback = "抱歉，我没有理解你的问题，可以再说一次吗？"
                logger.info("通用对话未返回内容，使用兜底回复")
                print(encode_chat_response(fallback), flush=True)
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

    def _find_recent_run_dirs(self, max_count: int = 3) -> List[Path]:
        """查找所有扫描根下最近的几个 run_dir（按修改时间）。"""
        run_dirs: List[Path] = []
        for scan_root in self.scan_dirs:
            if not scan_root.exists():
                continue
            try:
                projects = list(scan_root.iterdir())
            except OSError:
                continue
            for project_dir in projects:
                if not project_dir.is_dir():
                    continue
                work_dir = project_dir / "work"
                if not work_dir.exists():
                    continue
                try:
                    children = list(work_dir.iterdir())
                except OSError:
                    continue
                for run_dir in children:
                    if run_dir.is_dir():
                        run_dirs.append(run_dir)
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return run_dirs[:max_count]

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
        """使用指定路径执行推理评估（委托 autotrain4.post_train）。"""
        run_dir = None
        if getattr(self, "run_ctx", None):
            run_dir = self.run_ctx.run_dir
        ctx = context_from_paths(
            new_dvp=new_dvp,
            image_dir=image_dir,
            task_type=(self.intent.task_type if self.intent else "") or "",
            work_dir=Path(new_dvp).parent,
            run_dir=run_dir,
            baseline_dvp=baseline_model_path,
            config_path=self.config_path,
            dataset_dir=image_dir,
        )
        action = "compare" if baseline_model_path else "report"
        try:
            result = run_post_train_action(
                action,
                ctx,
                inference_url=self.args.inference_api_url,
                progress=not self.args.no_progress,
                report_path=report_path or self.args.inference_report_path or None,
            )
            for w in result.get("warnings") or []:
                logger.warning("%s", w)
            summary = result.get("summary") or {}
            logger.info("========== 推理评估完成 ==========")
            logger.info("报告保存路径: %s", result.get("report_path"))
            logger.info("总图片数: %s", summary.get("total_images"))
            logger.info(
                "成功: %s, 失败: %s",
                summary.get("new_success_count"),
                (summary.get("total_images") or 0) - (summary.get("new_success_count") or 0),
            )
            if result.get("compare"):
                logger.info("baseline 成功: %s", summary.get("baseline_success_count"))
                logger.info(
                    "一致: %s, 不一致: %s",
                    summary.get("match_count"),
                    summary.get("mismatch_count"),
                )
        except Exception as e:
            logger.exception("推理评估失败: %s", e)

    def _interactive_post_train(self) -> None:
        """训练成功后，通过 LLM 询问用户是否生成 inference report 或与 baseline 对比。"""
        logger.info("========== 训练完成，进入后续操作询问 ==========")
        print(emit_stage("train_done"), flush=True)

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
            print("[POST_TRAIN_RESULT] 无法解析需求，不执行后续操作")
            return

        action = decision.get("action", "exit")
        if action == "exit":
            print("[POST_TRAIN_RESULT] 用户结束，不执行后续操作")
            return
        if action == "chat":
            print(f"[POST_TRAIN_RESULT] {decision.get('reply', '收到')}")
            return

        # 4. 执行动作（评估走 post_train）
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



def save_exception_fallback(
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
        failed_dir = Path(__file__).resolve().parent.parent / "failed_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(dataset_dir=failed_dir, folder_name="early-failure")
        if exc is not None:
            ctx.save_exception(exc=exc, exit_code=exit_code)
        else:
            ctx.save_exception(message=message, exit_code=exit_code)
        logger.info("早期失败异常已保存到: %s", ctx.exception_path)
    except Exception:
        pass


