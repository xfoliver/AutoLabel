"""
训练后评估（Post-Train / Test-Latest）统一入口。

CLI（auto_train4.0）与 UI（PostTrainWorker）共用同一套：
  - resolve_post_train_context：从 auto_latest 配置解析模型/数据/baseline
  - run_inference_report / run_baseline_compare：InferenceClient + InferenceEvaluator

不要在 backend 或 4.0 里再复制路径探测与 evaluate 流程。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .common import (
    find_baseline_model,
    find_latest_config,
    find_latest_dvp,
    resolve_path,
)
from .constants import DEFAULT_INFERENCE_URL
from .evaluator import ComparisonReport, InferenceEvaluator
from .inference_client import InferenceClient

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str, str], None]


@dataclass
class PostTrainContext:
    """训练后评估所需路径（UI / CLI 共用字段）。"""

    config_path: str
    task_type: str
    dataset_dir: str
    work_dir: str
    new_dvp: str
    baseline_dvp: Optional[str] = None
    # 报告默认落点；UI 历史行为是 work_dir，CLI test 可指定独立 run_dir
    run_dir: Optional[str] = None
    image_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("run_dir"):
            d["run_dir"] = d["work_dir"]
        if not d.get("image_dir"):
            d["image_dir"] = d["dataset_dir"]
        return d

    @property
    def report_run_dir(self) -> Path:
        return Path(self.run_dir or self.work_dir)

    @property
    def test_image_dir(self) -> Path:
        return Path(self.image_dir or self.dataset_dir)


def resolve_post_train_context(
    *,
    latest_config: Optional[Union[str, Path]] = None,
    task_type: str = "",
    baseline_model: Optional[Union[str, Path]] = None,
    find_baseline: bool = True,
    run_dir: Optional[Union[str, Path]] = None,
    image_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解析训练后评估所需路径。

    Returns:
        (context_dict, error_message)
        context 含: config_path, task_type, dataset_dir, work_dir, run_dir,
                    new_dvp, baseline_dvp, image_dir
    """
    cfg_path = _resolve_config_path(latest_config, task_type)
    if not cfg_path:
        return None, "未找到 config/auto_latest_*.json，无法做训练后评估"

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        return None, f"读取配置失败: {e}"

    project = cfg.get("project", {}) or {}
    resolved_task = (project.get("task_type") or task_type or "").strip()
    dataset_raw = project.get("dataset_dir")
    work_raw = project.get("work_dir")

    if not dataset_raw or not work_raw:
        return None, "配置缺少 project.dataset_dir 或 project.work_dir"

    dataset_path = resolve_path(dataset_raw)
    work_path = resolve_path(work_raw)

    if not dataset_path.exists():
        return None, f"测试集目录不存在: {dataset_path}"
    if not work_path.exists():
        return None, f"工作目录不存在: {work_path}"

    new_dvp = find_latest_dvp(work_path)
    if not new_dvp:
        return None, f"工作目录下未找到 .dvp 模型: {work_path}"

    baseline_path = _resolve_baseline(
        baseline_model=baseline_model,
        task_type=resolved_task,
        find_baseline=find_baseline,
    )

    img = resolve_path(image_dir) if image_dir else dataset_path
    rd = resolve_path(run_dir) if run_dir else work_path

    ctx = PostTrainContext(
        config_path=str(cfg_path),
        task_type=resolved_task,
        dataset_dir=str(dataset_path),
        work_dir=str(work_path),
        new_dvp=str(new_dvp),
        baseline_dvp=str(baseline_path) if baseline_path else None,
        run_dir=str(rd),
        image_dir=str(img),
    )
    return ctx.to_dict(), None


def context_from_paths(
    *,
    new_dvp: Union[str, Path],
    image_dir: Union[str, Path],
    task_type: str = "",
    work_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    baseline_dvp: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """从已知路径构造 context（训练后交互已定位模型时用）。"""
    new_p = resolve_path(new_dvp)
    img_p = resolve_path(image_dir)
    work_p = resolve_path(work_dir) if work_dir else new_p.parent
    run_p = resolve_path(run_dir) if run_dir else work_p
    ds_p = resolve_path(dataset_dir) if dataset_dir else img_p
    base = resolve_path(baseline_dvp) if baseline_dvp else None
    cfg = str(resolve_path(config_path)) if config_path else ""
    return PostTrainContext(
        config_path=cfg,
        task_type=task_type or "",
        dataset_dir=str(ds_p),
        work_dir=str(work_p),
        new_dvp=str(new_p),
        baseline_dvp=str(base) if base and base.exists() else (
            str(baseline_dvp) if baseline_dvp else None
        ),
        run_dir=str(run_p),
        image_dir=str(img_p),
    ).to_dict()


def run_inference_report(
    context: Dict[str, Any],
    *,
    inference_url: str = DEFAULT_INFERENCE_URL,
    progress: bool = True,
    on_progress: Optional[ProgressCallback] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """仅对新模型生成 inference report。"""
    return _run_evaluation(
        context,
        compare=False,
        inference_url=inference_url,
        progress=progress,
        on_progress=on_progress,
        report_path=report_path,
    )


def run_baseline_compare(
    context: Dict[str, Any],
    *,
    inference_url: str = DEFAULT_INFERENCE_URL,
    progress: bool = True,
    on_progress: Optional[ProgressCallback] = None,
    report_path: Optional[Union[str, Path]] = None,
    fallback_to_report_if_no_baseline: bool = True,
) -> Dict[str, Any]:
    """新模型与 baseline 对比；无 baseline 时可回退为仅 report。"""
    return _run_evaluation(
        context,
        compare=True,
        inference_url=inference_url,
        progress=progress,
        on_progress=on_progress,
        report_path=report_path,
        fallback_to_report_if_no_baseline=fallback_to_report_if_no_baseline,
    )


def run_post_train_action(
    action: str,
    context: Dict[str, Any],
    *,
    inference_url: str = DEFAULT_INFERENCE_URL,
    progress: bool = True,
    on_progress: Optional[ProgressCallback] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """按 action 分发：report | compare（及别名）。"""
    act = (action or "report").strip().lower()
    if act in ("compare", "compare_baseline", "baseline"):
        return run_baseline_compare(
            context,
            inference_url=inference_url,
            progress=progress,
            on_progress=on_progress,
            report_path=report_path,
        )
    return run_inference_report(
        context,
        inference_url=inference_url,
        progress=progress,
        on_progress=on_progress,
        report_path=report_path,
    )


def default_report_path(context: Dict[str, Any], *, compare: bool) -> Path:
    """报告默认路径：run_dir 下 inference_report / baseline_compare_report。"""
    run_dir = Path(context.get("run_dir") or context.get("work_dir") or ".")
    name = "baseline_compare_report.json" if compare else "inference_report.json"
    return run_dir / name


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _resolve_config_path(
    latest_config: Optional[Union[str, Path]],
    task_type: str,
) -> Optional[Path]:
    if latest_config:
        p = resolve_path(latest_config)
        if p.exists():
            return p
        logger.warning("指定配置不存在: %s", p)
    return find_latest_config(task_type or "")


def _resolve_baseline(
    *,
    baseline_model: Optional[Union[str, Path]],
    task_type: str,
    find_baseline: bool,
) -> Optional[Path]:
    if baseline_model:
        p = resolve_path(baseline_model)
        if p.exists():
            return p
        logger.warning("指定 baseline 不存在: %s", p)
        # 显式指定但无效时，不再自动找，避免 silent 换模型
        return None
    if find_baseline and task_type:
        return find_baseline_model(task_type)
    return None


def _run_evaluation(
    context: Dict[str, Any],
    *,
    compare: bool,
    inference_url: str,
    progress: bool,
    on_progress: Optional[ProgressCallback],
    report_path: Optional[Union[str, Path]],
    fallback_to_report_if_no_baseline: bool = True,
) -> Dict[str, Any]:
    """执行评估并返回结构化结果（不抛到 UI 线程外时由调用方处理异常）。"""
    new_dvp = Path(context["new_dvp"])
    image_dir = Path(context.get("image_dir") or context["dataset_dir"])
    baseline = context.get("baseline_dvp")
    task_type = context.get("task_type") or "model"

    effective_compare = bool(compare)
    warnings: list[str] = []
    if effective_compare and not baseline:
        msg = (
            f"未找到任务「{task_type}」对应的 baseline 模型，"
            + ("已改为仅生成 inference report" if fallback_to_report_if_no_baseline else "无法对比")
        )
        warnings.append(msg)
        logger.warning(msg)
        if fallback_to_report_if_no_baseline:
            effective_compare = False
        else:
            raise FileNotFoundError(msg)

    if not new_dvp.exists():
        raise FileNotFoundError(f"模型文件不存在: {new_dvp}")
    if not image_dir.exists():
        raise FileNotFoundError(f"测试图片目录不存在: {image_dir}")

    out_path = (
        resolve_path(report_path)
        if report_path
        else default_report_path(context, compare=effective_compare)
    )

    start = time.time()
    client = InferenceClient(base_url=inference_url, progress=progress and on_progress is None)
    evaluator = InferenceEvaluator(client=client)
    if on_progress is not None:
        # 有外部进度回调时关掉内置 tqdm，避免双进度
        evaluator.progress = False
        client.progress = False

    report: ComparisonReport = evaluator.evaluate(
        new_model_path=new_dvp,
        image_dir=image_dir,
        baseline_model_path=baseline if effective_compare else None,
        on_progress=on_progress,
    )
    report.save(out_path)
    elapsed = round(time.time() - start, 2)
    report_dict = report.to_dict()
    report_dict["_elapsed_sec"] = elapsed
    summary = report_dict.get("summary") or {}

    action_name = "compare" if effective_compare else "report"
    logger.info("========== 推理评估完成 ==========")
    logger.info("动作: %s", action_name)
    logger.info("报告: %s", out_path)
    logger.info(
        "总图=%s 成功=%s baseline成功=%s 一致=%s 不一致=%s 耗时=%ss",
        summary.get("total_images"),
        summary.get("new_success_count"),
        summary.get("baseline_success_count"),
        summary.get("match_count"),
        summary.get("mismatch_count"),
        elapsed,
    )

    return {
        "ok": True,
        "action": action_name,
        "report_path": str(out_path),
        "report": report_dict,
        "summary": summary,
        "elapsed_sec": elapsed,
        "warnings": warnings,
        "compare": effective_compare,
        "new_dvp": str(new_dvp),
        "image_dir": str(image_dir),
        "baseline_dvp": baseline if effective_compare else None,
    }
