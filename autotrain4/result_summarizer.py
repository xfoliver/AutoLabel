"""
将训练 / 质检 / 评估结构化结果整理成助手可读中文摘要。

设计：
- 模板摘要始终可用（不依赖 LLM、不阻塞）
- 可选 LLM 润色：失败时回退模板
- 输入均为 dict / 简单标量，便于 PyQt 与 CLI 复用
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_MAX_ISSUES = 8
_MAX_HIST = 8
_MAX_MISMATCH_SAMPLES = 5
_MAX_LLM_CHARS = 3500


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def summarize_quality(report: Dict[str, Any]) -> str:
    """数据质检报告 → 助手摘要（纯模板）。"""
    if not isinstance(report, dict):
        return "数据质检已结束，但报告格式异常，无法生成摘要。"

    passed = bool(report.get("pass"))
    grade = report.get("grade") or "?"
    try:
        score = float(report.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    task = str(report.get("task_type") or "未知任务")
    dataset = str(report.get("dataset_dir") or "")
    summary = report.get("summary") or {}
    stats = report.get("stats") or {}
    issues = report.get("issues") or []
    hist = report.get("label_histogram") or {}

    n_img = summary.get("n_images", stats.get("n_images", 0))
    n_err = summary.get("blocking_issues", 0)
    n_warn = summary.get("warnings", 0)
    n_cls = summary.get("n_classes", len(hist) if isinstance(hist, dict) else 0)
    imb = summary.get("imbalance_ratio", stats.get("imbalance_ratio"))
    label_mode = summary.get("label_mode") or (
        "folder" if task == "分类" else "json"
    )
    is_cls = label_mode == "folder" or task == "分类"
    pair = summary.get("pair_rate", stats.get("pair_rate"))

    verdict = "通过" if passed else "未通过"
    lines: List[str] = [
        f"【数据质检】{verdict} · 等级 {grade} · 得分 {score:.2f}",
        f"· 任务：{task}",
    ]
    if dataset:
        lines.append(f"· 目录：{_short_path(dataset)}")
    if is_cls:
        lines.append(f"· 规模：图片 {n_img} · 类别 {n_cls} · 标注方式 文件夹分类")
    else:
        pair_txt = f"{float(pair):.0%}" if isinstance(pair, (int, float)) else "N/A"
        n_json = summary.get("n_jsons", stats.get("n_jsons", "—"))
        lines.append(
            f"· 规模：图片 {n_img} · JSON {n_json} · 配对率 {pair_txt} · 类别 {n_cls}"
        )
    if isinstance(imb, (int, float)) and n_cls:
        lines.append(f"· 类别不平衡比 max/min：{float(imb):.1f}")
    lines.append(f"· 问题：阻断 {n_err} · 警告 {n_warn}")

    issue_lines = _format_issues(issues, limit=_MAX_ISSUES)
    if issue_lines:
        lines.append("· 主要问题：")
        lines.extend(issue_lines)
    elif passed:
        lines.append("· 未发现阻断或警告项，数据集训练就绪性良好。")

    if isinstance(hist, dict) and hist:
        top = list(hist.items())[:_MAX_HIST]
        hist_txt = "、".join(f"{k}({v})" for k, v in top)
        extra = f" 等共 {len(hist)} 类" if len(hist) > _MAX_HIST else ""
        lines.append(f"· 类别分布（节选）：{hist_txt}{extra}")

    lines.append("· 建议：" + _quality_advice(passed, n_err, n_warn, issues, is_cls, pair))
    return "\n".join(lines)


def summarize_train(
    *,
    success: bool,
    training_completed: bool = False,
    elapsed_sec: Optional[float] = None,
    run_dir: Optional[Union[str, Path]] = None,
    exit_code: Optional[int] = None,
    task_type: Optional[str] = None,
    extra_note: Optional[str] = None,
) -> str:
    """训练 / 子进程任务结束 → 助手摘要（纯模板）。

    会尽量从 run_dir 读取 config / exception / 模型文件名。
    """
    meta = _load_run_meta(run_dir)
    task = task_type or meta.get("task_type") or "未知"
    dataset = meta.get("dataset_dir") or ""
    model_name = meta.get("dvp_name") or ""
    exc_msg = meta.get("exception_message") or ""
    exc_type = meta.get("exception_type") or ""

    elapsed_txt = _fmt_elapsed(elapsed_sec)

    if training_completed and success:
        lines = [
            f"【训练完成】成功{elapsed_txt}",
            f"· 任务：{task}",
        ]
        if dataset:
            lines.append(f"· 数据：{_short_path(dataset)}")
        if model_name:
            lines.append(f"· 模型：{model_name}")
        if run_dir:
            lines.append(f"· 产物目录：{_short_path(str(run_dir))}")
        lines.append("· 建议：可生成推理报告，或与 Baseline 对比；也可直接结束。")
        if extra_note:
            lines.append(f"· 备注：{extra_note}")
        return "\n".join(lines)

    if not success:
        code_txt = f" · 退出码 {exit_code}" if exit_code is not None else ""
        lines = [
            f"【训练失败】未完成{elapsed_txt}{code_txt}",
            f"· 任务：{task}",
        ]
        if dataset:
            lines.append(f"· 数据：{_short_path(dataset)}")
        if run_dir:
            lines.append(f"· 诊断目录：{_short_path(str(run_dir))}")
        if exc_type or exc_msg:
            head = exc_msg.strip().splitlines()[0] if exc_msg else ""
            if len(head) > 160:
                head = head[:157] + "…"
            detail = f"{exc_type}: {head}" if exc_type and head else (exc_type or head)
            if detail:
                lines.append(f"· 原因摘要：{detail}")
        lines.append(
            "· 建议：打开 run 目录查看 exception.json / 截图 / 录屏；"
            "确认数据集路径、平台 EXE 与 LLM 配置后重试。"
        )
        if extra_note:
            lines.append(f"· 备注：{extra_note}")
        return "\n".join(lines)

    # 成功但非完整训练（chat / locate / dry-run / test 等）
    lines = [
        f"【任务结束】已完成{elapsed_txt}",
        f"· 识别任务类型：{task}" if task and task != "未知" else "· 本次未启动完整模型训练流程",
    ]
    if run_dir:
        lines.append(f"· 相关目录：{_short_path(str(run_dir))}")
    if model_name:
        lines.append(f"· 相关模型：{model_name}")
    lines.append("· 说明：若上方已有助手答复，请以对话内容为准；本条为流程收尾摘要。")
    if extra_note:
        lines.append(f"· 备注：{extra_note}")
    return "\n".join(lines)


def summarize_eval(
    report: Optional[Dict[str, Any]] = None,
    *,
    success: bool = True,
    report_path: Optional[Union[str, Path]] = None,
    error_message: Optional[str] = None,
    action: Optional[str] = None,
    task_type: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
) -> str:
    """训练后推理 / Baseline 对比 → 助手摘要。"""
    elapsed_txt = _fmt_elapsed(elapsed_sec)

    if not success:
        lines = [
            f"【训练后评估失败】{elapsed_txt}".strip(),
        ]
        if task_type:
            lines.append(f"· 任务：{task_type}")
        if error_message:
            head = str(error_message).strip().splitlines()[0]
            if len(head) > 180:
                head = head[:177] + "…"
            lines.append(f"· 原因：{head}")
        lines.append(
            "· 建议：确认 DLCV Test 推理服务（默认 http://127.0.0.1:9890）已启动，"
            "且模型 .dvp 与测试图目录有效后重试。"
        )
        return "\n".join(lines)

    data = report
    if data is None and report_path:
        data = _read_json(report_path)
    if not isinstance(data, dict):
        path_txt = _short_path(str(report_path)) if report_path else ""
        return (
            f"【训练后评估完成】{elapsed_txt}\n"
            f"· 报告已保存{('：' + path_txt) if path_txt else ''}\n"
            "· 未能解析报告内容，请直接打开 JSON 查看。"
        )

    summary = data.get("summary") or {}
    # 兼容报告本身就是扁平 summary 的情况
    if not summary and any(k in data for k in ("total_images", "match_count", "new_success_count")):
        summary = data

    total = int(summary.get("total_images") or 0)
    new_ok = int(summary.get("new_success_count") or 0)
    new_fail = max(0, total - new_ok)
    base_ok = summary.get("baseline_success_count")
    match_c = summary.get("match_count")
    mismatch_c = summary.get("mismatch_count")
    base_model = data.get("baseline_model_path") or ""
    # 仅在有真实 baseline 路径、或 action 明确为 compare 时视为对比报告。
    # 注意：inference-only 报告里也可能带 match_count=0 占位，不能据此误判。
    compare = bool(base_model) or (action == "compare")
    if action == "report":
        compare = bool(base_model)

    new_model = data.get("new_model_path") or ""
    image_dir = data.get("image_dir") or ""

    title = "【Baseline 对比完成】" if compare else "【推理报告完成】"
    lines = [f"{title}{elapsed_txt}".strip()]
    if task_type:
        lines.append(f"· 任务：{task_type}")
    if new_model:
        lines.append(f"· 新模型：{Path(str(new_model)).name}")
    if compare and base_model:
        lines.append(f"· Baseline：{Path(str(base_model)).name}")
    if image_dir:
        lines.append(f"· 测试图：{_short_path(str(image_dir))}")

    if total > 0:
        rate = new_ok / total
        lines.append(f"· 新模型推理：成功 {new_ok}/{total}（{rate:.0%}）· 失败 {new_fail}")
    else:
        lines.append("· 新模型推理：未统计到图片（total_images=0）")

    if compare and match_c is not None and mismatch_c is not None:
        compared = int(match_c) + int(mismatch_c)
        if compared > 0:
            lines.append(
                f"· 与 Baseline：一致 {match_c} · 不一致 {mismatch_c}"
                f"（一致率 {int(match_c) / compared:.0%}）"
            )
        else:
            lines.append(f"· 与 Baseline：一致 {match_c} · 不一致 {mismatch_c}")
        if base_ok is not None and total > 0:
            lines.append(f"· Baseline 推理成功：{base_ok}/{total}")

        mismatches = data.get("mismatched_images") or []
        if isinstance(mismatches, list) and mismatches:
            names = []
            for item in mismatches[:_MAX_MISMATCH_SAMPLES]:
                if isinstance(item, dict):
                    names.append(str(item.get("image") or "?"))
                else:
                    names.append(str(item))
            more = (
                f" 等 {len(mismatches)} 张"
                if len(mismatches) > _MAX_MISMATCH_SAMPLES
                else ""
            )
            lines.append(f"· 不一致样例：{'、'.join(names)}{more}")

    if report_path:
        lines.append(f"· 报告路径：{_short_path(str(report_path))}")

    lines.append("· 建议：" + _eval_advice(compare, total, new_ok, match_c, mismatch_c))
    return "\n".join(lines)


def polish_with_llm(
    template_text: str,
    *,
    kind: str,
    context: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    temperature: float = 0.2,
) -> Optional[str]:
    """用 LLM 把模板摘要润色成更自然的助手回复；失败返回 None。"""
    if not template_text or not str(template_text).strip():
        return None
    client = llm_client
    if client is None:
        try:
            from autotrain4.llm import LLMClient

            client = LLMClient(timeout=45)
        except Exception:
            return None
    if not getattr(client, "available", False):
        return None

    compact = _compact_context(kind, context or {})
    system = (
        "你是 DLCV 视觉训练工作台的中文助手。"
        "根据给定的结构化结果与草稿摘要，写一段简洁专业的中文总结给用户。"
        "要求：\n"
        "1. 保留关键数字与结论（通过/失败、得分、成功数、一致率等），不要编造未提供的指标。\n"
        "2. 指出最重要的 1～3 个问题或亮点，并给出可执行下一步。\n"
        "3. 使用短段落或项目符号，总长度控制在 180～350 字。\n"
        "4. 不要输出 Markdown 代码块，不要说自己是 AI。"
    )
    user = (
        f"场景：{kind}\n\n"
        f"【草稿摘要】\n{template_text}\n\n"
        f"【结构化要点】\n{compact}\n\n"
        "请输出最终助手回复："
    )
    try:
        answer = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
    except Exception as e:
        logger.warning("摘要 LLM 润色失败: %s", e)
        return None

    if not answer or not str(answer).strip():
        return None
    text = str(answer).strip()
    # 去掉意外代码围栏
    if text.startswith("```"):
        parts = text.split("\n")
        if parts and parts[0].startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip() == "```":
            parts = parts[:-1]
        text = "\n".join(parts).strip()
    return text or None


def summarize_quality_with_optional_llm(
    report: Dict[str, Any],
    *,
    use_llm: bool = False,
    llm_client: Any = None,
) -> str:
    base = summarize_quality(report)
    if not use_llm:
        return base
    polished = polish_with_llm(
        base, kind="data_quality", context={"report": _trim_quality(report)}, llm_client=llm_client
    )
    return polished or base


def summarize_train_with_optional_llm(
    *,
    use_llm: bool = False,
    llm_client: Any = None,
    **kwargs: Any,
) -> str:
    base = summarize_train(**kwargs)
    if not use_llm:
        return base
    polished = polish_with_llm(
        base,
        kind="train_finished",
        context={
            "success": kwargs.get("success"),
            "training_completed": kwargs.get("training_completed"),
            "elapsed_sec": kwargs.get("elapsed_sec"),
            "run_dir": str(kwargs.get("run_dir") or ""),
            "exit_code": kwargs.get("exit_code"),
            "meta": _load_run_meta(kwargs.get("run_dir")),
        },
        llm_client=llm_client,
    )
    return polished or base


def summarize_eval_with_optional_llm(
    report: Optional[Dict[str, Any]] = None,
    *,
    use_llm: bool = False,
    llm_client: Any = None,
    **kwargs: Any,
) -> str:
    base = summarize_eval(report, **kwargs)
    if not use_llm:
        return base
    ctx_report = report
    if ctx_report is None and kwargs.get("report_path"):
        ctx_report = _read_json(kwargs["report_path"])
    polished = polish_with_llm(
        base,
        kind="post_train_eval",
        context={
            "success": kwargs.get("success", True),
            "action": kwargs.get("action"),
            "task_type": kwargs.get("task_type"),
            "report": _trim_eval(ctx_report) if isinstance(ctx_report, dict) else {},
            "error_message": kwargs.get("error_message"),
        },
        llm_client=llm_client,
    )
    return polished or base


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _short_path(path: str, keep: int = 64) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if len(text) <= keep:
        return text
    return "…" + text[-(keep - 1) :]


def _fmt_elapsed(elapsed_sec: Optional[float]) -> str:
    if elapsed_sec is None:
        return ""
    try:
        sec = float(elapsed_sec)
    except (TypeError, ValueError):
        return ""
    if sec < 0:
        return ""
    if sec < 60:
        return f" · 耗时 {sec:.1f}s"
    minutes = int(sec // 60)
    rem = sec - minutes * 60
    if minutes < 60:
        return f" · 耗时 {minutes}分{rem:.0f}秒"
    hours = minutes // 60
    minutes = minutes % 60
    return f" · 耗时 {hours}小时{minutes}分"


def _format_issues(issues: Any, limit: int = _MAX_ISSUES) -> List[str]:
    if not isinstance(issues, list) or not issues:
        return []
    # error > warn > info
    order = {"error": 0, "warn": 1, "info": 2}
    sorted_issues = sorted(
        [i for i in issues if isinstance(i, dict)],
        key=lambda x: (order.get(str(x.get("severity")), 9), str(x.get("code") or "")),
    )
    lines: List[str] = []
    for iss in sorted_issues[:limit]:
        sev = str(iss.get("severity") or "info").upper()
        code = iss.get("code") or ""
        msg = str(iss.get("message") or "").strip()
        if len(msg) > 100:
            msg = msg[:97] + "…"
        sug = str(iss.get("suggestion") or "").strip()
        head = f"  - [{sev}] {code}: {msg}" if code else f"  - [{sev}] {msg}"
        lines.append(head)
        if sug:
            if len(sug) > 80:
                sug = sug[:77] + "…"
            lines.append(f"    建议：{sug}")
    rest = len(sorted_issues) - limit
    if rest > 0:
        lines.append(f"  - …另有 {rest} 项，详见质检面板")
    return lines


def _quality_advice(
    passed: bool,
    n_err: int,
    n_warn: int,
    issues: Any,
    is_cls: bool,
    pair: Any,
) -> str:
    if passed and n_warn == 0:
        return "可直接用于开训；若类别较少建议再核对标注一致性。"
    if passed:
        return "无阻断项，可开训；建议先处理警告项以降低训练风险。"
    tips: List[str] = []
    codes = {
        str(i.get("code") or "")
        for i in (issues or [])
        if isinstance(i, dict) and i.get("severity") == "error"
    }
    if any("pair" in c for c in codes) or (
        not is_cls and isinstance(pair, (int, float)) and pair < 0.9
    ):
        tips.append("优先修复图与 JSON 配对")
    if any("corrupt" in c or "image" in c for c in codes):
        tips.append("清理无法解码的坏图")
    if any("empty" in c or "label" in c for c in codes):
        tips.append("补全空标注")
    if any("imbalance" in c or "tail" in c or "class" in c for c in codes):
        tips.append("补充少类样本或调整采样")
    if not tips:
        tips.append("按质检面板中的 error 项逐条修复")
    tips.append("修复后重新质检再开训")
    return "；".join(tips) + "。"


def _eval_advice(
    compare: bool,
    total: int,
    new_ok: int,
    match_c: Any,
    mismatch_c: Any,
) -> str:
    if total <= 0:
        return "检查测试图片目录是否为空或路径是否正确。"
    if new_ok == 0:
        return "新模型全部推理失败，检查推理服务、模型文件与图片格式。"
    if not compare:
        if new_ok / total >= 0.95:
            return "推理成功率较高，可抽样人工复核后考虑上线或设为 Baseline。"
        return "关注失败图片，必要时回到标注/训练配置排查。"
    try:
        mc, mm = int(match_c or 0), int(mismatch_c or 0)
    except (TypeError, ValueError):
        return "查看对比报告中的不一致样例，决定是否替换 Baseline。"
    compared = mc + mm
    if compared <= 0:
        return "未形成有效对比对，请确认 Baseline 与测试集。"
    rate = mc / compared
    if rate >= 0.9 and new_ok / total >= 0.9:
        return "与 Baseline 高度一致且推理稳定，可考虑晋升或保持当前 Baseline。"
    if mm > mc:
        return "不一致较多：请抽查 mismatched 样例，确认是提升还是回退后再决定是否替换 Baseline。"
    return "存在部分不一致，建议结合业务指标与难例再决定模型取舍。"


def _load_run_meta(run_dir: Optional[Union[str, Path]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "task_type": None,
        "dataset_dir": None,
        "dvp_name": None,
        "exception_message": None,
        "exception_type": None,
    }
    if not run_dir:
        return out
    root = Path(str(run_dir))
    if not root.is_dir():
        return out

    for name in ("config.json", "runtime_info.json"):
        data = _read_json(root / name)
        if not isinstance(data, dict):
            continue
        project = data.get("project") if isinstance(data.get("project"), dict) else {}
        out["task_type"] = out["task_type"] or project.get("task_type") or data.get("task_type")
        out["dataset_dir"] = (
            out["dataset_dir"]
            or project.get("dataset_dir")
            or data.get("dataset_dir")
        )

    try:
        dvps = sorted(root.glob("*.dvp"), key=lambda p: p.stat().st_mtime, reverse=True)
        if dvps:
            out["dvp_name"] = dvps[0].name
    except Exception:
        pass

    exc = _read_json(root / "exception.json")
    if isinstance(exc, dict):
        out["exception_type"] = (
            exc.get("type") or exc.get("exc_type") or exc.get("error_type")
        )
        msg = (
            exc.get("message")
            or exc.get("error")
            or exc.get("msg")
            or exc.get("detail")
        )
        if not msg and isinstance(exc.get("exception"), dict):
            msg = exc["exception"].get("message") or exc["exception"].get("type")
        if not msg:
            # 常见：整份就是 traceback 文本字段
            msg = exc.get("traceback") or exc.get("stack")
            if isinstance(msg, list):
                msg = "\n".join(str(x) for x in msg[:5])
        out["exception_message"] = str(msg) if msg else json.dumps(exc, ensure_ascii=False)[:300]

    return out


def _read_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    try:
        p = Path(path)
        if not p.is_file():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _trim_quality(report: Dict[str, Any]) -> Dict[str, Any]:
    issues = report.get("issues") or []
    if isinstance(issues, list):
        slim_issues = []
        for iss in issues[:12]:
            if not isinstance(iss, dict):
                continue
            slim_issues.append(
                {
                    "code": iss.get("code"),
                    "severity": iss.get("severity"),
                    "message": iss.get("message"),
                    "count": iss.get("count"),
                    "suggestion": iss.get("suggestion"),
                    "samples": (iss.get("samples") or [])[:3],
                }
            )
    else:
        slim_issues = []
    hist = report.get("label_histogram") or {}
    if isinstance(hist, dict):
        hist = dict(list(hist.items())[:15])
    return {
        "pass": report.get("pass"),
        "grade": report.get("grade"),
        "score": report.get("score"),
        "task_type": report.get("task_type"),
        "dataset_dir": report.get("dataset_dir"),
        "summary": report.get("summary") or {},
        "issues": slim_issues,
        "label_histogram": hist,
    }


def _trim_eval(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    summary = report.get("summary") or {}
    mismatches = report.get("mismatched_images") or []
    slim_mis = []
    if isinstance(mismatches, list):
        for item in mismatches[:8]:
            if isinstance(item, dict):
                slim_mis.append({"image": item.get("image")})
            else:
                slim_mis.append({"image": str(item)})
    return {
        "new_model_path": report.get("new_model_path"),
        "baseline_model_path": report.get("baseline_model_path"),
        "image_dir": report.get("image_dir"),
        "summary": {
            k: summary.get(k)
            for k in (
                "total_images",
                "new_success_count",
                "baseline_success_count",
                "match_count",
                "mismatch_count",
            )
            if k in summary or True
        },
        "mismatched_images": slim_mis,
    }


def _compact_context(kind: str, context: Dict[str, Any]) -> str:
    try:
        text = json.dumps(context, ensure_ascii=False, default=str)
    except Exception:
        text = str(context)
    if len(text) > _MAX_LLM_CHARS:
        text = text[: _MAX_LLM_CHARS - 1] + "…"
    return f"kind={kind}\n{text}"
