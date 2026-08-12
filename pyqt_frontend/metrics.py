"""训练指标：从 prompt / 日志刷新 UI。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def update_metrics_from_prompt(window: Any) -> None:
    """根据 prompt 内容更新任务类型和预计耗时指标；清空输入时不重置。"""
    text = window.prompt_input.toPlainText().strip()
    if not text:
        return

    # 任务类型识别
    upper = text.upper()
    task_types = ["DETR", "ISEG", "SSEG", "OCR", "CLS", "DET"]
    detected = None
    for t in task_types:
        if t in upper:
            detected = t
            break
    if detected:
        window.metric_labels["task_type"].setText(detected)
        if hasattr(window, "data_quality_panel"):
            window.data_quality_panel.set_task_hint(detected)

    # 预计耗时：根据类型给一个经验值
    estimates = {
        "OCR": "约 3 分钟",
        "CLS": "约 4 分钟",
        "DET": "约 6 分钟",
        "DETR": "约 8 分钟",
        "ISEG": "约 10 分钟",
        "SSEG": "约 10 分钟",
    }
    if detected:
        window.metric_labels["estimate"].setText(estimates.get(detected, "-"))


def update_metrics_from_log(window: Any, line: str) -> None:
    """根据日志输出更新任务类型、匹配数据集、预计耗时指标。"""
    # 任务类型：优先精确匹配 task_type=
    m = re.search(r"task_type\s*=\s*([A-Z]+)", line)
    if m:
        window.metric_labels["task_type"].setText(m.group(1))
        if hasattr(window, "data_quality_panel"):
            window.data_quality_panel.set_task_hint(m.group(1))

    # 任务类型：从日志里的中文任务描述推断
    task_type_map = {
        "分类": "CLS",
        "目标检测": "DET",
        "实例分割": "ISEG",
        "语义分割": "SSEG",
        "文字识别": "OCR",
        "文本识别": "OCR",
        "旋转框": "ROTATE",
        "关键点": "KPT",
    }
    for cn, en in task_type_map.items():
        if cn in line:
            window.metric_labels["task_type"].setText(en)
            if hasattr(window, "data_quality_panel"):
                window.data_quality_panel.set_task_hint(en)
            break

    # 任务类型：从项目名 / URL / project_id 里的类型代码推断
    m = re.search(r"[/-](\d-)?(OCR|CLS|DET|DETR|ISEG|SSEG|ROTATE)[/_-]", line.upper())
    if m:
        window.metric_labels["task_type"].setText(m.group(2))
        if hasattr(window, "data_quality_panel"):
            window.data_quality_panel.set_task_hint(m.group(2))

    # 匹配数据集：从 dataset_dir 路径里提取
    m = re.search(r"dataset_dir\s*=\s*(.+?)(?:\s|$)", line)
    if m:
        path = Path(m.group(1).strip().strip('"'))
        if path.name == "训练集" or path.name == "train":
            ds_name = path.parent.name
        else:
            ds_name = path.name
        window.metric_labels["dataset"].setText(ds_name)
        if hasattr(window, "data_quality_panel"):
            # 优先完整路径，便于质检直接扫盘
            window.data_quality_panel.set_dataset_hint(str(path))

    # 匹配数据集：从 "创建项目: X-CLS-..." 提取
    m = re.search(r"创建项目[:：]\s*([\w\-]+)", line)
    if m:
        window.metric_labels["dataset"].setText(m.group(1))
        if hasattr(window, "data_quality_panel"):
            window.data_quality_panel.set_dataset_hint(m.group(1))

    # 训练进度 -> 更新预计剩余时间
    m = re.search(r"训练进度[:：]\s*(\d+(?:\.\d+)?)%", line)
    if m:
        pct = float(m.group(1))
        if pct > 0 and pct < 100:
            total_sec = window._estimate_total_seconds()
            remain = int(total_sec * (100 - pct) / 100)
            window.metric_labels["estimate"].setText(f"剩余约 {remain} 秒")
        elif pct >= 100:
            window.metric_labels["estimate"].setText("即将完成")


def estimate_total_seconds(window: Any) -> int:
    """根据当前任务类型估算总训练时长（秒）。"""
    task_type = window.metric_labels["task_type"].text()
    mapping = {
        "OCR": 180,
        "CLS": 240,
        "DET": 360,
        "DETR": 480,
        "ISEG": 600,
        "SSEG": 600,
    }
    return mapping.get(task_type, 300)
