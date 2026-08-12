"""LLM 领域任务：意图解析、数据集匹配、类别对齐（与传输层分离）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class SupportsChat(Protocol):
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]: ...


def safe_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    logger.warning("LLM 返回内容 JSON 解析失败: %s", text[:200])
    return None


_INTENT_SYSTEM = """你是 DLCV 视觉训练工作台的意图理解助手。
你全权理解用户自然语言；下游代码不再用关键词猜意图。
先判断用户真正想做什么，再填 JSON。

【intent_action】
- train：明确要训练/开训/启动训练。
- dry_run：只要生成配置/演练、不真正开训。
- test：测试/评估/验证已有模型。
- locate：只查找/定位数据集或路径，不要训练。
- chat：闲聊或纯概念问答。

【needs_training】仅 train 为 true；test/locate/chat/dry_run 为 false。
不要因为提到数据集、任务类型、盘符就设为 true。

【tasks】
- train/test/locate/dry_run：1 条或多条。
- chat：tasks 必须为 []。

【返回 JSON，不要附加说明】
{
  "needs_training": true,
  "intent_action": "train",
  "tasks": [
    {
      "task_type": "分类/目标检测/OCR/实例分割/语义分割/旋转框检测/DETR 或空字符串",
      "categories": [],
      "dataset_hint": "用户提到的项目名或数据描述摘要",
      "dataset_path": "用户写出的完整路径字面量，否则空字符串",
      "output_dir": "模型输出目录完整路径，否则空字符串",
      "mode": "train|dry_run|test|locate|chat",
      "needs_training": true,
      "data_analysis": true,
      "mentions_augmentation": false,
      "data_augmentation": [],
      "baseline_model": "",
      "wants_baseline": false,
      "location_drives": ["Z:\\\\"],
      "location_path_fragments": ["zzh"],
      "location_name_keywords": ["易格技术"]
    }
  ]
}

【字段细则】
1. task_type 映射：cls→分类，det→目标检测，OCR→OCR，iseg→实例分割，sseg→语义分割，rotate→旋转框检测，detr→DETR。
2. categories 禁止臆造。
3. dataset_path：仅用户写出的完整路径；口语位置不要编路径，填 location_* 与 dataset_hint。
4. location_drives：用户提到的盘符，统一成 Z:\\\\ 这种形式；没有则 []。
5. location_path_fragments：目录名片段（如 zzh、data）；没有则 []。
6. location_name_keywords：项目/客户/场景名（如 易格技术、手机检测）；没有则 []。
7. output_dir：用户明确指定输出目录时填，否则空字符串。
8. 数据增强：仅用户明确要求时 mentions_augmentation=true，data_augmentation 从下列选：调整图像大小、随机比例缩放、随机翻转、随机固定角度旋转、随机裁剪、随机调整亮度、随机调整对比度、随机JPG编码；否则 false 与 []。
9. baseline_model 仅明确 .dvp 路径；只说对比则 wants_baseline=true 且路径为空。
10. 多任务拆成多个 tasks；一件事一个元素。
11. 拿不准且无开训意愿：locate 或 chat，切勿默认 train。
"""


def normalize_intent_payload(parsed: Any) -> Optional[Dict[str, Any]]:
    """统一成 {needs_training, intent_action?, tasks}。"""
    if not parsed:
        return None
    if isinstance(parsed, list):
        return {
            "needs_training": True,
            "intent_action": "train",
            "tasks": parsed,
        }
    if not isinstance(parsed, dict):
        return None

    if "tasks" not in parsed:
        task = dict(parsed)
        action = str(
            task.get("mode")
            or task.get("intent_action")
            or ("train" if task.get("needs_training", True) else "chat")
        ).strip().lower()
        needs = bool(task.get("needs_training", action in {"train", "training"}))
        return {
            "needs_training": needs,
            "intent_action": action if action in {"train", "test", "locate", "chat"} else (
                "train" if needs else "chat"
            ),
            "tasks": [task],
        }

    tasks = parsed.get("tasks")
    if tasks is None:
        tasks = []
    if not isinstance(tasks, list):
        return None

    intent_action = str(parsed.get("intent_action") or "").strip().lower()
    if "needs_training" in parsed:
        needs_training = bool(parsed.get("needs_training"))
    elif intent_action == "train":
        needs_training = True
    elif intent_action in {"test", "locate", "chat"}:
        needs_training = False
    else:
        needs_training = any(
            isinstance(t, dict)
            and str(t.get("mode", "train")).lower() in {"train", "training"}
            and bool(t.get("needs_training", True))
            for t in tasks
        ) if tasks else False

    if not intent_action:
        if needs_training:
            intent_action = "train"
        elif any(
            isinstance(t, dict) and str(t.get("mode", "")).lower() in {"locate", "find", "search"}
            for t in tasks
        ):
            intent_action = "locate"
        elif any(
            isinstance(t, dict) and str(t.get("mode", "")).lower() in {
                "test", "testing", "evaluate", "eval", "infer", "inference"
            }
            for t in tasks
        ):
            intent_action = "test"
        else:
            intent_action = "chat" if not tasks else "train"

    return {
        "needs_training": needs_training,
        "intent_action": intent_action,
        "tasks": tasks,
    }


def extract_intent(client: SupportsChat, prompt: str) -> Optional[Dict[str, Any]]:
    content = client.chat([
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    parsed = safe_parse_json(content) if content else None
    return normalize_intent_payload(parsed)


def select_dataset(
    client: SupportsChat,
    prompt: str,
    intent: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    system_msg = (
        "你是 DLCV 训练平台的数据集匹配助手。请根据用户意图，从候选数据集中选出最匹配的一个。\n"
        "返回 JSON 格式：\n"
        "{\n"
        '  "folder": "选中的文件夹名称（候选的 folder 字段）",\n'
        '  "path": "选中项的完整 path 字段（必须填写候选中的 path，用于消歧）",\n'
        '  "reasoning": "简要说明为什么选这个文件夹",\n'
        '  "confidence": 0.95\n'
        "}\n"
        "注意：\n"
        "1. folder / path 必须来自候选列表，不要编造；只要候选含 path 字段就必须回传该 path。\n"
        "2. 候选可能来自多个磁盘根目录或深层子目录；若存在同名 folder，必须返回对应的 path。\n"
        "3. 优先匹配用户提到的项目名、客户名、盘符/目录线索（如易格、zzh、手机检测）；"
        "不要仅因为文件夹名含任务缩写（如 5-ISEG）就忽略更匹配的项目目录。\n"
        "4. 如果没有任何候选匹配，返回 null。\n"
        "5. 只返回 JSON，不要附加说明。"
    )
    user_msg = (
        f"用户原始输入：{prompt}\n"
        f"解析出的意图：{json.dumps(intent, ensure_ascii=False)}\n"
        f"候选数据集：{json.dumps(candidates, ensure_ascii=False)}"
    )
    content = client.chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    parsed = safe_parse_json(content) if content else None
    return parsed if isinstance(parsed, dict) else None


def match_categories(
    client: SupportsChat,
    user_categories: List[str],
    dataset_categories: List[str],
) -> Optional[Dict[str, Optional[str]]]:
    if not user_categories or not dataset_categories:
        return None
    system_msg = (
        "你是一个类别对齐助手。请判断用户提到的每个类别对应数据集中的哪个类别。\n"
        "如果数据集中没有对应类别，请填 null。\n"
        '返回 JSON 格式：{"类别A": "dataset_cat_a", "类别B": null}'
    )
    user_msg = (
        f"用户类别：{json.dumps(user_categories, ensure_ascii=False)}\n"
        f"数据集类别：{json.dumps(dataset_categories, ensure_ascii=False)}"
    )
    content = client.chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    parsed = safe_parse_json(content) if content else None
    if not isinstance(parsed, dict):
        return None
    return {k: v if v in dataset_categories else None for k, v in parsed.items()}
