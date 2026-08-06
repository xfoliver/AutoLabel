"""
LLM 客户端。

支持任意 OpenAI 兼容接口，通过环境变量配置：
    AUTOTRAIN_LLM_BASE_URL
    AUTOTRAIN_LLM_API_KEY
    AUTOTRAIN_LLM_MODEL

或 DeepSeek 专用环境变量：
    DEEPSEEK_API_KEY
    DEEPSEEK_MODEL
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


class LLMClient:
    """OpenAI 兼容的轻量 LLM 客户端。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = (
            base_url
            or os.environ.get("AUTOTRAIN_LLM_BASE_URL", "")
            or DEEPSEEK_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("AUTOTRAIN_LLM_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self.model = (
            model
            or os.environ.get("AUTOTRAIN_LLM_MODEL", "")
            or os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL)
        )
        self.timeout = timeout
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    @classmethod
    def deepseek(
        cls,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> "LLMClient":
        return cls(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            model=model or os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL),
            timeout=timeout,
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("未安装 openai 包，无法使用 LLM 功能") from e

        if not self.api_key:
            raise RuntimeError("LLM API key 未配置，请设置 DEEPSEEK_API_KEY 或 AUTOTRAIN_LLM_API_KEY")

        self._client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self._client

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
        if not self.available:
            raise RuntimeError("LLM 未配置 base_url / api_key，无法调用")
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def extract_intent(self, prompt: str) -> Optional[Dict[str, Any]]:
        """解析用户自然语言意图，返回完整结构（含 needs_training + tasks）。

        必须返回 dict，供 nlu 读取顶层 needs_training；不要只丢 tasks 列表。
        """
        system_msg = (
            "你是 DLCV 视觉训练工作台的意图理解助手。先理解用户**真正想做什么**，"
            "再映射到动作；不要把出现「OCR/训练集/数据集/cls」等词就默认当成要开训。\n"
            "\n"
            "【先判断动作 intent_action，再填字段】\n"
            "- train：用户明确要求训练/开训/跑训练/生成并启动训练流程"
            "（如「训练一个 OCR」「帮我训 cls」「开始训练」）。\n"
            "- test：用户明确要求测试/评估/验证/推理已有模型"
            "（如「测试一下最新模型」「评估这个 dvp」）。\n"
            "- locate：用户只要求查找/定位/找到/看看数据集或文件在哪，"
            "**不要训练、不要建项目、不要开 WebBridge**"
            "（如「帮我定位 OCR 训练集」「zzh 里的数据在哪」「找一下实例分割数据集」）。"
            "即使用户提到了任务类型（OCR/cls/iseg…）或「训练集」字样，只要没有开训意愿，仍是 locate。\n"
            "- chat：闲聊、问概念、与平台数据/训练流程无关、或信息不足且无需扫盘"
            "（如「什么是 mAP」「你好」）。\n"
            "\n"
            "【needs_training】\n"
            "- 仅 train 时为 true。\n"
            "- test / locate / chat 一律为 false。\n"
            "- 不要因为提到数据集、任务类型、盘符就设为 true。\n"
            "\n"
            "【tasks】\n"
            "- train / test / locate：通常 1 条或多条；每条对应一个可执行目标。\n"
            "- chat：tasks 必须为 []。\n"
            "- locate 时也要填 task_type（若用户说了 OCR/检测等）、dataset_hint/路径线索，"
            "mode 必须为 \"locate\"，needs_training 为 false。\n"
            "\n"
            "【返回 JSON，不要附加说明】\n"
            "{\n"
            '  "needs_training": false,\n'
            '  "intent_action": "locate",\n'
            '  "tasks": [\n'
            "    {\n"
            '      "task_type": "分类/目标检测/OCR/实例分割/语义分割/旋转框检测/DETR 之一，未知则空字符串",\n'
            '      "categories": ["仅用户明确要区分的类别"],\n'
            '      "dataset_hint": "项目名/文件夹名/路径片段，如 zzh、易格技术",\n'
            '      "dataset_path": "仅当用户写出含盘符的完整路径字面量时填写，否则空字符串",\n'
            '      "output_dir": "用户指定的模型输出目录；locate 一般为空",\n'
            '      "mode": "train | test | locate | chat",\n'
            '      "needs_training": false,\n'
            '      "data_analysis": true,\n'
            '      "mentions_augmentation": false,\n'
            '      "data_augmentation": [],\n'
            '      "baseline_model": "",\n'
            '      "wants_baseline": false\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "【字段细则】\n"
            "1. task_type 简写映射：cls→分类，det→目标检测，OCR→OCR，"
            "iseg/ISEG/instance→实例分割，sseg/SSEG/semantic→语义分割，"
            "rotate→旋转框检测，detr→DETR。iseg 与 sseg 勿混。\n"
            "2. categories 禁止臆造。\n"
            "3. dataset_path：只有用户写出完整路径字面量（如 Z:\\\\proj\\\\2-训练集）才填；"
            "口语「z盘 zzh 文件夹中的易格」→ dataset_hint 收项目名/片段，dataset_path 留空。"
            "勿把输出目录填进 dataset_path，勿改写成本地示例目录。\n"
            "4. dataset_hint：zzh、易格技术、文件夹名等；已有真实 dataset_path 时可填末级目录名。\n"
            "5. output_dir：用户明确说模型输出/保存到某处时填完整路径，否则 \"\"。\n"
            "6. mentions_augmentation / data_augmentation：仅当用户明确要求数据增强"
            "（随机翻转、裁剪等）时 mentions=true 并填标准选项："
            "调整图像大小、随机比例缩放、随机翻转、随机固定角度旋转、随机裁剪、"
            "随机调整亮度、随机调整对比度、随机JPG编码；否则 mentions=false 且列表 []。\n"
            "7. baseline_model：明确 .dvp 路径才填；只说 baseline/对比则路径 \"\" 且 wants_baseline=true。\n"
            "8. data_analysis：训练前是否分析数据，默认 true；locate/chat 可 false。\n"
            "9. 多任务时 tasks 多元素；用户只要一件事就只返回一个元素。\n"
            "10. 拿不准时：若含定位/查找语气且无开训动词 → locate；"
            "若完全无关 → chat；切勿默认 train。\n"
        )
        content = self.chat([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ])
        parsed = self._safe_parse_json(content) if content else None
        if not parsed:
            return None

        # 统一成 {needs_training, intent_action?, tasks}
        if isinstance(parsed, list):
            return {
                "needs_training": True,
                "intent_action": "train",
                "tasks": parsed,
            }
        if not isinstance(parsed, dict):
            return None

        if "tasks" not in parsed:
            # 旧版：单任务对象
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
            # 无顶层字段时：看 tasks 是否像训练
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

    def select_dataset(
        self,
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
        content = self.chat([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        return self._safe_parse_json(content) if content else None

    def match_categories(
        self,
        user_categories: List[str],
        dataset_categories: List[str],
    ) -> Optional[Dict[str, Optional[str]]]:
        if not self.available or not user_categories or not dataset_categories:
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
        content = self.chat([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        parsed = self._safe_parse_json(content) if content else None
        if not parsed:
            return None
        return {k: v if v in dataset_categories else None for k, v in parsed.items()}

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning("LLM 返回内容 JSON 解析失败: %s", text[:200])
        return None
