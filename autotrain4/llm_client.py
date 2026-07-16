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

    def extract_intent(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        system_msg = (
            "你是一个 DLCV 训练平台任务解析助手。请从用户输入中提取所有训练任务，并返回 JSON。\n"
            "格式如下：\n"
            "{\n"
            '  "needs_training": true,\n'
            '  "tasks": [\n'
            "    {\n"
            '      "task_type": "分类/目标检测/OCR/实例分割/语义分割/旋转框检测/DETR 之一",\n'
            '      "categories": ["类别1", "类别2"],\n'
            '      "dataset_hint": "用户提到的数据集文件夹名，没有则为空字符串",\n'
            '      "mode": "train",\n'
            '      "data_analysis": true,\n'
            '      "mentions_augmentation": false,\n'
            '      "data_augmentation": [],\n'
            '      "baseline_model": "用户指定的基准模型 .dvp 路径，没有则为空字符串",\n'
            '      "wants_baseline": false\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "注意：\n"
            "1. needs_training 是布尔值：只有当用户明确想启动训练/测试任务时才为 true；"
            "如果用户只是闲聊、询问概念、求助与训练无关的问题，则为 false。\n"
            "2. tasks 是任务列表。如果用户只要求一个任务，也返回只含一个元素的列表。\n"
            "3. 每个任务的 categories 只返回用户明确想区分的类别，不要臆造。\n"
            "4. 如果用户用 cls、det、OCR、iseg、sseg、rotate、detr 等简写，正确映射为中文任务类型。\n"
            "5. data_analysis 表示是否需要在训练前执行数据分析，默认 true。\n"
            "6. mentions_augmentation 是布尔值，只有当用户明确提到该任务需要\"数据增强\"、\"随机翻转\"、\"随机裁剪\"等"
            "增强相关需求时才为 true；否则必须为 false。\n"
            "7. data_augmentation 是数据增强选项列表，可选："
            "调整图像大小、随机比例缩放、随机翻转、随机固定角度旋转、随机裁剪、"
            "随机调整亮度、随机调整对比度、随机JPG编码。"
            "只有当该任务的 mentions_augmentation 为 true 时才填写；否则必须返回空列表。\n"
            "8. baseline_model：如果用户明确给出 .dvp 路径就返回；只提到 baseline/基准/对比但无路径时返回空字符串，并把 wants_baseline 设为 true。\n"
            "9. 如果用户想测试或对比模型，mode 返回 test；想训练则返回 train。\n"
            "10. 只返回 JSON，不要附加说明。"
        )
        content = self.chat([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ])
        parsed = self._safe_parse_json(content) if content else None
        if not parsed:
            return None
        # 兼容旧版返回单个对象的情况
        if isinstance(parsed, dict) and "tasks" not in parsed:
            return [parsed]
        tasks = parsed.get("tasks") if isinstance(parsed, dict) else parsed
        return tasks if isinstance(tasks, list) else None

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
            '  "folder": "选中的文件夹名称（必须来自候选列表）",\n'
            '  "reasoning": "简要说明为什么选这个文件夹",\n'
            '  "confidence": 0.95\n'
            "}\n"
            "注意：\n"
            "1. folder 必须是候选列表中真实存在的名称，不要编造。\n"
            "2. 如果没有任何候选匹配，返回 null。\n"
            "3. 只返回 JSON，不要附加说明。"
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
