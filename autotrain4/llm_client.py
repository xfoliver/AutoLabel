"""
可选的 LLM 客户端，用于自然语言理解和类别语义对齐。

支持任意 OpenAI 兼容接口，通过环境变量配置：
    AUTOTRAIN_LLM_BASE_URL
    AUTOTRAIN_LLM_API_KEY
    AUTOTRAIN_LLM_MODEL

如果未配置，则所有 LLM 方法返回 None，调用方应使用规则/向量兜底逻辑。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI 兼容的轻量 LLM 客户端。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = (base_url or os.environ.get("AUTOTRAIN_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("AUTOTRAIN_LLM_API_KEY", "")
        self.model = model or os.environ.get("AUTOTRAIN_LLM_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self._client: Any = None

    @property
    def available(self) -> bool:
        """是否配置了可用的 LLM。"""
        return bool(self.base_url and self.api_key)

    def _ensure_client(self) -> Any:
        """延迟导入并创建 openai 客户端。"""
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("未安装 openai 包，无法使用 LLM 功能") from e

        self._client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self._client

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
        """调用 LLM chat completion，返回文本内容。"""
        if not self.available:
            return None
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("LLM 调用失败: %s", e)
            return None

    def extract_intent(self, prompt: str) -> Optional[Dict[str, Any]]:
        """从用户自然语言中提取意图。"""
        system_msg = (
            "你是一个 DLCV 训练平台任务解析助手。请从用户输入中提取以下字段并返回 JSON，"
            "格式如下：\n"
            "{\n"
            '  "task_type": "分类/目标检测/OCR/实例分割/语义分割/旋转框检测/DETR 之一",\n'
            '  "categories": ["类别1", "类别2"],\n'
            '  "dataset_hint": "用户提到的数据集文件夹名，没有则为空字符串",\n'
            '  "mode": "train"\n'
            "}\n"
            "注意：\n"
            "1. categories 只返回用户明确想区分的类别，不要臆造。\n"
            "2. 如果用户用 cls、det、OCR、iseg、sseg、rotate、detr 等简写，正确映射为中文任务类型。\n"
            "3. 只返回 JSON，不要附加说明。"
        )
        content = self.chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
        )
        if not content:
            return None
        return self._safe_parse_json(content)

    def match_categories(
        self,
        user_categories: List[str],
        dataset_categories: List[str],
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        让 LLM 判断用户类别与数据集类别的对应关系。
        返回 {user_cat: dataset_cat_or_None}。
        """
        if not self.available or not user_categories or not dataset_categories:
            return None
        system_msg = (
            "你是一个类别对齐助手。请判断用户提到的每个类别对应数据集中的哪个类别。\n"
            "如果数据集中没有对应类别，请填 null。\n"
            "返回 JSON 格式：{\"类别A\": \"dataset_cat_a\", \"类别B\": null}"
        )
        user_msg = (
            f"用户类别：{json.dumps(user_categories, ensure_ascii=False)}\n"
            f"数据集类别：{json.dumps(dataset_categories, ensure_ascii=False)}"
        )
        content = self.chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )
        if not content:
            return None
        parsed = self._safe_parse_json(content)
        if not parsed:
            return None
        # 只保留有效映射
        return {k: v if v in dataset_categories else None for k, v in parsed.items()}

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
        """从可能包含 markdown 代码块的文本中安全解析 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 块
        import re

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试提取最后一个 { ... }
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        logger.warning("LLM 返回内容 JSON 解析失败: %s", text[:200])
        return None
