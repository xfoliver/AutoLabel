"""
推理结果评估与基准对比。

对同一张测试集，分别用新模型和基准模型跑推理，
然后对比两者结果是否一致，并生成 JSON 报告。
"""

import json
import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .inference_client import InferenceClient

logger = logging.getLogger(__name__)


def _float_to_decimal_str(value: float) -> str:
    """把 float 转成普通小数字符串，避免科学计数法。"""
    if value != value:  # NaN
        return "0"
    if value == float("inf"):
        return "1"
    if value == float("-inf"):
        return "0"
    try:
        # str(float) 会给出最短且能 round-trip 的十进制表示
        d = Decimal(str(value))
    except InvalidOperation:
        return "0"
    # 四舍五入保留 8 位小数，再展开成不带指数的普通小数
    d = d.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _indent_text(text: str, prefix: str) -> str:
    """给多行文本的每一行加上前缀缩进。"""
    return "\n".join(prefix + line for line in text.split("\n"))


def _format_json_no_scientific(obj: Any, indent: int = 2, level: int = 0) -> str:
    """把 Python 对象格式化为 JSON 字符串，浮点数一律用普通小数表示。

    避免 Python 默认的 `7.0995488385960925e-06` 这种科学计数法，
    方便直接阅读 score / probability 对应的百分比。
    """
    pad = " " * (indent * level)
    next_pad = " " * (indent * (level + 1))

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            key = json.dumps(str(k), ensure_ascii=False)
            val = _format_json_no_scientific(v, indent, level + 1)
            first, *rest = val.split("\n")
            item = f"{next_pad}{key}: {first}"
            if rest:
                item += "\n" + _indent_text("\n".join(rest), next_pad)
            items.append(item)
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        items = [
            _indent_text(_format_json_no_scientific(v, indent, level + 1), next_pad)
            for v in obj
        ]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"


    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "null"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _float_to_decimal_str(obj)

    return json.dumps(obj, ensure_ascii=False)


@dataclass
class ComparisonReport:
    """对比报告。"""

    new_model_path: str
    baseline_model_path: Optional[str]
    image_dir: str
    total_images: int = 0
    new_success_count: int = 0
    baseline_success_count: int = 0
    match_count: int = 0
    mismatch_count: int = 0
    new_failed_images: List[str] = field(default_factory=list)
    baseline_failed_images: List[str] = field(default_factory=list)
    mismatched_images: List[Dict[str, Any]] = field(default_factory=list)
    per_image_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_model_path": self.new_model_path,
            "baseline_model_path": self.baseline_model_path,
            "image_dir": self.image_dir,
            "summary": {
                "total_images": self.total_images,
                "new_success_count": self.new_success_count,
                "baseline_success_count": self.baseline_success_count,
                "match_count": self.match_count,
                "mismatch_count": self.mismatch_count,
                "new_failed_images": self.new_failed_images,
                "baseline_failed_images": self.baseline_failed_images,
            },
            "mismatched_images": self.mismatched_images,
            "per_image_results": self.per_image_results,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """保存报告到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_format_json_no_scientific(self.to_dict()))
        logger.info("对比报告已保存: %s", path)
        return path


class InferenceEvaluator:
    """推理结果评估器。"""

    def __init__(self, client: Optional[InferenceClient] = None):
        self.client = client or InferenceClient()
        # 默认跟随 client 的进度条设置
        self.progress = getattr(self.client, "progress", True)

    def evaluate(
        self,
        new_model_path: Union[str, Path],
        image_dir: Union[str, Path],
        baseline_model_path: Optional[Union[str, Path]] = None,
        on_progress: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> ComparisonReport:
        """
        评估新模型推理结果，并可选择与基准模型对比。

        Args:
            new_model_path: 新训练好的 .dvp 模型路径。
            image_dir: 测试图片目录。
            baseline_model_path: 可选，基准模型 .dvp 路径。
            on_progress: 可选进度回调 (current, total, image_name, model_name)。

        Returns:
            ComparisonReport 对象。
        """
        new_model_path = Path(new_model_path).resolve()
        image_dir = Path(image_dir).resolve()
        baseline_model_path = Path(baseline_model_path).resolve() if baseline_model_path else None

        logger.info("========== 推理评估开始 ==========")
        logger.info("新模型: %s", new_model_path)
        if baseline_model_path:
            logger.info("基准模型: %s", baseline_model_path)
        logger.info("测试图片目录: %s", image_dir)

        # 有 GUI 回调时关掉 tqdm，改由回调/日志输出百分比，避免进度只出现在终端
        use_tqdm = self.progress and on_progress is None
        new_results = self.client.infer_batch(
            new_model_path,
            image_dir,
            progress=use_tqdm,
            on_progress=on_progress,
        )
        baseline_results: Optional[List[Dict[str, Any]]] = None
        if baseline_model_path:
            baseline_results = self.client.infer_batch(
                baseline_model_path,
                image_dir,
                progress=use_tqdm,
                on_progress=on_progress,
            )

        return self._build_report(
            new_results=new_results,
            baseline_results=baseline_results,
            new_model_path=str(new_model_path),
            baseline_model_path=str(baseline_model_path) if baseline_model_path else None,
            image_dir=str(image_dir),
        )

    def _build_report(
        self,
        new_results: List[Dict[str, Any]],
        baseline_results: Optional[List[Dict[str, Any]]],
        new_model_path: str,
        baseline_model_path: Optional[str],
        image_dir: str,
    ) -> ComparisonReport:
        """根据两次推理结果构建对比报告。"""
        report = ComparisonReport(
            new_model_path=new_model_path,
            baseline_model_path=baseline_model_path,
            image_dir=image_dir,
            total_images=len(new_results),
        )

        baseline_by_image: Dict[str, Dict[str, Any]] = {}
        if baseline_results:
            baseline_by_image = {r["image"]: r for r in baseline_results}

        for new_r in new_results:
            image_name = new_r["image"]
            new_success = new_r.get("success", False)
            new_result = new_r.get("result") if new_success else None

            if new_success:
                report.new_success_count += 1
            else:
                report.new_failed_images.append(image_name)

            entry: Dict[str, Any] = {
                "image": image_name,
                "new_model": {
                    "success": new_success,
                    "result": new_result,
                    "error": new_r.get("error") if not new_success else None,
                },
            }

            if image_name in baseline_by_image:
                base_r = baseline_by_image[image_name]
                base_success = base_r.get("success", False)
                base_result = base_r.get("result") if base_success else None

                if base_success:
                    report.baseline_success_count += 1
                else:
                    report.baseline_failed_images.append(image_name)

                is_match = self._compare_results(new_result, base_result)
                if is_match:
                    report.match_count += 1
                else:
                    report.mismatch_count += 1
                    report.mismatched_images.append({
                        "image": image_name,
                        "new_result": new_result,
                        "baseline_result": base_result,
                    })

                entry["baseline_model"] = {
                    "success": base_success,
                    "result": base_result,
                    "error": base_r.get("error") if not base_success else None,
                    "match": is_match,
                }

            report.per_image_results.append(entry)

        logger.info("========== 推理评估汇总 ==========")
        logger.info("总图片数: %s", report.total_images)
        logger.info("新模型成功: %s", report.new_success_count)
        if baseline_model_path:
            logger.info("基准模型成功: %s", report.baseline_success_count)
            logger.info("结果一致: %s", report.match_count)
            logger.info("结果不一致: %s", report.mismatch_count)

        return report

    @staticmethod
    def _compare_results(new_result: Any, baseline_result: Any, score_threshold: float = 0.05) -> bool:
        """
        对比两个推理结果是否一致。

        对 OCR / CLS 等任务，只取置信度最高的 top-1 结果比较：
        1. category_name 不同 -> 不一致。
        2. category_name 相同但 score 差距 > score_threshold -> 不一致。
        3. 其他情况 -> 一致。

        如果结果结构不是列表或 top-1 项不可比较，则回退到 JSON 字符串归一化对比。

        Args:
            new_result: 新模型推理结果。
            baseline_result: 基准模型推理结果。
            score_threshold: score 差异阈值，默认 0.05。
        """
        new_results = new_result.get("results") if isinstance(new_result, dict) else None
        baseline_results = baseline_result.get("results") if isinstance(baseline_result, dict) else None

        if not isinstance(new_results, list) or not isinstance(baseline_results, list):
            # 回退：JSON 字符串归一化对比
            try:
                return json.dumps(new_result, sort_keys=True, ensure_ascii=False) == \
                       json.dumps(baseline_result, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                return new_result == baseline_result

        if len(new_results) == 0 or len(baseline_results) == 0:
            return len(new_results) == len(baseline_results)

        def _top_item(results: List[Any]) -> Any:
            scored = [
                (idx, item.get("score"))
                for idx, item in enumerate(results)
                if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
            ]
            if scored:
                return results[max(scored, key=lambda x: x[1])[0]]
            return results[0]

        new_top = _top_item(new_results)
        baseline_top = _top_item(baseline_results)

        if not isinstance(new_top, dict) or not isinstance(baseline_top, dict):
            try:
                return json.dumps(new_top, sort_keys=True, ensure_ascii=False) == \
                       json.dumps(baseline_top, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                return new_top == baseline_top

        new_name = new_top.get("category_name")
        baseline_name = baseline_top.get("category_name")
        if new_name != baseline_name:
            return False

        new_score = new_top.get("score")
        baseline_score = baseline_top.get("score")
        if isinstance(new_score, (int, float)) and isinstance(baseline_score, (int, float)):
            if abs(new_score - baseline_score) > score_threshold:
                return False

        return True


def compare_models(
    new_model_path: Union[str, Path],
    baseline_model_path: Union[str, Path],
    image_dir: Union[str, Path],
    report_path: Optional[Union[str, Path]] = None,
    inference_api_url: str = "http://127.0.0.1:9890",
) -> Path:
    """
    便捷函数：对比两个模型并保存报告。

    Returns:
        报告文件路径。
    """
    client = InferenceClient(base_url=inference_api_url)
    evaluator = InferenceEvaluator(client=client)
    report = evaluator.evaluate(
        new_model_path=new_model_path,
        image_dir=image_dir,
        baseline_model_path=baseline_model_path,
    )

    if report_path is None:
        report_path = Path(new_model_path).parent / "inference_report.json"

    return report.save(report_path)
