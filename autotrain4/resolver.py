"""
语义数据集解析器（LLM 直接驱动，无规则/向量兜底）。

根据用户意图，在扫描目录下找到最匹配的数据集文件夹：
    1. 按任务类型过滤候选文件夹（仅做候选范围收敛，非语义兜底）。
    2. 调用 LLM 直接选择最匹配的文件夹。
    3. 调用 LLM 对齐用户类别与数据集类别。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import DATASET_CANDIDATES, IMAGE_EXTENSIONS, TASK_FOLDER_KEYWORDS
from .llm_client import LLMClient
from .nlu import TrainIntent

logger = logging.getLogger(__name__)


@dataclass
class DatasetMatch:
    """数据集匹配结果。"""

    folder: Path
    dataset_dir: Path
    task_type: str
    categories: List[str] = field(default_factory=list)
    matched_user_categories: Dict[str, Optional[str]] = field(default_factory=dict)
    folder_score: float = 0.0
    category_score: float = 0.0
    total_score: float = 0.0
    total_images: int = 0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder": str(self.folder),
            "dataset_dir": str(self.dataset_dir),
            "task_type": self.task_type,
            "categories": self.categories,
            "matched_user_categories": self.matched_user_categories,
            "folder_score": round(self.folder_score, 3),
            "category_score": round(self.category_score, 3),
            "total_score": round(self.total_score, 3),
            "total_images": self.total_images,
            "reasoning": self.reasoning,
        }


class DatasetResolver:
    """数据集解析器（第二层 LLM）。"""

    def __init__(
        self,
        scan_dir: Path,
        llm_client: Optional[LLMClient] = None,
    ):
        self.scan_dir = Path(scan_dir).resolve()
        self.llm = llm_client or LLMClient()

    def resolve(
        self,
        intent: TrainIntent,
        auto_threshold: float = 0.55,
        task_context: Optional[str] = None,
    ) -> DatasetMatch:
        """
        解析最匹配的数据集。
        直接由 LLM 在候选文件夹中做最终选择。
        """
        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法解析数据集匹配（已取消规则/向量兜底）")

        candidates = self._find_task_folders(intent.task_type)
        if not candidates:
            raise ValueError(
                f"在 {self.scan_dir} 下未找到任务类型 {intent.task_type} 的文件夹"
            )

        candidate_infos = [self._build_candidate_info(folder) for folder in candidates]
        logger.info("第二层 LLM：在 %d 个候选数据集中选择最匹配项", len(candidate_infos))

        selection = self.llm.select_dataset(
            prompt=task_context if task_context is not None else intent.raw_prompt,
            intent=intent.to_task_dict(),
            candidates=candidate_infos,
        )
        logger.debug("LLM 原始选择结果: %s", selection)
        if not selection:
            raise RuntimeError("LLM 数据集选择失败，无法继续（已取消兜底策略）")

        selected_folder_name = selection.get("folder")
        if not selected_folder_name:
            raise RuntimeError(f"LLM 未返回有效数据集选择: {selection}")

        selected_folder = self.scan_dir / selected_folder_name
        if not selected_folder.exists() or not selected_folder.is_dir():
            raise ValueError(f"LLM 选择的文件夹不存在: {selected_folder}")

        confidence = float(selection.get("confidence", 0.0))
        if confidence < auto_threshold:
            raise ValueError(
                f"LLM 对数据集匹配不够确信（置信度 {confidence:.3f} < {auto_threshold}）。"
                f"候选文件夹: {[c['folder'] for c in candidate_infos]}。请用 --folder 指定。"
            )

        match = self._score_folder(selected_folder, intent)
        match.total_score = confidence
        match.reasoning = str(selection.get("reasoning", ""))

        # 如果用户明确指定了类别，要求至少一半类别能被数据集覆盖
        if intent.categories and match.categories:
            matched_count = sum(1 for v in match.matched_user_categories.values() if v is not None)
            coverage = matched_count / len(intent.categories)
            if coverage < 0.5:
                raise ValueError(
                    f"数据集类别不匹配: 用户要求 {intent.categories}，"
                    f"数据集 {match.folder.name} 只匹配到 {matched_count}/{len(intent.categories)} 个"
                    f"（数据集类别: {match.categories}）。"
                )

        return match

    def _find_task_folders(self, task_type: str) -> List[Path]:
        """根据任务类型过滤文件夹（整词匹配，避免 det 误匹配 detr）。"""
        if not self.scan_dir.exists():
            raise ValueError(f"扫描目录不存在: {self.scan_dir}")

        keywords = TASK_FOLDER_KEYWORDS.get(task_type, [task_type.lower()])
        candidates: List[Path] = []
        for folder in self.scan_dir.iterdir():
            if not folder.is_dir():
                continue
            name_lower = folder.name.lower()
            for kw in keywords:
                kw_lower = kw.lower()
                if re.search(
                    rf"(?<![a-z0-9\u4e00-\u9fa5]){re.escape(kw_lower)}(?![a-z0-9\u4e00-\u9fa5])",
                    name_lower,
                ):
                    candidates.append(folder)
                    break
        return candidates

    def _build_candidate_info(self, folder: Path) -> Dict[str, Any]:
        """构造候选文件夹信息，供 LLM 选择。"""
        dataset_dir = self._find_dataset_dir(folder)
        categories, total_images = self._detect_categories(dataset_dir or folder)
        return {
            "folder": folder.name,
            "dataset_dir": str(dataset_dir or folder),
            "categories": categories,
            "total_images": total_images,
        }

    def _find_dataset_dir(self, folder: Path) -> Optional[Path]:
        """在文件夹内探测数据集子目录。"""
        for candidate in DATASET_CANDIDATES:
            cand_path = folder / candidate
            if cand_path.exists() and cand_path.is_dir():
                return cand_path
        return None

    def _detect_categories(self, dataset_dir: Path) -> Tuple[List[str], int]:
        """探测数据集内部类别（按子文件夹）和图片总数。

        分类任务：子文件夹即类别，统计每个子文件夹内的图片。
        OCR/检测等任务：图片可能直接放在根目录，找不到子文件夹类别时回退统计根目录图片。
        """
        categories: List[str] = []
        total = 0
        if not dataset_dir.exists():
            return categories, total

        # 1. 先按分类任务风格探测子文件夹类别
        for subdir in dataset_dir.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.lower() in {"annotations", "labels", "image", "images", "ok", "ng"}:
                continue
            count = sum(
                1
                for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if count > 0:
                categories.append(subdir.name)
                total += count

        # 2. 没找到分类式子文件夹时，回退统计根目录图片（OCR/检测风格）
        if not categories:
            root_count = sum(
                1
                for f in dataset_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if root_count > 0:
                total = root_count

        return categories, total

    def _score_folder(self, folder: Path, intent: TrainIntent) -> DatasetMatch:
        """对单个文件夹构造匹配结果。"""
        dataset_dir = self._find_dataset_dir(folder)
        if dataset_dir is None:
            dataset_dir = folder

        dataset_cats, total_images = self._detect_categories(dataset_dir)

        matched: Dict[str, Optional[str]] = {}
        if intent.categories and dataset_cats:
            mapping = self.llm.match_categories(intent.categories, dataset_cats)
            if mapping is not None:
                matched = mapping
            else:
                matched = {cat: None for cat in intent.categories}
        elif intent.categories:
            matched = {cat: None for cat in intent.categories}

        return DatasetMatch(
            folder=folder,
            dataset_dir=dataset_dir,
            task_type=intent.task_type,
            categories=dataset_cats,
            matched_user_categories=matched,
            total_images=total_images,
        )
