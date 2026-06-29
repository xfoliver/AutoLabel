"""
语义数据集解析器。

根据用户意图，在扫描目录下找到最匹配的数据集文件夹。
匹配策略：
    1. 任务类型过滤（分类只找分类文件夹）
    2. 文件夹名语义匹配
    3. 数据集内部类别匹配（精确 → 模糊 → 向量语义 → LLM 兜底）
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

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
        }


class SemanticEmbedder:
    """基于 sentence-transformers 的语义编码器（懒加载）。"""

    _instance: Optional["SemanticEmbedder"] = None

    def __new__(cls, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._model_name = model_name
        return cls._instance

    def _init(self):
        if self._initialized:
            return
        logger.info("正在加载语义模型: %s", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._initialized = True
        except Exception as e:
            logger.error("加载语义模型失败: %s", e)
            raise

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._init()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def similarity(self, a: str, b: str) -> float:
        """计算两个文本的语义相似度（余弦），范围 [0, 1]。"""
        import numpy as np

        vecs = self.encode([a, b])
        v1, v2 = np.array(vecs[0]), np.array(vecs[1])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(dot / norm) if norm > 0 else 0.0


class DatasetResolver:
    """数据集解析器。"""

    # 各层匹配的阈值
    EXACT_MATCH_SCORE = 1.0
    FUZZY_THRESHOLD = 80.0  # rapidfuzz ratio 0-100
    EMBEDDING_THRESHOLD = 0.72

    def __init__(
        self,
        scan_dir: Path,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[SemanticEmbedder] = None,
    ):
        self.scan_dir = Path(scan_dir).resolve()
        self.llm = llm_client or LLMClient()
        self.embedder = embedder or SemanticEmbedder()

    def resolve(
        self,
        intent: TrainIntent,
        top_k: int = 3,
        auto_threshold: float = 0.75,
    ) -> DatasetMatch:
        """
        解析最匹配的数据集。
        如果 top1 分数 >= auto_threshold，直接返回；
        否则抛出异常，提示用户多个候选。
        """
        candidates = self._find_task_folders(intent.task_type)
        if not candidates:
            raise ValueError(
                f"在 {self.scan_dir} 下未找到任务类型 {intent.task_type} 的文件夹"
            )

        matches = [self._score_folder(folder, intent) for folder in candidates]
        matches = sorted(matches, key=lambda m: m.total_score, reverse=True)

        logger.info("数据集匹配 top-%d:", top_k)
        for i, m in enumerate(matches[:top_k], 1):
            logger.info("  #%d %s (score=%.3f, folder=%.3f, cat=%.3f)",
                        i, m.folder.name, m.total_score, m.folder_score, m.category_score)

        if not matches:
            raise ValueError("没有可用候选")

        best = matches[0]

        # 如果用户明确指定了类别，要求至少一半类别能被数据集覆盖。
        # 但数据集本身没有可探测类别时（如 OCR / DET / SEG 的标注文件组织），
        # 无法做子文件夹级别的校验，跳过覆盖检查，只保留语义匹配分。
        if intent.categories and best.categories:
            matched_count = sum(1 for v in best.matched_user_categories.values() if v is not None)
            coverage = matched_count / len(intent.categories)
            if coverage < 0.5:
                raise ValueError(
                    f"数据集类别不匹配: 用户要求 {intent.categories}，"
                    f"数据集 {best.folder.name} 只匹配到 {matched_count}/{len(intent.categories)} 个"
                    f"（数据集类别: {best.categories}）。"
                )

        if best.total_score < auto_threshold and len(matches) > 1:
            # 如果最高分不够高且存在竞争者，列出候选让用户选择
            others = [m.folder.name for m in matches[:top_k]]
            raise ValueError(
                f"对数据集匹配不够确信（最高分 {best.total_score:.3f} < {auto_threshold}）。"
                f"候选文件夹: {others}。请用 --folder 指定。"
            )

        return best

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
                # 整词匹配：关键词前后必须是字母数字中文的边界
                if re.search(rf"(?<![a-z0-9\u4e00-\u9fa5]){re.escape(kw_lower)}(?![a-z0-9\u4e00-\u9fa5])", name_lower):
                    candidates.append(folder)
                    break
        return candidates

    def _find_dataset_dir(self, folder: Path) -> Optional[Path]:
        """在文件夹内探测数据集子目录。"""
        for candidate in DATASET_CANDIDATES:
            cand_path = folder / candidate
            if cand_path.exists() and cand_path.is_dir():
                return cand_path
        return None

    def _detect_categories(self, dataset_dir: Path) -> Tuple[List[str], int]:
        """探测数据集内部类别（按子文件夹）和图片总数。"""
        categories: List[str] = []
        total = 0
        for subdir in dataset_dir.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.lower() in {"annotations", "labels", "image", "images", "ok", "ng"}:
                continue
            count = sum(1 for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if count > 0:
                categories.append(subdir.name)
                total += count
        return categories, total

    def _score_folder(self, folder: Path, intent: TrainIntent) -> DatasetMatch:
        """对单个文件夹打分。"""
        dataset_dir = self._find_dataset_dir(folder)
        if dataset_dir is None:
            # 没有标准训练集子目录，尝试 folder 本身
            dataset_dir = folder

        dataset_cats, total_images = self._detect_categories(dataset_dir)

        # 文件夹名语义分
        folder_score = self._score_folder_name(folder.name, intent)

        # 类别匹配分
        cat_score, matched = self._score_categories(intent.categories, dataset_cats)

        # 综合分：类别匹配权重更高
        total_score = folder_score * 0.3 + cat_score * 0.7

        return DatasetMatch(
            folder=folder,
            dataset_dir=dataset_dir,
            task_type=intent.task_type,
            categories=dataset_cats,
            matched_user_categories=matched,
            folder_score=folder_score,
            category_score=cat_score,
            total_score=total_score,
            total_images=total_images,
        )

    def _score_folder_name(self, folder_name: str, intent: TrainIntent) -> float:
        """根据文件夹名与用户意图的语义相似度打分。"""
        # 如果用户给了数据集提示，直接匹配提示
        if intent.dataset_hint and intent.dataset_hint.lower() in folder_name.lower():
            return 1.0

        # 用 embedding 比较 folder_name 和 prompt / categories 的语义
        try:
            query = intent.raw_prompt
            if intent.categories:
                query = " ".join(intent.categories)
            score = self.embedder.similarity(folder_name, query)
            return score
        except Exception as e:
            logger.debug("文件夹名语义匹配失败: %s", e)
            return 0.0

    def _score_categories(
        self,
        user_categories: List[str],
        dataset_categories: List[str],
    ) -> Tuple[float, Dict[str, Optional[str]]]:
        """
        计算用户类别与数据集类别的匹配分数。
        每个用户类别都会找到最佳匹配的数据集类别。
        """
        if not user_categories:
            # 用户没指定类别，只要数据集有类别就给一个基础分
            return 0.5 if dataset_categories else 0.0, {}

        if not dataset_categories:
            return 0.0, {cat: None for cat in user_categories}

        matched: Dict[str, Optional[str]] = {}
        per_scores: List[float] = []

        for user_cat in user_categories:
            best_cat, best_score = self._match_single_category(user_cat, dataset_categories)
            matched[user_cat] = best_cat if best_score >= 0.5 else None
            per_scores.append(best_score)

        # 平均分，同时惩罚未匹配的类别
        avg_score = sum(per_scores) / len(per_scores)
        coverage = sum(1 for v in matched.values() if v is not None) / len(user_categories)
        final_score = avg_score * 0.7 + coverage * 0.3
        return final_score, matched

    def _match_single_category(
        self,
        user_cat: str,
        dataset_categories: List[str],
    ) -> Tuple[Optional[str], float]:
        """为单个用户类别找到最佳数据集类别。"""
        user_cat_norm = user_cat.strip().lower()

        # 1. 精确匹配
        for dcat in dataset_categories:
            if dcat.strip().lower() == user_cat_norm:
                return dcat, self.EXACT_MATCH_SCORE

        # 2. 模糊匹配
        best_fuzzy: Optional[str] = None
        best_fuzzy_score = 0.0
        for dcat in dataset_categories:
            score = fuzz.ratio(user_cat, dcat)
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy = dcat
        if best_fuzzy and best_fuzzy_score >= self.FUZZY_THRESHOLD:
            return best_fuzzy, best_fuzzy_score / 100.0

        # 3. 向量语义匹配
        try:
            best_emb: Optional[str] = None
            best_emb_score = 0.0
            for dcat in dataset_categories:
                score = self.embedder.similarity(user_cat, dcat)
                if score > best_emb_score:
                    best_emb_score = score
                    best_emb = dcat
            if best_emb and best_emb_score >= self.EMBEDDING_THRESHOLD:
                return best_emb, best_emb_score
        except Exception as e:
            logger.debug("类别语义匹配失败: %s", e)

        # 4. LLM 兜底
        if self.llm.available:
            try:
                mapping = self.llm.match_categories([user_cat], dataset_categories)
                if mapping and mapping.get(user_cat) in dataset_categories:
                    return mapping[user_cat], 0.85
            except Exception as e:
                logger.debug("LLM 类别匹配失败: %s", e)

        # 返回最佳模糊/语义结果（即使低于阈值），供上层参考
        if best_fuzzy and best_fuzzy_score > 50:
            return best_fuzzy, best_fuzzy_score / 100.0
        return None, 0.0
