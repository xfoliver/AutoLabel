"""
语义数据集解析器（LLM 直接驱动，无规则/向量兜底）。

根据用户意图，在一个或多个扫描根目录下找到最匹配的数据集文件夹：
    1. 按任务类型过滤候选文件夹（仅做候选范围收敛，非语义兜底）。
    2. 调用 LLM 直接选择最匹配的文件夹。
    3. 调用 LLM 对齐用户类别与数据集类别。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .common import (
    DATASET_CANDIDATES,
    IMAGE_EXTENSIONS,
    TASK_FOLDER_KEYWORDS,
    coerce_existing_dir,
    looks_like_filesystem_path,
    normalize_scan_dirs,
)
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
    """数据集解析器（第二层 LLM）。

    支持单个或多个扫描根目录（方案 A：多根浅扫）。
    每个根下只扫描直接子目录，不递归。
    """

    def __init__(
        self,
        scan_dir: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        llm_client: Optional[LLMClient] = None,
        scan_dirs: Optional[Sequence[Union[str, Path]]] = None,
    ):
        # scan_dirs 优先；否则把 scan_dir 当作单根或多根输入规范化
        if scan_dirs is not None:
            roots = normalize_scan_dirs(list(scan_dirs))
        elif scan_dir is not None:
            roots = normalize_scan_dirs(scan_dir)
        else:
            roots = normalize_scan_dirs()

        self.scan_dirs: List[Path] = roots
        # 兼容旧代码：scan_dir 指向第一个根
        self.scan_dir: Path = self.scan_dirs[0]
        self.llm = llm_client or LLMClient()

    def resolve(
        self,
        intent: TrainIntent,
        auto_threshold: float = 0.55,
        task_context: Optional[str] = None,
    ) -> DatasetMatch:
        """
        解析最匹配的数据集。

        优先使用用户/LLM 给出的显式路径；否则在 scan_dirs 候选中由 LLM 选择。
        """
        explicit = self.resolve_explicit_path(intent)
        if explicit is not None:
            return explicit

        if not self.llm.available:
            raise RuntimeError("LLM 未配置，无法解析数据集匹配（已取消规则/向量兜底）")

        candidates = self._find_task_folders(intent.task_type)
        if not candidates:
            roots_disp = ", ".join(str(p) for p in self.scan_dirs)
            raise ValueError(
                f"在扫描根 [{roots_disp}] 下未找到任务类型 {intent.task_type} 的文件夹"
            )

        candidate_infos = [self._build_candidate_info(folder) for folder in candidates]
        logger.info(
            "第二层 LLM：在 %d 个扫描根、%d 个候选数据集中选择最匹配项",
            len(self.scan_dirs),
            len(candidate_infos),
        )

        selection = self.llm.select_dataset(
            prompt=task_context if task_context is not None else intent.raw_prompt,
            intent=intent.to_task_dict(),
            candidates=candidate_infos,
        )
        logger.debug("LLM 原始选择结果: %s", selection)
        if not selection:
            raise RuntimeError("LLM 数据集选择失败，无法继续（已取消兜底策略）")

        selected_folder = self._resolve_selected_folder(selection, candidate_infos)
        confidence = float(selection.get("confidence", 0.0))
        if confidence < auto_threshold:
            raise ValueError(
                f"LLM 对数据集匹配不够确信（置信度 {confidence:.3f} < {auto_threshold}）。"
                f"候选: {[c.get('path') or c.get('folder') for c in candidate_infos]}。"
                f"请用 --folder 指定。"
            )

        match = self._score_folder(selected_folder, intent)
        match.total_score = confidence
        match.reasoning = str(selection.get("reasoning", ""))
        self._ensure_category_coverage(match, intent)
        return match

    def resolve_explicit_path(
        self,
        intent: TrainIntent,
        raw_path: Optional[str] = None,
    ) -> Optional[DatasetMatch]:
        """若意图或入参包含可用的数据集路径，则直接构造匹配结果并跳过扫描。"""
        candidates_raw: List[str] = []
        if raw_path:
            candidates_raw.append(str(raw_path).strip())
        for value in (
            getattr(intent, "dataset_path", "") or "",
            getattr(intent, "dataset_hint", "") or "",
        ):
            text = str(value).strip()
            if text and text not in candidates_raw:
                candidates_raw.append(text)

        for text in candidates_raw:
            if not text:
                continue
            # hint 仅是短名时，在各 scan_dir 下相对解析；绝对/像路径的同样走多根搜索
            folder = coerce_existing_dir(text, *self.scan_dirs)

            if folder is None:
                if looks_like_filesystem_path(text) or Path(text).is_absolute() or re.match(r"^[A-Za-z]:", text):
                    raise ValueError(f"指定的数据集路径不存在或不是目录: {text}")
                continue

            match = self._score_folder(folder, intent)
            match.total_score = 1.0
            match.reasoning = f"使用用户指定路径: {folder}"
            logger.info("跳过数据集扫描，使用显式路径: %s -> dataset_dir=%s", folder, match.dataset_dir)
            self._ensure_category_coverage(match, intent)
            return match
        return None

    def resolve_folder_arg(self, folder_name: str, intent: TrainIntent) -> DatasetMatch:
        """解析 --folder：支持绝对路径，或任一 scan_dir 下的相对名。"""
        text = str(folder_name or "").strip()
        if not text:
            raise ValueError("folder 不能为空")

        folder = coerce_existing_dir(text, *self.scan_dirs)
        if folder is None:
            roots_disp = ", ".join(str(p) for p in self.scan_dirs)
            raise ValueError(f"指定文件夹不存在: {text}（scan_dirs=[{roots_disp}]）")

        match = self._score_folder(folder, intent)
        match.total_score = 1.0
        match.reasoning = f"使用 --folder 指定: {folder}"
        self._ensure_category_coverage(match, intent)
        return match

    def _ensure_category_coverage(self, match: DatasetMatch, intent: TrainIntent) -> None:
        """如果用户明确指定了类别，要求至少一半类别能被数据集覆盖。"""
        if intent.categories and match.categories:
            matched_count = sum(1 for v in match.matched_user_categories.values() if v is not None)
            coverage = matched_count / len(intent.categories)
            if coverage < 0.5:
                raise ValueError(
                    f"数据集类别不匹配: 用户要求 {intent.categories}，"
                    f"数据集 {match.folder.name} 只匹配到 {matched_count}/{len(intent.categories)} 个"
                    f"（数据集类别: {match.categories}）。"
                )

    def _find_task_folders(self, task_type: str) -> List[Path]:
        """根据任务类型在所有扫描根下过滤直接子文件夹（整词匹配，避免 det 误匹配 detr）。"""
        existing_roots = [p for p in self.scan_dirs if p.exists()]
        missing = [p for p in self.scan_dirs if not p.exists()]
        for root in missing:
            logger.warning("扫描根目录不存在，已跳过: %s", root)
        if not existing_roots:
            roots_disp = ", ".join(str(p) for p in self.scan_dirs)
            raise ValueError(f"扫描目录均不存在: [{roots_disp}]")

        keywords = TASK_FOLDER_KEYWORDS.get(task_type, [task_type.lower()])
        candidates: List[Path] = []
        seen = set()
        for root in existing_roots:
            try:
                entries = list(root.iterdir())
            except OSError as exc:
                logger.warning("无法读取扫描根 %s: %s", root, exc)
                continue
            for folder in entries:
                if not folder.is_dir():
                    continue
                name_lower = folder.name.lower()
                matched = False
                for kw in keywords:
                    kw_lower = kw.lower()
                    if re.search(
                        rf"(?<![a-z0-9一-龥]){re.escape(kw_lower)}(?![a-z0-9一-龥])",
                        name_lower,
                    ):
                        matched = True
                        break
                if not matched:
                    continue
                key = str(folder.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(folder)
        return candidates

    def _build_candidate_info(self, folder: Path) -> Dict[str, Any]:
        """构造候选文件夹信息，供 LLM 选择。

        多根时同名文件夹可能出现在不同盘，因此同时提供 basename 与绝对 path。
        """
        folder = Path(folder)
        try:
            folder_resolved = folder.resolve()
        except OSError:
            folder_resolved = folder
        dataset_dir = self._find_dataset_dir(folder_resolved)
        categories, total_images = self._detect_categories(dataset_dir or folder_resolved)
        scan_root = self._locate_scan_root(folder_resolved)
        return {
            "folder": folder_resolved.name,
            "path": str(folder_resolved),
            "scan_root": str(scan_root) if scan_root is not None else "",
            "dataset_dir": str(dataset_dir or folder_resolved),
            "categories": categories,
            "total_images": total_images,
        }

    def _locate_scan_root(self, folder: Path) -> Optional[Path]:
        """返回包含该文件夹的扫描根（若可判定）。"""
        try:
            folder_res = folder.resolve()
        except OSError:
            folder_res = folder
        for root in self.scan_dirs:
            try:
                root_res = root.resolve()
            except OSError:
                root_res = root
            try:
                folder_res.relative_to(root_res)
                return root_res
            except ValueError:
                continue
        return None

    def _resolve_selected_folder(
        self,
        selection: Dict[str, Any],
        candidate_infos: List[Dict[str, Any]],
    ) -> Path:
        """把 LLM 返回的 folder/path 解析为真实候选目录。"""
        raw_path = str(selection.get("path") or "").strip()
        raw_folder = str(selection.get("folder") or "").strip()
        token = raw_path or raw_folder
        if not token:
            raise RuntimeError(f"LLM 未返回有效数据集选择: {selection}")

        by_path = {str(c.get("path", "")).lower(): c for c in candidate_infos if c.get("path")}
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidate_infos:
            by_name.setdefault(str(c.get("folder", "")).lower(), []).append(c)

        # 1) 完整 path 精确匹配
        hit = by_path.get(token.lower())
        if hit is None and raw_path:
            hit = by_path.get(raw_path.lower())
        # 2) 规范化 path 再比一次
        if hit is None:
            try:
                norm = str(Path(token).resolve()).lower()
                hit = by_path.get(norm)
            except OSError:
                pass
        # 3) basename；多根重名时若只有 path 能消歧则报错
        if hit is None:
            name_hits = by_name.get(Path(token).name.lower(), [])
            if len(name_hits) == 1:
                hit = name_hits[0]
            elif len(name_hits) > 1:
                paths = [h.get("path") for h in name_hits]
                raise ValueError(
                    f"LLM 选择的文件夹名在多个扫描根下重名: {Path(token).name} -> {paths}。"
                    f"请改用 --folder 指定绝对路径，或让模型返回 path 字段。"
                )

        if hit is None:
            # 4) 后缀/包含匹配（LLM 有时返回 path 的尾段）
            for c in candidate_infos:
                p = str(c.get("path", ""))
                if p and (p.lower().endswith(token.lower()) or token.lower() in p.lower()):
                    hit = c
                    break

        if hit is None:
            raise ValueError(
                f"LLM 选择的文件夹不在候选列表中: {token}；"
                f"候选 path={[c.get('path') for c in candidate_infos]}"
            )

        selected = Path(str(hit["path"]))
        if not selected.exists() or not selected.is_dir():
            raise ValueError(f"LLM 选择的文件夹不存在: {selected}")
        return selected

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
        """对单个文件夹构造匹配结果。

        folder 既可以是「项目目录」（其下再找 训练集/），也可以直接是训练集目录本身。
        """
        folder = Path(folder).resolve()
        nested = self._find_dataset_dir(folder)
        if nested is not None:
            project_dir = folder
            dataset_dir = nested
        else:
            # 用户直接指向训练集目录：自身既是 project 也是 dataset，
            # work/ 会建在该目录下，避免误用上级大目录名。
            project_dir = folder
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
            folder=project_dir,
            dataset_dir=dataset_dir,
            task_type=intent.task_type,
            categories=dataset_cats,
            matched_user_categories=matched,
            total_images=total_images,
        )
