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
    NL_DEFAULT_MAX_CANDIDATES,
    NL_DEFAULT_MAX_DEPTH,
    NL_DEFAULT_MAX_DIRS_VISITED,
    NL_DEFAULT_TOP_K,
    NL_SCORE_GAP_FALLBACK,
    NL_SINGLE_CANDIDATE_SCORE,
    NL_SKIP_DIR_NAMES,
    TASK_FOLDER_KEYWORDS,
    LocationClues,
    coerce_existing_dir,
    extract_location_clues,
    fuzzy_name_match,
    looks_like_filesystem_path,
    normalize_scan_dirs,
    resolve_search_anchors,
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

    支持：
    1. 显式绝对路径
    2. NL 锚点有界深搜（盘符 + 片段/项目名）
    3. 多根浅扫（每个根 depth=1）+ 任务关键字 + LLM 选择
    """

    def __init__(
        self,
        scan_dir: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        llm_client: Optional[LLMClient] = None,
        scan_dirs: Optional[Sequence[Union[str, Path]]] = None,
        *,
        enable_nl_locate: bool = True,
        scan_depth: int = NL_DEFAULT_MAX_DEPTH,
        max_dirs_visited: int = NL_DEFAULT_MAX_DIRS_VISITED,
        max_candidates: int = NL_DEFAULT_MAX_CANDIDATES,
        drive_roots: Optional[Sequence[Union[str, Path]]] = None,
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
        self.enable_nl_locate = bool(enable_nl_locate)
        self.scan_depth = max(1, int(scan_depth))
        self.max_dirs_visited = max(10, int(max_dirs_visited))
        self.max_candidates = max(1, int(max_candidates))
        # 测试可注入：按 clues.drives 顺序映射盘符根，避免依赖真实 Z:\
        self.drive_roots: Optional[List[Path]] = (
            [Path(p) for p in drive_roots] if drive_roots is not None else None
        )

    def resolve(
        self,
        intent: TrainIntent,
        auto_threshold: float = 0.55,
        task_context: Optional[str] = None,
    ) -> DatasetMatch:
        """
        解析最匹配的数据集。

        优先级：显式路径 → NL 锚点深搜 → scan_dirs 浅扫 + LLM。
        """
        explicit = self.resolve_explicit_path(intent)
        if explicit is not None:
            return explicit

        prompt_for_llm = task_context if task_context is not None else intent.raw_prompt

        if self.enable_nl_locate:
            nl_match = self._resolve_via_nl_locate(
                intent,
                auto_threshold=auto_threshold,
                task_context=prompt_for_llm,
            )
            if nl_match is not None:
                return nl_match

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
            prompt=prompt_for_llm,
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

    def _resolve_via_nl_locate(
        self,
        intent: TrainIntent,
        auto_threshold: float,
        task_context: str,
    ) -> Optional[DatasetMatch]:
        """锚点有界深搜；无线索或无候选时返回 None，由调用方 fallback 浅扫。"""
        clues = extract_location_clues(
            prompt=intent.raw_prompt or "",
            dataset_hint=getattr(intent, "dataset_hint", "") or "",
            task_type=intent.task_type or "",
        )
        if not clues.has_searchable_signal():
            return None

        anchors = resolve_search_anchors(
            clues,
            self.scan_dirs,
            drive_roots=self.drive_roots,
        )
        if not anchors:
            logger.info("NL 定位有线索但无锚点，回退浅扫: %s", clues.to_dict())
            return None

        discovered = self.discover_dataset_candidates(
            anchors,
            task_type=intent.task_type,
            clues=clues,
        )
        if not discovered:
            logger.info(
                "NL 深搜未找到候选（anchors=%s），回退浅扫",
                [str(a) for a in anchors],
            )
            return None

        return self._pick_from_discovered(
            discovered,
            intent=intent,
            clues=clues,
            auto_threshold=auto_threshold,
            task_context=task_context,
        )

    def discover_dataset_candidates(
        self,
        anchors: Sequence[Path],
        task_type: str,
        clues: Optional[LocationClues] = None,
    ) -> List[Dict[str, Any]]:
        """在锚点下有界遍历，收集 dataset-like 候选信息。"""
        clues = clues or LocationClues(task_type=task_type)
        task_kws = TASK_FOLDER_KEYWORDS.get(task_type, [task_type.lower()] if task_type else [])
        visited = 0
        seen_paths: set = set()
        scored: List[Dict[str, Any]] = []

        for anchor in anchors:
            anchor = Path(anchor)
            if not anchor.exists():
                continue
            queue: List[Tuple[Path, int]] = [(anchor, 0)]
            while queue:
                if visited >= self.max_dirs_visited:
                    logger.warning(
                        "NL 深搜达到 max_dirs_visited=%d，提前停止",
                        self.max_dirs_visited,
                    )
                    break
                if len(scored) >= self.max_candidates:
                    break

                current, depth = queue.pop(0)
                try:
                    cur_res = current.resolve()
                except OSError:
                    cur_res = current
                key = str(cur_res).lower()
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                visited += 1

                if self._should_skip_dir(cur_res):
                    continue

                reasons = self._dataset_like_reasons(cur_res, task_kws, clues)
                if reasons:
                    info = self._build_candidate_info(cur_res)
                    info["match_reasons"] = reasons
                    info["depth"] = depth
                    info["nl_score"] = self._score_discovered_candidate(info, clues, task_kws)
                    # 无图片的候选降权但仍保留（结构像训练集时可能图片在更深层已被 dataset_dir 统计）
                    scored.append(info)

                if depth >= self.scan_depth:
                    continue

                try:
                    children = list(cur_res.iterdir())
                except OSError:
                    continue
                for child in children:
                    try:
                        if child.is_symlink():
                            continue
                        if not child.is_dir():
                            continue
                        if child.name.startswith("."):
                            continue
                        if child.name.lower() in NL_SKIP_DIR_NAMES:
                            continue
                    except OSError:
                        continue
                    queue.append((child, depth + 1))

            if len(scored) >= self.max_candidates:
                break

        scored.sort(key=lambda c: float(c.get("nl_score", 0.0)), reverse=True)
        scored = self._collapse_nested_candidates(scored)
        if len(scored) > self.max_candidates:
            scored = scored[: self.max_candidates]

        logger.info(
            "NL 深搜完成: visited=%d candidates=%d top=%s",
            visited,
            len(scored),
            [
                (c.get("path"), round(float(c.get("nl_score", 0.0)), 2))
                for c in scored[:5]
            ],
        )
        return scored

    @staticmethod
    def _collapse_nested_candidates(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同一链条上保留项目目录，去掉其下的 1-训练集 等子候选。"""
        if len(scored) <= 1:
            return scored

        def _key(c: Dict[str, Any]) -> str:
            try:
                return str(Path(str(c.get("path", ""))).resolve()).lower()
            except OSError:
                return str(c.get("path", "")).lower()

        # 已按分数降序；优先保留高分，若 A 是 B 的父路径则丢弃子路径
        kept: List[Dict[str, Any]] = []
        kept_keys: List[str] = []
        for c in scored:
            k = _key(c)
            if not k:
                continue
            skip = False
            # 若已有祖先在 kept 中，跳过当前（子目录）
            for pk in kept_keys:
                if k != pk and (k.startswith(pk.rstrip("\\/") + "\\") or k.startswith(pk.rstrip("/") + "/")):
                    skip = True
                    break
            if skip:
                continue
            # 若当前是已保留项的祖先，替换掉子项
            new_kept: List[Dict[str, Any]] = []
            new_keys: List[str] = []
            for prev, pk in zip(kept, kept_keys):
                if pk != k and (pk.startswith(k.rstrip("\\/") + "\\") or pk.startswith(k.rstrip("/") + "/")):
                    continue  # drop child
                new_kept.append(prev)
                new_keys.append(pk)
            new_kept.append(c)
            new_keys.append(k)
            kept, kept_keys = new_kept, new_keys

        kept.sort(key=lambda c: float(c.get("nl_score", 0.0)), reverse=True)
        return kept

    @staticmethod
    def _should_skip_dir(path: Path) -> bool:
        name = path.name.lower()
        if name in NL_SKIP_DIR_NAMES:
            return True
        if name.startswith("."):
            return True
        return False

    def _dataset_like_reasons(
        self,
        folder: Path,
        task_kws: Sequence[str],
        clues: LocationClues,
    ) -> List[str]:
        """返回该目录像数据集的原因列表；空列表表示不入围。"""
        reasons: List[str] = []
        name = folder.name
        name_lower = name.lower()

        for kw in task_kws:
            kw_lower = kw.lower()
            if re.search(
                rf"(?<![a-z0-9一-龥]){re.escape(kw_lower)}(?![a-z0-9一-龥])",
                name_lower,
            ):
                reasons.append(f"task_keyword:{kw}")
                break

        nested = self._find_dataset_dir(folder)
        if nested is not None:
            _, imgs = self._detect_categories(nested)
            if imgs > 0 or nested.exists():
                reasons.append(f"dataset_subdir:{nested.name}")

        if any(tok in name_lower for tok in ("训练", "train", "dataset")):
            _, imgs = self._detect_categories(folder)
            if imgs > 0:
                reasons.append("name_train_with_images")

        # 路径段命中用户项目名：本身不足以入围，但若同时有结构/任务信号会在 score 体现；
        # 若仅有 keyword 命中且其下存在 DATASET_CANDIDATES，上面已覆盖。
        # 额外：folder 自身就是 1-训练集 且有图
        if name in DATASET_CANDIDATES or name_lower in {c.lower() for c in DATASET_CANDIDATES}:
            _, imgs = self._detect_categories(folder)
            if imgs > 0:
                reasons.append("self_is_dataset_dir")

        path_str = str(folder)
        if clues.name_keywords:
            hit_kw = any(
                fuzzy_name_match(seg, kw)
                for seg in folder.parts
                for kw in clues.name_keywords
            )
            if hit_kw and (nested is not None or any(r.startswith("task_keyword") for r in reasons)):
                reasons.append("path_keyword_with_structure")

        # 去重 reasons 但保持
        return list(dict.fromkeys(reasons))

    def _score_discovered_candidate(
        self,
        info: Dict[str, Any],
        clues: LocationClues,
        task_kws: Sequence[str],
    ) -> float:
        score = 0.0
        path = str(info.get("path") or "")
        folder_name = str(info.get("folder") or "")
        parts_l = [p.lower() for p in Path(path).parts] if path else []

        for kw in clues.name_keywords:
            kw_l = kw.lower()
            if any(fuzzy_name_match(p, kw) for p in Path(path).parts):
                score += 3.0
            elif fuzzy_name_match(folder_name, kw):
                score += 3.0

        for frag in clues.path_fragments:
            if any(fuzzy_name_match(p, frag) for p in Path(path).parts):
                score += 1.5

        for kw in task_kws:
            if re.search(
                rf"(?<![a-z0-9一-龥]){re.escape(kw.lower())}(?![a-z0-9一-龥])",
                folder_name.lower(),
            ) or any(kw.lower() in p for p in parts_l):
                score += 2.0
                break

        reasons = info.get("match_reasons") or []
        if any(str(r).startswith("dataset_subdir") for r in reasons):
            score += 2.0
        if any(str(r) == "self_is_dataset_dir" for r in reasons):
            score += 1.5

        if int(info.get("total_images") or 0) > 0:
            score += 1.0

        depth = int(info.get("depth") or 0)
        score -= 0.1 * depth
        return score

    def _pick_from_discovered(
        self,
        discovered: List[Dict[str, Any]],
        intent: TrainIntent,
        clues: LocationClues,
        auto_threshold: float,
        task_context: str,
    ) -> DatasetMatch:
        """从深搜候选中选出最终 DatasetMatch。"""
        if not discovered:
            raise ValueError("NL 深搜候选为空")

        top = discovered[:NL_DEFAULT_TOP_K]
        best = top[0]
        best_score = float(best.get("nl_score", 0.0))
        second_score = float(top[1].get("nl_score", 0.0)) if len(top) > 1 else -999.0

        # 单一高分或显著领先：规则直选
        if len(top) == 1 and best_score >= NL_SINGLE_CANDIDATE_SCORE:
            return self._match_from_candidate_info(
                best,
                intent,
                reasoning=f"NL 定位唯一候选 (score={best_score:.2f}): {best.get('path')}",
                total_score=max(best_score / 10.0, 0.9),
            )

        if len(top) > 1 and (best_score - second_score) >= NL_SCORE_GAP_FALLBACK and best_score >= NL_SINGLE_CANDIDATE_SCORE:
            return self._match_from_candidate_info(
                best,
                intent,
                reasoning=(
                    f"NL 定位规则优选 (score={best_score:.2f}, gap={best_score - second_score:.2f}): "
                    f"{best.get('path')}"
                ),
                total_score=max(best_score / 10.0, 0.85),
            )

        if not self.llm.available:
            if best_score >= NL_SINGLE_CANDIDATE_SCORE:
                return self._match_from_candidate_info(
                    best,
                    intent,
                    reasoning=f"NL 定位无 LLM，采用最高分候选: {best.get('path')}",
                    total_score=max(best_score / 10.0, 0.8),
                )
            raise RuntimeError(
                "LLM 未配置，且 NL 深搜无法唯一确定数据集。"
                f"候选: {[c.get('path') for c in top]}"
            )

        logger.info(
            "NL 深搜 %d 个候选，交由 LLM 消歧（top score=%.2f）",
            len(top),
            best_score,
        )
        # 附带线索，帮助模型优先项目名
        enriched_prompt = (
            f"{task_context}\n"
            f"[位置线索] drives={clues.drives}, fragments={clues.path_fragments}, "
            f"keywords={clues.name_keywords}"
        )
        selection = self.llm.select_dataset(
            prompt=enriched_prompt,
            intent=intent.to_task_dict(),
            candidates=top,
        )
        if not selection:
            if best_score - second_score >= NL_SCORE_GAP_FALLBACK * 0.5 and best_score >= 2.0:
                return self._match_from_candidate_info(
                    best,
                    intent,
                    reasoning=f"LLM 选择失败，回退 NL 最高分: {best.get('path')}",
                    total_score=max(best_score / 10.0, 0.75),
                )
            raise RuntimeError("LLM 数据集选择失败，无法继续（NL 深搜候选消歧失败）")

        try:
            selected_folder = self._resolve_selected_folder(selection, top)
        except ValueError:
            if best_score >= 2.0:
                selected_folder = Path(str(best["path"]))
                selection = {
                    "reasoning": f"LLM 返回无效，回退最高分 {best.get('path')}",
                    "confidence": max(best_score / 10.0, 0.7),
                }
            else:
                raise

        confidence = float(selection.get("confidence", 0.0) or 0.0)
        if confidence < auto_threshold and best_score >= NL_SINGLE_CANDIDATE_SCORE:
            # NL 分数够高时放宽
            confidence = max(confidence, 0.8)

        if confidence < auto_threshold:
            raise ValueError(
                f"LLM 对 NL 定位候选不够确信（置信度 {confidence:.3f} < {auto_threshold}）。"
                f"候选: {[c.get('path') for c in top]}。请用更明确的路径或 --folder。"
            )

        match = self._score_folder(selected_folder, intent)
        match.total_score = confidence
        match.reasoning = str(selection.get("reasoning", "") or f"NL+LLM 选择: {selected_folder}")
        self._ensure_category_coverage(match, intent)
        return match

    def _match_from_candidate_info(
        self,
        info: Dict[str, Any],
        intent: TrainIntent,
        reasoning: str,
        total_score: float,
    ) -> DatasetMatch:
        folder = Path(str(info.get("path") or info.get("folder")))
        match = self._score_folder(folder, intent)
        match.total_score = float(total_score)
        match.reasoning = reasoning
        self._ensure_category_coverage(match, intent)
        logger.info("NL 定位命中: %s -> dataset_dir=%s", match.folder, match.dataset_dir)
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
                # LLM 常把「z盘zzh的易格」脑补成 Z:\zzh\易格技术 这类不存在路径。
                # 硬失败会阻断 NL 锚点深搜；改为跳过并继续后续策略。
                if looks_like_filesystem_path(text) or Path(text).is_absolute() or re.match(
                    r"^[A-Za-z]:", text
                ):
                    logger.warning(
                        "显式数据集路径不存在，跳过并尝试 NL/浅扫定位: %s",
                        text,
                    )
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
