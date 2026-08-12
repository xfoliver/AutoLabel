"""
训练集数据质量检查（规则引擎，不依赖大模型）。

独立于 auto_train4.0 训练流水线，可被 CLI ``check_dataset.py`` 调用，
后续若要接入训练门禁，只需 import 本模块的 ``check_dataset``。

检查范围（训练就绪性 + 分布风险）：
  - 有效图片数量 / 坏图（无法解码、空文件）
  - 【分类 CLS】只按「一类一文件夹」统计类别与长尾，不扫描、不比对 JSON
  - 【检测/分割/OCR/旋转】图↔JSON 配对 + JSON 字段/几何
  - 空标注、几何越界、shape_type 不匹配（非分类）
  - 类别直方图与长尾（分类=文件夹计数；检测/分割=shapes.label）
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .dataset_fs import (
    DATASET_CANDIDATES,
    IMAGE_EXTENSIONS,
    find_dataset_dir as _fs_find_dataset_dir,
)
from .task_types import normalize_task_type as _normalize_task_type_core

logger = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

# 扫描时跳过的目录名（小写）
_SKIP_DIR_NAMES = {
    "work",
    "checkpoints",
    "__pycache__",
    ".git",
    "node_modules",
    "baseline models",
    "models",
    "model",
    "输出",
    "模型输出",
    "模型",
    "annotations",
    "labels",
    "images",
    "ok",
    "ng",
}

# 分类任务在统计子目录时额外跳过的名字
_CLS_SKIP_DIR_NAMES = _SKIP_DIR_NAMES | {
    "train",
    "val",
    "test",
    "valid",
    "训练集",
    "验证集",
    "测试集",
}

# 任务 -> 允许的 shape_type（空集合表示不校验类型）
_ALLOWED_SHAPE_TYPES: Dict[str, Set[str]] = {
    "OCR": set(),  # OCR 主要看 flags，shapes 可选
    "分类": set(),
    "目标检测": {"rectangle"},
    "DETR": {"rectangle", "polygon"},
    "实例分割": {"polygon", "rectangle"},
    "语义分割": {"polygon", "rectangle", "circle"},
    "旋转框检测": {"rotation"},
}

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class QualityPolicy:
    """可配置阈值。"""

    min_images: int = 10
    min_pair_rate: float = 0.9
    # 配对率低于该值升为 error（默认更严的硬门禁）
    error_pair_rate: float = 0.5
    max_corrupt_ratio: float = 0.02
    max_empty_label_ratio: float = 0.2
    warn_empty_label_ratio: float = 0.05
    min_per_class: int = 5
    warn_imbalance_ratio: float = 20.0
    error_imbalance_ratio: float = 100.0
    geom_tol_px: float = 2.0
    min_box_side: float = 1.0
    max_samples: int = 20
    # 图像解码：sample | full | off
    image_decode: str = "sample"
    decode_sample_size: int = 50
    # 是否把「仅警告」也算作不通过（默认否，仅 error 不通过）
    fail_on_warning: bool = False
    seed: int = 42


@dataclass
class Issue:
    code: str
    severity: str  # error | warn | info
    message: str
    count: int = 0
    samples: List[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    task_type: str
    dataset_dir: str
    scanned_at: str
    pass_: bool = True  # JSON 字段名 "pass"
    grade: str = "F"
    score: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    label_histogram: Dict[str, int] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "dataset_dir": self.dataset_dir,
            "scanned_at": self.scanned_at,
            "pass": self.pass_,
            "grade": self.grade,
            "score": round(self.score, 4),
            "summary": self.summary,
            "stats": self.stats,
            "label_histogram": self.label_histogram,
            "issues": [i.to_dict() for i in self.issues],
            "policy": self.policy,
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def normalize_task_type(task: Optional[str]) -> str:
    """将用户输入规范为内部任务名；无法识别时原样返回 strip 结果。"""
    return _normalize_task_type_core(task, unknown="keep")


def find_dataset_dir(path: Union[str, Path]) -> Path:
    """
    解析实际训练集目录。

    优先进入 DATASET_CANDIDATES（训练集/train/…），避免停在项目根
    （根下往往还有 work/模型，且分类的类文件夹在「训练集」内）。
    实现见 ``autotrain4.dataset_fs.find_dataset_dir``（单源）。
    """
    return _fs_find_dataset_dir(path, require_images=True)

def _has_direct_images(directory: Path) -> bool:
    try:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def _has_class_subdirs(directory: Path) -> bool:
    """是否存在「子文件夹内有图」的分类结构。"""
    try:
        for p in directory.iterdir():
            if not p.is_dir():
                continue
            if p.name.lower() in _CLS_SKIP_DIR_NAMES or p.name in _CLS_SKIP_DIR_NAMES:
                continue
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    return True
    except OSError:
        return False
    return False


def _count_images_flat_or_nested(directory: Path, max_files: int = 0) -> int:
    n = 0
    try:
        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            if any(part.lower() in _SKIP_DIR_NAMES for part in p.parts):
                continue
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                n += 1
                if max_files and n >= max_files:
                    return n
    except OSError:
        return n
    return n


def _iter_images(dataset_dir: Path) -> List[Path]:
    images: List[Path] = []
    for p in dataset_dir.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(dataset_dir).parts
        if any(part.lower() in _SKIP_DIR_NAMES for part in rel_parts[:-1]):
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return sorted(images)


def _iter_jsons(dataset_dir: Path) -> List[Path]:
    jsons: List[Path] = []
    for p in dataset_dir.rglob("*.json"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(dataset_dir).parts
        if any(part.lower() in _SKIP_DIR_NAMES for part in rel_parts[:-1]):
            continue
        jsons.append(p)
    return sorted(jsons)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# 坏图
# ---------------------------------------------------------------------------


def check_image_file(path: Path) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    返回 (status, size_wh)。
    status: OK | EMPTY_FILE | TOO_SMALL_FILE | DECODE_FAIL | NO_PIL | INVALID_SIZE
    """
    try:
        size = path.stat().st_size
    except OSError:
        return "DECODE_FAIL", None
    if size == 0:
        return "EMPTY_FILE", None
    if size < 32:
        return "TOO_SMALL_FILE", None
    if Image is None:
        return "NO_PIL", None
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if w < 1 or h < 1:
                return "INVALID_SIZE", (w, h)
            return "OK", (w, h)
    except Exception:
        return "DECODE_FAIL", None


# ---------------------------------------------------------------------------
# JSON / 标注
# ---------------------------------------------------------------------------


def _load_label_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except Exception as e:
            return None, f"encoding: {e}"
    except OSError as e:
        return None, f"io: {e}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json: {e}"
    if not isinstance(data, dict):
        return None, "json: root is not object"
    return data, None


def _ocr_has_label(data: dict) -> bool:
    flags = data.get("flags")
    if isinstance(flags, dict):
        for k, v in flags.items():
            # 文本在 key 上（DLCV OCR 常见：{ "识别文字": true }）
            if str(k).strip() and str(k).strip() not in {"", "null"}:
                if v is True or v is False or v is None or str(v).strip() != "":
                    return True
            if v is not None and str(v).strip() and str(v).lower() not in {"true", "false"}:
                return True
    shapes = data.get("shapes")
    if isinstance(shapes, list):
        for sh in shapes:
            if not isinstance(sh, dict):
                continue
            if str(sh.get("label") or "").strip():
                return True
            pts = sh.get("points")
            if isinstance(pts, list) and len(pts) >= 2:
                return True
    return False


def _points_ok(points: Any) -> bool:
    if not isinstance(points, list) or not points:
        return False
    for pt in points:
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            return False
        try:
            float(pt[0])
            float(pt[1])
        except (TypeError, ValueError):
            return False
    return True


def _shape_geometry_issues(
    shape: dict,
    w: Optional[int],
    h: Optional[int],
    tol: float,
    min_side: float,
) -> List[str]:
    """返回该 shape 的几何问题代码列表（可空）。"""
    problems: List[str] = []
    points = shape.get("points")
    if not _points_ok(points):
        problems.append("BAD_POINTS")
        return problems

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if w is not None and h is not None and w > 0 and h > 0:
        for x, y in zip(xs, ys):
            if x < -tol or y < -tol or x > w + tol or y > h + tol:
                problems.append("OOB")
                break

    st = str(shape.get("shape_type") or "").lower()
    if st == "rectangle" and len(points) >= 2:
        bw = abs(xs[1] - xs[0])
        bh = abs(ys[1] - ys[0])
        if bw < min_side or bh < min_side:
            problems.append("DEGENERATE")
    elif st in {"polygon", "rotation"} and len(points) >= 3:
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw < min_side or bh < min_side:
            problems.append("DEGENERATE")
    return problems


def validate_label_data(
    data: dict,
    task_type: str,
    *,
    image_wh: Optional[Tuple[int, int]] = None,
    policy: Optional[QualityPolicy] = None,
) -> Dict[str, Any]:
    """
    校验单份标注。返回:
      has_label, labels[], shape_types Counter-able list,
      problems[] (str codes), dim_mismatch bool
    """
    policy = policy or QualityPolicy()
    result: Dict[str, Any] = {
        "has_label": False,
        "labels": [],
        "shape_types": [],
        "problems": [],
        "dim_mismatch": False,
        "n_shapes": 0,
    }

    w = h = None
    if image_wh:
        w, h = image_wh
    jw, jh = data.get("imageWidth"), data.get("imageHeight")
    try:
        jw_i = int(jw) if jw is not None else None
        jh_i = int(jh) if jh is not None else None
    except (TypeError, ValueError):
        jw_i = jh_i = None
        result["problems"].append("BAD_IMAGE_SIZE_FIELD")

    if w and h and jw_i and jh_i:
        if abs(w - jw_i) > 1 or abs(h - jh_i) > 1:
            result["dim_mismatch"] = True
    if w is None and jw_i:
        w = jw_i
    if h is None and jh_i:
        h = jh_i

    if task_type == "OCR":
        result["has_label"] = _ocr_has_label(data)
        if not result["has_label"]:
            result["problems"].append("EMPTY_LABEL")
        # OCR flags 文本可作为「类」统计（截断过长 key）
        flags = data.get("flags")
        if isinstance(flags, dict):
            for k in flags.keys():
                lab = str(k).strip()
                if lab:
                    result["labels"].append(lab if len(lab) <= 64 else lab[:61] + "...")
        return result

    if task_type == "分类":
        result["has_label"] = True
        return result

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        result["problems"].append("NO_SHAPES_FIELD")
        result["has_label"] = False
        return result

    result["n_shapes"] = len(shapes)
    allowed = _ALLOWED_SHAPE_TYPES.get(task_type, set())

    if len(shapes) == 0:
        result["has_label"] = False
        result["problems"].append("EMPTY_LABEL")
        return result

    any_valid = False
    for sh in shapes:
        if not isinstance(sh, dict):
            result["problems"].append("BAD_SHAPE")
            continue
        label = str(sh.get("label") or "").strip()
        st = str(sh.get("shape_type") or "").strip().lower()
        if st:
            result["shape_types"].append(st)
        if label:
            result["labels"].append(label)

        if allowed and st and st not in allowed:
            result["problems"].append("BAD_SHAPE_TYPE")

        if task_type == "旋转框检测":
            if st != "rotation":
                result["problems"].append("BAD_SHAPE_TYPE")
            if "direction" not in sh:
                result["problems"].append("MISSING_DIRECTION")
            else:
                try:
                    float(sh["direction"])
                except (TypeError, ValueError):
                    result["problems"].append("BAD_DIRECTION")
            pts = sh.get("points")
            if not _points_ok(pts) or len(pts) < 4:
                result["problems"].append("BAD_POINTS")
            else:
                any_valid = True
        elif task_type == "目标检测":
            pts = sh.get("points")
            if not _points_ok(pts) or len(pts) < 2:
                result["problems"].append("BAD_POINTS")
            else:
                any_valid = True
        else:
            # DETR / 分割等
            pts = sh.get("points")
            need = 3 if st == "polygon" else 2
            if st == "circle":
                need = 2
            if not _points_ok(pts) or len(pts) < need:
                result["problems"].append("BAD_POINTS")
            else:
                any_valid = True

        for g in _shape_geometry_issues(sh, w, h, policy.geom_tol_px, policy.min_box_side):
            result["problems"].append(g)

        if label:
            any_valid = True

    result["has_label"] = any_valid and not (
        len(shapes) > 0 and all(str(s.get("label") or "").strip() == "" for s in shapes if isinstance(s, dict))
        and task_type != "语义分割"
    )
    # 有 shape 但全无 label：仍算有几何，检测任务标 EMPTY 倾向
    if len(shapes) > 0 and not any(str(s.get("label") or "").strip() for s in shapes if isinstance(s, dict)):
        if task_type in {"目标检测", "DETR", "实例分割", "旋转框检测"}:
            result["problems"].append("MISSING_LABEL_NAME")

    if not result["has_label"] and "EMPTY_LABEL" not in result["problems"]:
        result["problems"].append("EMPTY_LABEL")

    return result


# ---------------------------------------------------------------------------
# 分类目录
# ---------------------------------------------------------------------------


def _detect_cls_histogram(dataset_dir: Path) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    # 仅统计「子目录里有图」的类文件夹
    class_dirs = []
    for d in subdirs:
        if d.name.lower() in _CLS_SKIP_DIR_NAMES or d.name in _CLS_SKIP_DIR_NAMES:
            continue
        n = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        if n > 0:
            class_dirs.append((d.name, n))
    if class_dirs:
        return {name: n for name, n in sorted(class_dirs)}
    # 无子类目录：整包算一个 unknown
    n = sum(1 for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    if n:
        return {"_root_": n}
    return {}


# ---------------------------------------------------------------------------
# 主检查
# ---------------------------------------------------------------------------


def check_dataset(
    dataset_dir: Union[str, Path],
    task_type: str,
    *,
    policy: Optional[QualityPolicy] = None,
    resolve_train_subdir: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> QualityReport:
    """
    对训练集目录做规则质检，返回 QualityReport。
    """
    policy = policy or QualityPolicy()
    task = normalize_task_type(task_type)
    if not task:
        raise ValueError("task_type 不能为空")

    root_in = Path(dataset_dir).expanduser().resolve()
    if not root_in.is_dir():
        raise FileNotFoundError(f"目录不存在: {root_in}")

    data_dir = find_dataset_dir(root_in) if resolve_train_subdir else root_in
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def prog(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    prog(f"扫描目录: {data_dir}  task={task}")

    issues: List[Issue] = []
    sample_cap = max(1, policy.max_samples)

    def add_issue(
        code: str,
        severity: str,
        message: str,
        samples: Optional[Sequence[str]] = None,
        count: int = 0,
        suggestion: str = "",
    ) -> None:
        ss = list(samples or [])[:sample_cap]
        issues.append(
            Issue(
                code=code,
                severity=severity,
                message=message,
                count=count or len(ss),
                samples=ss,
                suggestion=suggestion,
            )
        )

    images = _iter_images(data_dir)
    n_images = len(images)
    # 分类：标签=文件夹名，不扫描、不比对 JSON
    # 其他任务：才收集旁路 JSON 做图↔标配对
    is_cls = task == "分类"
    jsons: List[Path] = [] if is_cls else _iter_jsons(data_dir)
    n_jsons = len(jsons)

    # ---- 基础数量 ----
    if n_images == 0:
        add_issue(
            "E_NO_IMAGES",
            "error",
            "未找到任何图片文件",
            suggestion="确认路径是否为训练集目录，或图片扩展名是否支持",
        )
        return _finalize_report(
            task, data_dir, now, issues, {}, policy,
            stats={
                "n_images": 0,
                "n_jsons": 0 if is_cls else n_jsons,
                "label_mode": "folder" if is_cls else "json",
                "resolved_from": str(root_in),
            },
        )

    if n_images < policy.min_images:
        add_issue(
            "E_TOO_FEW_IMAGES",
            "error",
            f"有效图片过少: {n_images} < min_images={policy.min_images}",
            count=n_images,
            suggestion="补充样本或降低 min_images 阈值",
        )
    elif n_images < policy.min_images * 2:
        add_issue(
            "W_FEW_IMAGES",
            "warn",
            f"图片数量偏少: {n_images}",
            count=n_images,
            suggestion="小样本训练不稳定，建议补充数据",
        )

    # ---- 分类任务 ----
    label_hist: Dict[str, int] = {}
    shape_type_counter: Counter = Counter()
    paired = 0
    missing_json: List[str] = []
    orphan_json: List[str] = []
    corrupt_json: List[str] = []
    empty_labels: List[str] = []
    bad_shape_type: List[str] = []
    missing_direction: List[str] = []
    bad_points: List[str] = []
    oob: List[str] = []
    degenerate: List[str] = []
    dim_mismatch: List[str] = []
    missing_label_name: List[str] = []

    corrupt_images: List[str] = []
    image_sizes: Dict[str, Tuple[int, int]] = {}
    ext_counter: Counter = Counter(p.suffix.lower() for p in images)

    # 解码策略
    decode_mode = (policy.image_decode or "sample").lower()
    rng = random.Random(policy.seed)
    if decode_mode == "off":
        to_decode: List[Path] = []
    elif decode_mode == "full":
        to_decode = list(images)
    else:
        k = min(len(images), max(0, policy.decode_sample_size))
        to_decode = rng.sample(list(images), k) if k < len(images) else list(images)

    if is_cls:
        prog(
            f"[分类] 按文件夹类别检查：图片 {n_images} 张；"
            f"解码检查 {len(to_decode)} 张 ({decode_mode})；跳过 JSON 配对"
        )
    else:
        prog(f"图片 {n_images} 张，JSON {n_jsons} 个；解码检查 {len(to_decode)} 张 ({decode_mode})")

    for img in to_decode:
        status, wh = check_image_file(img)
        if status in {"EMPTY_FILE", "TOO_SMALL_FILE", "DECODE_FAIL", "INVALID_SIZE"}:
            corrupt_images.append(_rel(img, data_dir))
        elif status == "OK" and wh:
            image_sizes[str(img)] = wh
        elif status == "NO_PIL":
            pass

    if Image is None and decode_mode != "off":
        add_issue(
            "W_NO_PIL",
            "warn",
            "未安装 Pillow，跳过坏图解码检查（仅做文件大小检查）",
            suggestion="pip install Pillow",
        )
        # 无 PIL 时仍做空文件检查
        for img in images:
            try:
                if img.stat().st_size == 0:
                    corrupt_images.append(_rel(img, data_dir))
            except OSError:
                corrupt_images.append(_rel(img, data_dir))

    if is_cls:
        # -------- 分类专用：只认「一类一文件夹」，不做图↔JSON --------
        label_hist = _detect_cls_histogram(data_dir)
        if not label_hist or (len(label_hist) == 1 and "_root_" in label_hist):
            add_issue(
                "W_CLS_NO_SUBDIR",
                "warn",
                "未检测到「一类一文件夹」结构，分类任务以子目录名为类别、不需要 JSON 标注",
                suggestion="将图片按类别放入子文件夹，例如 train/猫/*.jpg、train/狗/*.jpg",
            )
        empty_class_dirs = []
        for d in data_dir.iterdir():
            if not d.is_dir() or d.name.lower() in _CLS_SKIP_DIR_NAMES or d.name in _CLS_SKIP_DIR_NAMES:
                continue
            n = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if n == 0:
                empty_class_dirs.append(d.name)
        if empty_class_dirs:
            add_issue(
                "W_EMPTY_CLASS_DIR",
                "warn",
                f"存在空类别目录: {len(empty_class_dirs)}",
                samples=empty_class_dirs,
                count=len(empty_class_dirs),
                suggestion="删除空目录或补图",
            )
        # 根目录散落图片（未进入任何类文件夹）
        loose = [
            _rel(p, data_dir)
            for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if loose and len(label_hist) > 0 and "_root_" not in label_hist:
            add_issue(
                "W_LOOSE_IMAGES",
                "warn",
                f"根目录下有 {len(loose)} 张未归类图片（分类不会使用这些图）",
                samples=loose,
                count=len(loose),
                suggestion="移入对应类别文件夹",
            )
        # 若目录里夹杂 JSON，明确提示「已忽略」，避免用户以为要配对
        stray_json = 0
        try:
            stray_json = sum(
                1
                for p in data_dir.rglob("*.json")
                if p.is_file()
                and not any(
                    part.lower() in _SKIP_DIR_NAMES
                    for part in p.relative_to(data_dir).parts[:-1]
                )
            )
        except OSError:
            stray_json = 0
        if stray_json > 0:
            add_issue(
                "I_CLS_IGNORE_JSON",
                "info",
                f"分类任务已忽略目录内 {stray_json} 个 JSON（类别以文件夹名为准，无需图↔JSON 配对）",
                count=stray_json,
                suggestion="可保留或删除这些 JSON，均不影响分类质检与训练就绪判断",
            )
        prog(
            f"[分类] 类别 {len([k for k in label_hist if k != '_root_'])} 个: "
            + ", ".join(f"{k}={v}" for k, v in list(label_hist.items())[:8])
        )
    else:
        # 图 json 配对：同 stem，json 可在同目录或任意子路径（stem 全局唯一优先）
        json_by_stem: Dict[str, List[Path]] = defaultdict(list)
        for j in jsons:
            json_by_stem[j.stem].append(j)

        image_by_stem: Dict[str, List[Path]] = defaultdict(list)
        for im in images:
            image_by_stem[im.stem].append(im)

        paired_stems = set(image_by_stem.keys()) & set(json_by_stem.keys())
        paired = sum(len(image_by_stem[s]) for s in paired_stems)
        # 更直观：有 json 的图片数
        n_img_with_json = 0
        for im in images:
            if im.stem in json_by_stem:
                n_img_with_json += 1
            else:
                missing_json.append(_rel(im, data_dir))
        paired = n_img_with_json

        for stem, jlist in json_by_stem.items():
            if stem not in image_by_stem:
                for j in jlist:
                    orphan_json.append(_rel(j, data_dir))

        prog(f"配对: {paired}/{n_images}，开始解析 JSON…")

        # 解析有图的 json
        checked = 0
        for im in images:
            jlist = json_by_stem.get(im.stem) or []
            if not jlist:
                continue
            # 优先同目录 json
            jpath = None
            for j in jlist:
                if j.parent == im.parent:
                    jpath = j
                    break
            if jpath is None:
                jpath = jlist[0]

            data, err = _load_label_json(jpath)
            rel_j = _rel(jpath, data_dir)
            if err or data is None:
                corrupt_json.append(rel_j)
                continue

            wh = image_sizes.get(str(im))
            # 未解码到的图，不强制 decode；几何用 json 内尺寸
            detail = validate_label_data(data, task, image_wh=wh, policy=policy)
            checked += 1

            for lab in detail["labels"]:
                label_hist[lab] = label_hist.get(lab, 0) + 1
            for st in detail["shape_types"]:
                shape_type_counter[st] += 1

            rel_im = _rel(im, data_dir)
            probs = detail["problems"]
            if not detail["has_label"] or "EMPTY_LABEL" in probs:
                empty_labels.append(rel_im)
            if "BAD_SHAPE_TYPE" in probs:
                bad_shape_type.append(rel_im)
            if "MISSING_DIRECTION" in probs or "BAD_DIRECTION" in probs:
                missing_direction.append(rel_im)
            if "BAD_POINTS" in probs:
                bad_points.append(rel_im)
            if "OOB" in probs:
                oob.append(rel_im)
            if "DEGENERATE" in probs:
                degenerate.append(rel_im)
            if detail.get("dim_mismatch"):
                dim_mismatch.append(rel_im)
            if "MISSING_LABEL_NAME" in probs:
                missing_label_name.append(rel_im)

        prog(f"已解析标注 {checked} 份")

    # ---- 汇总规则 ----
    # 分类：无配对概念；其他任务：paired / n_images
    if is_cls:
        pair_rate = None  # type: ignore
        paired = n_images  # 仅表示纳入统计的图片数，非 JSON 配对
    else:
        pair_rate = paired / n_images if n_images else 0.0

    if decode_mode == "off":
        corrupt_ratio = 0.0
    elif decode_mode == "full":
        corrupt_ratio = len(corrupt_images) / n_images if n_images else 0.0
    else:
        # 抽样比例估计
        corrupt_ratio = len(corrupt_images) / max(len(to_decode), 1)

    if not is_cls:
        # 配对率为 0 且目录像「一类一文件夹」时，优先提示任务类型选错
        # （例如分类集却选了 det，前端会误报 E_PAIR_RATE）
        cls_hist_hint: Dict[str, int] = {}
        looks_like_cls = False
        if pair_rate is not None and pair_rate <= 0.0 and n_jsons == 0:
            try:
                cls_hist_hint = _detect_cls_histogram(data_dir)
                looks_like_cls = bool(
                    cls_hist_hint
                    and "_root_" not in cls_hist_hint
                    and len(cls_hist_hint) >= 2
                )
            except OSError:
                cls_hist_hint = {}
                looks_like_cls = False

        if looks_like_cls:
            add_issue(
                "E_TASK_MISMATCH_CLS",
                "error",
                (
                    f"当前任务为「{task}」，但目录呈「一类一文件夹」且无 JSON"
                    f"（图 {n_images} 张、类文件夹 {len(cls_hist_hint)} 个）。"
                    "分类任务不应检查图↔JSON 配对。"
                ),
                count=n_images,
                suggestion="请将任务类型改为「分类 / cls」后重新检查；分类以子目录名为类别",
            )
        elif pair_rate is not None and pair_rate < policy.error_pair_rate:
            add_issue(
                "E_PAIR_RATE",
                "error",
                f"图↔JSON 配对率过低: {pair_rate:.1%} ({paired}/{n_images})",
                samples=missing_json,
                count=len(missing_json),
                suggestion=(
                    "为未标注图片补 JSON，或移出训练集；"
                    "若这是分类数据集，请把任务类型改为「分类/cls」（分类不检查 JSON 配对）"
                ),
            )
        elif pair_rate is not None and pair_rate < policy.min_pair_rate:
            add_issue(
                "W_PAIR_RATE",
                "warn",
                f"图↔JSON 配对率偏低: {pair_rate:.1%} ({paired}/{n_images})",
                samples=missing_json,
                count=len(missing_json),
                suggestion="建议补全标注，避免平台跳过或当负样本行为不一致",
            )
        elif missing_json:
            add_issue(
                "I_MISSING_JSON",
                "info",
                f"有 {len(missing_json)} 张图缺少同名 JSON",
                samples=missing_json,
                count=len(missing_json),
            )

        if orphan_json:
            add_issue(
                "W_ORPHAN_JSON",
                "warn",
                f"有 {len(orphan_json)} 个 JSON 无对应图片",
                samples=orphan_json,
                count=len(orphan_json),
                suggestion="清理残留标注或找回图片",
            )

        if corrupt_json:
            add_issue(
                "E_CORRUPT_JSON",
                "error",
                f"有 {len(corrupt_json)} 个 JSON 无法解析",
                samples=corrupt_json,
                count=len(corrupt_json),
                suggestion="检查编码与 JSON 语法",
            )

        n_paired_for_empty = max(paired - len(corrupt_json), 1)
        empty_ratio = len(empty_labels) / n_paired_for_empty if paired else 0.0
        # 上面分母粗糙，改用 paired 成功解析近似
        n_ok_parse = paired - len(corrupt_json)
        if n_ok_parse > 0:
            empty_ratio = len(empty_labels) / n_ok_parse
        else:
            empty_ratio = 0.0

        if empty_ratio >= policy.max_empty_label_ratio and empty_labels:
            add_issue(
                "E_EMPTY_LABEL",
                "error",
                f"空标注比例过高: {empty_ratio:.1%} ({len(empty_labels)})",
                samples=empty_labels,
                count=len(empty_labels),
                suggestion="OCR 请确认 flags 有文本；检测/分割请确认 shapes 非空",
            )
        elif empty_ratio >= policy.warn_empty_label_ratio and empty_labels:
            add_issue(
                "W_EMPTY_LABEL",
                "warn",
                f"存在空标注: {empty_ratio:.1%} ({len(empty_labels)})",
                samples=empty_labels,
                count=len(empty_labels),
            )

        if bad_shape_type:
            sev = "error" if len(bad_shape_type) / max(paired, 1) > 0.5 else "warn"
            add_issue(
                "E_SHAPE_TYPE" if sev == "error" else "W_SHAPE_TYPE",
                sev,
                f"shape_type 与任务不匹配: {len(bad_shape_type)} 个文件",
                samples=bad_shape_type,
                count=len(bad_shape_type),
                suggestion=f"任务 {task} 允许类型: {sorted(_ALLOWED_SHAPE_TYPES.get(task, []))}",
            )

        if task == "旋转框检测" and missing_direction:
            add_issue(
                "E_MISSING_DIRECTION",
                "error",
                f"旋转框缺少 direction: {len(missing_direction)}",
                samples=missing_direction,
                count=len(missing_direction),
                suggestion="为 rotation 标注补充 direction 字段",
            )

        if bad_points:
            add_issue(
                "W_BAD_POINTS",
                "warn",
                f"点集非法或点数不足: {len(bad_points)}",
                samples=bad_points,
                count=len(bad_points),
            )
        if oob:
            add_issue(
                "W_OOB",
                "warn",
                f"标注坐标越出图像范围: {len(oob)}",
                samples=oob,
                count=len(oob),
                suggestion="检查 imageWidth/Height 或框坐标",
            )
        if degenerate:
            add_issue(
                "W_DEGENERATE",
                "warn",
                f"存在退化框/过小区域: {len(degenerate)}",
                samples=degenerate,
                count=len(degenerate),
            )
        if dim_mismatch:
            add_issue(
                "W_DIM_MISMATCH",
                "warn",
                f"JSON 中宽高与图片实际尺寸不一致: {len(dim_mismatch)}",
                samples=dim_mismatch,
                count=len(dim_mismatch),
            )
        if missing_label_name:
            add_issue(
                "W_MISSING_LABEL_NAME",
                "warn",
                f"shape 缺少 label 名称: {len(missing_label_name)}",
                samples=missing_label_name,
                count=len(missing_label_name),
            )

    # 坏图
    if corrupt_images:
        ratio = corrupt_ratio
        sev = "error" if ratio >= policy.max_corrupt_ratio else "warn"
        # 全量模式下用真实比例；抽样用估计
        add_issue(
            "E_CORRUPT_IMAGE" if sev == "error" else "W_CORRUPT_IMAGE",
            sev,
            f"坏图/无法解码: {len(corrupt_images)}"
            + (f"（抽样估计占比 {ratio:.1%}）" if decode_mode == "sample" else f"（占比 {ratio:.1%}）"),
            samples=corrupt_images,
            count=len(corrupt_images),
            suggestion="删除或修复损坏文件",
        )

    # 长尾（OCR 的 flags 文本是内容不是固定类别，只做直方图展示，不做长尾门禁）
    hist_for_tail = {k: v for k, v in label_hist.items() if k != "_root_" and v > 0}
    rare_classes: List[str] = []
    imbalance_ratio = 1.0
    if task == "OCR":
        pass
    elif hist_for_tail:
        counts = list(hist_for_tail.values())
        mn, mx = min(counts), max(counts)
        imbalance_ratio = mx / max(mn, 1)
        rare_classes = sorted([k for k, v in hist_for_tail.items() if v < policy.min_per_class])
        if rare_classes:
            add_issue(
                "W_RARE_CLASS",
                "warn",
                f"稀有类（<{policy.min_per_class}）: {len(rare_classes)} — "
                + ", ".join(f"{k}={hist_for_tail[k]}" for k in rare_classes[:8]),
                samples=[f"{k}:{hist_for_tail[k]}" for k in rare_classes],
                count=len(rare_classes),
                suggestion="为稀有类补样本，或合并类别",
            )
        if imbalance_ratio >= policy.error_imbalance_ratio and len(hist_for_tail) >= 2:
            add_issue(
                "E_IMBALANCE",
                "error",
                f"类别极端长尾: max/min={imbalance_ratio:.1f}",
                samples=[f"{k}:{v}" for k, v in sorted(hist_for_tail.items(), key=lambda x: -x[1])[:sample_cap]],
                count=len(hist_for_tail),
                suggestion="平衡各类样本或调整采样策略",
            )
        elif imbalance_ratio >= policy.warn_imbalance_ratio and len(hist_for_tail) >= 2:
            add_issue(
                "W_IMBALANCE",
                "warn",
                f"类别长尾: max/min={imbalance_ratio:.1f}",
                samples=[f"{k}:{v}" for k, v in sorted(hist_for_tail.items(), key=lambda x: -x[1])[:sample_cap]],
                count=len(hist_for_tail),
                suggestion="关注少类召回；可补数或 class weight",
            )
    elif task == "分类":
        add_issue(
            "E_NO_CLASSES",
            "error",
            "未统计到任何类别",
            suggestion="检查分类文件夹结构",
        )
    elif paired > 0 and not hist_for_tail:
        add_issue(
            "W_NO_LABELS",
            "warn",
            "配对成功但未收集到任何 label 名称",
            suggestion="检查 shapes[].label 是否填写",
        )

    stats = {
        "n_images": n_images,
        "n_jsons": 0 if is_cls else n_jsons,
        "n_paired": None if is_cls else paired,
        "pair_rate": None if is_cls else round(float(pair_rate or 0.0), 4),
        "label_mode": "folder" if is_cls else "json",
        "n_missing_json": 0 if is_cls else len(missing_json),
        "n_orphan_json": 0 if is_cls else len(orphan_json),
        "n_corrupt_json": 0 if is_cls else len(corrupt_json),
        "n_empty_label": 0 if is_cls else len(empty_labels),
        "n_corrupt_image": len(corrupt_images),
        "corrupt_ratio_est": round(corrupt_ratio, 4),
        "n_classes": len(hist_for_tail),
        "imbalance_ratio": round(imbalance_ratio, 4),
        "rare_classes": rare_classes,
        "images_by_ext": dict(ext_counter),
        "shape_types": {} if is_cls else dict(shape_type_counter),
        "decode_mode": decode_mode,
        "n_decoded": len(to_decode),
        "resolved_from": str(root_in),
        "dataset_dir": str(data_dir),
    }

    return _finalize_report(task, data_dir, now, issues, hist_for_tail or label_hist, policy, stats=stats)


def _finalize_report(
    task: str,
    data_dir: Path,
    now: str,
    issues: List[Issue],
    label_hist: Dict[str, int],
    policy: QualityPolicy,
    stats: Dict[str, Any],
) -> QualityReport:
    n_err = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warn")
    score = _score_report(stats, issues, policy, task)
    grade = _grade(score, n_err)
    passed = n_err == 0 and (not policy.fail_on_warning or n_warn == 0)

    # 无图等致命情况
    if stats.get("n_images", 0) == 0:
        passed = False
        grade = "F"
        score = 0.0

    summary = {
        "score": round(score, 4),
        "grade": grade,
        "pass": passed,
        "n_images": stats.get("n_images", 0),
        "n_jsons": stats.get("n_jsons", 0),
        "pair_rate": stats.get("pair_rate"),  # 分类为 None
        "label_mode": stats.get("label_mode", "json"),
        "n_classes": stats.get("n_classes", len(label_hist)),
        "blocking_issues": n_err,
        "warnings": n_warn,
        "imbalance_ratio": stats.get("imbalance_ratio", 1.0),
    }

    # severity 排序: error, warn, info
    order = {"error": 0, "warn": 1, "info": 2}
    issues_sorted = sorted(issues, key=lambda x: (order.get(x.severity, 9), x.code))

    return QualityReport(
        task_type=task,
        dataset_dir=str(data_dir),
        scanned_at=now,
        pass_=passed,
        grade=grade,
        score=score,
        summary=summary,
        stats=stats,
        issues=issues_sorted,
        label_histogram=dict(sorted(label_hist.items(), key=lambda x: (-x[1], x[0]))),
        policy=asdict(policy),
    )


def _score_report(
    stats: Dict[str, Any],
    issues: List[Issue],
    policy: QualityPolicy,
    task: str,
) -> float:
    """可解释的简单扣分：从 1.0 起扣。"""
    score = 1.0
    n_img = stats.get("n_images") or 0
    if n_img <= 0:
        return 0.0

    if task != "分类":
        pair_rate = stats.get("pair_rate")
        if pair_rate is not None:
            pair_rate = float(pair_rate)
            if pair_rate < policy.min_pair_rate:
                score -= 0.15 * (policy.min_pair_rate - pair_rate) / max(policy.min_pair_rate, 1e-6)
            if pair_rate < policy.error_pair_rate:
                score -= 0.25

    corrupt_ratio = float(stats.get("corrupt_ratio_est") or 0.0)
    score -= min(0.3, corrupt_ratio * 5)

    if task != "分类":
        n_empty = int(stats.get("n_empty_label") or 0)
        n_paired = max(int(stats.get("n_paired") or 0), 1)
        empty_ratio = n_empty / n_paired
        score -= min(0.25, empty_ratio * 1.2)

    imb = float(stats.get("imbalance_ratio") or 1.0)
    if imb >= policy.warn_imbalance_ratio:
        score -= 0.08
    if imb >= policy.error_imbalance_ratio:
        score -= 0.12

    for iss in issues:
        if iss.severity == "error":
            score -= 0.06
        elif iss.severity == "warn":
            score -= 0.02

    if n_img < policy.min_images:
        score -= 0.2

    return _clamp01(score)


def _grade(score: float, n_err: int) -> str:
    if n_err > 0 and score < 0.4:
        return "F"
    if n_err > 0:
        return "D"
    if score >= 0.9:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.6:
        return "C"
    return "D"


def format_report_text(report: QualityReport) -> str:
    """人类可读摘要。"""
    d = report.to_dict()
    is_cls = d.get("task_type") == "分类" or d.get("summary", {}).get("label_mode") == "folder"
    pair = d["summary"].get("pair_rate")
    if is_cls:
        count_line = (
            f"图片:     {d['summary'].get('n_images')}  "
            f"标注方式: 文件夹分类（不检查 JSON 配对）"
        )
    else:
        pair_txt = f"{pair:.1%}" if isinstance(pair, (int, float)) else "N/A"
        count_line = (
            f"图片:     {d['summary'].get('n_images')}  JSON: {d['summary'].get('n_jsons')}  "
            f"配对率: {pair_txt}"
        )
    lines = [
        "=" * 60,
        "训练集数据质量报告",
        "=" * 60,
        f"目录:     {d['dataset_dir']}",
        f"任务:     {d['task_type']}",
        f"时间:     {d['scanned_at']}",
        f"结论:     {'PASS' if d['pass'] else 'FAIL'}  grade={d['grade']}  score={d['score']:.2f}",
        count_line,
        f"类别数:   {d['summary'].get('n_classes')}  不平衡比 max/min: {d['summary'].get('imbalance_ratio')}",
        f"阻断项:   {d['summary'].get('blocking_issues')}  警告: {d['summary'].get('warnings')}",
        "-" * 60,
    ]
    if d["label_histogram"]:
        lines.append("类别直方图 (top 20):")
        items = list(d["label_histogram"].items())[:20]
        for k, v in items:
            lines.append(f"  {v:6d}  {k}")
        if len(d["label_histogram"]) > 20:
            lines.append(f"  … 共 {len(d['label_histogram'])} 类")
        lines.append("-" * 60)

    if d["stats"].get("shape_types"):
        lines.append(f"shape_type 统计: {d['stats']['shape_types']}")
        lines.append("-" * 60)

    if not d["issues"]:
        lines.append("未发现问题。")
    else:
        lines.append("问题列表:")
        for iss in d["issues"]:
            badge = {"error": "[ERROR]", "warn": "[WARN] ", "info": "[INFO] "}.get(
                iss["severity"], "[????] "
            )
            lines.append(f"  {badge} {iss['code']}: {iss['message']}")
            if iss.get("suggestion"):
                lines.append(f"           建议: {iss['suggestion']}")
            if iss.get("samples"):
                preview = ", ".join(iss["samples"][:5])
                more = "" if len(iss["samples"]) <= 5 else f" …(+{len(iss['samples'])-5})"
                lines.append(f"           样例: {preview}{more}")
    lines.append("=" * 60)
    return "\n".join(lines)
