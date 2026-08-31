"""quality_risk heuristics — never treat as model_confidence."""

from __future__ import annotations

from typing import List, Optional, Sequence

from autolabel.models import ShapeRecord


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def compute_quality_risk(
    points: Sequence[Sequence[float]],
    *,
    image_width: int,
    image_height: int,
    min_area: float = 25.0,
) -> float:
    risk = 0.0
    area = polygon_area(points)
    if area < min_area:
        risk += 0.45
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if not xs:
        return 1.0
    margin = 2.0
    if min(xs) <= margin or min(ys) <= margin:
        risk += 0.15
    if max(xs) >= image_width - 1 - margin or max(ys) >= image_height - 1 - margin:
        risk += 0.15
    if area > 0 and len(points) > 40:
        risk += min(0.2, (len(points) - 40) * 0.005)
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    if bw < 3 or bh < 3:
        risk += 0.25
    return float(min(1.0, risk))


def bucket_for_shape(
    model_confidence: Optional[float],
    quality_risk: float,
    *,
    high: float = 0.7,
    low: float = 0.3,
) -> str:
    if model_confidence is None:
        return "sample"
    if model_confidence >= high and quality_risk < 0.5:
        return "suggest_accept"
    if model_confidence < low:
        return "problem"
    return "review"


def summarize_image_bucket(shapes: List[ShapeRecord]) -> str:
    if not shapes:
        return "empty"
    buckets = [bucket_for_shape(s.model_confidence, s.quality_risk) for s in shapes]
    if any(b == "problem" for b in buckets):
        return "problem"
    if any(b == "review" for b in buckets):
        return "review"
    if all(b == "suggest_accept" for b in buckets):
        return "suggest_accept"
    return "sample"
