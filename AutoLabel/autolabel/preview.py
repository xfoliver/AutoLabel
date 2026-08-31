"""Preview sampling (3–5 images)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence


def sample_preview_images(
    pending: Sequence[Path],
    count: int = 5,
    *,
    seed: Optional[int] = None,
) -> List[Path]:
    items = list(pending)
    if not items:
        return []
    k = min(max(int(count), 1), len(items))
    if seed is None:
        # Deterministic-ish spread: take evenly spaced indices
        if k == len(items):
            return items
        step = len(items) / k
        return [items[min(int(i * step), len(items) - 1)] for i in range(k)]
    rng = random.Random(seed)
    return rng.sample(items, k)
