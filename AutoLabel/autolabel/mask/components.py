"""Connected-components fallback (NOT default for ISEG)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from autolabel.models import InstancePred


def class_mask_to_pseudo_instances(
    class_mask: np.ndarray,
    *,
    class_id: int,
    class_name: str,
    min_area: int = 16,
) -> List[InstancePred]:
    """Split a single-class binary/class mask via connectedComponents.

    Warning: adjacent same-class objects will merge — only for explicit fallback.
    """
    binary = (class_mask == class_id).astype(np.uint8) if class_mask.max() > 1 else (class_mask > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    out: List[InstancePred] = []
    next_id = 1
    for lab in range(1, n):
        m = (labels == lab).astype(np.uint8)
        if int(m.sum()) < min_area:
            continue
        out.append(
            InstancePred(
                instance_id=next_id,
                class_id=class_id,
                class_name=class_name,
                mask=m,
                score=None,
            )
        )
        next_id += 1
    return out
