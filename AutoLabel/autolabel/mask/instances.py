"""Instance list ↔ instance-id mask."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from autolabel.mask.io import resize_mask_nn
from autolabel.models import InstancePred


def instances_to_id_mask(instances: List[InstancePred], height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.int32)
    for inst in instances:
        m = inst.mask
        if m.shape[0] != height or m.shape[1] != width:
            m = resize_mask_nn(m, width, height)
        canvas[m > 0] = int(inst.instance_id)
    return canvas


def id_mask_to_instances(
    id_mask: np.ndarray,
    meta: Dict[int, Dict],
    *,
    default_class_name: str = "object",
) -> List[InstancePred]:
    """meta: instance_id -> {class_id, class_name, score?}"""
    out: List[InstancePred] = []
    ids = [int(i) for i in np.unique(id_mask) if int(i) != 0]
    for iid in ids:
        info = meta.get(iid) or {}
        binary = (id_mask == iid).astype(np.uint8)
        out.append(
            InstancePred(
                instance_id=iid,
                class_id=int(info.get("class_id") or 1),
                class_name=str(info.get("class_name") or default_class_name),
                mask=binary,
                score=float(info["score"]) if info.get("score") is not None else None,
            )
        )
    return out
