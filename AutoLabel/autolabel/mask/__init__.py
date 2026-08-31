"""Mask package."""

from autolabel.mask.color_instances import colored_mask_to_instances, merge_overlapping_instances
from autolabel.mask.components import class_mask_to_pseudo_instances
from autolabel.mask.grabcut_refine import grabcut_refine_mask
from autolabel.mask.instances import id_mask_to_instances, instances_to_id_mask
from autolabel.mask.io import load_mask_png, resize_mask_nn, save_mask_npy, save_mask_png
from autolabel.mask.refine import refine_binary_mask

__all__ = [
    "colored_mask_to_instances",
    "merge_overlapping_instances",
    "class_mask_to_pseudo_instances",
    "grabcut_refine_mask",
    "id_mask_to_instances",
    "instances_to_id_mask",
    "load_mask_png",
    "resize_mask_nn",
    "save_mask_npy",
    "save_mask_png",
    "refine_binary_mask",
]
