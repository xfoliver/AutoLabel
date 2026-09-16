"""SAM3 debug overlays and bbox-rectangle detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from autolabel.debug_overlay import (  # noqa: E402
    count_sam3_raw,
    looks_like_axis_aligned_rectangle,
    resolve_preview_mask_path,
    summarize_shapes,
    write_sam_debug_overlays,
)
from autolabel.models import InstancePred, ShapeRecord, Stage1Result, TransformMeta  # noqa: E402


def test_rectangle_detection():
    rect = [[10.0, 20.0], [10.0, 40.0], [50.0, 40.0], [50.0, 20.0]]
    blob = [[10.0, 20.0], [12.0, 35.0], [40.0, 42.0], [48.0, 18.0], [22.0, 16.0]]
    assert looks_like_axis_aligned_rectangle(rect)
    assert not looks_like_axis_aligned_rectangle(blob)
    assert not looks_like_axis_aligned_rectangle(rect[:3])


def test_summarize_shapes_counts_rectangles():
    shapes = [
        ShapeRecord(label="weld", points=[[10, 10], [10, 40], [50, 40], [50, 10]]),
        ShapeRecord(label="weld", points=[[1, 1], [2, 8], [9, 7], [8, 2], [4, 0]]),
    ]
    info = summarize_shapes(shapes)
    assert info["final_shapes"] == 2
    assert info["rectangle_n"] == 1
    assert info["four_point_n"] == 1
    assert info["point_counts"] == [4, 5]


def test_count_sam3_raw_empty_when_no_instances():
    result = Stage1Result(
        instances=[],
        transform=TransformMeta.identity_for(100, 80),
        raw={"instances": []},
    )
    counts = count_sam3_raw(result)
    assert counts["raw_boxes"] == 0
    assert counts["raw_masks"] == 0
    assert "vision_boxes" not in counts


def test_resolve_preview_mask_prefers_color_overlay(tmp_path: Path):
    overlay = tmp_path / "all_overlay.jpg"
    overlay.write_bytes(b"jpeg")
    tiny = tmp_path / "mask_0.png"
    tiny.write_bytes(b"png")
    raw = {
        "overlay_path": str(overlay),
        "instances": [{"mask_path": str(tiny)}],
    }
    picked = resolve_preview_mask_path(raw=raw)
    assert picked == overlay


def test_resolve_preview_mask_keeps_chat_image_rawmask(tmp_path: Path):
    rawmask = tmp_path / "img_rawmask.png"
    rawmask.write_bytes(b"png")
    debug = tmp_path / "debug"
    debug.mkdir()
    (debug / "sam_raw_mask_overlay.jpg").write_bytes(b"jpg")
    picked = resolve_preview_mask_path(
        raw={},
        debug_dir=debug,
        colored_mask_path=rawmask,
    )
    assert picked == rawmask


def test_resolve_preview_mask_falls_back_to_debug_union(tmp_path: Path):
    debug = tmp_path / "debug"
    debug.mkdir()
    union = debug / "sam_raw_mask_overlay.jpg"
    union.write_bytes(b"jpg")
    tiny = tmp_path / "mask_0.png"
    tiny.write_bytes(b"png")
    picked = resolve_preview_mask_path(
        raw={"instances": [{"mask_path": str(tiny)}]},
        debug_dir=debug,
    )
    assert picked == union


def test_resolve_preview_mask_prefers_filtered_debug_over_worker_rainbow(tmp_path: Path):
    debug = tmp_path / "debug"
    debug.mkdir()
    filtered = debug / "sam_raw_mask_overlay.jpg"
    filtered.write_bytes(b"filtered")
    worker = tmp_path / "worker_overlay.jpg"
    worker.write_bytes(b"rainbow")
    picked = resolve_preview_mask_path(
        raw={"overlay_path": str(worker)},
        debug_dir=debug,
    )
    assert picked == filtered


def test_write_sam_debug_overlays(tmp_path: Path):
    img = np.full((80, 100, 3), 30, dtype=np.uint8)
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), img)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:50, 15:60] = 1
    result = Stage1Result(
        instances=[InstancePred(1, 1, "weld", mask, score=0.8)],
        transform=TransformMeta.identity_for(100, 80),
        raw={"instances": [{"bbox": [15, 20, 60, 50], "score": 0.8, "mask": True}]},
    )
    shapes = [
        ShapeRecord(
            label="weld",
            points=[[15.0, 20.0], [20.0, 48.0], [58.0, 49.0], [55.0, 22.0], [30.0, 18.0]],
        )
    ]
    out = tmp_path / "debug"
    payload = write_sam_debug_overlays(src, out_dir=out, result=result, shapes=shapes)
    assert payload["raw_masks"] == 1
    assert payload["final_shapes"] == 1
    assert payload["rectangle_n"] == 0
    assert (out / "sam_raw_mask_overlay.jpg").is_file()
    assert (out / "sam_raw_bbox_overlay.jpg").is_file()
    assert (out / "final_polygon_overlay.jpg").is_file()
    assert (out / "debug_summary.json").is_file()
    raw_vis = cv2.imread(str(out / "sam_raw_mask_overlay.jpg"))
    poly_vis = cv2.imread(str(out / "final_polygon_overlay.jpg"))
    assert raw_vis is not None and poly_vis is not None
    # Cyan pixel-mask (BGR ~255,220,0) vs class-colored polygon overlay.
    assert int(raw_vis[35, 30, 0]) > 150
    assert int(raw_vis[35, 30, 0]) > int(poly_vis[35, 30, 0])


def test_coord_debug_composites_all_instance_masks(tmp_path: Path):
    from autolabel.coord_debug import write_coord_debug

    img = np.full((40, 60, 3), 30, dtype=np.uint8)
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), img)
    m1 = np.zeros((40, 60), dtype=np.uint8)
    m1[5:15, 5:20] = 255
    m2 = np.zeros((40, 60), dtype=np.uint8)
    m2[20:35, 25:50] = 255
    p1 = tmp_path / "a_sam3_mask_0.png"
    p2 = tmp_path / "a_sam3_mask_1.png"
    cv2.imwrite(str(p1), m1)
    cv2.imwrite(str(p2), m2)
    out = tmp_path / "dbg"
    write_coord_debug(
        src,
        out_dir=out,
        shapes=[],
        sam3_raw={"instances": [{"mask_path": str(p1)}, {"mask_path": str(p2)}], "image_size": [60, 40]},
        mask_path=p1,
    )
    overlay = cv2.imread(str(out / "04_sam3_mask_overlay.jpg"))
    union = cv2.imread(str(out / "03_sam3_raw_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert overlay is not None and union is not None
    assert int(union[10, 10]) > 0
    assert int(union[28, 35]) > 0
    assert overlay[10, 10].max() > 80
    assert overlay[28, 35].max() > 80
