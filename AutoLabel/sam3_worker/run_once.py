"""Stage A: one-shot SAM3 inference using the sam3 Conda Python.

    C:\\dlcv\\envs\\sam3\\python.exe AutoLabel\\sam3_worker\\run_once.py ^
        --image D:\\data\\001.jpg --prompt phone
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from sam3_worker.inference import get_engine  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot SAM3 text-prompt segmentation")
    parser.add_argument("--image", required=True, help="image path")
    parser.add_argument("--prompt", required=True, help="short visual prompt, e.g. phone")
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument(
        "--out",
        default="",
        help="directory for masks / overlay (default: AutoLabel/runs/sam3)",
    )
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else HERE.parent / "runs" / "sam3"
    engine = get_engine()
    print("Loading SAM3…", flush=True)
    engine.load()
    print(f"Device: {engine.device}", flush=True)
    packed = engine.segment(
        image_path=args.image,
        prompt=args.prompt,
        score_threshold=args.score_threshold,
        save_dir=out,
    )
    summary = {
        "prompt": packed["prompt"],
        "image_path": packed["image_path"],
        "image_size": packed["image_size"],
        "n_instances": len(packed["instances"]),
        "instances": [
            {"score": x["score"], "bbox": x["bbox"], "area": x["area"], "mask_path": x["mask_path"]}
            for x in packed["instances"]
        ],
        "overlay_path": packed.get("overlay_path"),
        "result_json": packed.get("result_json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
