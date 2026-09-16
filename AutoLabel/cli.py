#!/usr/bin/env python
"""AutoLabel CLI — headless plan / preview / batch / commit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autolabel.capability import CapabilityError
from autolabel.paths import backends_path, ensure_user_config, settings_path
from autolabel.inference.base import build_backend_from_config, load_backends_config
from autolabel.orchestrator import AutoLabelOrchestrator
from autolabel.plan import format_plan_text


def _load_settings(path: Path | None) -> dict:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    ensure_user_config()
    local = settings_path()
    if local.exists():
        with local.open("r", encoding="utf-8") as f:
            return json.load(f)
    example = ROOT / "config" / "settings.example.json"
    if example.exists():
        with example.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoLabel MVP CLI (ISEG)")
    parser.add_argument("--dataset", required=True, help="数据集目录（含图片）")
    parser.add_argument("--label", default="object", help="类别名")
    parser.add_argument("--description", default="", help="目标语义描述")
    parser.add_argument("--exclude", default="", help="排除项，逗号分隔")
    parser.add_argument("--prompt", default="", help="自然语言（可选，覆盖手工字段）")
    parser.add_argument("--backend", default=None, help="引擎：sam3.1，或设置里添加的 API 引擎 id / chat-image / mock")
    parser.add_argument("--backends-config", default=None, help="backends.json 路径")
    parser.add_argument("--settings", default=None, help="LLM settings.json")
    parser.add_argument("--preview", action="store_true", help="试标 3~5 张")
    parser.add_argument("--batch", action="store_true", help="全量（需 --yes 确认试标）")
    parser.add_argument("--yes", action="store_true", help="确认试标并允许全量")
    parser.add_argument("--commit", action="store_true", help="Commit work JSON 到旁路")
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--intent-mode", default=None, help="auto | llm | simple")
    parser.add_argument("--score-threshold", type=float, default=None, help="SAM3.1 score threshold")
    parser.add_argument("--min-area", type=int, default=None, help="drop tiny instances (px)")
    args = parser.parse_args(argv)

    backends_path_arg = Path(args.backends_config) if args.backends_config else backends_path()
    bcfg = load_backends_config(backends_path_arg if backends_path_arg.exists() else None)
    settings = _load_settings(Path(args.settings) if args.settings else None)
    if args.intent_mode:
        settings["intent_mode"] = args.intent_mode
    backend = build_backend_from_config(bcfg, args.backend, settings=settings)
    if args.score_threshold is not None and hasattr(backend, "score_threshold"):
        backend.score_threshold = float(args.score_threshold)
        if getattr(backend, "options", None) is not None:
            backend.options["score_threshold"] = float(args.score_threshold)
    if args.min_area is not None and hasattr(backend, "min_area"):
        backend.min_area = int(args.min_area)
        if getattr(backend, "options", None) is not None:
            backend.options["min_area"] = int(args.min_area)

    orch = AutoLabelOrchestrator(
        backends_config=bcfg,
        backend=backend,
        backend_id=args.backend,
        settings=settings,
        log=lambda m: print(m, flush=True),
    )

    if args.prompt:
        orch.parse_prompt(args.prompt, dataset_path=args.dataset)
        if orch.state.intent:
            orch.state.intent.preview_count = args.preview_count
    else:
        excludes = [x.strip() for x in args.exclude.split(",") if x.strip()]
        orch.make_manual_intent(
            label=args.label,
            description=args.description,
            dataset_path=args.dataset,
            exclude=excludes,
            preview_count=args.preview_count,
        )

    try:
        plan = orch.build_plan(dataset_path=args.dataset)
    except CapabilityError as e:
        print(f"[ERROR] Capability Check 失败: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[ERROR] Plan 失败: {e}", file=sys.stderr)
        return 1

    print("===== PLAN =====")
    print(format_plan_text(plan))
    if args.plan_only:
        return 0

    if args.preview or args.batch or args.commit:
        results = orch.run_preview()
        print(f"试标完成: {len(results)} 张 → {orch.state.run_dir}")
        for r in results:
            print(f"  - {r.image_path.name}: {len(r.shapes)} shapes, bucket={r.review.bucket if r.review else '?'}")

    if args.batch:
        if not args.yes:
            print("全量需要 --yes 确认试标结果", file=sys.stderr)
            return 3
        orch.confirm_preview()
        report = orch.run_batch()
        print("===== REPORT =====")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.commit:
        if not orch.state.mapping:
            print("无 mapping，请先 --batch --yes", file=sys.stderr)
            return 4
        summary = orch.commit()
        print("===== COMMIT =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
