#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集数据质量检查 — 独立 CLI（不接入 auto_train4.0）。

用法示例:
  python check_dataset.py --folder "AI训练平台测试/3-DET/训练集" --task det
  python check_dataset.py --folder "AI训练平台测试/2-CLS" --task cls -o report.json
  python check_dataset.py --folder "AI训练平台测试/1-OCR/训练集" --task ocr --decode full
  python check_dataset.py --folder "AI训练平台测试/7-ROTATE" --task rotate --json

退出码:
  0  通过（无 error；若 --strict 则也无 warn）
  1  未通过（存在 error，或 --strict 下存在 warn）
  2  参数/路径错误
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_on_path() -> Path:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="训练集数据质量检查（规则引擎，独立于 auto_train4.0）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--folder",
        "-f",
        required=True,
        help="数据集目录（项目根或训练集子目录均可）",
    )
    p.add_argument(
        "--task",
        "-t",
        required=True,
        help="任务类型: ocr/cls/det/detr/iseg/sseg/rotate 或中文名",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="将完整 JSON 报告写入该路径",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="print_json",
        help="向 stdout 打印 JSON（默认打印可读文本）",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="警告也视为不通过（fail_on_warning）",
    )
    p.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="最少图片数，默认 10",
    )
    p.add_argument(
        "--min-pair-rate",
        type=float,
        default=0.9,
        help="配对率警告阈值，默认 0.9",
    )
    p.add_argument(
        "--error-pair-rate",
        type=float,
        default=0.5,
        help="配对率错误阈值，默认 0.5",
    )
    p.add_argument(
        "--min-per-class",
        type=int,
        default=5,
        help="每类最少样本/实例数，默认 5",
    )
    p.add_argument(
        "--warn-imbalance",
        type=float,
        default=20.0,
        help="长尾警告: max/min 阈值，默认 20",
    )
    p.add_argument(
        "--decode",
        choices=("sample", "full", "off"),
        default="sample",
        help="坏图解码策略: sample(默认)/full/off",
    )
    p.add_argument(
        "--decode-sample-size",
        type=int,
        default=50,
        help="sample 模式下解码张数，默认 50",
    )
    p.add_argument(
        "--no-resolve-subdir",
        action="store_true",
        help="不自动探测 训练集/train 等子目录，直接检查 --folder",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="少打印进度",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_on_path()
    from autotrain4.data_quality import (
        QualityPolicy,
        check_dataset,
        format_report_text,
        normalize_task_type,
    )

    args = build_parser().parse_args(argv)
    folder = Path(args.folder).expanduser()
    if not folder.exists():
        print(f"[ERROR] 路径不存在: {folder}", file=sys.stderr)
        return 2

    task = normalize_task_type(args.task)
    if not task:
        print(f"[ERROR] 无法识别任务类型: {args.task}", file=sys.stderr)
        return 2

    policy = QualityPolicy(
        min_images=args.min_images,
        min_pair_rate=args.min_pair_rate,
        error_pair_rate=args.error_pair_rate,
        min_per_class=args.min_per_class,
        warn_imbalance_ratio=args.warn_imbalance,
        image_decode=args.decode,
        decode_sample_size=args.decode_sample_size,
        fail_on_warning=bool(args.strict),
    )

    def progress(msg: str) -> None:
        if not args.quiet and not args.print_json:
            print(f"  … {msg}", file=sys.stderr)

    try:
        report = check_dataset(
            folder,
            task,
            policy=policy,
            resolve_train_subdir=not args.no_resolve_subdir,
            progress_cb=progress if not args.quiet else None,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    payload = report.to_dict()

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.print_json:
            print(f"[OK] JSON 报告已写入: {out}", file=sys.stderr)

    def _safe_print(text: str) -> None:
        """Avoid UnicodeEncodeError on Windows GBK consoles."""
        try:
            print(text)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))

    if args.print_json:
        _safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _safe_print(format_report_text(report))

    return 0 if report.pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
