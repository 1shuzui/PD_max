#!/usr/bin/env python3
"""构建 AIGC 无水印鲁棒性离线实验；默认只预览。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_detection.workflows.aigc_document import MINIMUM_SOURCE_GROUPS, build_aigc_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 AIGC 文档离线实验，默认 dry-run")
    parser.add_argument("--images-dir", default="app/ai_detection/images")
    parser.add_argument("--apply", action="store_true", help="写入隔离元数据、派生图片和实验清单")
    parser.add_argument(
        "--max-controls-per-class",
        type=int,
        default=MINIMUM_SOURCE_GROUPS,
        help="每类角落扰动对照的最大独立组数",
    )
    args = parser.parse_args(argv)
    if args.max_controls_per_class < 1:
        parser.error("--max-controls-per-class 必须大于 0")
    result = build_aigc_experiment(
        Path(args.images_dir),
        apply=args.apply,
        maximum_controls_per_class=args.max_controls_per_class,
    )
    print(json.dumps({
        key: value for key, value in result.items() if key not in {"manifest", "experiment"}
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
