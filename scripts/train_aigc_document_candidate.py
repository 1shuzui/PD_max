#!/usr/bin/env python3
"""训练 AIGC 文档离线候选；不会改写生产 v3。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_detection.workflows.aigc_document import train_aigc_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 AIGC 文档离线候选")
    parser.add_argument("--images-dir", default="app/ai_detection/images")
    parser.add_argument("--experiment", default="app/ai_detection/images/aigc_experiment_manifest.json")
    parser.add_argument("--output-dir", default="app/ai_detection/models/aigc_candidates")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--run-id", help="可复现输出目录名")
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold 必须介于 0 和 1 之间")
    result = train_aigc_candidate(
        Path(args.images_dir),
        Path(args.experiment),
        Path(args.output_dir),
        threshold=args.threshold,
        run_id=args.run_id,
    )
    print(json.dumps({
        "output_dir": result["output_dir"],
        "candidate_only": result["candidate_only"],
        "active_model_unchanged": result["active_model_unchanged"],
        "gate_passed": result["gates"]["passed"],
        "blockers": result["gates"]["blockers"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
