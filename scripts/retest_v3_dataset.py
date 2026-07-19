#!/usr/bin/env python3
"""Run the active v3 model through the production OCR/ROI path and write a result package."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.v1.routes.ai_detection import EngineContainer, ensure_ai_detection_runtime
from app.ai_detection.workflows.v3_evaluation import ProductionV3Evaluator, run_v3_retest


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="app/ai_detection/images")
    parser.add_argument("--output", default="app/ai_detection/models/evaluations/v3")
    args = parser.parse_args()

    await ensure_ai_detection_runtime()
    engine = EngineContainer.instance
    reader = EngineContainer.ocr_reader
    if engine is None or reader is None:
        raise RuntimeError("图片检测运行时未加载")
    report = await asyncio.to_thread(
        run_v3_retest,
        ProductionV3Evaluator(engine, reader),
        Path(args.images),
        Path(args.output),
        model_version=str(getattr(engine, "_xgb_path", "")),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
