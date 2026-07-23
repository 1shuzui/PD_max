#!/usr/bin/env python3
"""显式校验本机 ROI sidecar，生成未来局部辅助模型的候选输入清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


FIELD_TYPES = {"amount", "name", "time"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sidecar(images_root: Path, sidecar_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        item = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "sidecar 无法读取"
    if not isinstance(item, dict):
        return None, "sidecar 必须为对象"
    relative = str(item.get("image_relative_path") or "").replace("\\", "/")
    image_path = (images_root / relative).resolve()
    try:
        image_path.relative_to(images_root.resolve())
    except ValueError:
        return None, "图片路径越界"
    if not image_path.is_file() or len(Path(relative).parts) != 2 or Path(relative).parts[0] not in {"normal", "tampered"}:
        return None, "图片不在 normal/tampered 白名单"
    digest = sha256(image_path)
    if item.get("image_sha256") != digest or sidecar_path.stem != digest:
        return None, "图片 SHA 不匹配"
    with Image.open(image_path) as image:
        width, height = image.size
    if int(item.get("width") or -1) != width or int(item.get("height") or -1) != height:
        return None, "图片尺寸不匹配"
    label = 0 if image_path.parent.name == "normal" else 1
    if item.get("image_label") != image_path.parent.name:
        return None, "图片标签不匹配"
    regions = item.get("manual_rois")
    if not isinstance(regions, list):
        return None, "manual_rois 必须为数组"
    validated = []
    for region in regions:
        if not isinstance(region, dict):
            return None, "ROI 必须为对象"
        if str(region.get("field_type") or "") == "other":
            continue
        if str(region.get("field_type") or "") not in FIELD_TYPES:
            return None, "ROI 字段类型无效"
        try:
            x1, y1, x2, y2 = (float(region[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return None, "ROI 坐标无效"
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            return None, "ROI 坐标越界"
        if not isinstance(region.get("is_tampered"), bool):
            return None, "ROI 缺少实际篡改真值"
        if not str(region.get("source") or "").strip():
            return None, "ROI 缺少来源"
        validated.append(dict(region))
    if label == 1 and not any(region["is_tampered"] for region in validated):
        return None, "篡改图片缺少实际篡改框"
    if label == 0 and any(region["is_tampered"] for region in validated):
        return None, "正常图片不能包含实际篡改框"
    return {
        "image_path": relative,
        "label": label,
        "image_sha256": digest,
        "width": width,
        "height": height,
        "manual_rois": validated,
        "source_sidecar": str(sidecar_path),
    }, None


def import_annotations(images_root: Path, annotations_root: Path) -> dict[str, Any]:
    accepted = []
    rejected = []
    for sidecar in sorted(annotations_root.glob("*.json")):
        item, reason = validate_sidecar(images_root, sidecar)
        if item is None:
            rejected.append({"sidecar": str(sidecar), "reason": reason})
        else:
            accepted.append(item)
    return {"schema_version": 1, "accepted": accepted, "rejected": rejected}


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验本机 ROI 标注；默认 dry-run，不会加入训练")
    parser.add_argument("--images-dir", default="app/ai_detection/images")
    parser.add_argument("--annotations-dir", default="app/ai_detection/locate_json/annotations/local")
    parser.add_argument("--apply", action="store_true", help="写出候选局部辅助模型输入清单")
    parser.add_argument("--output", help="--apply 时必须指定的候选清单路径")
    args = parser.parse_args(argv)
    if args.apply and not args.output:
        parser.error("--apply 必须同时指定 --output；该工具不会自动加入训练")
    result = import_annotations(Path(args.images_dir).resolve(), Path(args.annotations_dir).resolve())
    result["dry_run"] = not args.apply
    if args.apply:
        write_atomic(Path(args.output).resolve(), result)
        result["output"] = str(Path(args.output).resolve())
    print(json.dumps({"accepted": len(result["accepted"]), "rejected": result["rejected"], "dry_run": result["dry_run"]}, ensure_ascii=False, indent=2))
    return 0 if not result["rejected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
