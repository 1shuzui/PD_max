#!/usr/bin/env python3
"""Incrementally register canonical AI-detection images without moving source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.reorganize_ai_dataset import (
        IMAGE_SUFFIXES,
        TEMPLATE_PHASH_HAMMING_THRESHOLD,
        base_stem,
        image_phash,
        phash_distance,
        sha256,
        source_for_path,
        validate_image,
    )
except ModuleNotFoundError:
    from reorganize_ai_dataset import (
        IMAGE_SUFFIXES,
        TEMPLATE_PHASH_HAMMING_THRESHOLD,
        base_stem,
        image_phash,
        phash_distance,
        sha256,
        source_for_path,
        validate_image,
    )


class PHashIndex:
    """用七段精确桶查询小 Hamming 距离候选，避免全量两两比较。"""

    _SEGMENTS = ((0, 9), (9, 18), (18, 27), (27, 36), (36, 45), (45, 54), (54, 64))

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._values: dict[str, int] = {}

    @classmethod
    def _keys(cls, value: int) -> Iterable[tuple[int, int]]:
        for number, (start, end) in enumerate(cls._SEGMENTS):
            width = end - start
            yield number, (value >> start) & ((1 << width) - 1)

    def add(self, key: str, value: int) -> None:
        self._values[key] = value
        for bucket in self._keys(value):
            self._buckets[bucket].add(key)

    def near(self, value: int, maximum_distance: int) -> list[tuple[str, int]]:
        candidates: set[str] = set()
        for bucket in self._keys(value):
            candidates.update(self._buckets.get(bucket, set()))
        return sorted(
            (
                (key, phash_distance(value, candidate))
                for key, candidate in ((key, self._values[key]) for key in candidates)
                if phash_distance(value, candidate) <= maximum_distance
            ),
            key=lambda item: (item[1], item[0]),
        )


def _deterministic_split(group_id: str) -> str:
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 15:
        return "test"
    if value < 30:
        return "validation"
    return "train"


def _group_id(relative_path: str, digest: str) -> str:
    key = f"{base_stem(Path(relative_path)).lower()}|{digest}"
    return f"source-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"entries": []}
    return data if isinstance(data, dict) else {"entries": []}


def _iter_canonical_images(images_dir: Path) -> Iterable[tuple[Path, int]]:
    for label, directory_name in ((0, "normal"), (1, "tampered")):
        directory = images_dir / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path, label


def _entry_key(entry: dict[str, Any]) -> str:
    return str(entry.get("path") or "").replace("\\", "/")


def _copy_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(entry, ensure_ascii=False))


def build_incremental_manifest(images_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    images_dir = Path(images_dir).resolve()
    manifest_path = images_dir / "dataset_manifest.json"
    previous = _read_manifest(manifest_path)
    previous_entries = [_copy_entry(item) for item in previous.get("entries", []) if isinstance(item, dict)]
    previous_by_path = {_entry_key(item): item for item in previous_entries if _entry_key(item)}
    entries_by_path = dict(previous_by_path)
    current_paths: set[str] = set()
    metadata_by_path: dict[str, dict[str, Any]] = {}
    index = PHashIndex()
    blockers: list[dict[str, Any]] = []
    changed_paths: list[str] = []

    # 对已有条目只补充检索元数据，split/group 永远沿用原记录。
    for path, label in _iter_canonical_images(images_dir):
        relative = path.relative_to(images_dir).as_posix()
        current_paths.add(relative)
        digest = sha256(path)
        existing = entries_by_path.get(relative)
        if existing and existing.get("sha256") == digest and int(existing.get("label", label)) == label:
            entry = existing
            if not entry.get("phash"):
                entry["phash"] = f"{image_phash(path):016x}"
            entry.setdefault("template_cluster", f"template-{str(entry['phash'])[:16]}")
            entry.setdefault("group_id", _group_id(relative, digest))
            entry.setdefault("parent_group_id", entry["group_id"])
        else:
            width, height, media_format = validate_image(path)
            entry = {
                "path": relative,
                "label": label,
                "class": "normal" if label == 0 else "tampered",
                "source": source_for_path(path, label),
                "sha256": digest,
                "phash": f"{image_phash(path):016x}",
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "format": media_format,
                "is_derived": False,
                "fixed_regression": False,
                "training_replay_regression": False,
            }
            entries_by_path[relative] = entry
            changed_paths.append(relative)
        metadata_by_path[relative] = entry
        index.add(relative, int(str(entry["phash"]), 16))

    existing_paths = set(previous_by_path)
    new_or_changed = [path for path in sorted(current_paths) if path in changed_paths]
    for relative in new_or_changed:
        entry = metadata_by_path[relative]
        label = int(entry["label"])
        p_hash = int(str(entry["phash"]), 16)
        near = [
            (other_path, distance)
            for other_path, distance in index.near(p_hash, TEMPLATE_PHASH_HAMMING_THRESHOLD)
            if other_path != relative
        ]
        same_label = [metadata_by_path[path] for path, _distance in near if int(metadata_by_path[path]["label"]) == label]
        cross_label = [metadata_by_path[path] for path, _distance in near if int(metadata_by_path[path]["label"]) != label]
        logical_group = next(
            (
                item.get("group_id")
                for path, item in metadata_by_path.items()
                if path != relative
                and int(item.get("label", -1)) == label
                and base_stem(Path(path)).lower() == base_stem(Path(relative)).lower()
            ),
            None,
        )
        related = cross_label or same_label
        related_group = logical_group or next((item.get("group_id") for item in related if item.get("group_id")), None)
        entry["group_id"] = str(related_group or _group_id(relative, str(entry["sha256"])))
        entry["parent_group_id"] = entry["group_id"]
        entry["template_cluster"] = str(
            next((item.get("template_cluster") for item in related if item.get("template_cluster")), f"template-{entry['phash']}"),
        )
        if cross_label:
            frozen = [item for item in cross_label if item.get("split") in {"validation", "test"} and _entry_key(item) in existing_paths]
            if frozen:
                blockers.append(
                    {
                        "path": relative,
                        "reason": "跨标签近似模板命中冻结 validation/test，不能重排既有 split",
                        "related_paths": [_entry_key(item) for item in frozen],
                    }
                )
                entry["split"] = "train"
                entry["split_reason"] = "blocked_cross_label_near_duplicate_requires_manual_resolution"
            else:
                entry["split"] = "train"
                entry["split_reason"] = "forced_cross_label_near_duplicate_training"
                for item in cross_label:
                    if _entry_key(item) in new_or_changed:
                        item["group_id"] = entry["group_id"]
                        item["parent_group_id"] = entry["group_id"]
                        item["split"] = "train"
                        item["split_reason"] = "forced_cross_label_near_duplicate_training"
        elif related and related[0].get("split") in {"train", "validation", "test"}:
            entry["split"] = str(related[0]["split"])
            entry["split_reason"] = "inherited_template_or_source_group"
        else:
            entry["split"] = _deterministic_split(entry["group_id"])
            entry["split_reason"] = "incremental_deterministic_group_assignment"

    entries = []
    for previous_entry in previous_entries:
        relative = _entry_key(previous_entry)
        if previous_entry.get("is_derived"):
            entries.append(previous_entry)
        elif relative in current_paths:
            entries.append(entries_by_path[relative])
    for relative in sorted(current_paths - existing_paths):
        entries.append(entries_by_path[relative])
    manifest = {
        "version": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(images_dir),
        "classes": {"0": "normal", "1": "tampered"},
        "split_policy": "incremental_sha256_phash_index_group_preserving_no_reassignment",
        "entries": entries,
    }
    return manifest, {
        "dry_run": True,
        "changed_paths": changed_paths,
        "blockers": blockers,
        "entry_count": len(entries),
        "existing_split_preserved": True,
    }


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="增量更新 AI 图片数据集清单；默认只预览")
    parser.add_argument("--images-dir", default="app/ai_detection/images")
    parser.add_argument("--apply", action="store_true", help="确认写入 dataset_manifest.json")
    args = parser.parse_args(argv)
    images_dir = Path(args.images_dir).resolve()
    manifest, summary = build_incremental_manifest(images_dir)
    summary["dry_run"] = not args.apply
    if args.apply and not summary["blockers"]:
        write_manifest_atomic(images_dir / "dataset_manifest.json", manifest)
        summary["written"] = True
    else:
        summary["written"] = False
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.apply and summary["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
