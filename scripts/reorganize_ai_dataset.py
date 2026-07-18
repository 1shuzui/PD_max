#!/usr/bin/env python3
"""Normalize the image dataset into raw, derived and group-safe splits.

Canonical source images live in ``images/normal`` and ``images/tampered``.
Legacy ``*_enhanced`` files are retained under ``images/derived`` for audit
purposes but never participate in validation or test metrics.  ``pptest`` is a
known production tampered source, so it is registered in the tampered training
set and recorded separately as a training replay regression sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "validation", "test")
DERIVED_SPLIT = "derived"
TEMPLATE_PHASH_HAMMING_THRESHOLD = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> Tuple[int, int, str]:
    with Image.open(path) as image:
        image.verify()
        return image.width, image.height, str(image.format or "").upper()


def image_phash(path: Path) -> int:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片生成模板簇: {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.float32(small))[:8, :8]
    median = float(np.median(coefficients.flatten()[1:]))
    bits = (coefficients > median).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def phash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def is_prebuilt_enhanced(path: Path) -> bool:
    return bool(re.search(r"_enhanced(?:_\d+)?$", path.stem, flags=re.IGNORECASE))


def base_stem(path: Path) -> str:
    return re.sub(r"_enhanced(?:_\d+)?$", "", path.stem, flags=re.IGNORECASE)


def iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def move_or_deduplicate(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != sha256(source):
            raise FileExistsError(f"目标文件已存在且内容不同: {target}")
        source.unlink()
        return
    shutil.move(str(source), str(target))


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and sha256(target) == sha256(source):
            return
        raise FileExistsError(f"训练集注册文件已存在且内容不同: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def move_legacy_images(images_dir: Path) -> None:
    """Move legacy root images and real vouchers into canonical class folders."""
    normal = images_dir / "normal"
    tampered = images_dir / "tampered"
    normal.mkdir(parents=True, exist_ok=True)
    tampered.mkdir(parents=True, exist_ok=True)

    sources: List[Tuple[Path, int]] = []
    for path in iter_images(images_dir):
        lower = path.name.lower()
        if lower.startswith("no"):
            sources.append((path, 0))
        elif lower.startswith("p"):
            sources.append((path, 1))
        else:
            raise ValueError(f"无法从文件名确定标签: {path}")

    real_dir = images_dir / "真实凭证"
    if real_dir.exists():
        sources.extend((path, 0) for path in iter_images(real_dir))

    for source, label in sources:
        move_or_deduplicate(source, (normal if label == 0 else tampered) / source.name)

    if real_dir.exists() and not any(real_dir.iterdir()):
        real_dir.rmdir()


def move_prebuilt_enhancements(images_dir: Path) -> None:
    """Keep legacy enhanced files auditable without treating them as raw data."""
    for class_name in ("normal", "tampered"):
        source_dir = images_dir / class_name
        derived_dir = images_dir / "derived" / class_name
        for path in list(iter_images(source_dir)):
            if is_prebuilt_enhanced(path):
                move_or_deduplicate(path, derived_dir / path.name)


def register_pptest(images_dir: Path, pptest_dir: Path) -> None:
    """Register production tampered samples as canonical train inputs.

    Preserve the user-provided pptest directory and use hard links where the
    filesystem permits, so the original evidence remains easy to inspect.
    """
    target_dir = images_dir / "tampered"
    for source in iter_images(pptest_dir):
        digest = sha256(source)[:12]
        safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", source.stem).strip("._") or "sample"
        target = target_dir / f"pptest__{digest}__{safe_stem}{source.suffix.lower()}"
        link_or_copy(source, target)


def make_group_id(paths: List[Path], hashes: Dict[Path, str]) -> str:
    logical = sorted(base_stem(path).lower() for path in paths)
    content = sorted(hashes[path] for path in paths)
    key = "|".join(logical + content)
    return f"source-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def source_for_path(path: Path, label: int) -> str:
    name = path.name.lower()
    if name.startswith("pptest__"):
        return "production_tampered"
    if label == 0 and name.startswith("no"):
        return "legacy_normal"
    if label == 1 and name.startswith("p"):
        return "legacy_tampered"
    return "real_certificate" if label == 0 else "curated_tampered"


def _is_near_duplicate_source(left: Path, right: Path) -> bool:
    """Detect a same-source tamper pair with only a localized pixel change."""
    left_img = cv2.imdecode(np.fromfile(left, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imdecode(np.fromfile(right, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None or left_img.shape != right_img.shape:
        return False

    difference = cv2.absdiff(left_img, right_img)
    return bool(
        float(np.mean(difference)) <= 3.0
        and float(np.mean(difference > 20)) <= 0.03
    )


def _template_clusters(raw_files: List[Tuple[Path, int]]) -> Dict[Path, str]:
    """Build deterministic visual-template clusters for split isolation."""
    paths = [path for path, _label in raw_files]
    phashes = {path: image_phash(path) for path in paths}
    parent: Dict[Path, Path] = {path: path for path in paths}

    def find(path: Path) -> Path:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: Path, right: Path) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if phash_distance(phashes[left], phashes[right]) <= TEMPLATE_PHASH_HAMMING_THRESHOLD:
                union(left, right)

    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for path in paths:
        grouped[find(path)].append(path)
    result: Dict[Path, str] = {}
    for members in grouped.values():
        signature = "|".join(sorted(f"{phashes[path]:016x}" for path in members))
        cluster_id = f"template-{hashlib.sha256(signature.encode()).hexdigest()[:16]}"
        for path in members:
            result[path] = cluster_id
    return result


def _group_rows(
    raw_files: List[Tuple[Path, int]],
    hashes: Dict[Path, str],
    template_clusters: Dict[Path, str],
):
    parent: Dict[Path, Path] = {path: path for path, _label in raw_files}

    def find(path: Path) -> Path:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: Path, right: Path) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_logical: Dict[Tuple[int, str], List[Path]] = defaultdict(list)
    by_hash: Dict[Tuple[int, str], List[Path]] = defaultdict(list)
    labels = {path: label for path, label in raw_files}
    for path, label in raw_files:
        by_logical[(label, base_stem(path).lower())].append(path)
        by_hash[(label, hashes[path])].append(path)
    for family in list(by_logical.values()) + list(by_hash.values()):
        for path in family[1:]:
            union(family[0], path)

    by_template: Dict[str, List[Path]] = defaultdict(list)
    for path in parent:
        by_template[template_clusters[path]].append(path)
    for family in by_template.values():
        for path in family[1:]:
            union(family[0], path)

    normal_paths = [path for path, label in raw_files if label == 0]
    tampered_paths = [path for path, label in raw_files if label == 1]
    for normal_path in normal_paths:
        for tampered_path in tampered_paths:
            if _is_near_duplicate_source(normal_path, tampered_path):
                union(normal_path, tampered_path)

    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for path in parent:
        grouped[find(path)].append(path)

    rows = []
    for paths in grouped.values():
        rows.append((make_group_id(paths, hashes), sorted(paths, key=lambda item: item.name.lower())))
    return rows


def _assign_splits(group_rows, labels: Dict[Path, int]) -> Tuple[Dict[str, str], Dict[str, str]]:
    split_by_group: Dict[str, str] = {}
    reason_by_group: Dict[str, str] = {}
    for label in (0, 1):
        rows = []
        for group_id, paths in group_rows:
            path_labels = {labels[path] for path in paths}
            if len(path_labels) > 1:
                split_by_group[group_id] = "train"
                reason_by_group[group_id] = "forced_near_duplicate_source_pair_training"
                continue
            if label not in path_labels:
                continue
            if any(source_for_path(path, label) == "production_tampered" for path in paths):
                split_by_group[group_id] = "train"
                reason_by_group[group_id] = "forced_production_tampered_training"
            else:
                rows.append((group_id, paths))

        rows.sort(key=lambda row: row[1])
        total = len(rows)
        test_count = max(1, round(total * 0.15)) if total >= 3 else 0
        validation_count = max(1, round(total * 0.15)) if total >= 5 else 0
        for index, (group_id, _paths) in enumerate(rows):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            split_by_group[group_id] = split
            reason_by_group[group_id] = "grouped_sha256_and_base_stem_stratified"
    return split_by_group, reason_by_group


def build_manifest(images_dir: Path) -> Dict[str, object]:
    raw_files: List[Tuple[Path, int]] = []
    derived_files: List[Tuple[Path, int]] = []
    for label, dirname in ((0, "normal"), (1, "tampered")):
        raw_files.extend((path, label) for path in iter_images(images_dir / dirname))
        derived_files.extend((path, label) for path in iter_images(images_dir / "derived" / dirname))

    hashes = {path: sha256(path) for path, _label in raw_files + derived_files}
    labels_by_hash: Dict[str, set[int]] = defaultdict(set)
    for path, label in raw_files + derived_files:
        labels_by_hash[hashes[path]].add(label)
    conflicts = [digest for digest, labels in labels_by_hash.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(f"发现跨标签重复图片，无法训练: {conflicts[:3]}")

    labels_by_path = {path: label for path, label in raw_files}
    template_clusters = _template_clusters(raw_files)
    group_rows = _group_rows(raw_files, hashes, template_clusters)
    split_by_group, reason_by_group = _assign_splits(group_rows, labels_by_path)
    group_by_raw_path = {
        path: group_id
        for group_id, paths in group_rows
        for path in paths
    }
    parent_group_by_key = {
        (label, base_stem(path).lower()): group_by_raw_path[path]
        for path, label in raw_files
    }

    entries: List[Dict[str, object]] = []
    for group_id, paths in sorted(group_rows, key=lambda row: row[0]):
        split = split_by_group[group_id]
        for path in paths:
            label = labels_by_path[path]
            width, height, media_format = validate_image(path)
            source = source_for_path(path, label)
            entry = {
                    "path": path.relative_to(images_dir).as_posix(),
                    "label": label,
                    "class": "normal" if label == 0 else "tampered",
                    "source": source,
                    "split": split,
                    "split_reason": reason_by_group[group_id],
                    "group_id": group_id,
                    "parent_group_id": group_id,
                    "sha256": hashes[path],
                    "size_bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "format": media_format,
                    "is_derived": False,
                    "fixed_regression": source in {"legacy_normal", "legacy_tampered"},
                    "training_replay_regression": source == "production_tampered",
                    "template_cluster": template_clusters[path],
            }
            existing_manifest = images_dir / "dataset_manifest.json"
            if existing_manifest.is_file():
                try:
                    previous_entries = json.loads(existing_manifest.read_text(encoding="utf-8")).get("entries", [])
                    previous_by_path = {str(item.get("path")): item for item in previous_entries}
                    if previous_by_path.get(entry["path"], {}).get("roi_sidecar"):
                        entry["roi_sidecar"] = previous_by_path[entry["path"]]["roi_sidecar"]
                except (OSError, json.JSONDecodeError):
                    pass
            entries.append(entry)

    for path, label in sorted(derived_files, key=lambda item: str(item[0]).lower()):
        parent_group_id = parent_group_by_key.get((label, base_stem(path).lower()))
        if not parent_group_id:
            raise ValueError(f"派生图未找到原图家族: {path}")
        width, height, media_format = validate_image(path)
        entries.append(
            {
                "path": path.relative_to(images_dir).as_posix(),
                "label": label,
                "class": "normal" if label == 0 else "tampered",
                "source": "legacy_prebuilt_augmentation",
                "split": DERIVED_SPLIT,
                "split_reason": "prebuilt_derived_excluded_from_metrics",
                "group_id": parent_group_id,
                "parent_group_id": parent_group_id,
                "sha256": hashes[path],
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "format": media_format,
                "is_derived": True,
                "fixed_regression": False,
                "training_replay_regression": False,
                "template_cluster": template_clusters.get(path),
            }
        )

    return {
        "version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(images_dir),
        "classes": {"0": "normal", "1": "tampered"},
        "split_policy": "raw_grouped_sha256_base_stem_and_near_duplicate_source_pairs_70_15_15_with_forced_training_pairs",
        "entries": entries,
    }


def materialize_splits(images_dir: Path, manifest: Dict[str, object]) -> None:
    split_root = images_dir / "splits"
    for split in SPLITS:
        for class_name in ("normal", "tampered"):
            target = split_root / split / class_name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    for entry in manifest["entries"]:
        entry = dict(entry)
        split = str(entry["split"])
        if split not in SPLITS or bool(entry.get("is_derived")):
            continue
        source = images_dir / str(entry["path"])
        target_name = f"{entry['group_id']}__{Path(str(entry['path'])).name}"
        link_or_copy(source, split_root / split / str(entry["class"]) / target_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="app/ai_detection/images")
    parser.add_argument("--pptest-dir", default="app/ai_detection/pptest")
    args = parser.parse_args()
    images_dir = Path(args.images_dir).resolve()
    pptest_dir = Path(args.pptest_dir).resolve()
    move_legacy_images(images_dir)
    move_prebuilt_enhancements(images_dir)
    register_pptest(images_dir, pptest_dir)
    manifest = build_manifest(images_dir)
    (images_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    materialize_splits(images_dir, manifest)
    counts = defaultdict(int)
    for entry in manifest["entries"]:
        counts[f"{entry['split']}/{entry['class']}"] += 1
    print(json.dumps({"entries": len(manifest["entries"]), "counts": dict(sorted(counts.items()))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
