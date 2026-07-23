"""AIGC 文档离线实验：构建去水印变体、训练候选并输出可审计结果包。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ai_detection.core.aigc_document import (
    DEFAULT_WATERMARK_REGION,
    FEATURE_NAMES,
    VARIANT_TYPES,
    apply_watermark_variant,
    extract_watermark_neutral_features,
    normalized_bbox_to_pixels,
)
from app.ai_detection.core.dataset_policy import AIGC_DOCUMENT_SOURCE, is_v3_baseline_sample


AIGC_SOURCE_FILES = frozenset(
    {f"tampered/{number}.png" for number in range(1, 20)} | {"tampered/20.jpg"}
)
EXPERIMENT_MANIFEST_NAME = "aigc_experiment_manifest.json"
EXPERIMENT_VERSION = 1
MINIMUM_SOURCE_GROUPS = 20
MAX_NORMAL_CONTROL_FPR = 0.02
MAX_WATERMARK_RECALL_GAP = 0.05
MAX_DERIVED_LONG_SIDE = 1600


@dataclass(frozen=True)
class AIGCRecord:
    """实验清单中的单张图片记录。"""

    record_id: str
    path: Path
    split: str
    group_id: str
    source_sha256: str
    business_label: int
    aigc_label: int
    stratum: str
    generator: str
    watermark_state: str
    variant_type: str
    watermark_regions: tuple[dict[str, Any], ...]
    is_derived: bool
    source_path: str
    template_family: str = "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 无法读取: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _image_path(images_root: Path, row: Mapping[str, Any]) -> Path:
    return (images_root / str(row.get("path") or "")).resolve()


def _watermark_regions() -> list[dict[str, Any]]:
    # 位置来自当前样本的人工观察；必须二审确认，且永远不是业务篡改字段。
    return [dict(DEFAULT_WATERMARK_REGION)]


def _copy_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _mark_aigc_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    marked = _copy_payload(row)
    marked["original_source"] = str(marked.get("original_source") or marked.get("source") or "unknown")
    marked["source"] = AIGC_DOCUMENT_SOURCE
    marked["fraud_type"] = AIGC_DOCUMENT_SOURCE
    marked["exclude_from_v3_training"] = True
    marked["exclude_from_v3_evaluation"] = True
    marked["exclude_from_v3_candidate_gate"] = True
    marked["aigc"] = {
        "generator": "doubao",
        "template_family": str(marked.get("template_cluster") or "unknown"),
        "watermark_state": "present",
        "watermark_removed": False,
        "watermark_regions": _watermark_regions(),
        "tampered_field_rois": [],
        "roi_truth_status": "unknown",
        "derivation": {"kind": "source", "derived_from": None},
    }
    return marked


def _source_record(row: Mapping[str, Any]) -> dict[str, Any]:
    aigc = row["aigc"]
    digest = str(row["sha256"])
    return {
        "record_id": f"aigc-source:{digest}",
        "path": str(row["path"]),
        "source_path": str(row["path"]),
        "source_sha256": digest,
        "sha256": digest,
        "split": str(row["split"]),
        "group_id": str(row["group_id"]),
        "parent_group_id": str(row.get("parent_group_id") or row["group_id"]),
        "business_label": 1,
        "aigc_label": 1,
        "stratum": "aigc_watermark_present",
        "generator": str(aigc["generator"]),
        "template_family": str(aigc["template_family"]),
        "watermark_state": "present",
        "variant_type": "original",
        "watermark_regions": _copy_payload(aigc["watermark_regions"]),
        "tampered_field_rois": [],
        "roi_truth_status": "unknown",
        "is_derived": False,
        "derivation": {"kind": "source", "derived_from": None},
    }


def _derived_path(role: str, source_sha256: str, variant_type: str) -> str:
    return f"derived/aigc_experiment/{role}/{source_sha256}/{variant_type}.jpg"


def _derived_record(
    row: Mapping[str, Any],
    *,
    role: str,
    variant_type: str,
    stratum: str,
    aigc_label: int,
    business_label: int,
    generator: str,
) -> dict[str, Any]:
    digest = str(row["sha256"])
    regions = _watermark_regions()
    return {
        "record_id": f"{role}:{digest}:{variant_type}",
        "path": _derived_path(role, digest, variant_type),
        "source_path": str(row["path"]),
        "source_sha256": digest,
        "sha256": None,
        "split": str(row["split"]),
        "group_id": str(row["group_id"]),
        "parent_group_id": str(row.get("parent_group_id") or row["group_id"]),
        "business_label": int(business_label),
        "aigc_label": int(aigc_label),
        "stratum": stratum,
        "generator": generator,
        "template_family": str(row.get("template_cluster") or "unknown"),
        "watermark_state": "removed" if aigc_label else "not_applicable",
        "variant_type": variant_type,
        "watermark_regions": regions,
        "tampered_field_rois": [],
        "roi_truth_status": "unknown" if aigc_label else "not_required_for_aigc_control",
        "is_derived": True,
        "derivation": {
            "kind": "watermark_robustness_variant" if aigc_label else "matched_corner_control",
            "derived_from": str(row["path"]),
            "source_sha256": digest,
            "transform": variant_type,
            "output_max_long_side": MAX_DERIVED_LONG_SIDE,
        },
    }


def _representative_rows(rows: Iterable[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    representatives: dict[str, Mapping[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (str(item.get("sha256") or ""), str(item.get("path") or ""))):
        representatives.setdefault(str(row.get("group_id") or ""), row)
    return list(representatives.values())[:count]


def _select_controls(source_rows: Sequence[Mapping[str, Any]], all_rows: Sequence[Mapping[str, Any]], maximum: int) -> list[tuple[str, Mapping[str, Any]]]:
    source_group_counts: dict[str, int] = defaultdict(int)
    for row in source_rows:
        source_group_counts[str(row["split"])] += 1
    total = max(1, sum(source_group_counts.values()))
    controls: list[tuple[str, Mapping[str, Any]]] = []
    for label, role in ((0, "normal_control"), (1, "real_local_tamper_control")):
        candidates = [
            row for row in all_rows
            if is_v3_baseline_sample(row)
            and int(row.get("label", -1)) == label
            and (label == 0 or bool(row.get("roi_sidecar")))
        ]
        for split, source_count in sorted(source_group_counts.items()):
            allocation = max(1, int(round(maximum * source_count / total)))
            controls.extend((role, row) for row in _representative_rows(
                (row for row in candidates if str(row.get("split")) == split), allocation
            ))
    return controls


def _resize_for_derived(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_DERIVED_LONG_SIDE:
        return image
    scale = MAX_DERIVED_LONG_SIDE / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _write_variant(images_root: Path, record: dict[str, Any]) -> None:
    source = (images_root / record["source_path"]).resolve()
    image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取派生源图: {source}")
    transformed = apply_watermark_variant(_resize_for_derived(image), str(record["variant_type"]), record["watermark_regions"])
    target = (images_root / record["path"]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", transformed, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"无法编码派生图: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        encoded.tofile(str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    record["sha256"] = sha256_file(target)
    record["width"] = int(transformed.shape[1])
    record["height"] = int(transformed.shape[0])


def validate_experiment_manifest(payload: Mapping[str, Any], images_root: str | Path | None = None) -> list[str]:
    """校验变体与源图不会跨 split，也不会把水印伪装成业务 ROI。"""
    errors: list[str] = []
    groups: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in payload.get("entries", []):
        if not isinstance(row, Mapping):
            errors.append("实验清单存在非对象条目")
            continue
        group = str(row.get("group_id") or "")
        split = str(row.get("split") or "")
        source_sha = str(row.get("source_sha256") or "")
        groups[group].add(split)
        source_splits[source_sha].add(split)
        for region in row.get("watermark_regions") or []:
            if not isinstance(region, Mapping) or region.get("field_type") != "other" or bool(region.get("is_tampered")):
                errors.append(f"水印区域不是 other 非篡改框: {row.get('record_id')}")
        if row.get("aigc_label") == 1 and row.get("tampered_field_rois"):
            errors.append(f"AIGC 样本不应伪造业务 ROI 真值: {row.get('record_id')}")
        if images_root is not None and row.get("is_derived"):
            path = (Path(images_root) / str(row.get("path") or "")).resolve()
            if not path.is_file():
                errors.append(f"派生图片不存在: {path}")
            elif row.get("sha256") and row["sha256"] != sha256_file(path):
                errors.append(f"派生图片 SHA 不匹配: {path}")
    errors.extend(f"group 跨 split: {group}" for group, splits in groups.items() if len(splits) > 1)
    errors.extend(f"源图跨 split: {source_sha}" for source_sha, splits in source_splits.items() if len(splits) > 1)
    return errors


def build_aigc_experiment(
    images_root: str | Path,
    *,
    apply: bool = False,
    maximum_controls_per_class: int = MINIMUM_SOURCE_GROUPS,
) -> dict[str, Any]:
    """标记原始 AIGC 图并按相同变换生成离线实验清单；默认 dry-run。"""
    root = Path(images_root).resolve()
    manifest_path = root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    all_rows = [row for row in manifest.get("entries", []) if isinstance(row, Mapping)]
    updated_entries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    for original in all_rows:
        row = _copy_payload(original)
        if str(row.get("path") or "") in AIGC_SOURCE_FILES:
            if not _image_path(root, row).is_file():
                missing_sources.append(str(row.get("path")))
            else:
                row = _mark_aigc_source_row(row)
                source_rows.append(row)
        updated_entries.append(row)
    if missing_sources:
        raise ValueError(f"AIGC 原图缺失: {', '.join(sorted(missing_sources))}")
    if len(source_rows) != len(AIGC_SOURCE_FILES):
        found = {str(row.get("path")) for row in source_rows}
        absent = sorted(AIGC_SOURCE_FILES - found)
        raise ValueError(f"manifest 未登记全部 AIGC 原图: {', '.join(absent)}")

    experiment_entries = [_source_record(row) for row in source_rows]
    for row in source_rows:
        for variant_type in VARIANT_TYPES:
            experiment_entries.append(_derived_record(
                row,
                role="aigc",
                variant_type=variant_type,
                stratum="aigc_watermark_removed",
                aigc_label=1,
                business_label=1,
                generator="doubao",
            ))
    for role, row in _select_controls(source_rows, updated_entries, maximum_controls_per_class):
        stratum = "normal_corner_control" if role == "normal_control" else "real_local_tamper_corner_control"
        for variant_type in VARIANT_TYPES:
            experiment_entries.append(_derived_record(
                row,
                role=role,
                variant_type=variant_type,
                stratum=stratum,
                aigc_label=0,
                business_label=int(row["label"]),
                generator="not_aigc",
            ))

    updated_manifest = _copy_payload(manifest)
    updated_manifest["entries"] = updated_entries
    experiment = {
        "version": EXPERIMENT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_only": True,
        "active_model_unchanged": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256_before_update": sha256_file(manifest_path),
        "source_manifest_sha256_after_update": None,
        "entries": experiment_entries,
    }
    errors = validate_experiment_manifest(experiment)
    if errors:
        raise ValueError("实验清单校验失败: " + "; ".join(errors))
    if apply:
        for record in experiment_entries:
            if record["is_derived"]:
                _write_variant(root, record)
        errors = validate_experiment_manifest(experiment, root)
        if errors:
            raise ValueError("派生样本校验失败: " + "; ".join(errors))
        _write_json_atomic(manifest_path, updated_manifest)
        experiment["source_manifest_sha256_after_update"] = sha256_file(manifest_path)
        _write_json_atomic(root / EXPERIMENT_MANIFEST_NAME, experiment)
    source_groups = len({str(row["group_id"]) for row in source_rows})
    return {
        "dry_run": not apply,
        "source_count": len(source_rows),
        "source_group_count": source_groups,
        "derived_count": sum(1 for row in experiment_entries if row["is_derived"]),
        "experiment_entry_count": len(experiment_entries),
        "excluded_from_v3_count": len(source_rows),
        "experiment_manifest": str(root / EXPERIMENT_MANIFEST_NAME),
        "manifest": updated_manifest,
        "experiment": experiment,
    }


def load_experiment_records(experiment_path: str | Path, images_root: str | Path) -> list[AIGCRecord]:
    root = Path(images_root).resolve()
    payload = _read_json(Path(experiment_path))
    errors = validate_experiment_manifest(payload, root)
    if errors:
        raise ValueError("AIGC 实验数据无效: " + "; ".join(errors))
    records = []
    for row in payload.get("entries", []):
        path = (root / str(row["path"])).resolve()
        if not path.is_file():
            continue
        records.append(AIGCRecord(
            record_id=str(row["record_id"]),
            path=path,
            split=str(row["split"]),
            group_id=str(row["group_id"]),
            source_sha256=str(row["source_sha256"]),
            business_label=int(row["business_label"]),
            aigc_label=int(row["aigc_label"]),
            stratum=str(row["stratum"]),
            generator=str(row["generator"]),
            watermark_state=str(row["watermark_state"]),
            variant_type=str(row["variant_type"]),
            watermark_regions=tuple(_copy_payload(item) for item in row.get("watermark_regions") or []),
            is_derived=bool(row["is_derived"]),
            source_path=str(row["source_path"]),
            template_family=str(row.get("template_family") or "unknown"),
        ))
    return records


def _feature_row(record: AIGCRecord) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(record.path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取实验图片: {record.path}")
    return extract_watermark_neutral_features(image, record.watermark_regions)


def _metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    positives = [row for row in items if int(row["aigc_label"]) == 1]
    negatives = [row for row in items if int(row["aigc_label"]) == 0]
    true_positive = sum(bool(row["predicted_aigc"]) for row in positives)
    false_positive = sum(bool(row["predicted_aigc"]) for row in negatives)
    recall = true_positive / len(positives) if positives else None
    specificity = (len(negatives) - false_positive) / len(negatives) if negatives else None
    return {
        "available": bool(items),
        "sample_count": len(items),
        "positive_sample_count": len(positives),
        "negative_sample_count": len(negatives),
        "aigc_recall": round(recall, 6) if recall is not None else None,
        "false_positive_rate": round(false_positive / len(negatives), 6) if negatives else None,
        "specificity": round(specificity, 6) if specificity is not None else None,
        "balanced_accuracy": round((recall + specificity) / 2.0, 6) if recall is not None and specificity is not None else None,
        "independent_group_count": len({str(row["group_id"]) for row in items}),
    }


def _watermark_pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["stratum"] in {"aigc_watermark_present", "aigc_watermark_removed"}:
            by_source[str(row["source_sha256"])].append(row)
    paired_present: list[float] = []
    paired_removed: list[float] = []
    for source_rows in by_source.values():
        present = [row for row in source_rows if row["stratum"] == "aigc_watermark_present"]
        removed = [row for row in source_rows if row["stratum"] == "aigc_watermark_removed"]
        if present and removed:
            paired_present.append(float(bool(present[0]["predicted_aigc"])))
            paired_removed.append(float(np.mean([bool(row["predicted_aigc"]) for row in removed])))
    present_recall = float(np.mean(paired_present)) if paired_present else None
    removed_recall = float(np.mean(paired_removed)) if paired_removed else None
    gap = present_recall - removed_recall if present_recall is not None and removed_recall is not None else None
    return {
        "paired_source_count": len(paired_present),
        "watermark_present_recall": round(present_recall, 6) if present_recall is not None else None,
        "watermark_removed_recall": round(removed_recall, 6) if removed_recall is not None else None,
        "watermark_recall_gap": round(gap, 6) if gap is not None else None,
    }


def evaluate_aigc_gates(rows: Sequence[Mapping[str, Any]], records: Sequence[AIGCRecord]) -> dict[str, Any]:
    """AIGC 候选门禁：样本不足或水印依赖时一律不可启用。"""
    metrics_by_stratum = {
        stratum: _metric([row for row in rows if row["stratum"] == stratum])
        for stratum in (
            "aigc_watermark_present",
            "aigc_watermark_removed",
            "normal_corner_control",
            "real_local_tamper_corner_control",
        )
    }
    pair_metrics = _watermark_pair_metrics(rows)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record.group_id].add(record.split)
    split_leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    distinct_generators = sorted({record.generator for record in records if record.aigc_label == 1 and record.generator != "not_aigc"})
    true_unwatermarked = any(
        record.aigc_label == 1 and record.watermark_state in {"absent", "unknown"} and not record.is_derived
        for record in records
    )
    generator_holdout = {
        "available": len(distinct_generators) >= 2 and true_unwatermarked,
        "reason": None if len(distinct_generators) >= 2 and true_unwatermarked else "需至少两个生成器且包含真实无水印 AIGC 来源",
    }
    blockers: list[str] = []
    for stratum in ("aigc_watermark_present", "aigc_watermark_removed"):
        count = metrics_by_stratum[stratum]["independent_group_count"]
        if count < MINIMUM_SOURCE_GROUPS:
            blockers.append(f"{stratum} 仅 {count} 个独立源组，少于 {MINIMUM_SOURCE_GROUPS}")
    gap = pair_metrics["watermark_recall_gap"]
    if gap is None:
        blockers.append("缺少带水印/去水印成对样本")
    elif gap > MAX_WATERMARK_RECALL_GAP:
        blockers.append(f"去水印召回比带水印低 {gap:.2%}，超过 {MAX_WATERMARK_RECALL_GAP:.0%}")
    normal_fpr = metrics_by_stratum["normal_corner_control"]["false_positive_rate"]
    if normal_fpr is None or normal_fpr > MAX_NORMAL_CONTROL_FPR:
        blockers.append(f"正常角落扰动误报率 {normal_fpr!r} 超过 {MAX_NORMAL_CONTROL_FPR:.0%}")
    if len(distinct_generators) < 2:
        blockers.append("AIGC 来源少于两个生成器")
    if not true_unwatermarked:
        blockers.append("缺少真实无水印 AIGC 来源")
    if split_leaks:
        blockers.append("实验样本 group 跨 split: " + ", ".join(split_leaks))
    return {
        "passed": not blockers,
        "candidate_only": True,
        "active_model_unchanged": True,
        "blockers": blockers,
        "strata": metrics_by_stratum,
        "watermark_pair": pair_metrics,
        "distinct_generators": distinct_generators,
        "true_unwatermarked_source_present": true_unwatermarked,
        "generator_holdout": generator_holdout,
        "group_split_leaks": split_leaks,
    }


def _new_candidate_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=0.35, class_weight="balanced", solver="liblinear", max_iter=500, random_state=42)),
    ])


def _generator_leaveout_metrics(records: Sequence[AIGCRecord], features: Mapping[str, np.ndarray], threshold: float) -> dict[str, Any]:
    """在数据足够后按生成器留出，避免把同一生成器当作泛化证据。"""
    generators = sorted({record.generator for record in records if record.aigc_label == 1 and record.generator != "not_aigc"})
    true_unwatermarked = any(
        record.aigc_label == 1 and record.watermark_state in {"absent", "unknown"} and not record.is_derived
        for record in records
    )
    if len(generators) < 2 or not true_unwatermarked:
        return {
            "available": False,
            "reason": "需至少两个生成器且包含真实无水印 AIGC 来源",
            "generators": generators,
            "folds": [],
        }
    folds = []
    for held_out in generators:
        train = [
            record for record in records
            if record.split == "train" and not (record.aigc_label == 1 and record.generator == held_out)
        ]
        target = [record for record in records if record.aigc_label == 1 and record.generator == held_out]
        if len({record.aigc_label for record in train}) != 2 or not target:
            folds.append({"held_out_generator": held_out, "available": False, "reason": "训练或留出集合缺少类别"})
            continue
        model = _new_candidate_model()
        model.fit(np.vstack([features[record.record_id] for record in train]), np.asarray([record.aigc_label for record in train]))
        scores = [float(model.predict_proba(features[record.record_id].reshape(1, -1))[0, 1]) for record in target]
        folds.append({
            "held_out_generator": held_out,
            "available": True,
            "sample_count": len(target),
            "independent_group_count": len({record.group_id for record in target}),
            "aigc_recall": round(float(np.mean([score >= threshold for score in scores])), 6),
        })
    return {"available": True, "reason": None, "generators": generators, "folds": folds}


def _annotate_row(row: Mapping[str, Any], target: Path) -> None:
    image = cv2.imdecode(np.fromfile(str(row["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return
    for region in row.get("watermark_regions") or []:
        x1, y1, x2, y2 = normalized_bbox_to_pixels(region, image.shape[1], image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (120, 120, 40), 2)
        cv2.putText(image, "watermark other", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 40), 1, cv2.LINE_AA)
    color = (40, 40, 220) if row["predicted_aigc"] else (40, 160, 50)
    cv2.putText(image, f"AIGC {float(row['aigc_score']):.3f} {row['proposed_result']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    cv2.putText(image, str(row["stratum"]), (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        encoded.tofile(str(target))


def _write_chart(output_dir: Path, strata: Mapping[str, Mapping[str, Any]]) -> None:
    canvas = np.full((360, 860, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, "AIGC offline strata metrics", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
    for index, (name, metric) in enumerate(strata.items()):
        y = 82 + index * 65
        recall = metric.get("aigc_recall")
        fpr = metric.get("false_positive_rate")
        cv2.putText(canvas, name, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
        value = float(recall if recall is not None else fpr if fpr is not None else 0.0)
        cv2.rectangle(canvas, (340, y - 16), (340 + int(value * 430), y + 2), (70, 120, 220), -1)
        cv2.putText(canvas, f"value={value:.3f} groups={metric['independent_group_count']}", (340, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imencode(".png", canvas)[1].tofile(str(output_dir / "strata_metrics.png"))


def _write_result_package(
    output_dir: Path,
    rows: list[dict[str, Any]],
    gates: Mapping[str, Any],
    *,
    threshold: float,
    train_count: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = {
        "all": _metric(rows),
        "splits": {split: _metric([row for row in rows if row["split"] == split]) for split in ("train", "validation", "test")},
        "strata": gates["strata"],
        "gates": gates,
        "threshold": threshold,
        "feature_names": list(FEATURE_NAMES),
        "training_sample_count": train_count,
        "candidate_only": True,
        "active_model_unchanged": True,
        "ocr_consistency_mode": "layout_proxy_without_production_ocr",
    }
    _write_json_atomic(output_dir / "metrics.json", metrics)
    _write_json_atomic(output_dir / "results.json", {"rows": rows, "candidate_only": True, "active_model_unchanged": True})
    columns = (
        "record_id", "split", "stratum", "image_path", "source_path", "group_id", "source_sha256",
        "template_family", "aigc_label", "business_label", "aigc_score", "predicted_aigc", "result", "proposed_result",
        "evidence_type", "watermark_state", "variant_type", "review_record_json",
    )
    with (output_dir / "per_image_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key) for key in columns},
                "review_record_json": json.dumps(row["review_record"], ensure_ascii=False),
            })
    annotated = output_dir / "annotated"
    annotated.mkdir()
    for row in rows:
        suffix = hashlib.sha256(str(row["record_id"]).encode("utf-8")).hexdigest()[:10]
        _annotate_row(row, annotated / f"{Path(str(row['image_path'])).stem}__{suffix}.jpg")
    _write_chart(output_dir, gates["strata"])
    blocker_lines = [f"- {reason}" for reason in gates["blockers"]] or ["- 无"]
    report = [
        "# AIGC 文档离线候选报告",
        "",
        f"- 训练样本：{train_count}",
        f"- 阈值：{threshold:.3f}",
        "- 候选仅离线评估，未写入活跃模型注册表。",
        f"- 门禁：{'通过' if gates['passed'] else '阻断'}",
        "",
        "## 阻断原因",
        *blocker_lines,
        "",
        "## 水印成对评估",
        f"- 成对源数：{gates['watermark_pair']['paired_source_count']}",
        f"- 带水印召回：{gates['watermark_pair']['watermark_present_recall']}",
        f"- 去水印召回：{gates['watermark_pair']['watermark_removed_recall']}",
        f"- 召回差：{gates['watermark_pair']['watermark_recall_gap']}",
        "",
        "图像特征在提取前统一中和水印区域；当前 OCR 一致性仅使用版式代理，不能据此声称已具备跨生成器泛化能力。",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def train_aigc_candidate(
    images_root: str | Path,
    experiment_path: str | Path,
    output_root: str | Path,
    *,
    threshold: float = 0.50,
    run_id: str | None = None,
) -> dict[str, Any]:
    """训练正则化整图 AIGC 候选，绝不修改 v3 模型或其注册表。"""
    records = load_experiment_records(experiment_path, images_root)
    train_records = [record for record in records if record.split == "train"]
    if len({record.aigc_label for record in train_records}) != 2:
        raise ValueError("AIGC train split 缺少正类或负类")
    features = {record.record_id: _feature_row(record) for record in records}
    train_features = np.vstack([features[record.record_id] for record in train_records])
    train_labels = np.asarray([record.aigc_label for record in train_records], dtype=np.int32)
    model = _new_candidate_model()
    model.fit(train_features, train_labels)
    rows: list[dict[str, Any]] = []
    for record in records:
        score = float(model.predict_proba(features[record.record_id].reshape(1, -1))[0, 1])
        predicted = score >= threshold
        proposed_result = "篡改" if predicted else "未触发AIGC拦截"
        rows.append({
            "record_id": record.record_id,
            "split": record.split,
            "stratum": record.stratum,
            "image_path": str(record.path),
            "source_path": record.source_path,
            "group_id": record.group_id,
            "source_sha256": record.source_sha256,
            "template_family": record.template_family,
            "aigc_label": record.aigc_label,
            "business_label": record.business_label,
            "aigc_score": round(score, 6),
            "predicted_aigc": predicted,
            "result": proposed_result,
            "proposed_result": proposed_result,
            "evidence_type": AIGC_DOCUMENT_SOURCE,
            "watermark_state": record.watermark_state,
            "variant_type": record.variant_type,
            "watermark_regions": list(record.watermark_regions),
            "review_record": {
                "schema_version": 2,
                "source": "aigc_offline_candidate",
                "evidence_type": AIGC_DOCUMENT_SOURCE,
                "aigc_score": round(score, 6),
                "candidate_only": True,
            },
        })
    gates = evaluate_aigc_gates(rows, records)
    gates["generator_holdout"] = _generator_leaveout_metrics(records, features, threshold)
    output = Path(output_root).resolve() / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    metrics = _write_result_package(output, rows, gates, threshold=threshold, train_count=len(train_records))
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "evidence_type": AIGC_DOCUMENT_SOURCE,
        "candidate_only": True,
        "active_model_unchanged": True,
    }, output / "candidate.joblib")
    return {
        "output_dir": str(output),
        "gates": gates,
        "metrics": metrics,
        "candidate_only": True,
        "active_model_unchanged": True,
    }
