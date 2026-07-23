"""离线 AIGC 文档实验使用的无状态图像变换和特征。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


VARIANT_TYPES = (
    "watermark_inpaint",
    "watermark_cover",
    "watermark_blur",
    "watermark_crop_resize",
    "watermark_jpeg_recompress",
    "watermark_scaled_reencode",
)

FEATURE_NAMES = (
    "ela_mean",
    "ela_p95",
    "residual_std",
    "residual_mad",
    "dct_high_frequency_ratio",
    "dct_neighbor_delta",
    "edge_density",
    "edge_orientation_entropy",
    "gray_entropy",
    "saturation_std",
    "color_channel_delta",
    "jpeg_roundtrip_error",
    "layout_horizontal_cv",
    "layout_vertical_cv",
    "layout_component_density",
)

DEFAULT_WATERMARK_REGION = {
    "field_type": "other",
    "x1": 0.70,
    "y1": 0.84,
    "x2": 0.995,
    "y2": 0.995,
    "is_tampered": False,
    "source": "doubao_watermark_default",
    "review_required": True,
}


def normalized_bbox_to_pixels(region: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    """将归一化框裁剪到图像范围内。"""
    x1 = int(round(float(region.get("x1", 0.0)) * width))
    y1 = int(round(float(region.get("y1", 0.0)) * height))
    x2 = int(round(float(region.get("x2", 1.0)) * width))
    y2 = int(round(float(region.get("y2", 1.0)) * height))
    x1 = min(max(0, x1), max(0, width - 1))
    y1 = min(max(0, y1), max(0, height - 1))
    x2 = min(max(x1 + 1, x2), width)
    y2 = min(max(y1 + 1, y2), height)
    return x1, y1, x2, y2


def neutralize_regions(image: np.ndarray, regions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """用修补中和指定区域，避免特征直接利用水印文字。"""
    if image is None or image.size == 0:
        raise ValueError("图片为空")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for region in regions:
        x1, y1, x2, y2 = normalized_bbox_to_pixels(region, image.shape[1], image.shape[0])
        padding_x = max(2, int((x2 - x1) * 0.04))
        padding_y = max(2, int((y2 - y1) * 0.04))
        cv2.rectangle(
            mask,
            (max(0, x1 - padding_x), max(0, y1 - padding_y)),
            (min(image.shape[1], x2 + padding_x), min(image.shape[0], y2 + padding_y)),
            255,
            -1,
        )
    if not np.any(mask):
        return image.copy()
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image.copy()


def apply_watermark_variant(
    image: np.ndarray,
    variant_type: str,
    regions: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """生成与对照样本可复用的角落扰动，输出尺寸保持不变。"""
    if variant_type not in VARIANT_TYPES:
        raise ValueError(f"不支持的 AIGC 变体: {variant_type}")
    if image is None or image.size == 0:
        raise ValueError("图片为空")
    result = image.copy()
    h, w = result.shape[:2]
    first_region = regions[0] if regions else DEFAULT_WATERMARK_REGION
    x1, y1, x2, y2 = normalized_bbox_to_pixels(first_region, w, h)
    if variant_type == "watermark_inpaint":
        return neutralize_regions(result, regions)
    if variant_type == "watermark_cover":
        ring_x1 = max(0, x1 - max(2, (x2 - x1) // 8))
        ring_y1 = max(0, y1 - max(2, (y2 - y1) // 8))
        ring = result[ring_y1:y2, ring_x1:x2]
        color = tuple(int(value) for value in np.median(ring.reshape(-1, 3), axis=0)) if ring.size else (240, 240, 240)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)
        return result
    if variant_type == "watermark_blur":
        crop = result[y1:y2, x1:x2]
        if crop.size:
            kernel = max(3, ((max(crop.shape[:2]) // 12) * 2) + 1)
            result[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (kernel, kernel), 0)
        return result
    if variant_type == "watermark_crop_resize":
        crop_height = max(1, min(h - 1, int(round(h * 0.88))))
        cropped = result[:crop_height, :]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    cleaned = neutralize_regions(result, regions)
    if variant_type == "watermark_jpeg_recompress":
        return _jpeg_roundtrip(cleaned, 72)
    scaled = cv2.resize(cleaned, (max(1, int(w * 0.68)), max(1, int(h * 0.68))), interpolation=cv2.INTER_AREA)
    return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_CUBIC)


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    histogram, _ = np.histogram(values.ravel(), bins=bins, range=(0, 256), density=False)
    probability = histogram.astype(np.float64) / max(1, histogram.sum())
    probability = probability[probability > 0]
    return float(-np.sum(probability * np.log2(probability))) / math.log2(bins)


def _dct_features(gray: np.ndarray) -> tuple[float, float]:
    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    coeff = np.abs(cv2.dct(resized.astype(np.float32)))
    total = float(coeff.sum()) + 1e-6
    high_frequency = float(coeff[96:, 96:].sum()) / total
    neighboring = np.abs(coeff[:, 1:] - coeff[:, :-1])
    return high_frequency, float(neighboring.mean() / 255.0)


def _layout_features(gray: np.ndarray) -> tuple[float, float, float]:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    horizontal = binary.mean(axis=1) / 255.0
    vertical = binary.mean(axis=0) / 255.0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = sum(1 for index in range(1, count) if 4 <= stats[index, cv2.CC_STAT_AREA] <= gray.size * 0.05)
    return (
        float(horizontal.std() / max(horizontal.mean(), 1e-6)),
        float(vertical.std() / max(vertical.mean(), 1e-6)),
        float(components / max(1, gray.size / 10_000)),
    )


def extract_watermark_neutral_features(image: np.ndarray, regions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """提取整图频域、噪声和版式代理特征，先统一中和角落区域。"""
    cleaned = neutralize_regions(image, regions)
    small = cv2.resize(cleaned, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    recompressed = _jpeg_roundtrip(small, 90)
    difference = cv2.absdiff(small, recompressed)
    residual = cv2.Laplacian(gray, cv2.CV_32F)
    high_frequency, neighbor_delta = _dct_features(gray)
    edges = cv2.Canny(gray, 80, 180)
    gradients_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = (np.arctan2(gradients_y, gradients_x) + np.pi) * (180.0 / (2.0 * np.pi))
    orientation_entropy = _entropy(angles.astype(np.uint8), bins=18)
    layout_horizontal_cv, layout_vertical_cv, component_density = _layout_features(gray)
    left = small[:, : small.shape[1] // 2].astype(np.float32)
    right = small[:, small.shape[1] // 2 :].astype(np.float32)
    values = np.array(
        (
            float(difference.mean() / 255.0),
            float(np.percentile(difference, 95) / 255.0),
            float(residual.std() / 32.0),
            float(np.median(np.abs(residual - np.median(residual))) / 32.0),
            high_frequency,
            neighbor_delta,
            float(np.mean(edges > 0)),
            orientation_entropy,
            _entropy(gray),
            float(hsv[:, :, 1].std() / 255.0),
            float(np.abs(left.mean(axis=(0, 1)) - right.mean(axis=(0, 1))).mean() / 255.0),
            float(difference.mean() / 255.0),
            layout_horizontal_cv,
            layout_vertical_cv,
            component_density,
        ),
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
