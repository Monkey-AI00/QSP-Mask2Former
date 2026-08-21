"""Synthetic highlight augmentation helpers shared by visualization scripts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]


@dataclass
class HighlightAugConfig:
    prob: float = 0.0
    spots_range: Tuple[int, int] = (1, 3)
    sigma_range: Tuple[int, int] = (30, 80)
    intensity_range: Tuple[int, int] = (150, 255)
    focus_on_object: bool = False
    object_bbox_shrink: float = 0.0
    clip_to_object: bool = False
    object_mask_dilate: int = 0
    object_mask_feather: int = 0


def _ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    """Convert a COCO polygon/bbox annotation to a binary mask."""
    segmentation = ann.get("segmentation", None)
    out = np.zeros((h, w), dtype=np.uint8)
    if isinstance(segmentation, list) and segmentation:
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
            cv2.fillPoly(out, [points.astype(np.int32)], color=1)
        if np.any(out > 0):
            return out

    bbox = ann.get("bbox", None)
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, box_w, box_h = [float(value) for value in bbox[:4]]
        x1 = int(max(0, min(w - 1, round(x))))
        y1 = int(max(0, min(h - 1, round(y))))
        x2 = int(max(0, min(w - 1, round(x + box_w - 1))))
        y2 = int(max(0, min(h - 1, round(y + box_h - 1))))
        if x2 >= x1 and y2 >= y1:
            out[y1 : y2 + 1, x1 : x2 + 1] = 1
    return out


def apply_synthetic_highlight(
    img_rgb: np.ndarray,
    cfg: HighlightAugConfig,
    dataset_dict: Optional[dict] = None,
) -> np.ndarray:
    """Add random Gaussian over-exposure spots to an RGB image."""
    del dataset_dict  # Reserved for future object-focused augmentation.
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb
    if float(cfg.prob) <= 0.0 or random.random() > float(cfg.prob):
        return img_rgb

    height, width = img_rgb.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)
    n_min, n_max = int(cfg.spots_range[0]), int(cfg.spots_range[1])
    if n_max < n_min:
        n_min, n_max = n_max, n_min
    n_spots = max(1, random.randint(max(1, n_min), max(1, n_max)))

    ys, xs = np.mgrid[0:height, 0:width]
    sigma_min, sigma_max = int(cfg.sigma_range[0]), int(cfg.sigma_range[1])
    if sigma_max < sigma_min:
        sigma_min, sigma_max = sigma_max, sigma_min
    intensity_min, intensity_max = int(cfg.intensity_range[0]), int(cfg.intensity_range[1])
    if intensity_max < intensity_min:
        intensity_min, intensity_max = intensity_max, intensity_min

    for _ in range(n_spots):
        center_x = random.uniform(0, max(1, width - 1))
        center_y = random.uniform(0, max(1, height - 1))
        sigma = float(max(1, random.randint(max(1, sigma_min), max(1, sigma_max))))
        amplitude = float(max(0, random.randint(max(0, intensity_min), max(0, intensity_max))))
        gaussian = np.exp(
            -((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2.0 * sigma * sigma)
        )
        heat += amplitude * gaussian.astype(np.float32)

    out = img_rgb.astype(np.float32) + heat[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
