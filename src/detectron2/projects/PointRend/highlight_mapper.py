#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四阶段：数据模拟增强（强光/过曝），强迫网络学会依赖形状先验。

实现：
- apply_synthetic_highlight: 在 RGB 图像上叠加 1~N 个高斯光斑，模拟局部过曝
- HighlightMapper: 继承 detectron2 的 DatasetMapper，在几何增强前注入强光（仅训练阶段）
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper


@dataclass
class HighlightAugConfig:
    prob: float = 0.5
    spots_range: Tuple[int, int] = (1, 3)
    sigma_range: Tuple[int, int] = (30, 80)
    intensity_range: Tuple[int, int] = (150, 255)
    # 是否优先把光斑落在目标（插头）上：需要 annotations/segmentation 或 bbox
    focus_on_object: bool = True
    # 当 focus_on_object=True 时，允许对目标 bbox 做一定收缩，减少打到边缘背景
    object_bbox_shrink: float = 0.15
    # 是否把高亮限制在目标区域（mask 优先，bbox 兜底）；可让“洞”更集中在插头而不是背景
    clip_to_object: bool = True
    # clip_to_object=True 时，可对 mask/bbox 做膨胀（像素），让边缘更自然
    object_mask_dilate: int = 15
    # clip_to_object=True 时的“软边”宽度（像素）。0 表示硬边裁剪；>0 表示边界处平滑衰减到0。
    # 建议：如果你想“只在插头上饱和、边缘不要晕开”，将 dilate=0 且 feather=0~5。
    object_mask_feather: int = 0


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _ann_to_mask(ann: dict, h: int, w: int) -> Optional[np.ndarray]:
    """
    将 COCO annotation 的 segmentation 转成二值 mask（H,W）。
    返回 uint8(0/1)，失败则返回 None。
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return None
    try:
        from pycocotools import mask as mask_util

        if isinstance(seg, list):
            # polygon(s): list[list[float]]
            rles = mask_util.frPyObjects(seg, h, w)
            rle = mask_util.merge(rles)
        elif isinstance(seg, dict):
            # RLE
            rle = seg
        else:
            return None
        m = mask_util.decode(rle)
        if m is None:
            return None
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)
    except Exception:
        return None


def _dilate_mask(mask01: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask01
    k = int(dilate_px) * 2 + 1
    try:
        import cv2

        kernel = np.ones((k, k), dtype=np.uint8)
        return (cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1) > 0).astype(np.uint8)
    except Exception:
        try:
            from scipy import ndimage

            return ndimage.binary_dilation(mask01.astype(bool), iterations=int(dilate_px)).astype(np.uint8)
        except Exception:
            return mask01


def _soft_edge_alpha(mask01: np.ndarray, feather_px: int) -> np.ndarray:
    """
    由二值 mask 生成 alpha mask（float32, [0,1]）：
    - mask 内部 alpha=1
    - mask 外部在 feather_px 范围内线性衰减到 0（超过范围为 0）
    """
    feather_px = int(max(0, feather_px))
    if feather_px <= 0:
        return mask01.astype(np.float32)

    m = (mask01 > 0).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)

    # 计算“mask 外部点到 mask 内部的距离”
    # OpenCV 的 distanceTransform 对非零像素计算到最近零像素的距离。
    # 因此对 inv（外部=1，内部=0）做 distanceTransform，得到外部点到边界的距离。
    dist_out: np.ndarray
    try:
        import cv2

        dist_out = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    except Exception:
        try:
            from scipy import ndimage

            dist_out = ndimage.distance_transform_edt(inv.astype(bool)).astype(np.float32)
        except Exception:
            # 最差 fallback：无距离变换时退化成硬边
            return mask01.astype(np.float32)

    # 外部：alpha = clamp(1 - dist/feather, 0, 1)，内部强制为 1
    alpha = (1.0 - dist_out / float(feather_px)).clip(0.0, 1.0)
    alpha[m > 0] = 1.0
    return alpha.astype(np.float32)


def _pick_object_target(
    dataset_dict: dict, h: int, w: int, shrink: float
) -> Optional[Tuple[int, int, Optional[np.ndarray]]]:
    """
    从 dataset_dict["annotations"] 里挑一个实例，并返回：
    - center (cx,cy)：优先从 mask 上采样；否则在 bbox（可收缩）内采样
    - object_mask01：如果有 segmentation/bbox，则返回目标区域 mask（0/1）
    """
    annos = dataset_dict.get("annotations", None)
    if not annos:
        return None

    # 过滤 iscrowd / 无 bbox 的标注
    candidates = [a for a in annos if a.get("iscrowd", 0) == 0 and "bbox" in a]
    if not candidates:
        return None
    ann = random.choice(candidates)

    # 优先用 segmentation 采样“真正在插头上”的点
    m = _ann_to_mask(ann, h=h, w=w)
    if m is not None and int(m.sum()) > 0:
        ys, xs = np.nonzero(m)
        idx = random.randrange(len(xs))
        return int(xs[idx]), int(ys[idx]), m.astype(np.uint8)

    # fallback: 用 bbox 采样（可收缩，减少落到背景）
    x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
    if bw <= 1 or bh <= 1:
        return None

    # shrink bbox
    s = max(0.0, min(0.49, float(shrink)))
    x1 = int(round(x + bw * s))
    y1 = int(round(y + bh * s))
    x2 = int(round(x + bw * (1.0 - s)))
    y2 = int(round(y + bh * (1.0 - s)))

    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = random.randint(x1, x2)
    cy = random.randint(y1, y2)
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1 : y2 + 1, x1 : x2 + 1] = 1
    return cx, cy, m


def apply_synthetic_highlight(image: np.ndarray, cfg: HighlightAugConfig, *, dataset_dict: Optional[dict] = None) -> np.ndarray:
    """
    在图像上随机生成高斯强光，模拟过曝。
    Args:
        image: uint8 RGB (H,W,3)
        cfg: 参数配置
        dataset_dict: Detectron2 dataset_dict（可选）。提供时可把过曝落在目标上。
    Returns:
        uint8 RGB
    """
    if cfg.prob <= 0:
        return image
    if random.random() > cfg.prob:
        return image

    img = image.astype(np.float32, copy=True)
    h, w = img.shape[:2]

    # 预计算网格，避免每个 spot 重复 meshgrid
    yy, xx = np.mgrid[0:h, 0:w]

    nmin, nmax = cfg.spots_range
    num_spots = random.randint(int(nmin), int(nmax))

    smin, smax = cfg.sigma_range
    imin, imax = cfg.intensity_range

    # =========================================================
    # 性能优化（关键）：
    # 1) clip+feather 需要 distanceTransform/EDT（很贵）
    # 2) 以前每个光斑都会重新从 annotations 解码 mask + 计算 alpha
    #    -> 1~3 个光斑会让同一张图重复做 1~3 次距离变换，benchmark 会被拖慢很多
    # 这里改为：每张图只生成一次 obj_mask/alpha，并在其上采样多个中心点
    # =========================================================
    obj_alpha: Optional[np.ndarray] = None
    obj_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (ys, xs)
    if cfg.focus_on_object and cfg.clip_to_object and dataset_dict is not None:
        target0 = _pick_object_target(dataset_dict, h=h, w=w, shrink=cfg.object_bbox_shrink)
        if target0 is not None:
            _cx0, _cy0, obj_mask01_0 = target0
            if obj_mask01_0 is not None:
                obj_mask01_0 = _dilate_mask(obj_mask01_0, cfg.object_mask_dilate)
                obj_alpha = _soft_edge_alpha(obj_mask01_0, cfg.object_mask_feather)
                ys0, xs0 = np.nonzero(obj_mask01_0)
                if len(xs0) > 0:
                    obj_coords = (ys0, xs0)

    for _ in range(num_spots):
        # 选择光斑中心：优先在目标内部采样；否则全图随机
        if cfg.focus_on_object and obj_coords is not None:
            ys, xs = obj_coords
            idx = random.randrange(len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])
        else:
            target = None
            if cfg.focus_on_object and dataset_dict is not None:
                target = _pick_object_target(dataset_dict, h=h, w=w, shrink=cfg.object_bbox_shrink)

            if target is None:
                cx = random.randint(int(w * 0.2), int(w * 0.8))
                cy = random.randint(int(h * 0.2), int(h * 0.8))
                obj_alpha = None
            else:
                cx, cy, obj_mask01 = target
                if cfg.clip_to_object and obj_mask01 is not None:
                    obj_mask01 = _dilate_mask(obj_mask01, cfg.object_mask_dilate)
                    obj_alpha = _soft_edge_alpha(obj_mask01, cfg.object_mask_feather)
                else:
                    obj_alpha = None
        sigma = max(1, random.randint(int(smin), int(smax)))
        intensity = random.randint(int(imin), int(imax))

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * float(sigma) ** 2)).astype(np.float32)
        if cfg.clip_to_object and obj_alpha is not None:
            gaussian = gaussian * obj_alpha
        highlight = gaussian[..., None] * float(intensity)  # (H,W,1)
        img += highlight

    return np.clip(img, 0, 255).astype(np.uint8)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


class HighlightMapper(DatasetMapper):
    """
    自定义 Mapper：训练时实时注入强光（仅 photometric，不改变几何与标注对齐逻辑）。

    用法：在 build_detection_train_loader(..., mapper=HighlightMapper(...)) 传入即可。
    """

    def __init__(
        self,
        cfg,
        is_train: bool = True,
        augmentations: Optional[List] = None,
        highlight_cfg: Optional[HighlightAugConfig] = None,
    ):
        # 允许外部显式传 augmentations（保持与项目现有 Trainer.build_train_loader 一致）
        if augmentations is None:
            augmentations = utils.build_augmentation(cfg, is_train)

        super().__init__(
            cfg,
            is_train=is_train,
            augmentations=augmentations,
        )

        if highlight_cfg is None:
            # 通过环境变量快速调参（不改 cfg 结构）
            highlight_cfg = HighlightAugConfig(
                prob=_env_float("HIGHLIGHT_PROB", 0.5),
                focus_on_object=_env_str("HIGHLIGHT_FOCUS", "object").lower() in ("obj", "object", "true", "1", "yes", "y"),
                object_bbox_shrink=_env_float("HIGHLIGHT_BBOX_SHRINK", 0.15),
                clip_to_object=_env_str("HIGHLIGHT_CLIP", "1").lower() in ("true", "1", "yes", "y", "clip"),
                object_mask_dilate=int(_env_float("HIGHLIGHT_DILATE", 15)),
                object_mask_feather=int(_env_float("HIGHLIGHT_FEATHER", 0)),
            )
        self.highlight_cfg = highlight_cfg

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # 仅训练阶段注入强光（制造“洞”，迫使 head 依赖形状先验）
        if self.is_train:
            image = apply_synthetic_highlight(image, self.highlight_cfg, dataset_dict=dataset_dict)

        # 走 detectron2 原始 DatasetMapper 的后续逻辑（几何增强 + annotation transform）
        sem_seg_gt = None
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk)

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


