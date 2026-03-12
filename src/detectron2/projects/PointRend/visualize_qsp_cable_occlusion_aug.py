#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 QSP 训练中的“粗线圈/粗电缆遮挡增强”效果。

功能：
- 从 COCO json 中按文件自然顺序或显式列表选择图片
- 读取每张图对应的实例 mask，并合并成 object mask
- 单独应用粗线圈遮挡增强（不改 GT）
- 导出：
  - orig/        原图
  - mask/        目标 mask 叠加图
  - aug/         粗线圈遮挡增强图
  - side_by_side/ 原图 + 增强图 并排对比

示例：
  PYTHONPATH=/home/user/sjw/Yolo_pointrend/detectron2:/home/user/sjw/Yolo_pointrend/detectron2/projects/PointRend:/home/user/sjw/Yolo_pointrend/Mask2Former \
  python /home/user/sjw/Yolo_pointrend/detectron2/projects/PointRend/visualize_qsp_cable_occlusion_aug.py \
    --dataset-root /home/user/sjw/Yolo_pointrend/detectron2/plug_train_merged \
    --json-file plug_train.json \
    --out-dir /home/user/sjw/Yolo_pointrend/detectron2/projects/PointRend/output/cable_aug_vis \
    --index-range 1-8 --index-base 1 --num 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO

_D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _D2_ROOT not in sys.path:
    sys.path.insert(0, _D2_ROOT)

_M2F_ROOT = os.path.abspath(os.path.join(_D2_ROOT, "..", "Mask2Former"))
if _M2F_ROOT not in sys.path:
    sys.path.insert(0, _M2F_ROOT)

from mask2former.data.dataset_mappers.plug_qsp_instance_dataset_mapper import apply_wire_occlusion


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _put_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    bar_h = min(34, out.shape[0])
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, min(24, bar_h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])

    def _pad_to_h(img: np.ndarray, hh: int) -> np.ndarray:
        if img.shape[0] == hh:
            return img
        pad = hh - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.concatenate([_pad_to_h(left_bgr, h), _pad_to_h(right_bgr, h)], axis=1)


def _overlay_mask(img_bgr: np.ndarray, mask01: np.ndarray, color_bgr=(0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    out = img_bgr.copy()
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return out
    out[m > 0] = (
        out[m > 0].astype(np.float32) * (1.0 - alpha) + np.asarray(color_bgr, dtype=np.float32) * alpha
    ).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color_bgr, thickness=2, lineType=cv2.LINE_AA)
    return out


def _natural_sort_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _read_coco_images(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data.get("images", []), key=lambda im: _natural_sort_key(str(im.get("file_name", ""))))


def _parse_index_range(s: str) -> Tuple[int, int]:
    ss = str(s).strip().replace(":", "-")
    a_str, b_str = ss.split("-", 1)
    a, b = int(a_str), int(b_str)
    if a > b:
        a, b = b, a
    return a, b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize QSP thick cable/coil occlusion augmentation.")
    p.add_argument("--dataset-root", required=True, help="数据集目录（图片和 COCO json 所在目录）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO 标注文件名（相对于 dataset-root）")
    p.add_argument("--out-dir", default="./output/cable_aug_vis", help="输出目录")
    p.add_argument("--num", type=int, default=8, help="最多可视化多少张")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--images", default="", help="可选：逗号分隔的图片路径列表（绝对路径或相对 dataset-root）")
    p.add_argument("--index-range", default="", help="可选：按自然顺序选择范围，如 21-25")
    p.add_argument("--index-base", type=int, default=1, choices=(0, 1), help="index-range 的基准，默认 1")
    p.add_argument("--wire-num-min", type=int, default=2)
    p.add_argument("--wire-num-max", type=int, default=4)
    p.add_argument("--wire-thickness-min", type=int, default=22)
    p.add_argument("--wire-thickness-max", type=int, default=42)
    p.add_argument("--wire-ctrl-pts-min", type=int, default=4)
    p.add_argument("--wire-ctrl-pts-max", type=int, default=6)
    p.add_argument("--wire-pad-x", type=int, default=80)
    p.add_argument("--wire-pad-y", type=int, default=45)
    p.add_argument("--wire-brightness-min", type=float, default=190.0)
    p.add_argument("--wire-brightness-max", type=float, default=250.0)
    return p.parse_args()


def _normalize_image_path(path_str: str, dataset_root: str) -> str:
    p = str(path_str).strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(dataset_root, p))


def _build_picks(dataset_root: str, json_path: str, args: argparse.Namespace) -> List[Dict]:
    images = _read_coco_images(json_path)
    if str(args.images).strip():
        wanted = [_normalize_image_path(x, dataset_root) for x in str(args.images).split(",") if str(x).strip()]
        wanted_set = set(wanted)
        picks = [im for im in images if os.path.abspath(os.path.join(dataset_root, str(im.get("file_name", "")))) in wanted_set]
    else:
        picks = images

    if str(args.index_range).strip():
        a, b = _parse_index_range(str(args.index_range))
        if int(args.index_base) == 1:
            a0, b0 = max(0, a - 1), max(0, b - 1)
        else:
            a0, b0 = max(0, a), max(0, b)
        picks = picks[a0 : b0 + 1]

    if int(args.num) > 0:
        picks = picks[: int(args.num)]
    return picks


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    dataset_root = os.path.abspath(args.dataset_root)
    json_path = os.path.join(dataset_root, args.json_file)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")

    picks = _build_picks(dataset_root, json_path, args)
    if not picks:
        raise RuntimeError("未找到可视化图片，请检查 images/index-range 参数")

    coco = COCO(json_path)

    aug_cfg = {
        "wire_num_min": int(args.wire_num_min),
        "wire_num_max": int(args.wire_num_max),
        "wire_thickness_min": int(args.wire_thickness_min),
        "wire_thickness_max": int(args.wire_thickness_max),
        "wire_ctrl_pts_min": int(args.wire_ctrl_pts_min),
        "wire_ctrl_pts_max": int(args.wire_ctrl_pts_max),
        "wire_pad_x": int(args.wire_pad_x),
        "wire_pad_y": int(args.wire_pad_y),
        "wire_brightness_min": float(args.wire_brightness_min),
        "wire_brightness_max": float(args.wire_brightness_max),
    }

    out_dir = os.path.abspath(args.out_dir)
    orig_dir = os.path.join(out_dir, "orig")
    mask_dir = os.path.join(out_dir, "mask")
    aug_dir = os.path.join(out_dir, "aug")
    side_dir = os.path.join(out_dir, "side_by_side")
    for d in (orig_dir, mask_dir, aug_dir, side_dir):
        _ensure_dir(d)

    print(f"输出目录: {out_dir}")
    print(f"粗线圈增强参数: {aug_cfg}")

    for i, im in enumerate(picks):
        file_name = str(im.get("file_name", ""))
        img_path = os.path.join(dataset_root, file_name)
        image_id = int(im["id"])
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[跳过] 读取失败: {img_path}")
            continue

        obj_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for ann in anns:
            if int(ann.get("iscrowd", 0)) != 0:
                continue
            obj_mask = np.maximum(obj_mask, coco.annToMask(ann).astype(np.uint8))

        aug_bgr = apply_wire_occlusion(bgr, obj_mask, aug_cfg)

        orig_vis = _put_label(bgr, "ORIGINAL")
        mask_vis = _put_label(_overlay_mask(bgr, obj_mask, color_bgr=(80, 220, 120), alpha=0.3), "OBJECT MASK")
        aug_vis = _put_label(aug_bgr, "THICK CABLE / COIL OCCLUSION")
        side = _side_by_side(orig_vis, aug_vis)

        stem = f"{i:03d}_{os.path.splitext(os.path.basename(file_name))[0]}"
        cv2.imwrite(os.path.join(orig_dir, f"{stem}.png"), orig_vis)
        cv2.imwrite(os.path.join(mask_dir, f"{stem}.png"), mask_vis)
        cv2.imwrite(os.path.join(aug_dir, f"{stem}.png"), aug_vis)
        cv2.imwrite(os.path.join(side_dir, f"{stem}.png"), side)
        print(f"[{i+1}/{len(picks)}] saved: {stem}.png")

    print("完成。优先查看 side_by_side/ 与 mask/ 目录。")


if __name__ == "__main__":
    main()
