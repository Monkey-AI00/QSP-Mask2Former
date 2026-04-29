#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化第四阶段：强光/过曝模拟增强效果。

功能：
- 从 COCO json 中随机抽样若干张图
- 读取原图
- 应用 highlight_mapper.apply_synthetic_highlight
- 保存：orig / aug / side_by_side（并排对比）

示例：
  PYTHONPATH=... conda run -n pointrend python visualize_highlight_aug.py \
    --dataset-root /home/users1/sjw/cursor/workspace/datasets/plug_train1 \
    --json-file plug_train.json \
    --out-dir /home/users1/sjw/cursor/workspace/outputs/output/highlight_vis \
    --num 8 --seed 0 --prob 0.8
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import cv2
import numpy as np

from highlight_mapper import HighlightAugConfig, apply_synthetic_highlight


def _read_coco_image_list(json_path: str) -> List[str]:
    import json

    with open(json_path, "r") as f:
        data = json.load(f)
    images = data.get("images", [])
    files = []
    for im in images:
        fn = im.get("file_name", "")
        if fn:
            files.append(str(fn))
    return files


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    w1, w2 = left_bgr.shape[1], right_bgr.shape[1]

    def pad_to_h(img: np.ndarray, H: int) -> np.ndarray:
        if img.shape[0] == H:
            return img
        pad = H - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    left = pad_to_h(left_bgr, h)
    right = pad_to_h(right_bgr, h)
    return np.concatenate([left, right], axis=1)


def _put_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (min(520, out.shape[1] - 1), 32), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize synthetic highlight augmentation.")
    p.add_argument("--dataset-root", required=True, help="数据集目录（图片和 COCO json 所在目录）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO 标注文件名（相对于 dataset-root）")
    p.add_argument("--out-dir", default="./output/highlight_vis", help="输出目录")
    p.add_argument("--num", type=int, default=8, help="抽样可视化张数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--prob", type=float, default=0.8, help="应用强光的概率（建议可视化时设高一点）")
    p.add_argument(
        "--focus",
        choices=["object", "any"],
        default="object",
        help="光斑位置：object=尽量落在标注目标上；any=全图随机",
    )
    p.add_argument("--bbox-shrink", type=float, default=0.15, help="focus=object 时 bbox 收缩比例（降低打到背景概率）")
    p.add_argument("--clip", action="store_true", help="把高亮限制在目标区域（mask/bbox）内")
    p.add_argument("--dilate", type=int, default=15, help="clip 时对目标 mask/bbox 膨胀像素（边缘更自然）")
    p.add_argument("--feather", type=int, default=0, help="clip 时目标边界软边像素（0=硬边；>0=平滑衰减）")
    p.add_argument("--spots", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"), help="光斑数量范围")
    p.add_argument("--sigma", type=int, nargs=2, default=[30, 80], metavar=("MIN", "MAX"), help="光斑 sigma 范围")
    p.add_argument("--intensity", type=int, nargs=2, default=[150, 255], metavar=("MIN", "MAX"), help="光斑强度范围")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_root = os.path.abspath(args.dataset_root)
    json_path = os.path.join(dataset_root, args.json_file)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")

    # 读取一次 json，便于按 file_name 找 annotations（用于 focus=object）
    import json

    with open(json_path, "r") as f:
        coco = json.load(f)

    files = _read_coco_image_list(json_path)
    if not files:
        raise RuntimeError("json 中未找到 images/file_name")

    num = min(args.num, len(files))
    picks = random.sample(files, k=num)

    out_dir = os.path.abspath(args.out_dir)
    orig_dir = os.path.join(out_dir, "orig")
    aug_dir = os.path.join(out_dir, "aug")
    side_dir = os.path.join(out_dir, "side_by_side")
    for d in [orig_dir, aug_dir, side_dir]:
        _ensure_dir(d)

    hcfg = HighlightAugConfig(
        prob=float(args.prob),
        spots_range=(int(args.spots[0]), int(args.spots[1])),
        sigma_range=(int(args.sigma[0]), int(args.sigma[1])),
        intensity_range=(int(args.intensity[0]), int(args.intensity[1])),
        focus_on_object=(args.focus == "object"),
        object_bbox_shrink=float(args.bbox_shrink),
        clip_to_object=bool(args.clip),
        object_mask_dilate=int(args.dilate),
        object_mask_feather=int(args.feather),
    )

    print(f"输出目录: {out_dir}")
    print(f"高光参数: prob={hcfg.prob}, spots={hcfg.spots_range}, sigma={hcfg.sigma_range}, intensity={hcfg.intensity_range}")

    for i, rel in enumerate(picks):
        img_path = os.path.join(dataset_root, rel)
        if not os.path.exists(img_path):
            print(f"[跳过] 找不到图片: {img_path}")
            continue

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[跳过] 读取失败: {img_path}")
            continue

        rgb = _bgr_to_rgb(bgr)

        dd = None
        if args.focus == "object":
            image_id = None
            for im in coco.get("images", []):
                if str(im.get("file_name", "")) == str(rel):
                    image_id = im.get("id")
                    break
            if image_id is not None:
                annos = [a for a in coco.get("annotations", []) if a.get("image_id") == image_id and a.get("iscrowd", 0) == 0]
                dd = {"file_name": img_path, "annotations": annos}

        aug_rgb = apply_synthetic_highlight(rgb, hcfg, dataset_dict=dd)
        aug_bgr = _rgb_to_bgr(aug_rgb)

        # 标注一下便于区分
        orig_vis = _put_label(bgr, "ORIGINAL")
        aug_vis = _put_label(aug_bgr, "HIGHLIGHT / OVEREXPOSURE")
        side = _side_by_side(orig_vis, aug_vis)

        stem = f"{i:03d}_" + os.path.splitext(os.path.basename(rel))[0]
        cv2.imwrite(os.path.join(orig_dir, f"{stem}.png"), orig_vis)
        cv2.imwrite(os.path.join(aug_dir, f"{stem}.png"), aug_vis)
        cv2.imwrite(os.path.join(side_dir, f"{stem}.png"), side)

        print(f"[{i+1}/{num}] saved: {stem}.png")

    print("完成。你可以优先看 side_by_side/ 目录。")


if __name__ == "__main__":
    main()


