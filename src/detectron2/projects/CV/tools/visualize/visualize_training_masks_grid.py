#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成训练中间阶段的 mask 可视化网格图（右图风格）。

功能：
- 从 COCO 训练集抽样若干图片
- 对每张图生成多组“训练态”外观增强（亮度/对比度/色偏/模糊）
- 将实例 mask 以半透明彩色叠加到图像上
- 输出整张网格图与每个 tile

示例：
  python -u visualize_training_masks_grid.py \
    --dataset-root /home/users1/sjw/cursor/workspace/datasets/gangkou/plug_train_merged_0429_train \
    --json-file plug_train.json \
    --out-dir /home/users1/sjw/cursor/workspace/outputs/gangkou/output/training_masks_vis \
    --rows 5 --cols 7 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize training-stage masks as a tiled grid.")
    p.add_argument("--dataset-root", required=True, help="数据集根目录（图片与 COCO json 同层）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO 标注文件名（相对 dataset-root）")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--rows", type=int, default=5, help="网格行数（抽样图片数）")
    p.add_argument("--cols", type=int, default=7, help="网格列数（每张图的增强版本数）")
    p.add_argument("--tile-width", type=int, default=220, help="每个 tile 宽度")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--mask-only", action="store_true", help="仅输出黑底白 mask（不叠加原图）")
    p.add_argument("--show-tile-label", action="store_true", help="显示每个小图左上角标签（默认关闭）")
    p.add_argument(
        "--image-names",
        default="",
        help="指定要可视化的图片文件名，逗号分隔（按 COCO 的 file_name 匹配）",
    )
    p.add_argument(
        "--image-list-file",
        default="",
        help="指定一个 txt 文件，每行一个 file_name，用于精确可视化指定图片",
    )
    return p.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_image_name_set(args: argparse.Namespace) -> set:
    """
    汇总用户指定的 file_name 列表：
    - --image-names: 逗号分隔
    - --image-list-file: 每行一个
    """
    out = set()

    csv_names = str(getattr(args, "image_names", "")).strip()
    if csv_names:
        for x in csv_names.split(","):
            name = str(x).strip()
            if name:
                out.add(name)

    list_file = str(getattr(args, "image_list_file", "")).strip()
    if list_file:
        p = os.path.abspath(list_file)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"image list file not found: {p}")
        with open(p, "r") as f:
            for line in f:
                name = str(line).strip()
                if not name or name.startswith("#"):
                    continue
                out.add(name)
    return out


def _ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    out = np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg, list) and len(seg) > 0:
        # polygon 格式：[[x1,y1,x2,y2,...], ...]
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(out, [pts.astype(np.int32)], color=1)
        if np.any(out > 0):
            return out

    # 若不是 polygon（如 RLE）或 polygon 异常，则回退 bbox，保证脚本可运行
    bbox = ann.get("bbox", None)
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, bw, bh = [float(v) for v in bbox[:4]]
        x1 = int(max(0, min(w - 1, round(x))))
        y1 = int(max(0, min(h - 1, round(y))))
        x2 = int(max(0, min(w - 1, round(x + bw - 1))))
        y2 = int(max(0, min(h - 1, round(y + bh - 1))))
        if x2 >= x1 and y2 >= y1:
            out[y1 : y2 + 1, x1 : x2 + 1] = 1
    return out


def _random_train_like_aug(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img_bgr.astype(np.float32)

    # 亮度 / 对比度
    alpha = rng.uniform(0.78, 1.28)  # contrast
    beta = rng.uniform(-20, 28)      # brightness shift
    out = out * alpha + beta

    # 轻度色偏（HSV）
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out_u8, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-8, 8))) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + int(rng.uniform(-35, 40)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + int(rng.uniform(-15, 22)), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 随机轻模糊，模拟成像扰动
    if rng.random() < 0.55:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=rng.uniform(0.6, 1.4))

    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_mask(img_bgr: np.ndarray, mask01: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.42) -> np.ndarray:
    out = img_bgr.copy()
    m = (mask01 > 0)
    if not np.any(m):
        return out
    color = np.array(color_bgr, dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours((mask01 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color_bgr, 2, lineType=cv2.LINE_AA)
    return out


def _resize_to_width(img_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    scale = float(width) / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (width, nh), interpolation=cv2.INTER_AREA)


def _title(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    bar_h = min(20, max(16, out.shape[0] // 10))
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (6, int(bar_h * 0.78)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _mask_only_bgr(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8) * 255
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def _stack_row(imgs: List[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] < h:
            pad = h - im.shape[0]
            im = cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(im)
    return np.concatenate(padded, axis=1)


def _stack_grid(rows: List[np.ndarray]) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < w:
            pad = w - r.shape[1]
            r = cv2.copyMakeBorder(r, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(r)
    return np.concatenate(padded, axis=0)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset_root = os.path.abspath(args.dataset_root)
    json_path = os.path.join(dataset_root, args.json_file)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")

    with open(json_path, "r") as f:
        coco = json.load(f)

    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        image_id = int(ann.get("image_id"))
        anns_by_image.setdefault(image_id, []).append(ann)

    images = []
    for im in coco.get("images", []):
        image_id = int(im.get("id"))
        if image_id in anns_by_image and len(anns_by_image[image_id]) > 0:
            images.append(im)
    if not images:
        raise RuntimeError("No annotated images found in COCO json.")

    specified_names = _parse_image_name_set(args)
    if specified_names:
        selected_images = [im for im in images if str(im.get("file_name", "")) in specified_names]
        if not selected_images:
            raise ValueError(
                "未匹配到任何指定图片。请检查 --image-names / --image-list-file 与 COCO 的 file_name 是否一致。"
            )
        missing = sorted(specified_names - {str(im.get("file_name", "")) for im in selected_images})
        if missing:
            print(f"[warn] 以下指定图片未在 COCO 中找到: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        images = selected_images
    else:
        random.shuffle(images)

    rows = min(int(args.rows), len(images))
    cols = max(1, int(args.cols))
    picks = images[:rows]

    out_dir = os.path.abspath(args.out_dir)
    tile_dir = os.path.join(out_dir, "tiles")
    _ensure_dir(out_dir)
    _ensure_dir(tile_dir)

    palette = [
        (70, 180, 255), (90, 230, 140), (220, 120, 255), (120, 220, 255),
        (140, 180, 120), (255, 150, 90), (200, 110, 250), (90, 240, 210),
    ]

    grid_rows: List[np.ndarray] = []
    for r, im in enumerate(picks):
        file_name = str(im.get("file_name"))
        img_path = os.path.join(dataset_root, file_name)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        # 合并该图所有实例为 1 张 mask（单类 plug 数据足够）
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_by_image.get(int(im.get("id")), []):
            mask = np.maximum(mask, _ann_to_mask(ann, h, w))

        row_tiles: List[np.ndarray] = []
        for c in range(cols):
            rng = random.Random((int(args.seed) + 1) * 100000 + r * 1000 + c)
            if bool(args.mask_only):
                vis = _mask_only_bgr(mask)
                tag = f"mask{c}"
            else:
                if c == 0:
                    aug = bgr
                    tag = "orig"
                else:
                    aug = _random_train_like_aug(bgr, rng)
                    tag = f"aug{c}"
                color = palette[c % len(palette)]
                vis = _overlay_mask(aug, mask, color_bgr=color, alpha=0.42)
            if bool(getattr(args, "show_tile_label", False)):
                vis = _title(vis, f"row{r+1} {tag}")
            vis = _resize_to_width(vis, int(args.tile_width))
            row_tiles.append(vis)

            tile_name = f"r{r+1:02d}_c{c+1:02d}_{os.path.splitext(os.path.basename(file_name))[0]}.png"
            cv2.imwrite(os.path.join(tile_dir, tile_name), vis)

        if row_tiles:
            grid_rows.append(_stack_row(row_tiles))

    if not grid_rows:
        raise RuntimeError("No rows generated. Please check dataset image paths in json.")

    grid = _stack_grid(grid_rows)
    out_path = os.path.join(out_dir, "training_masks_grid.png")
    cv2.imwrite(out_path, grid)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

