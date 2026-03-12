#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成“强曝光评测集”（离线）：
- 读取原始数据集目录（图片 + COCO json）
- 对每张图片应用强光/过曝模拟（建议 prob=1.0）
- 输出到新目录 out_root（保持 file_name 不变），json 直接复制即可

这样做的好处：
1) 不需要改 detectron2 的 test loader / evaluator
2) 标注完全不变（只改像素），可直接对比 AP/可视化

示例：
  PYTHONPATH=... conda run -n pointrend python make_highlight_dataset.py \
    --in-root /home/users1/sjw/cursor/workspace/datasets/plug_train1 \
    --json-file plug_train.json \
    --out-root /home/users1/sjw/cursor/workspace/datasets/plug_train1_highlight \
    --prob 1.0 --focus object --clip --dilate 20
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List

import cv2
import numpy as np

from highlight_mapper import HighlightAugConfig, apply_synthetic_highlight


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make an offline highlighted dataset (pixel-only).")
    p.add_argument("--in-root", required=True, help="原始数据集目录（图片+COCO json）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO json 文件名（相对 in-root）")
    p.add_argument("--out-root", required=True, help="输出数据集目录（会创建）")
    p.add_argument("--prob", type=float, default=1.0, help="应用强光概率（评测集建议 1.0）")
    p.add_argument("--focus", choices=["object", "any"], default="object", help="光斑中心：object=尽量落在目标上")
    p.add_argument("--bbox-shrink", type=float, default=0.15, help="focus=object 时 bbox 收缩比例")
    p.add_argument("--clip", action="store_true", help="把高亮限制在目标区域（mask/bbox）内")
    p.add_argument("--dilate", type=int, default=20, help="clip 时对目标 mask/bbox 膨胀像素")
    p.add_argument("--feather", type=int, default=0, help="clip 时目标边界软边像素（0=硬边；>0=平滑衰减）")
    p.add_argument("--spots", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"))
    p.add_argument("--sigma", type=int, nargs=2, default=[30, 80], metavar=("MIN", "MAX"))
    p.add_argument("--intensity", type=int, nargs=2, default=[150, 255], metavar=("MIN", "MAX"))
    p.add_argument("--overwrite", action="store_true", help="若 out-root 存在则覆盖（危险）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    json_path = os.path.join(in_root, args.json_file)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")

    if os.path.exists(out_root):
        if not args.overwrite:
            raise FileExistsError(f"out-root already exists: {out_root} (use --overwrite to replace)")
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    import json

    with open(json_path, "r") as f:
        coco = json.load(f)

    # 建索引：file_name -> image_id, image_id -> annotations
    fn_to_id: Dict[str, int] = {}
    for im in coco.get("images", []):
        fn = str(im.get("file_name", ""))
        if fn:
            fn_to_id[fn] = int(im.get("id"))

    id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        iid = int(ann.get("image_id"))
        id_to_anns.setdefault(iid, []).append(ann)

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

    print(f"in_root : {in_root}")
    print(f"out_root: {out_root}")
    print(f"params  : prob={hcfg.prob}, focus={args.focus}, clip={hcfg.clip_to_object}, dilate={hcfg.object_mask_dilate}")

    n = 0
    for im in coco.get("images", []):
        fn = str(im.get("file_name", ""))
        if not fn:
            continue
        src = os.path.join(in_root, fn)
        dst = os.path.join(out_root, fn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[skip] failed to read: {src}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        dd = None
        if args.focus == "object":
            iid = fn_to_id.get(fn)
            if iid is not None:
                dd = {"file_name": src, "annotations": id_to_anns.get(iid, [])}

        aug_rgb = apply_synthetic_highlight(rgb, hcfg, dataset_dict=dd)
        aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst, aug_bgr)
        n += 1

    # 复制 json（不改 file_name）
    shutil.copy2(json_path, os.path.join(out_root, args.json_file))
    print(f"done. wrote {n} images + copied {args.json_file}")


if __name__ == "__main__":
    main()


