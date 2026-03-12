#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线扩充 plug 数据集（仅做“像素级”增强，不做几何变换）：
- 读取 in_root 下的 COCO json + 图片
- 对每张图片生成 N 份增强副本（默认强光/过曝模拟），标注直接复用（bbox/segmentation 不变）
- 输出到 out_root，并写出新的 COCO json（默认文件名仍为 plug_train.json，方便直接训练）

为什么这样做是安全的？
- 只做光照/颜色变化，不改变几何结构，因此 segmentation/bbox 不需要变换。
- 适合你当前的“强光鲁棒性”扩充；如果要旋转/缩放/裁剪等几何增强，必须同步变换 polygon/RLE。

示例：
  PYTHONPATH=/home/user/sjw/Yolo_pointrend/detectron2:/home/user/sjw/Yolo_pointrend/detectron2/projects/PointRend \
  python -u expand_plug_dataset_photometric.py \
    --in-root /home/user/sjw/Yolo_pointrend/detectron2/plug_train1 \
    --json-file plug_train.json \
    --out-root /home/user/sjw/Yolo_pointrend/detectron2/plug_train1_aug \
    --copies 3 \
    --include-original \
    --prob 1.0 --focus object --clip --dilate 20 --feather 0 \
    --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from highlight_mapper import HighlightAugConfig, apply_synthetic_highlight


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand plug COCO dataset with pixel-only augmentations.")
    p.add_argument("--in-root", required=True, help="原始数据集目录（图片+COCO json）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO json 文件名（相对 in-root）")
    p.add_argument("--out-root", required=True, help="输出数据集目录（会创建）")
    p.add_argument("--out-json", default="", help="输出 COCO json 文件名（相对 out-root，默认沿用 --json-file）")
    p.add_argument("--copies", type=int, default=3, help="每张图生成多少份增强副本（>=0）")
    p.add_argument("--include-original", action="store_true", help="是否把原图也复制进 out-root（并写入 json）")
    p.add_argument("--overwrite", action="store_true", help="若 out-root 已存在则覆盖（危险）")

    # highlight params (pixel-only)
    p.add_argument("--prob", type=float, default=1.0, help="应用强光概率（扩充集建议 1.0）")
    p.add_argument("--focus", choices=["object", "any"], default="object", help="光斑中心：object=尽量落在目标上")
    p.add_argument("--bbox-shrink", type=float, default=0.15, help="focus=object 时 bbox 收缩比例")
    p.add_argument("--clip", action="store_true", help="把高亮限制在目标区域（mask/bbox）内")
    p.add_argument("--dilate", type=int, default=20, help="clip 时对目标 mask/bbox 膨胀像素")
    p.add_argument("--feather", type=int, default=0, help="clip 时目标边界软边像素")
    p.add_argument("--spots", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"))
    p.add_argument("--sigma", type=int, nargs=2, default=[30, 80], metavar=("MIN", "MAX"))
    p.add_argument("--intensity", type=int, nargs=2, default=[150, 255], metavar=("MIN", "MAX"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _build_ann_index(coco: dict) -> Tuple[Dict[int, List[dict]], Dict[str, dict]]:
    id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        iid = int(ann.get("image_id"))
        id_to_anns.setdefault(iid, []).append(ann)
    fn_to_img: Dict[str, dict] = {}
    for im in coco.get("images", []):
        fn = str(im.get("file_name", ""))
        if fn:
            fn_to_img[fn] = im
    return id_to_anns, fn_to_img


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    json_in = os.path.join(in_root, str(args.json_file))
    if not os.path.isfile(json_in):
        raise FileNotFoundError(f"COCO json not found: {json_in}")

    out_json_name = str(args.out_json).strip() or str(args.json_file)
    json_out = os.path.join(out_root, out_json_name)

    if os.path.exists(out_root):
        if not bool(args.overwrite):
            raise FileExistsError(f"out-root already exists: {out_root} (use --overwrite to replace)")
        shutil.rmtree(out_root)
    _ensure_dir(out_root)

    coco = _read_json(json_in)
    id_to_anns, fn_to_img = _build_ann_index(coco)

    # build highlight cfg
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

    copies = int(args.copies)
    if copies < 0:
        copies = 0

    # new coco skeleton
    coco_out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": [],
        "annotations": [],
    }

    next_img_id = 1
    next_ann_id = 1

    def add_image_and_anns(*, src_fn: str, dst_fn: str, orig_img_id: int, w: int, h: int) -> None:
        nonlocal next_img_id, next_ann_id
        img_entry = {
            "id": next_img_id,
            "file_name": dst_fn.replace("\\", "/"),
            "width": int(w),
            "height": int(h),
        }
        # copy over optional fields if present
        orig_im = fn_to_img.get(src_fn, {})
        for k in ["license", "coco_url", "flickr_url", "date_captured"]:
            if k in orig_im:
                img_entry[k] = orig_im[k]

        coco_out["images"].append(img_entry)

        for ann in id_to_anns.get(int(orig_img_id), []):
            aa = copy.deepcopy(ann)
            aa["id"] = next_ann_id
            aa["image_id"] = next_img_id
            coco_out["annotations"].append(aa)
            next_ann_id += 1

        next_img_id += 1

    n_written = 0
    for im in coco.get("images", []):
        src_rel = str(im.get("file_name", ""))
        if not src_rel:
            continue
        src_abs = os.path.join(in_root, src_rel)
        bgr = cv2.imread(src_abs, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[skip] failed to read: {src_abs}")
            continue

        h, w = bgr.shape[:2]
        orig_img_id = int(im.get("id"))

        # optionally include original
        if bool(args.include_original):
            dst_rel = src_rel
            dst_abs = os.path.join(out_root, dst_rel)
            _ensure_dir(os.path.dirname(dst_abs))
            if not os.path.isfile(dst_abs):
                cv2.imwrite(dst_abs, bgr)
            add_image_and_anns(src_fn=src_rel, dst_fn=dst_rel, orig_img_id=orig_img_id, w=w, h=h)
            n_written += 1

        # dataset_dict for focus on object
        dd = None
        if args.focus == "object":
            dd = {"file_name": src_abs, "annotations": id_to_anns.get(orig_img_id, [])}

        # create copies
        p = Path(src_rel)
        stem, suf = p.stem, p.suffix or ".png"
        parent = str(p.parent) if str(p.parent) != "." else ""
        for k in range(copies):
            # ensure different randomness per copy
            random.seed(int(args.seed) * 1000003 + orig_img_id * 997 + k * 131)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            aug_rgb = apply_synthetic_highlight(rgb, hcfg, dataset_dict=dd)
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

            new_name = f"{stem}__hl{k:02d}{suf}"
            dst_rel = os.path.join(parent, new_name) if parent else new_name
            dst_abs = os.path.join(out_root, dst_rel)
            _ensure_dir(os.path.dirname(dst_abs))
            cv2.imwrite(dst_abs, aug_bgr)
            add_image_and_anns(src_fn=src_rel, dst_fn=dst_rel, orig_img_id=orig_img_id, w=w, h=h)
            n_written += 1

    _ensure_dir(os.path.dirname(json_out))
    _write_json(json_out, coco_out)
    print(f"done. wrote images={n_written}, json={json_out}")


if __name__ == "__main__":
    main()


