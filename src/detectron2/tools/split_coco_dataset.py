#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a flat COCO-style dataset into train/val directories.

Assumptions:
- Source directory contains images in the root directory.
- Source directory contains one COCO json file.
- image["file_name"] points to a file under the same source directory.

Example:
  python3 src/detectron2/tools/split_coco_dataset.py \
    --src-dir /home/user/sjw/workspace/datasets/guowang/charge_test_merged_0310_0314 \
    --json-name charge_train.json \
    --train-ratio 0.8 \
    --out-train-dir /home/user/sjw/workspace/datasets/guowang/charge_test_merged_0310_0314_train \
    --out-val-dir /home/user/sjw/workspace/datasets/guowang/charge_test_merged_0310_0314_val \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a flat COCO dataset into train/val directories.")
    p.add_argument("--src-dir", required=True, help="Source dataset directory.")
    p.add_argument("--json-name", default="charge_train.json", help="COCO json filename in source directory.")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio in (0, 1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for image split.")
    p.add_argument("--out-train-dir", required=True, help="Output train dataset directory.")
    p.add_argument("--out-val-dir", required=True, help="Output val dataset directory.")
    p.add_argument("--out-train-json", default="", help="Train json filename. Default: same as --json-name")
    p.add_argument("--out-val-json", default="", help="Val json filename. Default: same as --json-name")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directories if they exist.")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_and_reindex_subset(
    *,
    images_subset: List[dict],
    image_id_to_anns: Dict[int, List[dict]],
    src_dir: Path,
    out_dir: Path,
    categories: List[dict],
    licenses: List[dict],
    info: dict,
    out_json_name: str,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    new_images: List[dict] = []
    new_annotations: List[dict] = []
    next_image_id = 1
    next_ann_id = 1

    for image in images_subset:
        old_image_id = int(image["id"])
        file_name = str(image["file_name"])
        src_image_path = src_dir / file_name
        if not src_image_path.exists():
            raise FileNotFoundError(f"Image referenced by json not found: {src_image_path}")

        shutil.copy2(src_image_path, out_dir / file_name)

        new_image = dict(image)
        new_image["id"] = next_image_id
        new_images.append(new_image)

        for ann in image_id_to_anns.get(old_image_id, []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = next_image_id
            new_annotations.append(new_ann)
            next_ann_id += 1

        next_image_id += 1

    out_json = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": new_images,
        "annotations": new_annotations,
    }
    out_json_path = out_dir / out_json_name
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    return len(new_images), len(new_annotations)


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    src_json_path = src_dir / str(args.json_name)
    out_train_dir = Path(args.out_train_dir).expanduser().resolve()
    out_val_dir = Path(args.out_val_dir).expanduser().resolve()
    out_train_json = str(args.out_train_json).strip() or str(args.json_name)
    out_val_json = str(args.out_val_json).strip() or str(args.json_name)

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source dataset directory not found: {src_dir}")
    if not src_json_path.exists():
        raise FileNotFoundError(f"Source json not found: {src_json_path}")
    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError(f"--train-ratio must be in (0, 1), got {args.train_ratio}")

    for out_dir in (out_train_dir, out_val_dir):
        if out_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output directory already exists: {out_dir} (use --overwrite)")
            shutil.rmtree(out_dir)

    coco = _load_json(src_json_path)
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    categories = list(coco.get("categories", []))
    licenses = list(coco.get("licenses", []))
    info = dict(coco.get("info", {}))

    image_id_to_anns: Dict[int, List[dict]] = {}
    for ann in annotations:
        image_id_to_anns.setdefault(int(ann["image_id"]), []).append(ann)

    rng = random.Random(int(args.seed))
    images_shuffled = list(images)
    rng.shuffle(images_shuffled)

    num_images = len(images_shuffled)
    num_train = int(round(num_images * float(args.train_ratio)))
    num_train = max(1, min(num_images - 1, num_train))

    train_images = images_shuffled[:num_train]
    val_images = images_shuffled[num_train:]

    train_num_images, train_num_annotations = _copy_and_reindex_subset(
        images_subset=train_images,
        image_id_to_anns=image_id_to_anns,
        src_dir=src_dir,
        out_dir=out_train_dir,
        categories=categories,
        licenses=licenses,
        info=info,
        out_json_name=out_train_json,
    )
    val_num_images, val_num_annotations = _copy_and_reindex_subset(
        images_subset=val_images,
        image_id_to_anns=image_id_to_anns,
        src_dir=src_dir,
        out_dir=out_val_dir,
        categories=categories,
        licenses=licenses,
        info=info,
        out_json_name=out_val_json,
    )

    print(f"[done] src={src_dir}")
    print(f"[done] train_dir={out_train_dir} images={train_num_images} annotations={train_num_annotations}")
    print(f"[done] val_dir={out_val_dir} images={val_num_images} annotations={val_num_annotations}")
    print(f"[done] train_json={out_train_dir / out_train_json}")
    print(f"[done] val_json={out_val_dir / out_val_json}")


if __name__ == "__main__":
    main()
