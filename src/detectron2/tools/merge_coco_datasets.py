#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple flat COCO-style datasets into one folder.

Assumptions:
- Each source directory contains images in the root directory.
- Each source directory contains one COCO json file, e.g. charge_train.json.
- image["file_name"] points to a file under the same source directory.

Features:
- Copies images into the output directory.
- Merges categories by category name.
- Filters empty category names and "_background_".
- Rebuilds image/annotation ids to avoid collisions.
- Renames output images on filename collision.

Example:
  python3 src/detectron2/tools/merge_coco_datasets.py \
    --src /home/user/sjw/workspace/datasets/guowang/charge_test_0310 \
    --src /home/user/sjw/workspace/datasets/guowang/charge_test_0314 \
    --json-name charge_train.json \
    --out-dir /home/user/sjw/workspace/datasets/guowang/charge_test_merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple flat COCO datasets into one directory.")
    p.add_argument(
        "--src",
        action="append",
        required=True,
        help="Source dataset directory. Pass this argument multiple times.",
    )
    p.add_argument("--json-name", default="charge_train.json", help="COCO json filename in each source directory.")
    p.add_argument("--out-dir", required=True, help="Merged output directory.")
    p.add_argument("--out-json", default="", help="Merged output json filename. Default: same as --json-name")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it already exists.")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _valid_category_name(name: object) -> bool:
    s = str(name or "").strip()
    return bool(s) and s != "_background_"


def _build_unique_filename(name: str, used_names: set[str], prefix: str) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = f"{prefix}_{stem}{suffix}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    idx = 2
    while True:
        candidate = f"{prefix}_{stem}_{idx}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def merge_datasets(src_dirs: List[Path], json_name: str, out_dir: Path, out_json_name: str) -> Tuple[int, int]:
    merged_images: List[dict] = []
    merged_annotations: List[dict] = []
    merged_categories: List[dict] = []
    licenses: List[dict] = []
    info: dict = {}

    category_name_to_new_id: Dict[str, int] = {}
    used_output_names: set[str] = set()
    next_image_id = 1
    next_ann_id = 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for src_dir in src_dirs:
        json_path = src_dir / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

        coco = _load_json(json_path)
        if not info:
            info = coco.get("info", {})
        if not licenses:
            licenses = coco.get("licenses", [])

        src_categories = coco.get("categories", [])
        src_cat_id_to_name = {
            int(cat["id"]): str(cat.get("name", "")).strip()
            for cat in src_categories
            if "id" in cat
        }

        src_cat_id_to_new_id: Dict[int, int] = {}
        for cat in src_categories:
            if "id" not in cat:
                continue
            cat_name = str(cat.get("name", "")).strip()
            if not _valid_category_name(cat_name):
                continue
            if cat_name not in category_name_to_new_id:
                new_id = len(category_name_to_new_id) + 1
                category_name_to_new_id[cat_name] = new_id
                merged_categories.append(
                    {
                        "id": new_id,
                        "name": cat_name,
                        "supercategory": str(cat.get("supercategory", "")),
                    }
                )
            src_cat_id_to_new_id[int(cat["id"])] = category_name_to_new_id[cat_name]

        image_id_map: Dict[int, int] = {}
        src_name = src_dir.name

        for image in coco.get("images", []):
            old_image_id = int(image["id"])
            src_file_name = str(image["file_name"])
            src_image_path = src_dir / src_file_name
            if not src_image_path.exists():
                raise FileNotFoundError(f"Image referenced by json not found: {src_image_path}")

            output_file_name = _build_unique_filename(Path(src_file_name).name, used_output_names, src_name)
            shutil.copy2(src_image_path, out_dir / output_file_name)

            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = output_file_name
            merged_images.append(new_image)
            image_id_map[old_image_id] = next_image_id
            next_image_id += 1

        for ann in coco.get("annotations", []):
            old_cat_id = int(ann.get("category_id", -1))
            if old_cat_id not in src_cat_id_to_new_id:
                continue

            old_image_id = int(ann["image_id"])
            if old_image_id not in image_id_map:
                continue

            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[old_image_id]
            new_ann["category_id"] = src_cat_id_to_new_id[old_cat_id]
            merged_annotations.append(new_ann)
            next_ann_id += 1

        valid_cat_names = sorted({src_cat_id_to_name[cid] for cid in src_cat_id_to_new_id})
        print(
            f"[merge] {src_dir} -> images={len(coco.get('images', []))} "
            f"annotations={len(coco.get('annotations', []))} valid_categories={valid_cat_names}"
        )

    merged = {
        "info": info,
        "licenses": licenses,
        "categories": merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    out_json_path = out_dir / out_json_name
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return len(merged_images), len(merged_annotations)


def main() -> None:
    args = parse_args()
    src_dirs = [Path(p).expanduser().resolve() for p in args.src]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_json_name = str(args.out_json).strip() or str(args.json_name).strip()

    for src_dir in src_dirs:
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Source directory not found: {src_dir}")

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir} (use --overwrite)")
        shutil.rmtree(out_dir)

    num_images, num_annotations = merge_datasets(
        src_dirs=src_dirs,
        json_name=str(args.json_name),
        out_dir=out_dir,
        out_json_name=out_json_name,
    )
    print(f"[done] out_dir={out_dir}")
    print(f"[done] out_json={out_dir / out_json_name}")
    print(f"[done] images={num_images} annotations={num_annotations}")


if __name__ == "__main__":
    main()
