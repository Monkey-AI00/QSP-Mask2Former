#!/usr/bin/env python3
"""
批量从 JSON 标注中删除指定标签。

支持两种常见格式：
1) LabelMe: 删除 data["shapes"] 中 label == target_label 的对象
2) COCO: 删除 categories 中 name == target_label 的类别，并同步删除对应 annotations

对 COCO 还会自动将剩余类别重新映射为从 1 开始的连续 id，便于后续训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 根节点不是对象: {path}")
    return data


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _is_labelme(data: dict[str, Any]) -> bool:
    return isinstance(data.get("shapes"), list)


def _is_coco(data: dict[str, Any]) -> bool:
    return all(isinstance(data.get(k), list) for k in ("images", "annotations", "categories"))


def _process_labelme(data: dict[str, Any], target_label: str) -> tuple[dict[str, Any], int]:
    shapes = data.get("shapes", [])
    kept = [shape for shape in shapes if str(shape.get("label", "")).strip() != target_label]
    removed = len(shapes) - len(kept)
    data["shapes"] = kept
    return data, removed


def _process_coco(data: dict[str, Any], target_label: str) -> tuple[dict[str, Any], int]:
    categories = data.get("categories", [])
    target_cat_ids = {
        int(cat["id"])
        for cat in categories
        if str(cat.get("name", "")).strip() == target_label
    }

    if not target_cat_ids:
        return data, 0

    kept_categories = [
        cat for cat in categories if int(cat.get("id", -1)) not in target_cat_ids
    ]
    kept_annotations = [
        ann for ann in data.get("annotations", [])
        if int(ann.get("category_id", -1)) not in target_cat_ids
    ]
    removed = len(data.get("annotations", [])) - len(kept_annotations)

    # 重新映射剩余类别 id 为连续 1..N，便于后续训练/检查
    old_to_new: dict[int, int] = {}
    remapped_categories = []
    for new_id, cat in enumerate(sorted(kept_categories, key=lambda x: int(x["id"])), start=1):
        old_id = int(cat["id"])
        old_to_new[old_id] = new_id
        cat_new = dict(cat)
        cat_new["id"] = new_id
        remapped_categories.append(cat_new)

    remapped_annotations = []
    for ann in kept_annotations:
        ann_new = dict(ann)
        ann_new["category_id"] = old_to_new[int(ann["category_id"])]
        remapped_annotations.append(ann_new)

    data["categories"] = remapped_categories
    data["annotations"] = remapped_annotations
    return data, removed


def _iter_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove a target label from LabelMe or COCO JSON annotations.")
    parser.add_argument("input_path", help="单个 JSON 文件，或包含 JSON 的目录")
    parser.add_argument("--label", default="plug", help="要删除的标签名称，默认 plug")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="原地修改输入 JSON。默认会输出到 --output-dir。",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录。目录输入时会保留相对路径；单文件输入时输出到该目录。",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if not args.in_place and not str(args.output_dir).strip():
        raise ValueError("未指定输出方式：请使用 --in-place，或提供 --output-dir")

    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    json_files = _iter_json_files(input_path)
    if not json_files:
        raise RuntimeError(f"未找到任何 JSON 文件: {input_path}")

    total_files_changed = 0
    total_labels_removed = 0

    for json_path in json_files:
        data = _load_json(json_path)

        if _is_labelme(data):
            data_new, removed = _process_labelme(data, str(args.label).strip())
            fmt = "LabelMe"
        elif _is_coco(data):
            data_new, removed = _process_coco(data, str(args.label).strip())
            fmt = "COCO"
        else:
            print(f"[skip] {json_path} 不是已支持的 LabelMe/COCO JSON")
            continue

        if args.in_place:
            out_path = json_path
        else:
            if input_path.is_file():
                out_path = output_dir / json_path.name  # type: ignore[operator]
            else:
                rel = json_path.relative_to(input_path)
                out_path = output_dir / rel  # type: ignore[operator]
                out_path.parent.mkdir(parents=True, exist_ok=True)

        _save_json(out_path, data_new)

        if removed > 0:
            total_files_changed += 1
            total_labels_removed += removed
            print(f"[ok] {fmt}: {json_path} -> {out_path} | removed={removed}")
        else:
            print(f"[ok] {fmt}: {json_path} -> {out_path} | removed=0")

    print("\n处理完成")
    print(f"修改文件数: {total_files_changed}")
    print(f"删除的 '{args.label}' 标注总数: {total_labels_removed}")


if __name__ == "__main__":
    main()
