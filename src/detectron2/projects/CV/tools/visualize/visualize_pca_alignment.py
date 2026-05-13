#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA 方向对齐可视化（对应 generate_mean_shape.py 的对齐逻辑）：
1) PCA 主轴
2) 旋转到竖直
3) 180° 头尾翻转检查
4) 对齐轮廓叠加

默认数据可直接使用：
  datasets/gangkou/plug_train_merged_0429_train/plug_train.json

指定图片示例：
  python visualize_pca_alignment.py \
    --dataset-root /home/users1/sjw/cursor/workspace/datasets/gangkou/plug_train_merged_0429_train \
    --json-file plug_train.json \
    --image-names "a.png,b.png" \
    --out /home/users1/sjw/cursor/workspace/outputs/gangkou/output/pca_alignment_vis/verify_selected.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize PCA orientation alignment workflow.")
    p.add_argument("--dataset-root", required=True, help="数据集根目录（图片+COCO json）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO 标注文件名（相对 dataset-root）")
    p.add_argument("--ann-ids", nargs="*", type=int, default=None, help="只处理指定 annotation id 列表")
    p.add_argument("--image-ids", nargs="*", type=int, default=None, help="只处理指定 image id 列表")
    p.add_argument("--image-names", default="", help="只处理指定 file_name，逗号分隔")
    p.add_argument("--image-list-file", default="", help="txt 文件，每行一个 file_name")
    p.add_argument("--demo-index", type=int, default=0, help="从过滤后候选中选择第几个样本作为步骤演示")
    p.add_argument("--out", required=True, help="输出 PNG 路径")
    p.add_argument("--pad", type=int, default=10, help="bbox 裁剪 padding")
    p.add_argument("--temp-size", nargs=2, type=int, default=[128, 128], metavar=("W", "H"))
    p.add_argument("--overlay-k", type=int, default=10, help="轮廓叠加样本数")
    p.add_argument(
        "--overlay-mode",
        choices=["compact", "diverse"],
        default="compact",
        help="叠加样本策略：compact 更规整，diverse 保留差异",
    )
    p.add_argument("--overlay-size", nargs=2, type=int, default=[200, 200], metavar=("W", "H"))
    p.add_argument("--target-fill", type=float, default=0.75, help="后规范化时目标占画布比例")
    p.add_argument("--dashed-overlay", action="store_true", default=True, help="最终叠加使用虚线轮廓")
    p.add_argument("--dash-len", type=int, default=8, help="虚线段长度")
    p.add_argument("--dash-gap", type=int, default=6, help="虚线段间隔")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    out = np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg, list) and len(seg) > 0:
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(out, [pts.astype(np.int32)], color=1)
        if int(out.sum()) > 0:
            return out

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


def safe_crop_with_pad(mask_full: np.ndarray, bbox_xywh: Sequence[float], pad: int) -> Optional[np.ndarray]:
    x, y, w, h = [int(round(v)) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return None
    H, W = mask_full.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    c = mask_full[y1:y2, x1:x2]
    if c.size == 0:
        return None
    return c


def resize_mask(mask: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    w, h = int(size_wh[0]), int(size_wh[1])
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def get_orientation(mask: np.ndarray) -> float:
    y, x = np.nonzero(mask > 0)
    if len(x) == 0:
        return 0.0
    data = np.vstack([x, y]).T.astype(np.float32)
    mean = np.mean(data, axis=0, keepdims=True)
    centered = data - mean
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return 0.0
    evals, evecs = np.linalg.eig(cov)
    order = np.argsort(evals)[::-1]
    major_axis = evecs[:, order[0]]
    return float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(
        mask.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rot


def rotate_180(mask: np.ndarray) -> np.ndarray:
    return cv2.rotate(mask, cv2.ROTATE_180)


def align_mask(mask: np.ndarray, prefer_bottom_heavy: bool = True) -> np.ndarray:
    angle = get_orientation(mask)
    rotate_angle = -angle + 90.0
    rotated = rotate_mask(mask, rotate_angle)
    h = rotated.shape[0]
    top_mass = float(np.sum(rotated[: h // 2, :]))
    bottom_mass = float(np.sum(rotated[h // 2 :, :]))
    if prefer_bottom_heavy:
        if top_mass > bottom_mass:
            rotated = rotate_180(rotated)
    else:
        if bottom_mass > top_mass:
            rotated = rotate_180(rotated)
    return rotated


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    m = ((mask > 0.3).astype(np.uint8) * 255)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def post_normalize_aligned_mask(
    mask: np.ndarray,
    size_wh: Tuple[int, int],
    target_fill: float = 0.75,
    thr: float = 0.35,
) -> np.ndarray:
    """
    仅用于可视化：对齐后二次归一化，提升叠加规整度。
    - 重新按前景 bbox 裁剪
    - 依据目标占比缩放
    - 重心对齐到画布中心
    """
    w, h = int(size_wh[0]), int(size_wh[1])
    canvas = np.zeros((h, w), dtype=np.float32)
    bin_m = (mask > float(thr)).astype(np.uint8)
    bb = _mask_bbox(bin_m)
    if bb is None:
        return canvas
    x1, y1, x2, y2 = bb
    crop = bin_m[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return canvas

    area = float(crop.sum())
    if area <= 1.0:
        return canvas
    desired = max(0.05, min(0.95, float(target_fill))) * float(w * h)
    scale = float(np.sqrt(desired / area))
    ch, cw = crop.shape[:2]
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))

    # 防止超界
    max_w = max(1, int(round(w * 0.96)))
    max_h = max(1, int(round(h * 0.96)))
    if nw > max_w or nh > max_h:
        s2 = min(float(max_w) / float(max(1, nw)), float(max_h) / float(max(1, nh)))
        nw = max(1, int(round(nw * s2)))
        nh = max(1, int(round(nh * s2)))

    rs = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
    rs_bin = (rs > 0.5).astype(np.uint8)
    ys, xs = np.where(rs_bin > 0)
    if len(xs) == 0:
        return canvas
    cx_src = float(xs.mean())
    cy_src = float(ys.mean())
    cx_dst = (w - 1) / 2.0
    cy_dst = (h - 1) / 2.0
    ox = int(round(cx_dst - cx_src))
    oy = int(round(cy_dst - cy_src))

    x_from = max(0, -ox)
    y_from = max(0, -oy)
    x_to = min(nw, w - ox)
    y_to = min(nh, h - oy)
    if x_to <= x_from or y_to <= y_from:
        return canvas
    dx1 = ox + x_from
    dy1 = oy + y_from
    dx2 = dx1 + (x_to - x_from)
    dy2 = dy1 + (y_to - y_from)
    canvas[dy1:dy2, dx1:dx2] = rs_bin[y_from:y_to, x_from:x_to].astype(np.float32)
    return canvas


def _draw_dashed_contour(
    canvas: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int,
    dash_gap: int,
) -> None:
    pts = contour.reshape(-1, 2)
    if pts.shape[0] < 2:
        return
    cycle = max(1, int(dash_len) + int(dash_gap))
    for i in range(pts.shape[0]):
        p0 = pts[i]
        p1 = pts[(i + 1) % pts.shape[0]]
        seg = p1.astype(np.float32) - p0.astype(np.float32)
        length = float(np.hypot(seg[0], seg[1]))
        if length <= 1e-6:
            continue
        direction = seg / length
        t = 0.0
        while t < length:
            t2 = min(length, t + float(max(1, dash_len)))
            a = p0.astype(np.float32) + direction * t
            b = p0.astype(np.float32) + direction * t2
            cv2.line(
                canvas,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            t += float(cycle)


def _overlay_aligned_contours(
    masks: Sequence[np.ndarray],
    size_wh: Tuple[int, int],
    dashed_overlay: bool = True,
    dash_len: int = 8,
    dash_gap: int = 6,
) -> np.ndarray:
    w, h = int(size_wh[0]), int(size_wh[1])
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    colors = [
        (255, 0, 0),
        (0, 170, 255),
        (0, 200, 0),
        (180, 0, 255),
        (255, 120, 0),
        (80, 200, 200),
        (220, 80, 120),
        (140, 140, 0),
    ]
    for i, m in enumerate(masks):
        mm = resize_mask(m, (w, h))
        bin_m = (mm > 0.35).astype(np.uint8)
        cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = colors[i % len(colors)]
        if dashed_overlay:
            for contour in cnts:
                _draw_dashed_contour(canvas, contour, c, thickness=2, dash_len=dash_len, dash_gap=dash_gap)
        else:
            cv2.drawContours(canvas, cnts, -1, c, 2, lineType=cv2.LINE_AA)
    return canvas


def _resolve_out_paths(out_arg: str) -> Dict[str, str]:
    """
    输入 --out（可为 .png 或目录），返回 5 张图的输出路径。
    """
    out_abs = os.path.abspath(out_arg)
    if out_abs.lower().endswith(".png"):
        out_dir = os.path.dirname(out_abs)
        stem = os.path.splitext(os.path.basename(out_abs))[0]
    else:
        out_dir = out_abs
        stem = "pca_alignment"
    os.makedirs(out_dir, exist_ok=True)
    return {
        "step1_raw": os.path.join(out_dir, f"{stem}_step1_raw.png"),
        "step2_rotated_vertical": os.path.join(out_dir, f"{stem}_step2_rotated_vertical.png"),
        "step3_rotated_180": os.path.join(out_dir, f"{stem}_step3_rotated_180.png"),
        "step4_chosen_direction": os.path.join(out_dir, f"{stem}_step4_chosen_direction.png"),
        "final_overlay": os.path.join(out_dir, f"{stem}_final_aligned_overlay.png"),
    }


def _parse_name_set(csv_names: str, list_file: str) -> set:
    out = set()
    if str(csv_names).strip():
        for x in str(csv_names).split(","):
            name = str(x).strip()
            if name:
                out.add(name)
    if str(list_file).strip():
        p = Path(str(list_file)).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"image list file not found: {p}")
        for line in p.read_text().splitlines():
            name = str(line).strip()
            if not name or name.startswith("#"):
                continue
            out.add(name)
    return out


def _features(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    area = float((mask > 0.35).sum()) / float(h * w + 1e-6)
    angle = get_orientation(mask)
    ys, xs = np.where(mask > 0.35)
    if len(xs) == 0:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    aspect = bw / max(1.0, bh)
    return np.asarray([area, np.sin(np.deg2rad(angle)), aspect], dtype=np.float32)


def _select_diverse(masks: Sequence[np.ndarray], k: int) -> List[np.ndarray]:
    if not masks:
        return []
    k = min(max(1, int(k)), len(masks))
    X = np.stack([_features(m) for m in masks], axis=0)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    sel = [int(np.argmax(X[:, 0]))]  # 面积最大的先选
    dist = np.linalg.norm(X - X[sel[0] : sel[0] + 1], axis=1)
    dist[sel[0]] = -1
    while len(sel) < k:
        idx = int(np.argmax(dist))
        if dist[idx] < 0:
            break
        sel.append(idx)
        d2 = np.linalg.norm(X - X[idx : idx + 1], axis=1)
        dist = np.minimum(dist, d2)
        for s in sel:
            dist[s] = -1
    return [masks[i] for i in sel]


def _select_compact(masks: Sequence[np.ndarray], k: int) -> List[np.ndarray]:
    if not masks:
        return []
    k = min(max(1, int(k)), len(masks))
    X = np.stack([m.reshape(-1).astype(np.float32) for m in masks], axis=0)
    mean = X.mean(axis=0, keepdims=True)
    d = ((X - mean) ** 2).mean(axis=1)
    idx = np.argsort(d)[:k]
    return [masks[int(i)] for i in idx]


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    dataset_root = os.path.abspath(args.dataset_root)
    ann_path = os.path.join(dataset_root, args.json_file)
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"COCO json not found: {ann_path}")

    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = {int(im["id"]): im for im in coco.get("images", [])}
    ann_id_set = set(int(x) for x in args.ann_ids) if args.ann_ids else None
    image_id_set = set(int(x) for x in args.image_ids) if args.image_ids else None
    image_name_set = _parse_name_set(args.image_names, args.image_list_file)
    if image_name_set:
        name_to_image_id = {str(im.get("file_name", "")): int(im["id"]) for im in coco.get("images", [])}
        matched_ids = {name_to_image_id[n] for n in image_name_set if n in name_to_image_id}
        missing = sorted(image_name_set - set(name_to_image_id.keys()))
        if missing:
            print(f"[warn] 这些 file_name 在 COCO 中不存在: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if image_id_set is None:
            image_id_set = matched_ids
        else:
            image_id_set = image_id_set.intersection(matched_ids)

    candidates: List[np.ndarray] = []
    for ann in coco.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        ann_id = int(ann.get("id", -1))
        if ann_id_set is not None and ann_id not in ann_id_set:
            continue
        img = images.get(int(ann.get("image_id", -1)))
        if img is None:
            continue
        if image_id_set is not None and int(img["id"]) not in image_id_set:
            continue
        h = int(img.get("height", 0))
        w = int(img.get("width", 0))
        if h <= 2 or w <= 2:
            continue
        m_full = _ann_to_mask(ann, h, w).astype(np.float32)
        if float(m_full.sum()) < 30:
            continue
        crop = safe_crop_with_pad(m_full, ann.get("bbox", [0, 0, 0, 0]), pad=int(args.pad))
        if crop is None:
            continue
        norm = resize_mask(crop, (int(args.temp_size[0]), int(args.temp_size[1])))
        candidates.append(norm)

    if not candidates:
        raise RuntimeError("未收集到有效实例，请检查数据集与标注。")

    demo_idx = int(np.clip(int(args.demo_index), 0, len(candidates) - 1))
    demo = candidates[demo_idx]
    angle = get_orientation(demo)
    rotate_angle = -angle + 90.0
    rotated = rotate_mask(demo, rotate_angle)
    rotated_180 = rotate_180(rotated)
    h = rotated.shape[0]
    top_mass = float(np.sum(rotated[: h // 2, :]))
    bottom_mass = float(np.sum(rotated[h // 2 :, :]))
    aligned = rotated if bottom_mass >= top_mass else rotated_180

    aligned_all = [align_mask(m, prefer_bottom_heavy=True) for m in candidates]
    overlay_size = (int(args.overlay_size[0]), int(args.overlay_size[1]))
    normalized_all = [
        post_normalize_aligned_mask(m, size_wh=overlay_size, target_fill=float(args.target_fill))
        for m in aligned_all
    ]
    if args.overlay_mode == "diverse":
        aligned_set = _select_diverse(normalized_all, int(args.overlay_k))
    else:
        aligned_set = _select_compact(normalized_all, int(args.overlay_k))
    p_final = _overlay_aligned_contours(
        aligned_set,
        size_wh=overlay_size,
        dashed_overlay=bool(args.dashed_overlay),
        dash_len=int(args.dash_len),
        dash_gap=int(args.dash_gap),
    )

    out_paths = _resolve_out_paths(args.out)
    cv2.imwrite(out_paths["step1_raw"], _mask_to_bgr(demo))
    cv2.imwrite(out_paths["step2_rotated_vertical"], _mask_to_bgr(rotated))
    cv2.imwrite(out_paths["step3_rotated_180"], _mask_to_bgr(rotated_180))
    cv2.imwrite(out_paths["step4_chosen_direction"], _mask_to_bgr(aligned))
    cv2.imwrite(out_paths["final_overlay"], p_final)

    for k, v in out_paths.items():
        print(f"saved[{k}]: {v}")


if __name__ == "__main__":
    main()

