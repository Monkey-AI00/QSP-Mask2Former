#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第五部分可视化：KMeans clustering on aligned masks

数据计算部分与 generate_mean_shape.py 一致：
- collect_aligned_masks(...)
- flatten (N, H*W)
- _run_kmeans(...)
- cluster mean (按簇内样本均值)

说明（重要）：
- 蓝点、椭圆、红色虚线等属于示意绘制元素，用于贴近论文表达；
- 真正算法输出是 labels、centers、cluster means。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _import_generate_mean_shape():
    here = Path(__file__).resolve()
    datasets_dir = here.parent.parent / "datasets"
    if str(datasets_dir) not in sys.path:
        sys.path.insert(0, str(datasets_dir))
    try:
        import generate_mean_shape as gms  # type: ignore
    except Exception:
        return None
    return gms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize KMeans on aligned masks.")
    p.add_argument("--ann", required=True, help="COCO annotation json path")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--bank-size", type=int, default=4, help="聚类数 K（>=4 时可输出 1/2/3/K）")
    p.add_argument("--target-size", nargs=2, type=int, default=[28, 28], metavar=("W", "H"))
    p.add_argument("--temp-size", nargs=2, type=int, default=[64, 64], metavar=("W", "H"))
    p.add_argument("--pad", type=int, default=10)
    p.add_argument("--min-wh", type=int, default=5)
    p.add_argument("--max-instances", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--mean-thr", type=float, default=0.35, help="mean mask 转轮廓阈值")
    p.add_argument("--sample-dots", type=int, default=8, help="每簇示意蓝点数量")
    p.add_argument("--show-cluster-label", action="store_true", help="显示每个cluster卡片底部文字（默认关闭）")
    return p.parse_args()


def _normalize01(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(x.max())
    if mx > 0:
        x = x / mx
    return x


def _ann_to_mask_local(ann: dict, h: int, w: int) -> np.ndarray:
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


def _safe_crop_with_pad(mask_full: np.ndarray, bbox_xywh: Sequence[float], pad: int) -> np.ndarray:
    x, y, w, h = [int(round(v)) for v in bbox_xywh]
    H, W = mask_full.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + max(0, w) + pad), min(H, y + max(0, h) + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=mask_full.dtype)
    return mask_full[y1:y2, x1:x2]


def _get_orientation(mask: np.ndarray) -> float:
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


def _align_mask_local(mask: np.ndarray, prefer_bottom_heavy: bool = True) -> np.ndarray:
    angle = _get_orientation(mask)
    rotate_angle = -angle + 90.0
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(
        mask.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    half = h // 2
    top_mass = float(np.sum(rotated[:half, :]))
    bottom_mass = float(np.sum(rotated[half:, :]))
    if prefer_bottom_heavy:
        if top_mass > bottom_mass:
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
    else:
        if bottom_mass > top_mass:
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
    return rotated


def _collect_aligned_masks_fallback(
    ann_file: str,
    target_size: Tuple[int, int],
    temp_size: Tuple[int, int],
    pad: int,
    min_wh: int,
    max_instances: int | None,
    log_every: int,
) -> np.ndarray:
    with open(ann_file, "r") as f:
        coco = __import__("json").load(f)
    images = {int(im["id"]): im for im in coco.get("images", [])}
    out: List[np.ndarray] = []
    for i, ann in enumerate(coco.get("annotations", [])):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        if max_instances is not None and len(out) >= int(max_instances):
            break
        img = images.get(int(ann.get("image_id", -1)))
        if img is None:
            continue
        h = int(img.get("height", 0))
        w = int(img.get("width", 0))
        if h <= 2 or w <= 2:
            continue
        m_full = _ann_to_mask_local(ann, h, w).astype(np.float32)
        crop = _safe_crop_with_pad(m_full, ann.get("bbox", [0, 0, 0, 0]), int(pad))
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch < int(min_wh) or cw < int(min_wh):
            continue
        norm = cv2.resize(crop, (int(temp_size[0]), int(temp_size[1])), interpolation=cv2.INTER_AREA)
        aligned = _align_mask_local(norm, prefer_bottom_heavy=True)
        final = cv2.resize(aligned, (int(target_size[0]), int(target_size[1])), interpolation=cv2.INTER_AREA)
        out.append(final.astype(np.float32))
        if log_every > 0 and (i % int(log_every) == 0):
            print(f"[fallback] processed anns {i+1}/{len(coco.get('annotations', []))}, collected={len(out)}")
    if not out:
        raise RuntimeError("fallback path collected zero aligned masks")
    return np.stack(out, axis=0).astype(np.float32)


def _run_kmeans_local(features: np.ndarray, k: int, seed: int = 42, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    n = int(features.shape[0])
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1,{n}], got {k}")
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(max_iter):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            m = labels == i
            if np.any(m):
                centers[i] = features[m].mean(axis=0)
            else:
                centers[i] = features[rng.integers(0, n)]
    return centers, labels


def _draw_dashed_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 7,
    dash_gap: int = 5,
) -> None:
    if pts.shape[0] < 2:
        return
    cycle = max(1, int(dash_len) + int(dash_gap))
    for i in range(pts.shape[0]):
        p0 = pts[i].astype(np.float32)
        p1 = pts[(i + 1) % pts.shape[0]].astype(np.float32)
        d = p1 - p0
        L = float(np.hypot(d[0], d[1]))
        if L <= 1e-6:
            continue
        u = d / L
        t = 0.0
        while t < L:
            t2 = min(L, t + float(max(1, dash_len)))
            a = p0 + u * t
            b = p0 + u * t2
            cv2.line(
                img,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            t += float(cycle)


def _render_mean_contour_card(
    mean_mask: np.ndarray,
    title: str,
    sample_dots: int = 8,
    card_size: Tuple[int, int] = (250, 260),
    thr: float = 0.35,
    show_title: bool = False,
) -> np.ndarray:
    w, h = card_size
    card = np.full((h, w, 3), 255, dtype=np.uint8)

    # 外圈椭圆（蓝色虚线）
    center = (w // 2, int(h * 0.48))
    axes = (int(w * 0.30), int(h * 0.34))
    for a in range(0, 360, 18):
        cv2.ellipse(card, center, axes, 0, a, min(a + 9, 360), (226, 170, 110), 1, cv2.LINE_AA)

    # 蓝点（示意该簇样本）
    n = max(3, int(sample_dots))
    for i in range(n):
        t = 2.0 * np.pi * float(i) / float(n)
        px = int(round(center[0] + np.cos(t) * axes[0] * 0.88))
        py = int(round(center[1] + np.sin(t) * axes[1] * 0.88))
        cv2.circle(card, (px, py), 4, (220, 80, 40), -1, cv2.LINE_AA)

    # 红色虚线：cluster mean contour
    m = _normalize01(mean_mask)
    m_bin = (m > float(thr)).astype(np.uint8)
    cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        # 映射到椭圆内部区域
        x, y, ww, hh = cv2.boundingRect(c.astype(np.int32))
        if ww > 0 and hh > 0:
            c[:, 0] = (c[:, 0] - x) / float(ww)
            c[:, 1] = (c[:, 1] - y) / float(hh)
            box_w, box_h = int(w * 0.42), int(h * 0.52)
            ox = center[0] - box_w // 2
            oy = center[1] - box_h // 2
            c[:, 0] = ox + c[:, 0] * box_w
            c[:, 1] = oy + c[:, 1] * box_h
            _draw_dashed_polyline(card, c, color=(70, 70, 245), thickness=2, dash_len=8, dash_gap=6)

    if show_title:
        cv2.putText(card, title, (w // 2 - 45, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
    return card


def _render_left_strip(aligned_masks: np.ndarray, max_show: int = 3) -> np.ndarray:
    # aligned masks 缩略图 + flatten 向量灰条
    N, H, W = aligned_masks.shape
    show_n = min(max_show, N)
    thumb_w, thumb_h = 90, 90
    gap = 10
    left_w = 260
    left_h = 340
    panel = np.full((left_h, left_w, 3), 255, dtype=np.uint8)

    y = 20
    for i in range(show_n):
        m = _normalize01(aligned_masks[i])
        im = (m * 255.0).astype(np.uint8)
        im = cv2.resize(im, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        x = 15
        panel[y : y + thumb_h, x : x + thumb_w] = im
        y += thumb_h + gap
        if i == 1 and show_n > 2:
            cv2.putText(panel, "...", (48, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (50, 50, 50), 2, cv2.LINE_AA)

    # flatten 可视化（取第一个样本）
    vec = aligned_masks[0].reshape(-1)
    vec = _normalize01(vec)
    bar_h, bar_w = 180, 20
    bar = (cv2.resize(vec[:, None], (bar_w, bar_h), interpolation=cv2.INTER_NEAREST) * 255.0).astype(np.uint8)
    bar = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
    bx, by = 150, 50
    panel[by : by + bar_h, bx : bx + bar_w] = bar
    cv2.rectangle(panel, (bx, by), (bx + bar_w, by + bar_h), (160, 160, 160), 1, cv2.LINE_AA)

    cv2.putText(panel, "Aligned masks", (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(panel, f"(N={N})", (110, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(panel, "Flatten", (138, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return panel


def _stack_h(imgs: Sequence[np.ndarray], gap: int = 16, bg: int = 255) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    out = np.full((h, w, 3), bg, dtype=np.uint8)
    x = 0
    for i, im in enumerate(imgs):
        out[0 : im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1]
        if i < len(imgs) - 1:
            x += gap
    return out


def main() -> None:
    args = parse_args()
    gms = _import_generate_mean_shape()

    ann = os.path.abspath(args.ann)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if gms is not None:
        aligned_masks = gms.collect_aligned_masks(
            ann_file=ann,
            target_size=(int(args.target_size[0]), int(args.target_size[1])),
            temp_size=(int(args.temp_size[0]), int(args.temp_size[1])),
            pad=int(args.pad),
            min_wh=int(args.min_wh),
            category_names=None,
            prefer_bottom_heavy=True,
            max_instances=int(args.max_instances) if args.max_instances is not None else None,
            log_every=int(args.log_every),
        )
    else:
        print("[warn] generate_mean_shape import failed (likely missing pycocotools), using fallback JSON pipeline.")
        aligned_masks = _collect_aligned_masks_fallback(
            ann_file=ann,
            target_size=(int(args.target_size[0]), int(args.target_size[1])),
            temp_size=(int(args.temp_size[0]), int(args.temp_size[1])),
            pad=int(args.pad),
            min_wh=int(args.min_wh),
            max_instances=int(args.max_instances) if args.max_instances is not None else None,
            log_every=int(args.log_every),
        )
    N, H, W = aligned_masks.shape

    K = int(args.bank_size)
    if K < 2:
        raise ValueError("--bank-size 至少为 2；若希望展示 ClusterK，建议 >=4")

    features = aligned_masks.reshape(N, -1).astype(np.float32)
    if gms is not None:
        _, labels = gms._run_kmeans(features, k=K, seed=int(args.seed), max_iter=50)
    else:
        _, labels = _run_kmeans_local(features, k=K, seed=int(args.seed), max_iter=50)

    means = []
    counts = []
    for ci in range(K):
        members = aligned_masks[labels == ci]
        if members.shape[0] == 0:
            means.append(np.zeros((H, W), dtype=np.float32))
            counts.append(0)
        else:
            means.append(members.mean(axis=0).astype(np.float32))
            counts.append(int(members.shape[0]))

    # 主图仅展示 Cluster1,2,3,K（与用户要求一致）
    show_idx = []
    for idx in [0, 1, 2, K - 1]:
        if 0 <= idx < K and idx not in show_idx:
            show_idx.append(idx)

    cards = []
    for idx in show_idx:
        title = f"Cluster {idx+1}"
        card = _render_mean_contour_card(
            means[idx],
            title=title,
            sample_dots=min(12, max(4, counts[idx] // max(1, N // 10))),
            card_size=(250, 260),
            thr=float(args.mean_thr),
            show_title=bool(args.show_cluster_label),
        )
        cards.append(card)

    left = _render_left_strip(aligned_masks, max_show=3)
    panel = _stack_h([left] + cards, gap=18, bg=255)
    cv2.putText(
        panel,
        "5. KMeans clustering on aligned masks",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "KMeans in feature space of aligned masks",
        (12, panel.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )

    panel_path = os.path.join(out_dir, "kmeans_aligned_masks_panel.png")
    cv2.imwrite(panel_path, panel)

    # 单独输出 Cluster1/2/3/K
    for idx in show_idx:
        single = _render_mean_contour_card(
            means[idx],
            title=f"Cluster {idx+1}",
            sample_dots=min(12, max(4, counts[idx] // max(1, N // 10))),
            card_size=(300, 300),
            thr=float(args.mean_thr),
            show_title=bool(args.show_cluster_label),
        )
        cv2.imwrite(os.path.join(out_dir, f"cluster_{idx+1}.png"), single)

    # 保存真实算法产物
    np.save(os.path.join(out_dir, "kmeans_labels.npy"), labels.astype(np.int64))
    np.save(os.path.join(out_dir, "kmeans_cluster_means.npy"), np.stack(means, axis=0).astype(np.float32))

    note = (
        "Difference note:\n"
        "1) Real algorithm outputs: labels, cluster means, flattened features.\n"
        "2) Visual-only decorations: blue dots, dashed ellipse, red dashed contour style.\n"
        "3) Cluster cards show Cluster1/2/3/K as requested; not all clusters are drawn on panel.\n"
    )
    with open(os.path.join(out_dir, "difference_note.txt"), "w", encoding="utf-8") as f:
        f.write(note)

    print(f"saved: {panel_path}")
    print(f"saved: {os.path.join(out_dir, 'kmeans_labels.npy')}")
    print(f"saved: {os.path.join(out_dir, 'kmeans_cluster_means.npy')}")
    for idx in show_idx:
        print(f"saved: {os.path.join(out_dir, f'cluster_{idx+1}.png')}")


if __name__ == "__main__":
    main()

