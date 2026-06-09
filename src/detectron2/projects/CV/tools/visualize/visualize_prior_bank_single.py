#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prior Bank 单图导出（不拼表格、不画图例）：
- 基于 generate_mean_shape.py 的 build_prior_bank 计算 [K,H,W]
- 每个模板单独输出一张热力图 PNG
- 可选保存 prior_bank.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

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
    p = argparse.ArgumentParser(description="Export prior bank templates as individual PNGs.")
    p.add_argument("--ann", required=True, help="COCO annotation json path")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--bank-size", type=int, default=4, help="prior bank 模板数 K")
    p.add_argument("--target-size", nargs=2, type=int, default=[28, 28], metavar=("W", "H"))
    p.add_argument("--temp-size", nargs=2, type=int, default=[64, 64], metavar=("W", "H"))
    p.add_argument("--pad", type=int, default=10)
    p.add_argument("--min-wh", type=int, default=5)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--cmap", choices=["jet", "hot", "viridis", "plasma", "magma", "inferno"], default="jet")
    p.add_argument(
        "--normalize-mode",
        choices=["per_template", "global"],
        default="per_template",
        help="每张单独归一化 or 全模板共享归一化",
    )
    p.add_argument("--export-contour", dest="export_contour", action="store_true", default=True, help="导出每个模板阈值轮廓图（默认开启）")
    p.add_argument("--no-export-contour", dest="export_contour", action="store_false", help="不导出阈值轮廓图")
    p.add_argument("--contour-thr", type=float, default=0.5, help="轮廓阈值，作用在[0,1]归一化模板上")
    p.add_argument("--contour-color", nargs=3, type=int, default=[255, 255, 255], metavar=("B", "G", "R"), help="轮廓颜色（BGR）")
    p.add_argument("--contour-thickness", type=int, default=1, help="轮廓线宽")
    p.add_argument("--contour-bg", choices=["black", "heatmap"], default="black", help="轮廓图背景类型")
    p.add_argument("--save-npy", action="store_true", default=True, help="保存 prior_bank.npy（默认开启）")
    return p.parse_args()


def _colormap_jet_like(img_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    cmap = cmap_map.get(str(cmap_name).lower(), cv2.COLORMAP_JET)
    return cv2.applyColorMap(img_u8, cmap)


def _to_u8(mask: np.ndarray, vmax: float) -> np.ndarray:
    m = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vmax > 0:
        m = m / float(vmax)
    m = np.clip(m, 0.0, 1.0)
    return (m * 255.0).astype(np.uint8)


def _normalize01(mask: np.ndarray, vmax: float) -> np.ndarray:
    m = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vmax > 0:
        m = m / float(vmax)
    return np.clip(m, 0.0, 1.0)


def _render_contour(mask01: np.ndarray, thr: float, bg_mode: str, heatmap_bgr: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int) -> np.ndarray:
    thr = float(np.clip(thr, 0.0, 1.0))
    bin_u8 = (mask01 >= thr).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if str(bg_mode).lower() == "heatmap":
        canvas = heatmap_bgr.copy()
    else:
        h, w = bin_u8.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if contours:
        cv2.drawContours(canvas, contours, -1, color_bgr, max(1, int(thickness)))
    return canvas


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


def _safe_crop_with_pad(mask_full: np.ndarray, bbox_xywh: List[float], pad: int) -> np.ndarray:
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
    axis = evecs[:, order[0]]
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def _align_mask_local(mask: np.ndarray) -> np.ndarray:
    angle = _get_orientation(mask)
    rotate_angle = -angle + 90.0
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotate_angle, 1.0)
    rotated = cv2.warpAffine(
        mask.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    half = h // 2
    if float(np.sum(rotated[:half, :])) > float(np.sum(rotated[half:, :])):
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
        coco = json.load(f)
    images = {int(im["id"]): im for im in coco.get("images", [])}
    out: List[np.ndarray] = []
    anns = coco.get("annotations", [])
    for i, ann in enumerate(anns):
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
        aligned = _align_mask_local(norm)
        final = cv2.resize(aligned, (int(target_size[0]), int(target_size[1])), interpolation=cv2.INTER_AREA)
        out.append(final.astype(np.float32))
        if log_every > 0 and i % int(log_every) == 0:
            print(f"[fallback] processed anns {i+1}/{len(anns)}, collected={len(out)}")
    if not out:
        raise RuntimeError("fallback collected zero aligned masks")
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


def _build_prior_bank_fallback(
    ann_file: str,
    bank_size: int,
    target_size: Tuple[int, int],
    temp_size: Tuple[int, int],
    pad: int,
    min_wh: int,
    max_instances: int | None,
    log_every: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    aligned = _collect_aligned_masks_fallback(
        ann_file=ann_file,
        target_size=target_size,
        temp_size=temp_size,
        pad=pad,
        min_wh=min_wh,
        max_instances=max_instances,
        log_every=log_every,
    )
    N, H, W = aligned.shape
    if int(bank_size) <= 1:
        return aligned.mean(axis=0).astype(np.float32)[None, ...], int(N)
    n_clusters = min(int(bank_size) - 1, int(N))
    mean_shape = aligned.mean(axis=0).astype(np.float32)
    flat = aligned.reshape(N, -1)
    _, labels = _run_kmeans_local(flat, k=n_clusters, seed=int(seed), max_iter=50)
    means = []
    sizes = []
    for i in range(n_clusters):
        m = aligned[labels == i]
        if m.shape[0] == 0:
            continue
        means.append(m.mean(axis=0).astype(np.float32))
        sizes.append(int(m.shape[0]))
    order = np.argsort(np.asarray(sizes))[::-1] if sizes else np.asarray([], dtype=np.int64)
    bank = [mean_shape]
    for idx in order:
        bank.append(means[int(idx)])
    return np.stack(bank, axis=0).astype(np.float32), int(N)


def _template_filename(i: int, k: int) -> str:
    if i == 0:
        return "template_T0_global_mean.png"
    if i == k - 1:
        return f"template_T{i}_clusterK_mean.png"
    return f"template_T{i}_cluster{i}_mean.png"


def _contour_filename(i: int, k: int) -> str:
    return _template_filename(i, k).replace(".png", "_contour.png")


def main() -> None:
    args = parse_args()
    gms = _import_generate_mean_shape()

    ann = os.path.abspath(args.ann)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if gms is not None:
        bank, n = gms.build_prior_bank(
            ann_file=ann,
            bank_size=int(args.bank_size),
            target_size=(int(args.target_size[0]), int(args.target_size[1])),
            temp_size=(int(args.temp_size[0]), int(args.temp_size[1])),
            pad=int(args.pad),
            min_wh=int(args.min_wh),
            category_names=None,
            prefer_bottom_heavy=True,
            max_instances=int(args.max_instances) if args.max_instances is not None else None,
            log_every=int(args.log_every),
            seed=int(args.seed),
        )
    else:
        print("[warn] generate_mean_shape import failed (likely missing pycocotools), using fallback JSON pipeline.")
        bank, n = _build_prior_bank_fallback(
            ann_file=ann,
            bank_size=int(args.bank_size),
            target_size=(int(args.target_size[0]), int(args.target_size[1])),
            temp_size=(int(args.temp_size[0]), int(args.temp_size[1])),
            pad=int(args.pad),
            min_wh=int(args.min_wh),
            max_instances=int(args.max_instances) if args.max_instances is not None else None,
            log_every=int(args.log_every),
            seed=int(args.seed),
        )
    # bank: [K,H,W]
    K = int(bank.shape[0])

    if bool(args.save_npy):
        npy_path = os.path.join(out_dir, "prior_bank.npy")
        np.save(npy_path, bank.astype(np.float32))
        print(f"saved: {npy_path}")

    if args.normalize_mode == "global":
        global_vmax = float(np.nanmax(bank))
        vmax_list: List[float] = [global_vmax] * K
    else:
        vmax_list = [float(np.nanmax(bank[i])) for i in range(K)]

    for i in range(K):
        norm01 = _normalize01(bank[i], vmax=vmax_list[i])
        u8 = (norm01 * 255.0).astype(np.uint8)
        color = _colormap_jet_like(u8, args.cmap)
        out_name = _template_filename(i, K)
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, color)
        print(f"saved: {out_path}")
        if bool(args.export_contour):
            bgr = tuple(int(np.clip(v, 0, 255)) for v in args.contour_color)
            contour_img = _render_contour(
                mask01=norm01,
                thr=float(args.contour_thr),
                bg_mode=str(args.contour_bg),
                heatmap_bgr=color,
                color_bgr=(bgr[0], bgr[1], bgr[2]),
                thickness=int(args.contour_thickness),
            )
            contour_path = os.path.join(out_dir, _contour_filename(i, K))
            cv2.imwrite(contour_path, contour_img)
            print(f"saved: {contour_path}")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"N_instances={n}\n"
            f"bank_shape={tuple(bank.shape)}\n"
            f"dtype={bank.dtype}\n"
            f"normalize_mode={args.normalize_mode}\n"
            f"cmap={args.cmap}\n"
            f"export_contour={bool(args.export_contour)}\n"
            f"contour_thr={float(args.contour_thr)}\n"
            f"contour_bg={args.contour_bg}\n"
            f"contour_color_bgr={tuple(int(np.clip(v, 0, 255)) for v in args.contour_color)}\n"
            f"contour_thickness={int(args.contour_thickness)}\n"
        )
    print(f"saved: {os.path.join(out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()

