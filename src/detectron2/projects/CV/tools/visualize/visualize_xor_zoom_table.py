#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XOR 局部放大可视化（仅 Raw + GT + XOR Zoom）：

- 不输出实例分割叠加图（Pred Overlay）
- 保留 Raw 图与 GT 图
- 每个方法只输出 XOR 误差局部放大图
- 固定颜色：FP=红色，FN=蓝色
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import visualize_predictions_under_highlight_table_layout as base_vis


# 固定 XOR 配色（BGR）
FP_COLOR: Tuple[int, int, int] = (0, 0, 255)      # 红
FN_COLOR: Tuple[int, int, int] = (255, 0, 0)      # 蓝
ZOOM_BOX_COLOR: Tuple[int, int, int] = (0, 255, 255)  # 黄


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XOR-only zoom table visualization.")
    p.add_argument("--dataset-root", required=True, help="高反光评测集目录（图片+json）")
    p.add_argument("--clean-root", default="", help="可选：clean 数据目录（用于去除目标外光晕）")
    p.add_argument("--dataset-name", default="plug_highlight_eval_xor_zoom", help="注册到 detectron2 的数据集名")
    p.add_argument("--json-file", default="plug_test.json", help="COCO json（相对 dataset-root）")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-rows", type=int, default=4)
    p.add_argument("--scan-max", type=int, default=500)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--index-range", default="", help="如 36-40")
    p.add_argument("--index-base", type=int, default=1, choices=(0, 1))
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--cell-width", type=int, default=520, help="每个格子目标宽度")
    # 兼容旧命令：以下 3 个参数保留但不再生效（当前版本不做局部放大）
    p.add_argument("--zoom-scale", type=float, default=2.5, help="已弃用：旧版 XOR 局部放大倍数（当前忽略）")
    p.add_argument("--zoom-pad", type=int, default=20, help="已弃用：旧版局部框外扩像素（当前忽略）")
    p.add_argument("--zoom-min-size", type=int, default=96, help="已弃用：旧版局部框最小边长（当前忽略）")
    p.add_argument("--xor-alpha", type=float, default=0.82, help="XOR 颜色叠加透明度")
    p.add_argument("--xor-bg-dim", type=float, default=0.55, help="XOR 背景亮度系数")

    # Mask2Former
    p.add_argument("--mask2former-root", required=True)
    p.add_argument("--config-mask2former", required=True)
    p.add_argument("--weights-mask2former", default="", help="Mask2Former 基线权重（可选）")
    p.add_argument("--weights-mask2former-sdf", default="", help="M2F+SDF 权重（可选）")
    p.add_argument("--weights-mask2former-geoloss", required=True, help="M2F+GeoLoss 权重")
    p.add_argument("--config-mask2former-qsp", default="")
    p.add_argument("--weights-mask2former-qsp", required=True, help="M2F+QSP 权重")
    p.add_argument("--prior-path-mask2former-qsp", default="")

    # PointRend
    p.add_argument("--config-pointrend", required=True)
    p.add_argument("--weights-base", required=True)
    p.add_argument("--weights-spg", default="")
    p.add_argument("--shape-prior-npy", default="")

    # Mask R-CNN
    p.add_argument("--config-maskrcnn", required=True)
    p.add_argument("--weights-maskrcnn", required=True)

    # MaskTransfiner（可选）
    p.add_argument("--transfiner-root", default="")
    p.add_argument("--config-transfiner", default="")
    p.add_argument("--weights-transfiner", default="")
    return p.parse_args()


def _compute_focus_box(gt01: np.ndarray, pr01: np.ndarray, pad: int, min_size: int) -> Tuple[int, int, int, int]:
    gt = (gt01 > 0)
    pr = (pr01 > 0)
    err = np.logical_xor(gt, pr)
    ys, xs = np.where(err if np.any(err) else gt)

    h, w = gt01.shape[:2]
    if ys.size == 0 or xs.size == 0:
        cx, cy = w // 2, h // 2
        half = max(1, int(min_size // 2))
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    else:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

    x1 -= int(pad)
    y1 -= int(pad)
    x2 += int(pad)
    y2 += int(pad)

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    if bw < int(min_size):
        extra = (int(min_size) - bw) // 2 + 1
        x1 -= extra
        x2 += extra
    if bh < int(min_size):
        extra = (int(min_size) - bh) // 2 + 1
        y1 -= extra
        y2 += extra

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _crop_zoom(img: np.ndarray, box: Tuple[int, int, int, int], scale: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = img[y1 : y2 + 1, x1 : x2 + 1]
    ch, cw = crop.shape[:2]
    nw = max(1, int(round(cw * float(scale))))
    nh = max(1, int(round(ch * float(scale))))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)


def _gt_panel(raw_bgr: np.ndarray, gt01: np.ndarray) -> np.ndarray:
    return base_vis._overlay_mask(raw_bgr, gt01, color_bgr=(0, 255, 0), alpha=0.38, outline=True)


def _make_xor_zoom_panel(
    raw_bgr: np.ndarray,
    gt01: np.ndarray,
    pr01: np.ndarray,
    *,
    method_name: str,
    zoom_scale: float,
    zoom_pad: int,
    zoom_min_size: int,
    xor_alpha: float,
    xor_bg_dim: float,
) -> Tuple[np.ndarray, int, int]:
    xor_full, fp_px, fn_px = base_vis._xor_vis(
        raw_bgr,
        gt01,
        pr01,
        fp_color=FP_COLOR,
        fn_color=FN_COLOR,
        alpha=float(xor_alpha),
        bg_dim=float(xor_bg_dim),
    )
    # 当前版本不做局部放大，直接使用与 raw 同尺寸的 XOR 全图
    panel = xor_full.copy()
    cv2.putText(
        panel,
        "FP:red FN:blue",
        (8, max(20, panel.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    panel = base_vis._title_bar(panel, f"{method_name} XOR Zoom")
    return panel, fp_px, fn_px


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    dataset_name, num_classes = base_vis.register_plug_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        json_filename=args.json_file,
    )
    dicts = base_vis.DatasetCatalog.get(dataset_name)
    if not dicts:
        raise RuntimeError("dataset is empty")

    cand = sorted(list(dicts), key=lambda x: base_vis._natural_sort_key(str(x.get("file_name", ""))))
    cand = cand[: max(1, int(args.scan_max))]
    if bool(getattr(args, "no_shuffle", False)):
        picks = list(cand)
    else:
        random.shuffle(cand)
        picks = cand

    if str(getattr(args, "index_range", "")).strip():
        a, b = base_vis._parse_index_range(str(args.index_range).strip())
        base = int(getattr(args, "index_base", 1))
        if base == 1:
            a0, b0 = max(0, a - 1), max(0, b - 1)
        else:
            a0, b0 = max(0, a), max(0, b)
        picks = picks[a0 : b0 + 1]
    else:
        picks = picks[: max(1, int(args.num_rows))]
    if not picks:
        raise RuntimeError("no valid images found for visualization")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # predictors
    maskrcnn_pred = base_vis._build_maskrcnn_predictor(
        config_file=os.path.abspath(str(args.config_maskrcnn).strip()),
        weights=os.path.abspath(str(args.weights_maskrcnn).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    base_pred = base_vis._build_pointrend_predictor(
        config_file=os.path.abspath(str(args.config_pointrend).strip()),
        weights=os.path.abspath(str(args.weights_base).strip()),
        mask_head_name="PointRendMaskHead",
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    spg_pred = None
    if str(getattr(args, "weights_spg", "")).strip():
        if str(getattr(args, "shape_prior_npy", "")).strip():
            os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(str(args.shape_prior_npy).strip())
        spg_pred = base_vis._build_pointrend_predictor(
            config_file=os.path.abspath(str(args.config_pointrend).strip()),
            weights=os.path.abspath(str(args.weights_spg).strip()),
            mask_head_name="ShapeAwareCoarseMaskHead",
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )

    m2f_geoloss_pred = base_vis._build_mask2former_predictor(
        mask2former_root=str(args.mask2former_root),
        config_file=os.path.abspath(str(args.config_mask2former).strip()),
        weights=os.path.abspath(str(args.weights_mask2former_geoloss).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    m2f_base_pred = None
    if str(getattr(args, "weights_mask2former", "")).strip():
        m2f_base_pred = base_vis._build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root),
            config_file=os.path.abspath(str(args.config_mask2former).strip()),
            weights=os.path.abspath(str(args.weights_mask2former).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )
    m2f_sdf_pred = None
    if str(getattr(args, "weights_mask2former_sdf", "")).strip():
        m2f_sdf_pred = base_vis._build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root),
            config_file=os.path.abspath(str(args.config_mask2former).strip()),
            weights=os.path.abspath(str(args.weights_mask2former_sdf).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )

    config_mask2former_qsp = str(getattr(args, "config_mask2former_qsp", "")).strip() or str(
        getattr(args, "config_mask2former", "")
    ).strip()
    qsp_prior_override = str(getattr(args, "prior_path_mask2former_qsp", "")).strip() or str(
        getattr(args, "shape_prior_npy", "")
    ).strip()
    m2f_qsp_pred = base_vis._build_mask2former_predictor(
        mask2former_root=str(args.mask2former_root),
        config_file=os.path.abspath(config_mask2former_qsp),
        weights=os.path.abspath(str(args.weights_mask2former_qsp).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
        prior_path_override=os.path.abspath(qsp_prior_override) if qsp_prior_override else "",
    )

    tf_root = str(getattr(args, "transfiner_root", "")).strip()
    tf_cfg = str(getattr(args, "config_transfiner", "")).strip()
    tf_w = str(getattr(args, "weights_transfiner", "")).strip()
    has_tf = bool(tf_root and tf_cfg and tf_w)
    transfiner_mask_map: Dict[str, np.ndarray] = {}
    if has_tf:
        pick_paths = [os.path.abspath(str(d.get("file_name", ""))) for d in picks if str(d.get("file_name", "")).strip()]
        transfiner_mask_map = base_vis._predict_transfiner_masks_subprocess(
            transfiner_root=tf_root,
            config_file=tf_cfg,
            weights=tf_w,
            image_paths=pick_paths,
            score_thr=float(args.score_thr),
        )

    rows_img: List[np.ndarray] = []
    stats: List[dict] = []

    for i, d in enumerate(picks):
        img_path = str(d.get("file_name", ""))
        raw_bgr = base_vis._safe_imread(img_path)
        if raw_bgr is None:
            continue
        h, w = raw_bgr.shape[:2]
        gt = base_vis._gt_mask_from_dict(d, h=h, w=w)
        if gt is None:
            continue

        # 输入图可选去光晕（仅显示）
        input_bgr = raw_bgr
        if str(getattr(args, "clean_root", "")).strip():
            clean_bgr = base_vis._load_clean_match(
                clean_root=str(args.clean_root).strip(),
                highlight_root=str(args.dataset_root).strip(),
                highlight_path=img_path,
            )
            if clean_bgr is not None:
                input_bgr = base_vis._remove_halo_with_clean(
                    img_highlight_bgr=raw_bgr,
                    img_clean_bgr=clean_bgr,
                    obj_mask01=gt,
                )

        base_mask, _ = base_vis._predict_best_mask(base_pred, raw_bgr)
        maskrcnn_mask, _ = base_vis._predict_best_mask(maskrcnn_pred, raw_bgr)
        m2f_geoloss_mask, _ = base_vis._predict_best_mask(m2f_geoloss_pred, raw_bgr)
        m2f_qsp_mask, _ = base_vis._predict_best_mask(m2f_qsp_pred, raw_bgr)
        m2f_base_mask = None if m2f_base_pred is None else base_vis._predict_best_mask(m2f_base_pred, raw_bgr)[0]
        m2f_sdf_mask = None if m2f_sdf_pred is None else base_vis._predict_best_mask(m2f_sdf_pred, raw_bgr)[0]
        spg_mask = None if spg_pred is None else base_vis._predict_best_mask(spg_pred, raw_bgr)[0]
        tf_mask = transfiner_mask_map.get(os.path.abspath(img_path), None) if has_tf else None

        def nz(m: Optional[np.ndarray]) -> np.ndarray:
            if m is None:
                return np.zeros_like(gt)
            if m.shape != gt.shape:
                return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return (m > 0).astype(np.uint8)

        method_masks: List[Tuple[str, np.ndarray]] = [
            ("Mask R-CNN", nz(maskrcnn_mask)),
            ("PointRend(Base)", nz(base_mask)),
            ("M2F+Geo.Loss", nz(m2f_geoloss_mask)),
            ("QSP-M2F", nz(m2f_qsp_mask)),
        ]
        if spg_pred is not None:
            method_masks.insert(2, ("SPG-PointRend", nz(spg_mask)))
        if m2f_base_pred is not None:
            method_masks.insert(-2, ("Mask2Former", nz(m2f_base_mask)))
        if m2f_sdf_pred is not None:
            method_masks.insert(-1, ("M2F+SDF", nz(m2f_sdf_mask)))
        if has_tf:
            method_masks.insert(1, ("MaskTransfiner", nz(tf_mask)))

        raw_panel = input_bgr.copy()
        gt_panel = _gt_panel(input_bgr, gt)
        cells: List[np.ndarray] = [
            base_vis._title_bar(raw_panel, "Raw"),
            base_vis._title_bar(gt_panel, "GroundTruth"),
        ]

        for method_name, pr in method_masks:
            panel, fp_px, fn_px = _make_xor_zoom_panel(
                input_bgr,
                gt,
                pr,
                method_name=method_name,
                zoom_scale=float(args.zoom_scale),
                zoom_pad=int(args.zoom_pad),
                zoom_min_size=int(args.zoom_min_size),
                xor_alpha=float(args.xor_alpha),
                xor_bg_dim=float(args.xor_bg_dim),
            )
            cells.append(panel)
            stats.append(
                {
                    "image_path": img_path,
                    "method": method_name,
                    "fp_pixels": int(fp_px),
                    "fn_pixels": int(fn_px),
                    "xor_pixels": int(fp_px + fn_px),
                    "gt_pixels": int((gt > 0).sum()),
                    "pred_pixels": int((pr > 0).sum()),
                }
            )

        cells = [base_vis._resize_to_width(c, int(args.cell_width)) for c in cells]
        row = base_vis._stack_row(cells)
        rows_img.append(row)
        stem = f"{i:02d}_" + os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_xor_zoom_row.png"), row)

    if not rows_img:
        raise RuntimeError("no valid rows were generated")

    grid = base_vis._stack_grid(rows_img)
    out_grid = os.path.join(out_dir, "xor_zoom_grid.png")
    cv2.imwrite(out_grid, grid)
    print(f"saved: {out_grid}")

    out_csv = os.path.join(out_dir, "xor_zoom_error_stats.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "method", "fp_pixels", "fn_pixels", "xor_pixels", "gt_pixels", "pred_pixels"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in stats:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()

