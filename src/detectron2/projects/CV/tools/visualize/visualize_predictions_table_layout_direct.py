#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多方法误差对比图（直接测试版，无高光筛选/模拟）：

- 不按高光分数重排样本
- 不做 clean-halo 去光晕处理
- 直接基于 dataset-root + json-file 全量/按索引可视化
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

import visualize_predictions_under_highlight_table_layout as base_vis


def _make_header_row(column_titles: List[str], cell_width: int, header_h: int = 54) -> np.ndarray:
    w = int(cell_width) * len(column_titles)
    h = int(header_h)
    header = np.full((h, w, 3), 245, dtype=np.uint8)
    # bottom border
    cv2.line(header, (0, h - 1), (w - 1, h - 1), (165, 165, 165), 2, cv2.LINE_AA)
    for i, title in enumerate(column_titles):
        x0 = i * int(cell_width)
        x1 = x0 + int(cell_width)
        cv2.line(header, (x0, 0), (x0, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        tx = x0 + max(6, (int(cell_width) - tw) // 2)
        ty = max(th + 6, (h + th) // 2 - 4)
        cv2.putText(header, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(header, (w - 1, 0), (w - 1, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    return header


def _add_row_tag(row_img: np.ndarray, tag: str) -> np.ndarray:
    out = row_img.copy()
    cv2.rectangle(out, (6, 6), (44, 28), (248, 248, 248), thickness=-1)
    cv2.rectangle(out, (6, 6), (44, 28), (178, 178, 178), thickness=1)
    cv2.putText(out, tag, (13, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Table-layout qualitative comparison (direct test, no highlight simulation).")
    p.add_argument("--dataset-root", required=True, help="测试数据集目录（图片+json）")
    p.add_argument("--dataset-name", default="plug_direct_eval_table_vis", help="注册到 detectron2 的数据集名称")
    p.add_argument("--json-file", default="plug_test.json", help="COCO json 文件名（相对 dataset-root）")
    p.add_argument("--out-dir", default="./output/pred_vis_table_layout_direct", help="输出目录")
    p.add_argument("--seed", type=int, default=0)

    # selection
    p.add_argument("--num-rows", type=int, default=4, help="输出多少行（多少张图）")
    p.add_argument("--scan-max", type=int, default=200, help="最多扫描多少张图")
    p.add_argument("--no-shuffle", action="store_true", help="按数据集原始顺序选择图片，不随机打乱")
    p.add_argument("--index-range", default="", help="可选：只可视化指定序号范围（如 15-20，闭区间）")
    p.add_argument("--index-base", type=int, default=1, choices=(0, 1), help="index-range 的序号基准（默认1）")

    # visualization
    p.add_argument("--cell-width", type=int, default=520, help="每个格子的宽度（像素）")
    p.add_argument(
        "--no-header",
        "--hide-header",
        dest="no_header",
        action="store_true",
        help="去掉图片上方的全局列图注；默认保留图注。",
    )
    p.add_argument(
        "--no-row-tags",
        "--hide-row-tags",
        dest="no_row_tags",
        action="store_true",
        help="去掉每行左上角的 S1/S2/... 标签；默认保留标签。",
    )
    p.add_argument("--pred-erode", type=int, default=0, help="可选：可视化前先腐蚀预测 mask（像素）")
    p.add_argument("--score-thr", type=float, default=0.5, help="预测分数阈值")
    p.add_argument("--mode", choices=["pred_only", "pred_xor"], default="pred_xor", help="pred_only 或 pred_xor")
    p.add_argument("--xor-fp-color", default="0,0,255", help="FP 颜色(B,G,R)")
    p.add_argument("--xor-fn-color", default="255,0,0", help="FN 颜色(B,G,R)")
    p.add_argument("--xor-alpha", type=float, default=0.75, help="FP/FN 叠加透明度")
    p.add_argument("--xor-bg-dim", type=float, default=0.6, help="XOR 背景亮度系数（0~1）")

    # Mask2Former
    p.add_argument("--mask2former-root", required=True, help="Mask2Former 项目根目录")
    p.add_argument("--config-mask2former", required=True, help="Mask2Former config yaml")
    p.add_argument("--weights-mask2former", default="", help="Mask2Former 基线权重（可选）")
    p.add_argument("--weights-mask2former-sdf", default="", help="M2F+SDF 权重（可选）")
    p.add_argument("--weights-mask2former-geoloss", required=True, help="M2F+Geo.Loss 权重")
    p.add_argument("--config-mask2former-qsp", default="", help="Mask2Former+QSP config yaml（可选）")
    p.add_argument("--weights-mask2former-qsp", required=True, help="Mask2Former+QSP 权重")
    p.add_argument("--prior-path-mask2former-qsp", default="", help="可选：覆盖 Mask2Former+QSP 的 PRIOR_PATH")

    # PointRend
    p.add_argument("--config-pointrend", required=True, help="PointRend config yaml")
    p.add_argument("--weights-base", required=True, help="PointRend(Base) 权重")
    p.add_argument("--weights-spg", default="", help="SPG-PointRend 权重（可选）")
    p.add_argument("--shape-prior-npy", default="", help="shape prior .npy（可选）")

    # Mask R-CNN
    p.add_argument("--config-maskrcnn", required=True, help="Mask R-CNN config yaml")
    p.add_argument("--weights-maskrcnn", required=True, help="Mask R-CNN 权重")

    # MaskTransfiner（可选）
    p.add_argument("--transfiner-root", default="", help="transfiner 项目根目录（可选）")
    p.add_argument("--config-transfiner", default="", help="transfiner config yaml（可选）")
    p.add_argument("--weights-transfiner", default="", help="transfiner 权重（可选）")
    return p.parse_args()


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
        picks = list(cand)

    if str(getattr(args, "index_range", "")).strip():
        a, b = base_vis._parse_index_range(str(args.index_range).strip())
        base = int(getattr(args, "index_base", 1))
        if base == 1:
            a0 = max(0, a - 1)
            b0 = max(0, b - 1)
        else:
            a0 = max(0, a)
            b0 = max(0, b)
        picks = picks[a0 : b0 + 1]
    else:
        picks = picks[: max(1, int(args.num_rows))]
    if not picks:
        raise RuntimeError("no valid images found for visualization")

    out_dir = os.path.abspath(args.out_dir)
    base_vis._ensure_dir(out_dir)
    fp_color = base_vis._parse_bgr_triplet(args.xor_fp_color)
    fn_color = base_vis._parse_bgr_triplet(args.xor_fn_color)

    # predictors
    maskrcnn_pred = base_vis._build_maskrcnn_predictor(
        config_file=os.path.abspath(str(args.config_maskrcnn).strip()),
        weights=os.path.abspath(str(args.weights_maskrcnn).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    base_pred = base_vis._build_pointrend_predictor(
        config_file=os.path.abspath(args.config_pointrend),
        weights=os.path.abspath(args.weights_base),
        mask_head_name="PointRendMaskHead",
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    m2f_geoloss_pred = base_vis._build_mask2former_predictor(
        mask2former_root=args.mask2former_root,
        config_file=os.path.abspath(args.config_mask2former),
        weights=os.path.abspath(args.weights_mask2former_geoloss),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    m2f_base_pred = None
    if str(getattr(args, "weights_mask2former", "")).strip():
        m2f_base_pred = base_vis._build_mask2former_predictor(
            mask2former_root=args.mask2former_root,
            config_file=os.path.abspath(args.config_mask2former),
            weights=os.path.abspath(str(args.weights_mask2former).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )
    m2f_sdf_pred = None
    if str(getattr(args, "weights_mask2former_sdf", "")).strip():
        m2f_sdf_pred = base_vis._build_mask2former_predictor(
            mask2former_root=args.mask2former_root,
            config_file=os.path.abspath(args.config_mask2former),
            weights=os.path.abspath(str(args.weights_mask2former_sdf).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )
    config_mask2former_qsp = str(getattr(args, "config_mask2former_qsp", "")).strip() or str(
        getattr(args, "config_mask2former", "")
    ).strip()
    if not config_mask2former_qsp:
        raise ValueError("提供 --weights-mask2former-qsp 时必须同时提供 --config-mask2former-qsp 或 --config-mask2former")
    shape_prior_npy = str(getattr(args, "shape_prior_npy", "")).strip()
    if shape_prior_npy:
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(shape_prior_npy)
    qsp_prior_override = str(getattr(args, "prior_path_mask2former_qsp", "")).strip() or shape_prior_npy
    m2f_qsp_pred = base_vis._build_mask2former_predictor(
        mask2former_root=args.mask2former_root,
        config_file=os.path.abspath(config_mask2former_qsp),
        weights=os.path.abspath(str(args.weights_mask2former_qsp).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
        prior_path_override=os.path.abspath(qsp_prior_override) if qsp_prior_override else "",
    )
    spg_pred = None
    if str(getattr(args, "weights_spg", "")).strip():
        spg_pred = base_vis._build_pointrend_predictor(
            config_file=os.path.abspath(args.config_pointrend),
            weights=os.path.abspath(str(args.weights_spg).strip()),
            mask_head_name="ShapeAwareCoarseMaskHead",
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )

    tf_root = str(getattr(args, "transfiner_root", "")).strip()
    tf_cfg = str(getattr(args, "config_transfiner", "")).strip()
    tf_w = str(getattr(args, "weights_transfiner", "")).strip()
    has_tf = bool(tf_root or tf_cfg or tf_w)
    if has_tf and not (tf_root and tf_cfg and tf_w):
        raise ValueError("启用 MaskTransfiner 时必须同时提供 --transfiner-root --config-transfiner --weights-transfiner")
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
    error_rows: List[dict] = []
    column_titles: List[str] = []
    for i, d in enumerate(picks):
        img_path = d["file_name"]
        img_bgr = base_vis._safe_imread(img_path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        gt = base_vis._gt_mask_from_dict(d, h=h, w=w)
        if gt is None:
            continue

        # 直接测试：输入图不做任何高光修正
        input_bgr = img_bgr

        base_mask, _ = base_vis._predict_best_mask(base_pred, img_bgr)
        maskrcnn_mask, _ = base_vis._predict_best_mask(maskrcnn_pred, img_bgr)
        m2f_geoloss_mask, _ = base_vis._predict_best_mask(m2f_geoloss_pred, img_bgr)
        m2f_base_mask = None
        if m2f_base_pred is not None:
            m2f_base_mask, _ = base_vis._predict_best_mask(m2f_base_pred, img_bgr)
        m2f_sdf_mask = None
        if m2f_sdf_pred is not None:
            m2f_sdf_mask, _ = base_vis._predict_best_mask(m2f_sdf_pred, img_bgr)
        m2f_qsp_mask, _ = base_vis._predict_best_mask(m2f_qsp_pred, img_bgr)
        spg_mask = None
        if spg_pred is not None:
            spg_mask, _ = base_vis._predict_best_mask(spg_pred, img_bgr)
        tf_mask = transfiner_mask_map.get(os.path.abspath(img_path), None) if has_tf else None

        # fill empty with zeros
        if base_mask is None:
            base_mask = np.zeros_like(gt)
        if maskrcnn_mask is None:
            maskrcnn_mask = np.zeros_like(gt)
        if m2f_geoloss_mask is None:
            m2f_geoloss_mask = np.zeros_like(gt)
        if m2f_base_pred is not None and m2f_base_mask is None:
            m2f_base_mask = np.zeros_like(gt)
        if m2f_sdf_pred is not None and m2f_sdf_mask is None:
            m2f_sdf_mask = np.zeros_like(gt)
        if m2f_qsp_mask is None:
            m2f_qsp_mask = np.zeros_like(gt)
        if spg_pred is not None and spg_mask is None:
            spg_mask = np.zeros_like(gt)
        if has_tf and (tf_mask is None):
            tf_mask = np.zeros_like(gt)
        if has_tf and tf_mask is not None and tf_mask.shape != gt.shape:
            tf_mask = cv2.resize(tf_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        ep = int(getattr(args, "pred_erode", 0))
        base_mask_vis = base_vis._erode_mask(base_mask, ep)
        maskrcnn_mask_vis = base_vis._erode_mask(maskrcnn_mask, ep)
        m2f_geoloss_mask_vis = base_vis._erode_mask(m2f_geoloss_mask, ep)
        m2f_base_mask_vis = base_vis._erode_mask(m2f_base_mask, ep) if (m2f_base_mask is not None) else None
        m2f_sdf_mask_vis = base_vis._erode_mask(m2f_sdf_mask, ep) if (m2f_sdf_mask is not None) else None
        m2f_qsp_mask_vis = base_vis._erode_mask(m2f_qsp_mask, ep)
        spg_mask_vis = base_vis._erode_mask(spg_mask, ep) if (spg_mask is not None) else None
        tf_mask_vis = base_vis._erode_mask(tf_mask, ep) if (tf_mask is not None) else None

        input_vis = input_bgr.copy()
        gt_vis = base_vis._overlay_mask(input_bgr, gt, color_bgr=(0, 255, 0), alpha=0.35, outline=True)

        method_masks: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]] = [
            ("Mask R-CNN", maskrcnn_mask, maskrcnn_mask_vis, (120, 200, 255)),
            ("PointRend", base_mask, base_mask_vis, (255, 170, 120)),
            ("Mask2Former+Geo.Loss", m2f_geoloss_mask, m2f_geoloss_mask_vis, (255, 200, 80)),
            ("QSP-Mask2Former (ours)", m2f_qsp_mask, m2f_qsp_mask_vis, (80, 210, 160)),
        ]
        if spg_mask is not None and spg_mask_vis is not None:
            method_masks.insert(2, ("SPG-PointRend", spg_mask, spg_mask_vis, (110, 215, 120)))
        if m2f_base_mask is not None and m2f_base_mask_vis is not None:
            method_masks.insert(-2, ("Mask2Former", m2f_base_mask, m2f_base_mask_vis, (190, 120, 235)))
        if m2f_sdf_mask is not None and m2f_sdf_mask_vis is not None:
            method_masks.insert(-1, ("Mask2Former+SDF", m2f_sdf_mask, m2f_sdf_mask_vis, (235, 150, 210)))
        if has_tf and tf_mask is not None and tf_mask_vis is not None:
            method_masks.insert(1, ("MaskTransfiner", tf_mask, tf_mask_vis, (200, 120, 255)))

        cells: List[np.ndarray] = [input_vis, gt_vis]
        local_titles: List[str] = ["Raw", "GT"]
        for method_name, raw_mask, vis_mask, color_bgr in method_masks:
            pred_vis = base_vis._overlay_mask(input_bgr, vis_mask, color_bgr=color_bgr, alpha=0.35, outline=True)
            if str(args.mode) == "pred_only":
                cells.append(pred_vis)
                local_titles.append(method_name)
            else:
                xor_img, fp_px, fn_px = base_vis._xor_vis(
                    input_bgr,
                    gt,
                    raw_mask,
                    fp_color=fp_color,
                    fn_color=fn_color,
                    alpha=float(args.xor_alpha),
                    bg_dim=float(args.xor_bg_dim),
                )
                xor_vis = xor_img
                cells.extend([pred_vis, xor_vis])
                local_titles.extend([f"{method_name} Pred", f"{method_name} XOR"])
                error_rows.append(
                    {
                        "image_path": str(img_path),
                        "method": method_name,
                        "fp_pixels": int(fp_px),
                        "fn_pixels": int(fn_px),
                        "xor_pixels": int(fp_px + fn_px),
                        "gt_pixels": int((gt > 0).sum()),
                        "pred_pixels": int((raw_mask > 0).sum()),
                    }
                )

        if not column_titles:
            column_titles = list(local_titles)

        cells = [base_vis._resize_to_width(c, int(args.cell_width)) for c in cells]
        row = base_vis._stack_row(cells)
        if not bool(args.no_row_tags):
            row = _add_row_tag(row, f"S{i + 1}")
        rows_img.append(row)

        stem = f"{i:02d}_" + os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_row.png"), row)

    if not rows_img:
        raise RuntimeError("no valid rows were generated")

    grid_body = base_vis._stack_grid(rows_img)
    if bool(args.no_header):
        grid = grid_body
    else:
        header = _make_header_row(column_titles, int(args.cell_width), header_h=54)
        grid = np.concatenate([header, grid_body], axis=0)
    if str(args.mode) == "pred_only":
        out_path = os.path.join(out_dir, "table_layout_grid_direct.png")
    else:
        out_path = os.path.join(out_dir, "table_layout_grid_direct_xor.png")
    cv2.imwrite(out_path, grid)
    print(f"saved: {out_path}")

    if str(args.mode) == "pred_xor":
        stats_csv = os.path.join(out_dir, "per_image_error_stats.csv")
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["image_path", "method", "fp_pixels", "fn_pixels", "xor_pixels", "gt_pixels", "pred_pixels"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in error_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"saved: {stats_csv}")


if __name__ == "__main__":
    main()

