#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.6a 专用：门控权重（Sigmoid 后）可视化与统计

输出（每张图）：
- input_image.png
- gate_weight_sigmoid.png            # [0,1] 灰度热图（论文建议对象）
- gate_weight_colormap.png           # 伪彩热图
- gate_weight_overlay.png            # 叠加到输入图
- boundary_ring.png                  # 目标边界环区域
- fig6a_panel.png                    # 拼图（输入/灰度/叠加）
- fig6a_metrics.txt                  # 统计（高光、边界、交集区域）
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

import visualize_mask2former_qsp_flow as flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fig.6a gate-weight visualization (sigmoid gate map + overlay).")
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", default="", help="single image path")
    p.add_argument("--images", default="", help="comma-separated image paths")
    p.add_argument("--image-list", default="", help="txt file with one image path per line")
    p.add_argument("--dataset-root", default="", help="optional image root for batch export")
    p.add_argument("--scan-max", type=int, default=200)
    p.add_argument("--index-range", default="", help="e.g. 36-40")
    p.add_argument("--index-base", type=int, default=1, choices=(0, 1))
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--num-images", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--prior-path", default="", help="override MODEL.MASK_FORMER.PRIOR_PATH")
    p.add_argument("--num-classes", type=int, default=1)

    p.add_argument("--overlay-alpha", type=float, default=0.45, help="gate overlay alpha")
    p.add_argument(
        "--gate-colormap",
        choices=["jet", "hot", "autumn", "turbo", "inferno", "magma", "plasma", "viridis"],
        default="viridis",
        help="门控热图伪彩风格（影响 gate_weight_colormap/overlay）",
    )
    p.add_argument(
        "--gate-contrast-mode",
        choices=["global", "minmax", "percentile"],
        default="percentile",
        help="仅用于伪彩/叠加图的对比增强方式（global=直接[0,1]）",
    )
    p.add_argument("--gate-p-low", type=float, default=2.0, help="percentile 下限（仅 percentile 模式生效）")
    p.add_argument("--gate-p-high", type=float, default=98.0, help="percentile 上限（仅 percentile 模式生效）")
    p.add_argument(
        "--gate-gamma",
        type=float,
        default=1.35,
        help="伪彩图 gamma（>1 压暗背景、增强高响应对比；仅影响伪彩/叠加）",
    )
    p.add_argument("--highlight-v-thr", type=int, default=245, help="HSV-V threshold for highlight mask")
    p.add_argument("--highlight-s-max", type=int, default=70, help="HSV-S max for highlight mask")
    p.add_argument("--boundary-width", type=int, default=3, help="boundary ring width (pixels)")
    p.add_argument("--topk", type=int, default=0, help="按 ratio_highlight_boundary_vs_non 自动导出 Top-K（0=关闭）")
    p.add_argument("--topk-copy-panels", action="store_true", help="开启后把 Top-K 的 fig6a_panel.png 复制到独立目录")
    p.add_argument(
        "--paired-n",
        type=int,
        default=10,
        help="用于配对统计可视化的样本数（展示原始点+均值+95%CI，0=关闭）",
    )
    p.add_argument(
        "--paired-sort-by",
        choices=["input_order", "ratio_highlight_boundary_vs_non"],
        default="input_order",
        help="配对样本选择顺序",
    )
    return p.parse_args()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _title_bar(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    bar_h = min(34, max(20, h // 6))
    cv2.rectangle(out, (0, 0), (w - 1, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        str(title),
        (8, max(16, bar_h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _to_u8_gray(arr01: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr01, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    a = np.clip(a, 0.0, 1.0)
    return np.round(a * 255.0).astype(np.uint8)


def _normalize_for_visual(
    arr01: np.ndarray,
    mode: str,
    p_low: float,
    p_high: float,
    gamma: float,
) -> np.ndarray:
    a = np.nan_to_num(arr01, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    a = np.clip(a, 0.0, 1.0)

    m = str(mode).strip().lower()
    if m == "minmax":
        mn = float(np.min(a))
        mx = float(np.max(a))
        if mx - mn > 1e-8:
            a = (a - mn) / (mx - mn)
        else:
            a = np.zeros_like(a, dtype=np.float32)
    elif m == "percentile":
        lo = float(np.percentile(a, float(p_low)))
        hi = float(np.percentile(a, float(p_high)))
        if hi - lo > 1e-8:
            a = (a - lo) / (hi - lo)
        else:
            a = np.zeros_like(a, dtype=np.float32)
        a = np.clip(a, 0.0, 1.0)

    g = float(max(1e-6, gamma))
    a = np.power(np.clip(a, 0.0, 1.0), g).astype(np.float32, copy=False)
    return np.clip(a, 0.0, 1.0)


def _apply_colormap(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    name = str(cmap_name).strip().lower()
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "autumn": cv2.COLORMAP_AUTUMN,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    cm = cmap_map.get(name, cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray_u8, cm)


def _overlay_colormap(
    bgr: np.ndarray,
    arr01: np.ndarray,
    alpha: float,
    cmap_name: str,
    contrast_mode: str,
    p_low: float,
    p_high: float,
    gamma: float,
) -> np.ndarray:
    a = float(max(0.0, min(1.0, alpha)))
    vis = _normalize_for_visual(arr01, mode=contrast_mode, p_low=p_low, p_high=p_high, gamma=gamma)
    gray = _to_u8_gray(vis)
    heat = _apply_colormap(gray, cmap_name)
    out = bgr.copy().astype(np.float32)
    out = out * (1.0 - a) + heat.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _boundary_ring(mask01: np.ndarray, width: int) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m)
    k = max(1, int(width))
    kernel = np.ones((3, 3), dtype=np.uint8)
    d = cv2.dilate(m, kernel, iterations=k)
    e = cv2.erode(m, kernel, iterations=k)
    ring = ((d - e) > 0).astype(np.uint8)
    return ring


def _filter_object_components(mask01: np.ndarray, min_area_abs: int = 400, min_area_ratio_to_max: float = 0.08) -> np.ndarray:
    """
    过滤预测掩码中的微小孤立连通域，避免边界可视化出现“小蓝圈”。
    保留面积 >= max(min_area_abs, max_area * min_area_ratio_to_max) 的连通域。
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m

    areas: List[int] = []
    for lab in range(1, num_labels):
        areas.append(int(stats[lab, cv2.CC_STAT_AREA]))
    max_area = max(areas) if areas else 0
    thr = int(max(float(min_area_abs), float(max_area) * float(min_area_ratio_to_max)))

    kept = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= thr:
            kept[labels == lab] = 1
    return kept


def _remove_small_components(mask01: np.ndarray, min_area: int = 20) -> np.ndarray:
    mask = (mask01 > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            kept[labels == lab] = 1
    return kept


def _draw_degraded_region_annotation(
    img_bgr: np.ndarray,
    highlight_mask01: np.ndarray,
    boundary_mask01: np.ndarray,
    inter_mask01: np.ndarray,
) -> np.ndarray:
    """
    生成 degraded-region annotation：
    - 黄色半透明填充：highlight
    - 蓝色线：boundary band
    - 粉色线：highlight ∩ boundary
    """
    out = img_bgr.copy().astype(np.float32)
    hi = (highlight_mask01 > 0).astype(np.uint8)
    bd = (boundary_mask01 > 0).astype(np.uint8)
    inter = (inter_mask01 > 0).astype(np.uint8)

    # Yellow fill for highlight
    hi_alpha = 0.30
    yellow_bgr = np.array([0.0, 255.0, 255.0], dtype=np.float32)
    m_hi = hi > 0
    out[m_hi] = out[m_hi] * (1.0 - hi_alpha) + yellow_bgr * hi_alpha
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    # 直接按像素掩码着色，避免轮廓法产生“偏移/重复线”
    # 仅去掉极小边界连通域，避免误删插头下半部分主边界。
    kernel = np.ones((3, 3), dtype=np.uint8)
    bd = _remove_small_components(bd, min_area=28)

    # 统一线宽：先构造蓝色边界线，再把“交集在线上”的部分用固定 thickness 重绘为粉线覆盖蓝线。
    bd_line = cv2.dilate(bd, kernel, iterations=1)
    inter_on_bd = ((inter > 0) & (bd_line > 0)).astype(np.uint8)
    # 先轻微平滑，再按轮廓固定线宽重绘，避免末端粗细不均。
    pink_seed = cv2.morphologyEx(inter_on_bd, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours_inter, _ = cv2.findContours(pink_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inter_line = np.zeros_like(pink_seed, dtype=np.uint8)
    if contours_inter:
        cv2.drawContours(inter_line, contours_inter, -1, 1, thickness=7, lineType=cv2.LINE_AA)
    bd_only = ((bd_line > 0) & (inter_line == 0)).astype(np.uint8)
    inter_only = (inter_line > 0).astype(np.uint8)

    out_u8[bd_only > 0] = np.array([255, 0, 0], dtype=np.uint8)      # blue
    out_u8[inter_only > 0] = np.array([180, 105, 255], dtype=np.uint8)  # pink (override blue)

    return out_u8


def _safe_mean(arr: np.ndarray, mask01: np.ndarray) -> float:
    m = (mask01 > 0)
    if not np.any(m):
        return float("nan")
    return float(arr[m].mean())


def _sort_ratio_desc(x: dict) -> float:
    v = x.get("ratio_highlight_boundary_vs_non", float("nan"))
    try:
        fv = float(v)
    except Exception:
        return float("-inf")
    if not np.isfinite(fv):
        return float("-inf")
    return fv


def _to_float(v) -> float:
    try:
        f = float(v)
    except Exception:
        return float("nan")
    if not np.isfinite(f):
        return float("nan")
    return f


def _mean_ci95(vals: List[float], z: float = 1.96) -> Tuple[float, float]:
    arr = np.asarray([_to_float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    half = float(z * std / np.sqrt(arr.size))
    return mean, half


def _stem_keep_date_only(image_path: str) -> str:
    """
    将文件名 stem 规整为“到日期为止”，去掉后续时间戳，便于重复生成时稳定覆盖。
    例：mecheye_2d_20260304_095725_717945 -> mecheye_2d_20260304
    """
    stem = os.path.splitext(os.path.basename(str(image_path)))[0]
    parts = stem.split("_")
    out: List[str] = []
    date_found = False
    for p in parts:
        out.append(p)
        if re.fullmatch(r"\d{8}", p):
            date_found = True
            break
    if date_found and out:
        return "_".join(out)
    return stem


def _extract_gate_and_mask(predictor, img_bgr: np.ndarray):
    model = predictor.model
    model.eval()

    with torch.no_grad():
        original_image = img_bgr
        if predictor.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = predictor.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(model.device)

        inputs = [{"image": image, "height": height, "width": width}]
        images = [x["image"].to(model.device) for x in inputs]
        images = [(x - model.pixel_mean) / model.pixel_std for x in images]
        images = flow.ImageList.from_tensors(images, model.size_divisibility)

        features = model.backbone(images.tensor)
        sem_seg_head = model.sem_seg_head
        mask_features, transformer_encoder_features, multi_scale_features = sem_seg_head.pixel_decoder.forward_features(features)
        if sem_seg_head.transformer_in_feature == "multi_scale_pixel_decoder":
            outputs = sem_seg_head.predictor(multi_scale_features, mask_features, None)
        elif sem_seg_head.transformer_in_feature == "transformer_encoder":
            assert transformer_encoder_features is not None
            outputs = sem_seg_head.predictor(transformer_encoder_features, mask_features, None)
        elif sem_seg_head.transformer_in_feature == "pixel_embedding":
            outputs = sem_seg_head.predictor(mask_features, mask_features, None)
        else:
            outputs = sem_seg_head.predictor(features[sem_seg_head.transformer_in_feature], mask_features, None)

        if "pred_prior_gates" not in outputs:
            raise RuntimeError("Model outputs do not contain pred_prior_gates. 请确认是启用 QSP 的配置。")

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        prior_gate_results = outputs.get("pred_prior_gates", None)

        mask_pred_results = torch.nn.functional.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        if prior_gate_results is not None and prior_gate_results.ndim == 4:
            prior_gate_results = torch.nn.functional.interpolate(
                prior_gate_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        image_size = images.image_sizes[0]
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = flow.sem_seg_postprocess(mask_pred_results[0], image_size, height, width)
        mask_cls_result = mask_cls_result.to(mask_pred_result)
        q_idx, best_mask, best_score = flow._select_best_query(model, mask_cls_result, mask_pred_result)

        if prior_gate_results is not None and prior_gate_results.ndim == 4:
            gate_map = flow.sem_seg_postprocess(prior_gate_results[0], image_size, height, width)[q_idx]
            gate_map = gate_map.detach().float().cpu().numpy()
            gate_map = np.clip(gate_map, 0.0, 1.0)
        elif prior_gate_results is not None:
            gate_mean = float(prior_gate_results[0, q_idx, 0].detach().float().cpu().item())
            gate_map = np.full((height, width), fill_value=np.clip(gate_mean, 0.0, 1.0), dtype=np.float32)
        else:
            gate_map = np.zeros((height, width), dtype=np.float32)

        final_mask = best_mask.detach().float().cpu().numpy()
        return gate_map.astype(np.float32), final_mask.astype(np.float32), int(q_idx), float(best_score)


def _save_case(
    *,
    img_path: str,
    out_dir: str,
    gate_map: np.ndarray,
    final_mask: np.ndarray,
    q_idx: int,
    score: float,
    overlay_alpha: float,
    gate_colormap: str,
    gate_contrast_mode: str,
    gate_p_low: float,
    gate_p_high: float,
    gate_gamma: float,
    highlight_v_thr: int,
    highlight_s_max: int,
    boundary_width: int,
) -> dict:
    _ensure_dir(out_dir)
    img_bgr = flow._imread_bgr(img_path)
    flow._imwrite(os.path.join(out_dir, "input_image.png"), img_bgr)

    # 保留原始 Sigmoid 灰度（不做对比增强，便于跨样本可比）
    gate_u8_raw = _to_u8_gray(gate_map)
    flow._imwrite(os.path.join(out_dir, "gate_weight_sigmoid.png"), gate_u8_raw)
    # 仅在伪彩/叠加图上做可视化增强，提高目标与背景对比
    gate_vis01 = _normalize_for_visual(
        gate_map,
        mode=gate_contrast_mode,
        p_low=float(gate_p_low),
        p_high=float(gate_p_high),
        gamma=float(gate_gamma),
    )
    gate_u8_vis = _to_u8_gray(gate_vis01)
    gate_colormap_bgr = _apply_colormap(gate_u8_vis, gate_colormap)
    flow._imwrite(os.path.join(out_dir, "gate_weight_colormap.png"), gate_colormap_bgr)

    overlay = _overlay_colormap(
        img_bgr,
        gate_map,
        alpha=float(overlay_alpha),
        cmap_name=gate_colormap,
        contrast_mode=gate_contrast_mode,
        p_low=float(gate_p_low),
        p_high=float(gate_p_high),
        gamma=float(gate_gamma),
    )
    flow._imwrite(os.path.join(out_dir, "gate_weight_overlay.png"), overlay)

    hi = flow._highlight_mask(img_bgr, v_thr=int(highlight_v_thr), s_max=int(highlight_s_max)).astype(np.uint8)
    final_mask_clean = _filter_object_components(final_mask, min_area_abs=400, min_area_ratio_to_max=0.08)
    bd = _boundary_ring(final_mask_clean, width=int(boundary_width))
    hibd = ((hi > 0) & (bd > 0)).astype(np.uint8)
    non = ((hi == 0) & (bd == 0)).astype(np.uint8)

    flow._imwrite(os.path.join(out_dir, "boundary_ring.png"), bd * 255)
    flow._imwrite(os.path.join(out_dir, "highlight_mask.png"), hi * 255)
    flow._imwrite(os.path.join(out_dir, "highlight_boundary_intersection.png"), hibd * 255)
    degraded_anno = _draw_degraded_region_annotation(img_bgr, hi, bd, hibd)
    flow._imwrite(os.path.join(out_dir, "degraded_region_annotation.png"), degraded_anno)

    g_all = float(gate_map.mean())
    g_hi = _safe_mean(gate_map, hi)
    g_bd = _safe_mean(gate_map, bd)
    g_hibd = _safe_mean(gate_map, hibd)
    g_non = _safe_mean(gate_map, non)

    def _ratio(a: float, b: float) -> float:
        if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(b) < 1e-12:
            return float("nan")
        return float(a / b)

    r_hi = _ratio(g_hi, g_non)
    r_bd = _ratio(g_bd, g_non)
    r_hibd = _ratio(g_hibd, g_non)

    # simple panel
    panel_cells = [
        _title_bar(img_bgr, "Input"),
        _title_bar(degraded_anno, "Degraded-region Annotation"),
        _title_bar(gate_colormap_bgr, "Gating Heatmap"),
        _title_bar(overlay, "Gate Overlay"),
    ]
    panel_cells = [flow._resize_to(c, (360, 360)) for c in panel_cells]
    panel = np.concatenate(panel_cells, axis=1)
    flow._imwrite(os.path.join(out_dir, "fig6a_panel.png"), panel)

    metrics_txt = os.path.join(out_dir, "fig6a_metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"image={os.path.abspath(img_path)}\n")
        f.write(f"query_idx={q_idx}\n")
        f.write(f"instance_score={score}\n")
        f.write(f"highlight_pixels={int(hi.sum())}\n")
        f.write(f"boundary_pixels={int(bd.sum())}\n")
        f.write(f"highlight_boundary_pixels={int(hibd.sum())}\n")
        f.write(f"gate_mean_all={g_all}\n")
        f.write(f"gate_mean_highlight={g_hi}\n")
        f.write(f"gate_mean_boundary={g_bd}\n")
        f.write(f"gate_mean_highlight_boundary={g_hibd}\n")
        f.write(f"gate_mean_non_highlight_non_boundary={g_non}\n")
        f.write(f"ratio_highlight_vs_non={r_hi}\n")
        f.write(f"ratio_boundary_vs_non={r_bd}\n")
        f.write(f"ratio_highlight_boundary_vs_non={r_hibd}\n")
        f.write("semantics=If gate mechanism works, ratios > 1 are expected in highlight/boundary regions.\n")

    return {
        "image": os.path.abspath(img_path),
        "out_dir": os.path.abspath(out_dir),
        "query_idx": int(q_idx),
        "instance_score": float(score),
        "gate_mean_all": float(g_all),
        "highlight_pixels": int(hi.sum()),
        "boundary_pixels": int(bd.sum()),
        "highlight_boundary_pixels": int(hibd.sum()),
        "gate_mean_highlight": float(g_hi),
        "gate_mean_boundary": float(g_bd),
        "gate_mean_highlight_boundary": float(g_hibd),
        "gate_mean_non_highlight_non_boundary": float(g_non),
        "ratio_highlight_vs_non": float(r_hi),
        "ratio_boundary_vs_non": float(r_bd),
        "ratio_highlight_boundary_vs_non": float(r_hibd),
    }


def _export_paired_stats(rows: List[dict], out_dir: str, paired_n: int, sort_by: str) -> None:
    n = int(max(0, paired_n))
    if n <= 0 or not rows:
        return

    if sort_by == "ratio_highlight_boundary_vs_non":
        ordered = sorted(rows, key=_sort_ratio_desc, reverse=True)
    else:
        ordered = list(rows)

    selected = ordered[:n]
    if not selected:
        return

    paired_csv = os.path.join(out_dir, "fig6a_paired10_points.csv")
    paired_fields = [
        "sample_rank",
        "image",
        "gate_mean_non_highlight_non_boundary",
        "gate_mean_highlight",
        "gate_mean_boundary",
        "gate_mean_highlight_boundary",
        "ratio_highlight_vs_non",
        "ratio_boundary_vs_non",
        "ratio_highlight_boundary_vs_non",
    ]
    with open(paired_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=paired_fields)
        w.writeheader()
        for i, r in enumerate(selected, start=1):
            row = {"sample_rank": i}
            for k in paired_fields[1:]:
                row[k] = r.get(k, "")
            w.writerow(row)
    print(f"saved: {paired_csv}")

    comps = [
        ("highlight vs non", "gate_mean_highlight", "gate_mean_non_highlight_non_boundary"),
        ("boundary vs non", "gate_mean_boundary", "gate_mean_non_highlight_non_boundary"),
        ("highlight∩boundary vs non", "gate_mean_highlight_boundary", "gate_mean_non_highlight_non_boundary"),
    ]
    lines: List[str] = []
    lines.append(f"selected_samples={len(selected)}")
    lines.append(f"sort_by={sort_by}")
    lines.append("")
    for title, key_a, key_b in comps:
        va = [_to_float(r.get(key_a, float("nan"))) for r in selected]
        vb = [_to_float(r.get(key_b, float("nan"))) for r in selected]
        valid = [(a, b) for a, b in zip(va, vb) if np.isfinite(a) and np.isfinite(b)]
        arr_a = [x[0] for x in valid]
        arr_b = [x[1] for x in valid]
        mean_a, ci_a = _mean_ci95(arr_a)
        mean_b, ci_b = _mean_ci95(arr_b)
        lines.append(f"[{title}]")
        lines.append(f"valid_pairs={len(valid)}")
        lines.append(f"mean_{key_a}={mean_a:.6f}, ci95_half={ci_a:.6f}")
        lines.append(f"mean_{key_b}={mean_b:.6f}, ci95_half={ci_b:.6f}")
        lines.append("")

    stats_txt = os.path.join(out_dir, "fig6a_paired10_stats.txt")
    with open(stats_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    print(f"saved: {stats_txt}")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip paired plot: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (title, key_a, key_b) in zip(axes, comps):
        pairs: List[Tuple[float, float]] = []
        for r in selected:
            a = _to_float(r.get(key_a, float("nan")))
            b = _to_float(r.get(key_b, float("nan")))
            if np.isfinite(a) and np.isfinite(b):
                pairs.append((a, b))

        for a, b in pairs:
            ax.plot([0, 1], [b, a], color="#b0b0b0", linewidth=1.0, alpha=0.8, zorder=1)
            ax.scatter([0, 1], [b, a], color="#4c78a8", s=14, alpha=0.9, zorder=2)

        arr_a = [x[0] for x in pairs]
        arr_b = [x[1] for x in pairs]
        mean_a, ci_a = _mean_ci95(arr_a)
        mean_b, ci_b = _mean_ci95(arr_b)

        if np.isfinite(mean_b):
            ax.errorbar(
                [0],
                [mean_b],
                yerr=[[ci_b], [ci_b]],
                fmt="s",
                color="#d62728",
                markersize=5,
                linewidth=1.8,
                capsize=4,
                zorder=3,
            )
        if np.isfinite(mean_a):
            ax.errorbar(
                [1],
                [mean_a],
                yerr=[[ci_a], [ci_a]],
                fmt="s",
                color="#d62728",
                markersize=5,
                linewidth=1.8,
                capsize=4,
                zorder=3,
            )

        ax.set_xticks([0, 1], ["non", "region"])
        ax.set_title(f"{title}\n(n={len(pairs)})", fontsize=10)
        ax.set_ylabel("mean gate value")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("Paired samples: raw points, mean and 95% CI", fontsize=12)
    paired_png = os.path.join(out_dir, "fig6a_paired10_mean_ci.png")
    fig.savefig(paired_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {paired_png}")


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)
    image_paths = flow._resolve_image_paths(args)
    if not image_paths:
        raise ValueError("未解析到输入图片。请提供 --image / --images / --image-list / --dataset-root。")

    cfg = flow._build_cfg(
        os.path.abspath(args.config_file),
        os.path.abspath(args.weights),
        float(args.score_thr),
        str(args.prior_path).strip(),
        int(args.num_classes),
    )
    predictor = flow.DefaultPredictor(cfg)
    flow.DetectionCheckpointer(predictor.model).load(cfg.MODEL.WEIGHTS)

    rows: List[dict] = []
    multi = len(image_paths) > 1
    for i, img_path in enumerate(image_paths):
        if multi:
            stem = _stem_keep_date_only(img_path)
            case_dir = os.path.join(args.out_dir, f"{i:03d}_{stem}")
        else:
            case_dir = args.out_dir

        img_bgr = flow._imread_bgr(img_path)
        gate_map, final_mask, q_idx, score = _extract_gate_and_mask(predictor, img_bgr)
        row = _save_case(
            img_path=img_path,
            out_dir=case_dir,
            gate_map=gate_map,
            final_mask=final_mask,
            q_idx=q_idx,
            score=score,
            overlay_alpha=float(args.overlay_alpha),
            gate_colormap=str(args.gate_colormap),
            gate_contrast_mode=str(args.gate_contrast_mode),
            gate_p_low=float(args.gate_p_low),
            gate_p_high=float(args.gate_p_high),
            gate_gamma=float(args.gate_gamma),
            highlight_v_thr=int(args.highlight_v_thr),
            highlight_s_max=int(args.highlight_s_max),
            boundary_width=int(args.boundary_width),
        )
        rows.append(row)
        print(f"[{i+1}/{len(image_paths)}] done: {row['out_dir']}")

    summary_csv = os.path.join(args.out_dir, "fig6a_gate_summary.csv")
    fields = [
        "image",
        "out_dir",
        "query_idx",
        "instance_score",
        "gate_mean_all",
        "highlight_pixels",
        "boundary_pixels",
        "highlight_boundary_pixels",
        "gate_mean_highlight",
        "gate_mean_boundary",
        "gate_mean_highlight_boundary",
        "gate_mean_non_highlight_non_boundary",
        "ratio_highlight_vs_non",
        "ratio_boundary_vs_non",
        "ratio_highlight_boundary_vs_non",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"saved: {summary_csv}")

    _export_paired_stats(
        rows=rows,
        out_dir=args.out_dir,
        paired_n=int(args.paired_n),
        sort_by=str(args.paired_sort_by),
    )

    topk = int(max(0, args.topk))
    if topk > 0 and rows:
        ranked = sorted(rows, key=_sort_ratio_desc, reverse=True)
        ranked = [r for r in ranked if np.isfinite(float(r.get("ratio_highlight_boundary_vs_non", float("nan"))))]
        ranked = ranked[:topk]
        topk_csv = os.path.join(args.out_dir, "fig6a_topk_by_ratio.csv")
        with open(topk_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in ranked:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"saved: {topk_csv}")

        topk_md = os.path.join(args.out_dir, "fig6a_topk_by_ratio.md")
        lines: List[str] = []
        lines.append("| rank | image | ratio_highlight_boundary_vs_non | ratio_highlight_vs_non | ratio_boundary_vs_non | out_dir |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for i, r in enumerate(ranked, start=1):
            lines.append(
                "| {rank} | {image} | {rhb:.6f} | {rh:.6f} | {rb:.6f} | {out_dir} |".format(
                    rank=i,
                    image=os.path.basename(str(r.get("image", ""))),
                    rhb=float(r.get("ratio_highlight_boundary_vs_non", float("nan"))),
                    rh=float(r.get("ratio_highlight_vs_non", float("nan"))),
                    rb=float(r.get("ratio_boundary_vs_non", float("nan"))),
                    out_dir=str(r.get("out_dir", "")),
                )
            )
        with open(topk_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved: {topk_md}")

        if bool(args.topk_copy_panels):
            topk_dir = os.path.join(args.out_dir, "topk_panels")
            _ensure_dir(topk_dir)
            for i, r in enumerate(ranked, start=1):
                src = os.path.join(str(r.get("out_dir", "")), "fig6a_panel.png")
                if os.path.isfile(src):
                    dst = os.path.join(
                        topk_dir,
                        f"{i:02d}_{os.path.splitext(os.path.basename(str(r.get('image', ''))))[0]}_panel.png",
                    )
                    shutil.copy2(src, dst)
            print(f"saved: {topk_dir}")


if __name__ == "__main__":
    main()

