#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流程图全量导出（含项目已有对应项 + 缺失项补生成），按符号命名：

已有对应项（项目中可直接取到）：
- m_raw_logits.png         : M_i^raw (raw mask logits)
- m_raw_sigma.png          : sigma(M_i^raw)
- p_align.png              : P_i^align
- g_i.png                  : g_i (effective gate)
- c_i.png                  : c_i (occluder / suppressor prob)
- m_fuse.png               : M_i^fuse

缺失项（脚本内补生成）：
- p_bank.png               : Prior bank 全部原型拼图（P1..Pk）
- omega_i.csv / omega_i.png: omega_i (bank weights)
- p_mix.png                : P_i^mix = sum_k omega_{i,k} P_k
- residual_cue.png         : sigma(M_i^raw) - P_i^align
- suppression_map.png      : (1 - c_i)
- delta_i.png              : delta_i 近似（由融合方程反解）
- v_i.png                  : V_i 真实图（QueryShapePriorFusion.query_visual_map）
- m_fuse_logits.png        : M_i^fuse (fused mask logits, before sigmoid)
- m_fuse.png               : sigma(M_i^fuse) (fused mask probability)
- m_fuse_minus_raw_logits.png : M_i^fuse - M_i^raw (fusion update)

补充：
- affine_params.txt / missing_panels_meta.txt 记录 a_i 与 A_i 等文本参数。
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional

import numpy as np
import torch

import visualize_mask2former_qsp_flow as flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export missing QSP flow panels.")
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", default="", help="single image path")
    p.add_argument("--images", default="", help="comma-separated image paths")
    p.add_argument("--image-list", default="", help="txt file with one image path per line")
    p.add_argument("--dataset-root", default="", help="optional image root for batch export")
    p.add_argument("--scan-max", type=int, default=200)
    p.add_argument("--index-range", default="")
    p.add_argument("--index-base", type=int, default=1, choices=(0, 1))
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--num-images", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--prior-path", default="", help="override MODEL.MASK_FORMER.PRIOR_PATH")
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--max-side", type=int, default=256, help="panel size")
    p.add_argument(
        "--suppression-vis-stretch",
        choices=["none", "minmax", "percentile"],
        default="percentile",
        help="suppression_map.png 可视化拉伸方式",
    )
    p.add_argument("--suppression-p-low", type=float, default=1.0, help="percentile 下限（percentile 模式）")
    p.add_argument("--suppression-p-high", type=float, default=99.0, help="percentile 上限（percentile 模式）")
    # 兼容现有训练/可视化命令行风格
    p.add_argument("--num-gpus", type=int, default=1, help="reserved for CLI compatibility")
    p.add_argument("--gpu-id", type=int, default=-1, help="GPU index to use, e.g. 0/1; -1 keeps current env")
    p.add_argument("--device", default="", help="override MODEL.DEVICE, e.g. cuda or cpu")
    return p.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _norm01(x: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx - mn < 1e-8):
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def _stretch_for_vis(x01: np.ndarray, mode: str, p_low: float, p_high: float) -> np.ndarray:
    a = np.clip(np.nan_to_num(x01, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False), 0.0, 1.0)
    m = str(mode).strip().lower()
    if m == "minmax":
        return _norm01(a)
    if m == "percentile":
        lo = float(np.percentile(a, float(p_low)))
        hi = float(np.percentile(a, float(p_high)))
        if hi - lo < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    return a


def _save_heat(name: str, arr01: np.ndarray, out_dir: str, max_side: int) -> None:
    flow._save_tensor_heat(name, np.clip(arr01, 0.0, 1.0), out_dir, max_side)


def _save_signed(name: str, arr: np.ndarray, out_dir: str, max_side: int) -> None:
    flow._save_signed_heat(name, arr.astype(np.float32, copy=False), out_dir, max_side)


def _render_signed_with_vmax(arr: np.ndarray, vmax: float, max_side: int) -> np.ndarray:
    """按指定统一 vmax 渲染 signed heatmap，避免不同图各自归一化。"""
    x = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    scale = float(max(abs(float(vmax)), 1e-8))
    z = np.clip(x / scale, -1.0, 1.0)
    vis = np.zeros((*z.shape, 3), dtype=np.uint8)

    neg = z < 0
    pos = ~neg
    t_neg = z[neg] + 1.0
    vis[..., 0][neg] = 255
    vis[..., 1][neg] = np.round(255.0 * t_neg).astype(np.uint8)
    vis[..., 2][neg] = np.round(255.0 * t_neg).astype(np.uint8)
    t_pos = z[pos]
    vis[..., 0][pos] = np.round(255.0 * (1.0 - t_pos)).astype(np.uint8)
    vis[..., 1][pos] = np.round(255.0 * (1.0 - t_pos)).astype(np.uint8)
    vis[..., 2][pos] = 255
    return flow._resize_to(vis, (max_side, max_side))


def _save_shared_signed_pair(
    raw: np.ndarray,
    fused: np.ndarray,
    out_dir: str,
    max_side: int,
) -> tuple[float, np.ndarray]:
    """保存 raw/fused logits 共用色阶，并返回共享 vmax 与融合更新量。"""
    raw_arr = np.asarray(raw, dtype=np.float32)
    fused_arr = np.asarray(fused, dtype=np.float32)
    vmax = float(max(np.max(np.abs(raw_arr)), np.max(np.abs(fused_arr)), 1e-8))
    flow._imwrite(
        os.path.join(out_dir, "m_raw_logits.png"),
        _render_signed_with_vmax(raw_arr, vmax, max_side),
    )
    flow._imwrite(
        os.path.join(out_dir, "m_fuse_logits.png"),
        _render_signed_with_vmax(fused_arr, vmax, max_side),
    )
    update = fused_arr - raw_arr
    _save_signed("m_fuse_minus_raw_logits.png", update, out_dir, max_side)
    return vmax, update


def _title_bar_local(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    bar_h = min(30, max(18, h // 6))
    import cv2

    cv2.rectangle(out, (0, 0), (w - 1, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        str(title),
        (6, max(14, bar_h - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _render_prior_bank_gallery(prior_bank: np.ndarray, cell_size: int = 96, pad: int = 8) -> np.ndarray:
    # prior_bank: [K, H, W], value in [0,1]
    k = int(prior_bank.shape[0])
    cells: List[np.ndarray] = []
    for i in range(k):
        gray = flow._to_u8_heat(np.clip(prior_bank[i], 0.0, 1.0))
        bgr = flow._colormap_jet(gray)
        bgr = flow._resize_to(bgr, (cell_size, cell_size))
        bgr = _title_bar_local(bgr, f"P{i+1}")
        cells.append(bgr)
    if not cells:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    h = cells[0].shape[0]
    spacer = np.full((h, pad, 3), 255, dtype=np.uint8)
    row = cells[0]
    for c in cells[1:]:
        row = np.concatenate([row, spacer, c], axis=1)
    return row


def _render_weights_bar(weights: List[float], w: int = 420, h: int = 160) -> np.ndarray:
    import cv2

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    vals = np.asarray(weights, dtype=np.float32)
    if vals.size == 0:
        return _title_bar_local(canvas, "omega_i")
    vals = np.clip(vals, 0.0, 1.0)
    n = int(vals.size)
    left = 20
    right = 20
    bottom = 26
    top = 30
    bar_area_w = max(10, w - left - right)
    bar_w = max(4, bar_area_w // max(1, n * 2))
    gap = max(3, bar_w)
    x = left
    for i in range(n):
        v = float(vals[i])
        bh = int(round(v * max(1, h - top - bottom)))
        y1 = h - bottom - bh
        y2 = h - bottom
        cv2.rectangle(canvas, (x, y1), (x + bar_w, y2), (88, 140, 255), thickness=-1)
        cv2.putText(canvas, f"{i+1}", (x, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        x += bar_w + gap
        if x + bar_w >= w - right:
            break
    return _title_bar_local(canvas, "omega_i")


def _compose_symbol_panel(out_dir: str, items: List[tuple[str, str]], cell: int = 220, pad: int = 8) -> None:
    tiles: List[np.ndarray] = []
    for title, filename in items:
        p = os.path.join(out_dir, filename)
        if not os.path.isfile(p):
            continue
        img = flow._imread_bgr(p)
        img = flow._resize_to(img, (cell, cell))
        img = _title_bar_local(img, title)
        tiles.append(img)
    if not tiles:
        return
    spacer = np.full((tiles[0].shape[0], pad, 3), 255, dtype=np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, spacer, t], axis=1)
    flow._imwrite(os.path.join(out_dir, "qsp_symbol_panel.png"), row)


def _extract_and_export_one(
    predictor,
    cfg,
    img_path: str,
    out_dir: str,
    max_side: int,
    suppression_vis_stretch: str,
    suppression_p_low: float,
    suppression_p_high: float,
) -> dict:
    _ensure_dir(out_dir)
    model = predictor.model
    model.eval()

    img_bgr = flow._imread_bgr(img_path)
    flow._imwrite(os.path.join(out_dir, "input_image.png"), img_bgr)

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

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_pred_raw_results = outputs.get("pred_masks_raw", None)
        prior_mask_results = outputs.get("pred_prior_masks", None)
        prior_visual_results = outputs.get("pred_prior_visual", None)
        prior_gate_results = outputs.get("pred_prior_gates", None)
        prior_occluder_results = outputs.get("pred_prior_occluders", None)
        prior_bank_weight_results = outputs.get("pred_prior_bank_weights", None)
        prior_param_results = outputs.get("pred_prior_params", None)

        mask_pred_results = torch.nn.functional.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        if mask_pred_raw_results is not None and mask_pred_raw_results.ndim == 4:
            mask_pred_raw_results = torch.nn.functional.interpolate(
                mask_pred_raw_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        if prior_mask_results is not None and prior_mask_results.ndim == 4:
            prior_mask_results = torch.nn.functional.interpolate(
                prior_mask_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        if prior_visual_results is not None and prior_visual_results.ndim == 4:
            prior_visual_results = torch.nn.functional.interpolate(
                prior_visual_results,
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
        if prior_occluder_results is not None and prior_occluder_results.ndim == 4:
            prior_occluder_results = torch.nn.functional.interpolate(
                prior_occluder_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        image_size = images.image_sizes[0]
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = flow.sem_seg_postprocess(mask_pred_results[0], image_size, height, width)
        mask_cls_result = mask_cls_result.to(mask_pred_result)
        q_idx, _, best_score = flow._select_best_query(model, mask_cls_result, mask_pred_result)

        fused_mask_logits = mask_pred_result[q_idx].detach().float().cpu().numpy()
        raw_mask_logits = fused_mask_logits.copy()
        if mask_pred_raw_results is not None:
            raw_mask_result = flow.sem_seg_postprocess(mask_pred_raw_results[0], image_size, height, width)
            raw_mask_logits = raw_mask_result[q_idx].detach().float().cpu().numpy()
        fused_mask_prob = flow._sigmoid(fused_mask_logits)
        raw_mask_prob = flow._sigmoid(raw_mask_logits)

        if prior_mask_results is not None:
            prior_mask_result = flow.sem_seg_postprocess(prior_mask_results[0], image_size, height, width)
            p_align = np.clip(prior_mask_result[q_idx].detach().float().cpu().numpy(), 0.0, 1.0)
        else:
            p_align = np.zeros_like(raw_mask_prob, dtype=np.float32)

        if prior_gate_results is not None:
            gate_map = flow.sem_seg_postprocess(prior_gate_results[0], image_size, height, width)[q_idx]
            gate_map = np.clip(gate_map.detach().float().cpu().numpy(), 0.0, 1.0)
        else:
            gate_map = np.ones_like(raw_mask_prob, dtype=np.float32)

        if prior_occluder_results is not None:
            c_map = flow.sem_seg_postprocess(prior_occluder_results[0], image_size, height, width)[q_idx]
            c_map = np.clip(c_map.detach().float().cpu().numpy(), 0.0, 1.0)
        else:
            c_map = np.zeros_like(raw_mask_prob, dtype=np.float32)

        # ===== Prior-bank branch: P_k / omega_i / P_mix =====
        p_bank_img: Optional[np.ndarray] = None
        p_mix: Optional[np.ndarray] = None
        selected_bank_idx = 0
        bank_weights_list: List[float] = []
        prior_path = str(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_PATH", "")).strip()
        if prior_path:
            prior_bank_t = flow.load_prior_tensor(prior_path).float().cpu()  # [K,1,H,W]
            prior_bank = np.clip(prior_bank_t[:, 0].numpy(), 0.0, 1.0)
            p_bank_img = _render_prior_bank_gallery(prior_bank, cell_size=max(72, min(120, max_side // 2)))
            if prior_bank_weight_results is not None:
                w = prior_bank_weight_results[0, q_idx].detach().float().cpu().numpy()
                bank_weights_list = [float(x) for x in w.tolist()]
                selected_bank_idx = int(np.argmax(w))
                p_mix = np.tensordot(w, prior_bank[: len(w)], axes=(0, 0)).astype(np.float32)
            else:
                p_mix = prior_bank[0].astype(np.float32)

        if p_bank_img is not None:
            flow._imwrite(os.path.join(out_dir, "p_bank.png"), p_bank_img)
        if bank_weights_list:
            wbar = _render_weights_bar(bank_weights_list)
            flow._imwrite(os.path.join(out_dir, "omega_i.png"), wbar)
            with open(os.path.join(out_dir, "omega_i.csv"), "w", newline="", encoding="utf-8") as f:
                ww = csv.writer(f)
                ww.writerow(["k", "omega_i_k"])
                for i, v in enumerate(bank_weights_list, start=1):
                    ww.writerow([i, float(v)])
        if p_mix is not None:
            _save_heat("p_mix.png", p_mix, out_dir, max_side)

        # ===== 按流程图符号命名的核心图 =====
        shared_logit_vmax, fusion_update_logits = _save_shared_signed_pair(
            raw=raw_mask_logits,
            fused=fused_mask_logits,
            out_dir=out_dir,
            max_side=max_side,
        )
        _save_heat("m_raw_sigma.png", raw_mask_prob, out_dir, max_side)
        _save_heat("p_align.png", p_align, out_dir, max_side)
        _save_heat("g_i.png", gate_map, out_dir, max_side)
        _save_heat("c_i.png", c_map, out_dir, max_side)
        flow._save_tensor_gray("c_i_gray.png", c_map, out_dir, max_side)
        _save_heat("m_fuse.png", fused_mask_prob, out_dir, max_side)

        # residual cue: sigma(M_raw) - P_align
        residual_cue = (raw_mask_prob - p_align).astype(np.float32)
        _save_signed("residual_cue.png", residual_cue, out_dir, max_side)

        # suppression map: (1 - c_i)
        suppression_map = np.clip(1.0 - c_map, 0.0, 1.0).astype(np.float32)
        suppression_map_vis = _stretch_for_vis(
            suppression_map,
            mode=suppression_vis_stretch,
            p_low=float(suppression_p_low),
            p_high=float(suppression_p_high),
        )
        _save_heat("suppression_map.png", suppression_map_vis, out_dir, max_side)
        flow._save_tensor_gray("suppression_map_gray.png", suppression_map, out_dir, max_side)

        # delta_i (approx): (M_fuse - M_raw) / (alpha * gate)
        alpha = float(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_ALPHA", 1.0))
        denom = np.maximum(alpha * gate_map, 1e-4)
        delta_i = (fused_mask_logits - raw_mask_logits) / denom
        delta_i = np.clip(delta_i, -1.0, 1.0).astype(np.float32)
        _save_signed("delta_i.png", delta_i, out_dir, max_side)

        # V_i：直接读取模型融合模块内部的 query_visual_map，不再重新构造 proxy。
        if prior_visual_results is not None and prior_visual_results.ndim == 4:
            v_i = flow.sem_seg_postprocess(prior_visual_results[0], image_size, height, width)[q_idx]
            v_i = v_i.detach().float().cpu().numpy().astype(np.float32)
            _save_signed("v_i.png", v_i, out_dir, max_side)
            v_i_available = True
        else:
            # 兼容没有导出 pred_prior_visual 的旧模型代码，明确标记为缺失而非伪造真实 V_i。
            v_i = None
            v_i_available = False

        # 输出一个按流程图顺序的整合拼图，便于论文核对
        _compose_symbol_panel(
            out_dir,
            items=[
                ("P_k", "p_bank.png"),
                ("omega_i", "omega_i.png"),
                ("P_i^mix", "p_mix.png"),
                ("P_i^align", "p_align.png"),
                ("M_i^raw", "m_raw_logits.png"),
                ("sigma(M_i^raw)", "m_raw_sigma.png"),
                ("residual cue", "residual_cue.png"),
                ("V_i", "v_i.png"),
                ("g_i", "g_i.png"),
                ("c_i", "c_i.png"),
                ("1-c_i", "suppression_map.png"),
                ("delta_i", "delta_i.png"),
                ("M_i^fuse logits", "m_fuse_logits.png"),
                ("M_i^fuse", "m_fuse.png"),
            ],
            cell=max(128, min(220, int(max_side))),
            pad=10,
        )

        # A_i / a_i 文本参数（如果模型有输出 pred_prior_params）
        with open(os.path.join(out_dir, "affine_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"query_idx={q_idx}\n")
            f.write(f"prior_path={prior_path}\n")
            f.write(f"selected_bank_idx={selected_bank_idx}\n")
            if bank_weights_list:
                f.write(f"prior_bank_weights={bank_weights_list}\n")
            if prior_param_results is not None:
                params_t = prior_param_results[0, q_idx].detach().float().cpu()
                f.write(f"a_i_raw={params_t.numpy().tolist()}\n")
                f.write(f"angle_rad={float(params_t[0].item())}\n")
                f.write(f"angle_deg={float(params_t[0].item() * 180.0 / np.pi)}\n")
                f.write(f"log_sx={float(params_t[1].item())}\n")
                f.write(f"log_sy={float(params_t[2].item())}\n")
                f.write(f"sx={float(torch.exp(params_t[1]).item())}\n")
                f.write(f"sy={float(torch.exp(params_t[2]).item())}\n")
                f.write(f"tx_raw={float(params_t[3].item())}\n")
                f.write(f"ty_raw={float(params_t[4].item())}\n")
                f.write(f"tx={float(torch.tanh(params_t[3]).item())}\n")
                f.write(f"ty={float(torch.tanh(params_t[4]).item())}\n")
                affine_t = flow.build_affine_from_query_params(
                    prior_param_results[:, q_idx : q_idx + 1].detach().float()
                )[0].detach().cpu().numpy()
                f.write(f"A_i={affine_t.tolist()}\n")

    with open(os.path.join(out_dir, "missing_panels_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"image={os.path.abspath(img_path)}\n")
        f.write(f"query_idx={q_idx}\n")
        f.write(f"instance_score={best_score}\n")
        f.write(f"prior_path={prior_path}\n")
        f.write(f"selected_bank_idx={selected_bank_idx}\n")
        f.write(f"prior_bank_weights={bank_weights_list}\n")
        f.write("m_raw_sigma_semantics=sigmoid(raw_mask_logits)\n")
        f.write("m_raw_logits_semantics=raw mask logits before sigmoid\n")
        f.write("p_align_semantics=aligned prior map from warping\n")
        f.write("g_i_semantics=effective gate after occluder suppression\n")
        f.write("c_i_semantics=occluder/suppressor probability\n")
        f.write("m_fuse_semantics=sigmoid(fused_mask_logits)\n")
        f.write("m_fuse_logits_semantics=fused mask logits before sigmoid\n")
        f.write(f"shared_logit_vmax={shared_logit_vmax}\n")
        f.write(f"fusion_update_logits_min={float(np.min(fusion_update_logits))}\n")
        f.write(f"fusion_update_logits_max={float(np.max(fusion_update_logits))}\n")
        f.write(f"fusion_update_logits_mean={float(np.mean(fusion_update_logits))}\n")
        f.write(f"fusion_update_logits_absmax={float(np.max(np.abs(fusion_update_logits)))}\n")
        f.write(f"c_i_min={float(np.min(c_map))}\n")
        f.write(f"c_i_max={float(np.max(c_map))}\n")
        f.write(f"c_i_mean={float(np.mean(c_map))}\n")
        f.write(f"suppression_map_min={float(np.min(suppression_map))}\n")
        f.write(f"suppression_map_max={float(np.max(suppression_map))}\n")
        f.write(f"suppression_map_mean={float(np.mean(suppression_map))}\n")
        f.write("residual_cue_semantics=sigmoid(raw_mask)-aligned_prior\n")
        f.write("suppression_map_semantics=1-occluder_prob\n")
        f.write("delta_i_semantics=approx((fused_logits-raw_logits)/(alpha*gate))\n")
        if v_i_available and v_i is not None:
            f.write("v_i_semantics=actual query_visual_map from QueryShapePriorFusion\n")
            f.write(f"v_i_min={float(np.min(v_i))}\n")
            f.write(f"v_i_max={float(np.max(v_i))}\n")
            f.write(f"v_i_mean={float(np.mean(v_i))}\n")
        else:
            f.write("v_i_semantics=unavailable: model output pred_prior_visual was not found\n")
        f.write(f"suppression_vis_stretch={suppression_vis_stretch}\n")
        f.write(f"suppression_p_low={float(suppression_p_low)}\n")
        f.write(f"suppression_p_high={float(suppression_p_high)}\n")

    return {
        "image": os.path.abspath(img_path),
        "out_dir": os.path.abspath(out_dir),
        "query_idx": int(q_idx),
        "instance_score": float(best_score),
        "selected_bank_idx": int(selected_bank_idx),
        "has_bank_weights": int(len(bank_weights_list) > 0),
    }


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)
    image_paths = flow._resolve_image_paths(args)
    if not image_paths:
        raise ValueError("未解析到输入图片。请提供 --image / --images / --image-list / --dataset-root。")

    # GPU/DEVICE 兼容入口：允许使用 --gpu-id 或 --device
    if int(args.gpu_id) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))

    cfg = flow._build_cfg(
        os.path.abspath(args.config_file),
        os.path.abspath(args.weights),
        float(args.score_thr),
        str(args.prior_path).strip(),
        int(args.num_classes),
    )
    if str(args.device).strip():
        cfg.MODEL.DEVICE = str(args.device).strip()
    predictor = flow.DefaultPredictor(cfg)
    flow.DetectionCheckpointer(predictor.model).load(cfg.MODEL.WEIGHTS)

    rows: List[dict] = []
    multi = len(image_paths) > 1
    for i, img_path in enumerate(image_paths):
        if multi:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            case_dir = os.path.join(args.out_dir, f"{i:03d}_{stem}")
        else:
            case_dir = args.out_dir
        row = _extract_and_export_one(
            predictor=predictor,
            cfg=cfg,
            img_path=img_path,
            out_dir=case_dir,
            max_side=int(max(64, args.max_side)),
            suppression_vis_stretch=str(args.suppression_vis_stretch),
            suppression_p_low=float(args.suppression_p_low),
            suppression_p_high=float(args.suppression_p_high),
        )
        rows.append(row)
        print(f"[{i+1}/{len(image_paths)}] done: {row['out_dir']}")

    summary_csv = os.path.join(args.out_dir, "missing_panels_summary.csv")
    fields = ["image", "out_dir", "query_idx", "instance_score", "selected_bank_idx", "has_bank_weights"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()

