#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 M2F-QSP（Mask2Former + Query-Aligned Shape Prior）的中间产物：

- Input Image
- Raw Mask Logits (pred_masks_raw)
- Aligned Prior (pred_prior_masks)
- Effective Gate (pred_prior_gates，已融合 occluder 抑制后的有效 gate)
- Effective Gate Overlay（effective_gate 叠加输入图）
- Sampling Prob (Uncertainty) 与 Sampling Prob (Support) 热图及叠加图（Fig.6b）
- Occluder Suppression (pred_prior_occluders，可视化线缆/遮挡抑制区域)
- Fused Mask Logits (pred_masks)
- Final Output Overlay

说明：
- 该脚本不会走 MaskFormer 推理后的封装结果，而是手动执行
  backbone -> sem_seg_head，直接拿 raw outputs 进行可视化。
- 会基于最终实例推理得分，选取“最高分实例对应的 query”来展示中间产物。
- 支持单图与批量导出；批量模式会输出 gate_summary.csv 统计表。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _D2_ROOT not in sys.path:
    sys.path.insert(0, _D2_ROOT)

_WORKSPACE_ROOT = os.path.abspath(os.path.join(_D2_ROOT, "..", ".."))
_M2F_ROOT = os.path.abspath(os.path.join(_WORKSPACE_ROOT, "..", "Mask2Former"))
if _M2F_ROOT not in sys.path:
    sys.path.insert(0, _M2F_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList

import mask2former  # noqa: F401
from mask2former import add_maskformer2_config
from mask2former.modeling.shape_prior_fusion import build_affine_from_query_params, load_prior_tensor


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx - mn < 1e-8):
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _to_u8_heat(x01: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x01, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _colormap_jet(gray_u8: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)  # type: ignore[attr-defined]
    import matplotlib.cm as cm

    jet = cm.get_cmap("jet")
    rgb = (jet(gray_u8.astype(np.float32) / 255.0)[..., :3] * 255.0).astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def _resize_to(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = int(size[0]), int(size[1])
    if _HAS_CV2:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)  # type: ignore[attr-defined]
    pil = Image.fromarray(img[:, :, ::-1] if img.ndim == 3 else img)
    pil2 = pil.resize((w, h), resample=Image.NEAREST)
    arr = np.array(pil2)
    if arr.ndim == 3:
        arr = arr[:, :, ::-1].copy()
    return arr


def _imread_bgr(path: str) -> np.ndarray:
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # type: ignore[attr-defined]
        if img is None:
            raise FileNotFoundError(f"image not found/readable: {path}")
        return img
    im = Image.open(path).convert("RGB")
    rgb = np.array(im, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def _imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if _HAS_CV2:
        cv2.imwrite(path, img)  # type: ignore[attr-defined]
        return
    arr = img[:, :, ::-1] if img.ndim == 3 else img
    Image.fromarray(arr).save(path)


def _overlay_mask(bgr: np.ndarray, mask01: np.ndarray, color_bgr=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    m = (mask01 > 0.5).astype(np.uint8)
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:, :] = color_bgr
    out[m > 0] = (out[m > 0] * (1 - alpha) + overlay[m > 0] * alpha).astype(np.uint8)
    return out


def _overlay_heatmap(
    bgr: np.ndarray,
    map01: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    将 [0,1] 热图以固定 colormap 叠加到输入图上。
    """
    x = np.clip(np.nan_to_num(map01, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    gray = _to_u8_heat(x)
    heat_bgr = _colormap_jet(gray)
    if heat_bgr.shape[:2] != bgr.shape[:2]:
        heat_bgr = _resize_to(heat_bgr, (bgr.shape[1], bgr.shape[0]))
    a = float(max(0.0, min(1.0, alpha)))
    out = (bgr.astype(np.float32) * (1.0 - a) + heat_bgr.astype(np.float32) * a).astype(np.uint8)
    return out


def _highlight_mask(img_bgr: np.ndarray, v_thr: int, s_max: int) -> np.ndarray:
    if _HAS_CV2:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # type: ignore[attr-defined]
        _h, s, v = cv2.split(hsv)  # type: ignore[attr-defined]
        hi = (v >= int(v_thr)) & (s <= int(s_max))
        return hi.astype(np.uint8)
    # PIL fallback: approximate using RGB max channel
    rgb = img_bgr[:, :, ::-1].astype(np.float32)
    v = np.max(rgb, axis=2)
    s = (np.max(rgb, axis=2) - np.min(rgb, axis=2))
    hi = (v >= float(v_thr)) & (s <= float(s_max))
    return hi.astype(np.uint8)


def _natural_sort_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _iter_images(root: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    out: List[str] = []
    for dp, dns, fns in os.walk(os.path.abspath(root)):
        dns.sort(key=_natural_sort_key)
        fns.sort(key=_natural_sort_key)
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


def _parse_index_range(s: str) -> Tuple[int, int]:
    ss = str(s).strip()
    if not ss:
        raise ValueError("empty index-range")
    if "-" in ss:
        a, b = ss.split("-", 1)
    elif ":" in ss:
        a, b = ss.split(":", 1)
    else:
        raise ValueError("index-range must be like '15-20' or '15:20'")
    ia = int(a.strip())
    ib = int(b.strip())
    if ib < ia:
        ia, ib = ib, ia
    return ia, ib


def _resolve_image_paths(args: argparse.Namespace) -> List[str]:
    picks: List[str] = []
    if str(getattr(args, "images", "")).strip():
        for t in str(args.images).split(","):
            tt = t.strip()
            if tt:
                picks.append(os.path.abspath(tt))
    if str(getattr(args, "image_list", "")).strip():
        p = os.path.abspath(str(args.image_list).strip())
        if not os.path.isfile(p):
            raise FileNotFoundError(f"--image-list not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and (not s.startswith("#")):
                    picks.append(os.path.abspath(s))
    if str(getattr(args, "dataset_root", "")).strip():
        imgs = _iter_images(str(args.dataset_root).strip())
        if bool(getattr(args, "no_shuffle", False)):
            picks.extend(imgs[: max(1, int(getattr(args, "scan_max", 200)))])
        else:
            # 稳定选择即可；该脚本重点是解释机制，不依赖随机
            picks.extend(imgs[: max(1, int(getattr(args, "scan_max", 200)))])
    if str(getattr(args, "image", "")).strip():
        picks.append(os.path.abspath(str(args.image).strip()))

    # 去重且保序
    uniq: List[str] = []
    seen = set()
    for p in picks:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    uniq = [p for p in uniq if os.path.isfile(p)]

    if str(getattr(args, "index_range", "")).strip():
        a, b = _parse_index_range(str(args.index_range).strip())
        base = int(getattr(args, "index_base", 1))
        if base == 1:
            a0 = max(0, a - 1)
            b0 = max(0, b - 1)
        else:
            a0 = max(0, a)
            b0 = max(0, b)
        uniq = uniq[a0 : b0 + 1]

    if int(getattr(args, "num_images", 0)) > 0:
        uniq = uniq[: int(args.num_images)]
    return uniq


def _build_cfg(config_file: str, weights: str, score_thr: float, prior_path: str, num_classes: int):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = float(score_thr)
    if str(prior_path).strip():
        cfg.MODEL.MASK_FORMER.PRIOR_ON = True
        cfg.MODEL.MASK_FORMER.PRIOR_PATH = str(prior_path).strip()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def _save_tensor_heat(name: str, arr01: np.ndarray, out_dir: str, max_side: int) -> None:
    _imwrite(os.path.join(out_dir, name), _render_tensor_heat(arr01, max_side))


def _save_tensor_heat_norm(name: str, arr: np.ndarray, out_dir: str, max_side: int) -> None:
    _imwrite(os.path.join(out_dir, name), _render_tensor_heat_norm(arr, max_side))


def _save_signed_heat(name: str, arr: np.ndarray, out_dir: str, max_side: int) -> None:
    _imwrite(os.path.join(out_dir, name), _render_signed_heat(arr, max_side))


def _render_tensor_heat(arr01: np.ndarray, max_side: int) -> np.ndarray:
    gray = _to_u8_heat(arr01)
    colored = _colormap_jet(gray)
    return _resize_to(colored, (max_side, max_side))


def _render_tensor_heat_norm(arr: np.ndarray, max_side: int) -> np.ndarray:
    return _render_tensor_heat(_norm01(arr), max_side)


def _render_signed_heat(arr: np.ndarray, max_side: int) -> np.ndarray:
    x = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    vmax = float(np.max(np.abs(x)))
    if (not np.isfinite(vmax)) or vmax < 1e-8:
        vis = np.full((*x.shape, 3), 255, dtype=np.uint8)
    else:
        z = np.clip(x / vmax, -1.0, 1.0)
        vis = np.zeros((*z.shape, 3), dtype=np.uint8)

        neg = z < 0
        pos = ~neg

        # BGR: negative -> blue, zero -> white, positive -> red
        t_neg = z[neg] + 1.0
        vis[..., 0][neg] = 255
        vis[..., 1][neg] = np.round(255.0 * t_neg).astype(np.uint8)
        vis[..., 2][neg] = np.round(255.0 * t_neg).astype(np.uint8)

        t_pos = z[pos]
        vis[..., 0][pos] = np.round(255.0 * (1.0 - t_pos)).astype(np.uint8)
        vis[..., 1][pos] = np.round(255.0 * (1.0 - t_pos)).astype(np.uint8)
        vis[..., 2][pos] = 255

    return _resize_to(vis, (max_side, max_side))


def _save_tensor_gray(name: str, arr01: np.ndarray, out_dir: str, max_side: int) -> None:
    _imwrite(os.path.join(out_dir, name), _render_tensor_gray(arr01, max_side))


def _save_binary_mask(name: str, mask01: np.ndarray, out_dir: str, max_side: int) -> None:
    _imwrite(os.path.join(out_dir, name), _render_binary_mask(mask01, max_side))


def _render_tensor_gray(arr01: np.ndarray, max_side: int) -> np.ndarray:
    gray = _to_u8_heat(arr01)
    return _resize_to(gray, (max_side, max_side))


def _render_binary_mask(mask01: np.ndarray, max_side: int) -> np.ndarray:
    mask_u8 = ((mask01 > 0.5).astype(np.uint8) * 255)
    return _resize_to(mask_u8, (max_side, max_side))


def _bgr_to_pil_rgb(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        rgb = np.repeat(img[:, :, None], 3, axis=2)
    else:
        rgb = img[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _fit_with_padding(img: np.ndarray, width: int, height: int, pad: int = 12) -> Image.Image:
    src = _bgr_to_pil_rgb(img)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    avail_w = max(1, width - 2 * pad)
    avail_h = max(1, height - 2 * pad)
    scale = min(avail_w / max(src.width, 1), avail_h / max(src.height, 1))
    new_w = max(1, int(round(src.width * scale)))
    new_h = max(1, int(round(src.height * scale)))
    resized = src.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    offset = ((width - new_w) // 2, (height - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def _draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2.0
    y = top + (bottom - top - text_h) / 2.0
    draw.text((x, y), text, font=font, fill=fill)


def _paste_labeled_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    img: np.ndarray,
    tile_w: int,
    tile_h: int,
    title_h: int,
    font,
) -> tuple[int, int, int, int]:
    outer = (x, y, x + tile_w, y + title_h + tile_h)
    draw.rounded_rectangle(outer, radius=14, fill=(251, 252, 255), outline=(160, 170, 185), width=2)
    _draw_centered_text(draw, (x + 6, y + 4, x + tile_w - 6, y + title_h), title, font, (18, 28, 45))
    inner = _fit_with_padding(img, tile_w - 16, tile_h - 16, pad=8)
    canvas.paste(inner, (x + 8, y + title_h + 8))
    return outer


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill=(70, 80, 95),
    width: int = 4,
) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    if x1 == x2 and y1 == y2:
        return
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux
    head_len = 14
    head_w = 8
    tip = (x2, y2)
    left = (x2 - ux * head_len + px * head_w, y2 - uy * head_len + py * head_w)
    right = (x2 - ux * head_len - px * head_w, y2 - uy * head_len - py * head_w)
    draw.polygon([tip, left, right], fill=fill)


def _save_dpc_flow_overview(
    out_dir: str,
    max_side: int,
    img_bgr: np.ndarray,
    raw_mask_prob: np.ndarray,
    prior_mask_prob: np.ndarray,
    occ_map: np.ndarray,
    gate_map: np.ndarray,
    residual_update: np.ndarray,
    fused_mask_prob: np.ndarray,
    overlay_bgr: np.ndarray,
) -> None:
    tile_img = max(128, min(int(max_side), 220))
    tile_w = tile_img + 18
    tile_h = tile_img + 18
    title_h = 34
    margin = 36
    gap = 24
    panel_gap = 90
    flow_gap = 70
    fusion_box_w = 126
    fusion_box_h = 72
    main_panel_w = 3 * tile_w + 2 * gap
    canvas_w = margin * 2 + tile_w + flow_gap + main_panel_w + panel_gap + fusion_box_w + flow_gap + tile_w + flow_gap + tile_w
    canvas_h = margin * 2 + title_h + 2 * tile_h + gap + 100

    canvas = Image.new("RGB", (canvas_w, canvas_h), (246, 248, 252))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

    draw.text((margin, 12), "QSP + Mask2Former flow with occluder suppression", font=title_font, fill=(20, 28, 38))

    input_y = margin + title_h + tile_h // 2
    input_box = _paste_labeled_tile(
        canvas,
        draw,
        margin,
        input_y,
        "Input image",
        _resize_to(img_bgr, (tile_img, tile_img)),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )

    panel_x = margin + tile_w + flow_gap
    panel_y = margin + 24
    panel_box = (panel_x - 18, panel_y - 18, panel_x + main_panel_w + 18, panel_y + title_h + 2 * tile_h + gap + 18)
    draw.rounded_rectangle(panel_box, radius=18, outline=(90, 110, 135), width=3, fill=(255, 255, 255))
    draw.text((panel_x, panel_y - 34), "Dynamic Prior Correction (DPC)", font=title_font, fill=(24, 38, 58))

    raw_box = _paste_labeled_tile(
        canvas,
        draw,
        panel_x,
        panel_y,
        "Raw mask logits",
        _render_tensor_heat(raw_mask_prob, tile_img),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )
    prior_box = _paste_labeled_tile(
        canvas,
        draw,
        panel_x + tile_w + gap,
        panel_y,
        "Aligned prior",
        _render_tensor_heat(prior_mask_prob, tile_img),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )
    occ_box = _paste_labeled_tile(
        canvas,
        draw,
        panel_x + 2 * (tile_w + gap),
        panel_y,
        "Occluder suppression",
        _render_tensor_heat(occ_map, tile_img),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )
    gate_box = _paste_labeled_tile(
        canvas,
        draw,
        panel_x + tile_w // 2,
        panel_y + title_h + tile_h + gap,
        "Effective gate",
        _render_tensor_heat(gate_map, tile_img),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )
    residual_box = _paste_labeled_tile(
        canvas,
        draw,
        panel_x + tile_w // 2 + tile_w + gap,
        panel_y + title_h + tile_h + gap,
        "Residual update",
        _render_signed_heat(residual_update, tile_img),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )

    fusion_x = panel_box[2] + panel_gap
    fusion_y = margin + canvas_h // 2 - fusion_box_h // 2 - 6
    fusion_box = (fusion_x, fusion_y, fusion_x + fusion_box_w, fusion_y + fusion_box_h)
    draw.rounded_rectangle(fusion_box, radius=16, fill=(225, 236, 248), outline=(110, 135, 165), width=3)
    _draw_centered_text(draw, fusion_box, "Prior fusion", title_font, (24, 44, 72))

    fused_x = fusion_box[2] + flow_gap
    fused_y = input_y
    fused_box = _paste_labeled_tile(
        canvas,
        draw,
        fused_x,
        fused_y,
        "Fused mask",
        _render_tensor_heat(fused_mask_prob, tile_img),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )
    final_x = fused_box[2] + flow_gap
    final_box = _paste_labeled_tile(
        canvas,
        draw,
        final_x,
        fused_y,
        "Final output",
        _resize_to(overlay_bgr, (tile_img, tile_img)),
        tile_w,
        tile_h,
        title_h,
        text_font,
    )

    dpc_center = (fusion_box[0], (fusion_box[1] + fusion_box[3]) // 2)
    for box in (raw_box, prior_box, occ_box, gate_box, residual_box):
        _draw_arrow(draw, (box[2], (box[1] + box[3]) // 2), dpc_center)
    _draw_arrow(draw, (input_box[2], (input_box[1] + input_box[3]) // 2), (raw_box[0] - 18, (raw_box[1] + raw_box[3]) // 2))
    _draw_arrow(draw, (fusion_box[2], (fusion_box[1] + fusion_box[3]) // 2), (fused_box[0] - 18, (fused_box[1] + fused_box[3]) // 2))
    _draw_arrow(draw, (fused_box[2], (fused_box[1] + fused_box[3]) // 2), (final_box[0] - 18, (final_box[1] + final_box[3]) // 2))

    canvas.save(os.path.join(out_dir, "dpc_flow_overview.png"))


def _make_affine_grid_prior(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=np.float32)
    step = max(4, min(h, w) // 8)
    grid[::step, :] = 1.0
    grid[:, ::step] = 1.0
    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    grid[:, 0] = 1.0
    grid[:, -1] = 1.0
    grid[h // 2, :] = 1.0
    grid[:, w // 2] = 1.0
    return grid


def _select_best_query(model, mask_cls_result: torch.Tensor, mask_pred_result: torch.Tensor) -> tuple[int, torch.Tensor, float]:
    """
    Mirror MaskFormer.instance_inference to locate the best query index.
    """
    scores = F.softmax(mask_cls_result, dim=-1)[:, :-1]
    labels = torch.arange(
        model.sem_seg_head.num_classes, device=model.device
    ).unsqueeze(0).repeat(model.num_queries, 1).flatten(0, 1)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(model.test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]
    topk_query_indices = topk_indices // model.sem_seg_head.num_classes
    mask_pred = mask_pred_result[topk_query_indices]

    pred_masks = (mask_pred > 0).float()
    mask_scores_per_image = (
        (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1)
        / (pred_masks.flatten(1).sum(1) + 1e-6)
    )
    final_scores = scores_per_image * mask_scores_per_image
    best_i = int(final_scores.argmax().item())
    q_idx = int(topk_query_indices[best_i].item())
    best_mask = pred_masks[best_i]
    best_score = float(final_scores[best_i].item())
    return q_idx, best_mask, best_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", default="", help="single image path")
    p.add_argument("--images", default="", help="comma-separated image paths")
    p.add_argument("--image-list", default="", help="txt file with one image path per line")
    p.add_argument("--dataset-root", default="", help="optional image root for batch export")
    p.add_argument("--scan-max", type=int, default=200, help="max images from dataset-root")
    p.add_argument("--index-range", default="", help="optional inclusive range, e.g. 36-40")
    p.add_argument("--index-base", type=int, default=1, choices=(0, 1), help="index base for --index-range")
    p.add_argument("--no-shuffle", action="store_true", help="placeholder for consistent CLI style")
    p.add_argument("--num-images", type=int, default=0, help="optional cap on resolved image list")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--prior-path", default="", help="override MODEL.MASK_FORMER.PRIOR_PATH")
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--max-side", type=int, default=256, help="visualization canvas size for small tensors")
    p.add_argument("--gate-overlay-alpha", type=float, default=0.45, help="alpha for effective gate overlay")
    p.add_argument("--highlight-v-thr", type=int, default=245, help="HSV-V threshold for highlight mask")
    p.add_argument("--highlight-s-max", type=int, default=70, help="HSV-S max for highlight mask")
    p.add_argument("--export-fig6a-only", action="store_true", help="仅导出 Fig.6a 相关产物（门控热图链路）")
    p.add_argument("--export-fig6b-only", action="store_true", help="仅导出 Fig.6b 相关产物（采样概率热图链路）")
    return p.parse_args()


def _run_one_image(
    *,
    predictor: DefaultPredictor,
    cfg,
    config_file: str,
    weights: str,
    image_path: str,
    out_dir: str,
    score_thr: float,
    max_side: int,
    gate_overlay_alpha: float,
    highlight_v_thr: int,
    highlight_s_max: int,
    export_fig6a_only: bool,
    export_fig6b_only: bool,
) -> dict:
    _ensure_dir(out_dir)
    model = predictor.model
    model.eval()
    img_bgr = _imread_bgr(image_path)
    _imwrite(os.path.join(out_dir, "input_image.png"), img_bgr)

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
        images = ImageList.from_tensors(images, model.size_divisibility)

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

        if "pred_prior_masks" not in outputs:
            raise RuntimeError(
                "Model outputs do not contain pred_prior_masks. "
                "请确认使用的是打开 PRIOR_ON 的 QSP 配置，并且 decoder 改动已生效。"
            )

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_pred_raw_results = outputs.get("pred_masks_raw", None)
        prior_mask_results = outputs.get("pred_prior_masks", None)
        prior_gate_results = outputs.get("pred_prior_gates", None)
        prior_occluder_results = outputs.get("pred_prior_occluders", None)
        prior_param_results = outputs.get("pred_prior_params", None)
        prior_bank_weight_results = outputs.get("pred_prior_bank_weights", None)

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        if mask_pred_raw_results is not None:
            mask_pred_raw_results = F.interpolate(
                mask_pred_raw_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        if prior_mask_results is not None:
            prior_mask_results = F.interpolate(
                prior_mask_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        if prior_gate_results is not None and prior_gate_results.ndim == 4:
            prior_gate_results = F.interpolate(
                prior_gate_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        if prior_occluder_results is not None and prior_occluder_results.ndim == 4:
            prior_occluder_results = F.interpolate(
                prior_occluder_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        image_size = images.image_sizes[0]
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = sem_seg_postprocess(mask_pred_results[0], image_size, height, width)
        mask_cls_result = mask_cls_result.to(mask_pred_result)

        q_idx, best_mask, best_score = _select_best_query(model, mask_cls_result, mask_pred_result)

        fused_mask_logits = mask_pred_result[q_idx].detach().float().cpu().numpy()
        fused_mask_prob = _sigmoid(fused_mask_logits)
        final_mask = best_mask.detach().float().cpu().numpy()
        mask_features_map = mask_features[0].detach().float().abs().mean(dim=0).cpu().numpy()
        mask_features_map = _norm01(mask_features_map)

        if mask_pred_raw_results is not None:
            raw_mask_result = sem_seg_postprocess(mask_pred_raw_results[0], image_size, height, width)
            raw_mask_logits = raw_mask_result[q_idx].detach().float().cpu().numpy()
            raw_mask_prob = _sigmoid(raw_mask_logits)
        else:
            raw_mask_logits = fused_mask_logits.copy()
            raw_mask_prob = fused_mask_prob

        residual_update = fused_mask_logits - raw_mask_logits

        if prior_mask_results is not None:
            prior_mask_result = sem_seg_postprocess(prior_mask_results[0], image_size, height, width)
            prior_mask_prob = prior_mask_result[q_idx].detach().float().cpu().numpy()
            prior_mask_prob = np.clip(prior_mask_prob, 0.0, 1.0)
        else:
            prior_mask_prob = np.zeros_like(fused_mask_prob, dtype=np.float32)

        if prior_gate_results is not None and prior_gate_results.ndim == 4:
            gate_map = sem_seg_postprocess(prior_gate_results[0], image_size, height, width)[q_idx]
            gate_map = gate_map.detach().float().cpu().numpy()
            gate_map = np.clip(gate_map, 0.0, 1.0)
            gate_mean = float(gate_map.mean())
        elif prior_gate_results is not None:
            gate_mean = float(prior_gate_results[0, q_idx, 0].detach().float().cpu().item())
            gate_map = np.full_like(fused_mask_prob, fill_value=np.clip(gate_mean, 0.0, 1.0), dtype=np.float32)
        else:
            gate_mean = 0.0
            gate_map = np.full_like(fused_mask_prob, fill_value=0.0, dtype=np.float32)

        if prior_occluder_results is not None and prior_occluder_results.ndim == 4:
            occ_map = sem_seg_postprocess(prior_occluder_results[0], image_size, height, width)[q_idx]
            occ_map = occ_map.detach().float().cpu().numpy()
            occ_map = np.clip(occ_map, 0.0, 1.0)
            occ_mean = float(occ_map.mean())
        else:
            occ_mean = 0.0
            occ_map = np.full_like(fused_mask_prob, fill_value=0.0, dtype=np.float32)

        align_params_raw = None
        align_params_effective = None
        affine_matrix = None
        selected_bank_idx = 0
        selected_prior = None
        warped_grid = None

        if prior_bank_weight_results is not None:
            selected_bank_idx = int(prior_bank_weight_results[0, q_idx].detach().float().cpu().argmax().item())

        prior_path = str(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_PATH", "")).strip()
        if prior_path:
            prior_bank = load_prior_tensor(prior_path).float()
            selected_bank_idx = max(0, min(selected_bank_idx, int(prior_bank.shape[0]) - 1))
            selected_prior = prior_bank[selected_bank_idx, 0].detach().cpu().numpy()

        if prior_param_results is not None:
            params_t = prior_param_results[0, q_idx].detach().float().cpu()
            align_params_raw = params_t.numpy()
            align_params_effective = {
                "angle_rad": float(params_t[0].item()),
                "angle_deg": float(params_t[0].item() * 180.0 / np.pi),
                "log_sx": float(params_t[1].item()),
                "log_sy": float(params_t[2].item()),
                "sx": float(torch.exp(params_t[1]).item()),
                "sy": float(torch.exp(params_t[2]).item()),
                "tx_raw": float(params_t[3].item()),
                "ty_raw": float(params_t[4].item()),
                "tx": float(torch.tanh(params_t[3]).item()),
                "ty": float(torch.tanh(params_t[4]).item()),
            }
            affine_t = build_affine_from_query_params(
                prior_param_results[:, q_idx : q_idx + 1].detach().float()
            )[0].detach().cpu()
            affine_matrix = affine_t.numpy()
            if selected_prior is not None:
                grid_prior = _make_affine_grid_prior(selected_prior.shape[0], selected_prior.shape[1])
                grid_t = torch.from_numpy(grid_prior[None, None, :, :]).float()
                grid_warped_t = F.grid_sample(
                    grid_t,
                    F.affine_grid(affine_t.unsqueeze(0), size=(1, 1, height, width), align_corners=False),
                    align_corners=False,
                    padding_mode="zeros",
                )[0, 0]
                warped_grid = np.clip(grid_warped_t.detach().cpu().numpy(), 0.0, 1.0)

    if not export_fig6b_only:
        _save_tensor_heat("raw_mask.png", raw_mask_prob, out_dir, max_side)
        _save_binary_mask("raw_mask_prediction.png", raw_mask_prob, out_dir, max_side)
        _save_tensor_heat("mask_features.png", mask_features_map, out_dir, max_side)
        _save_tensor_heat("aligned_prior.png", prior_mask_prob, out_dir, max_side)
        _save_tensor_heat("prior_gate.png", gate_map, out_dir, max_side)
        _save_tensor_heat("effective_gate.png", gate_map, out_dir, max_side)
        _save_tensor_heat("prior_occluder_abs.png", occ_map, out_dir, max_side)
        _save_tensor_heat_norm("prior_occluder.png", occ_map, out_dir, max_side)
        _save_signed_heat("residual_update.png", residual_update, out_dir, max_side)
        _save_tensor_heat("fused_mask.png", fused_mask_prob, out_dir, max_side)
        if selected_prior is not None:
            _save_tensor_gray("selected_prior.png", np.clip(selected_prior, 0.0, 1.0), out_dir, max_side)
        if warped_grid is not None:
            _save_tensor_gray("affine_grid.png", warped_grid, out_dir, max_side)

    _imwrite(os.path.join(out_dir, "final_mask.png"), (final_mask * 255).astype(np.uint8))
    overlay = _overlay_mask(img_bgr, final_mask.astype(np.float32), color_bgr=(0, 255, 0), alpha=0.45)
    if not export_fig6b_only:
        _imwrite(os.path.join(out_dir, "final_overlay.png"), overlay)
        gate_overlay = _overlay_heatmap(img_bgr, gate_map, alpha=gate_overlay_alpha)
        _imwrite(os.path.join(out_dir, "effective_gate_overlay.png"), gate_overlay)

    hi = _highlight_mask(img_bgr, v_thr=highlight_v_thr, s_max=highlight_s_max)
    hi_f = hi.astype(np.float32)
    hi_count = int(hi.sum())
    uncertainty_map = -np.abs(raw_mask_logits.astype(np.float32))
    # Fig.6b(a): norm(uncertainty) * highlight_mask
    prob_unc = _norm01(uncertainty_map) * hi_f
    prob_unc = _norm01(prob_unc)
    # Fig.6b(b): norm(gate * |prior - raw|) * highlight_mask
    support_map = gate_map.astype(np.float32) * np.abs(prior_mask_prob.astype(np.float32) - raw_mask_prob.astype(np.float32))
    prob_sup = _norm01(support_map) * hi_f
    prob_sup = _norm01(prob_sup)
    if not export_fig6a_only:
        _save_tensor_heat("sampling_prob_uncertainty.png", prob_unc, out_dir, max_side)
        _save_tensor_heat("sampling_prob_support.png", prob_sup, out_dir, max_side)
        _imwrite(
            os.path.join(out_dir, "sampling_prob_uncertainty_overlay.png"),
            _overlay_heatmap(img_bgr, prob_unc, alpha=gate_overlay_alpha),
        )
        _imwrite(
            os.path.join(out_dir, "sampling_prob_support_overlay.png"),
            _overlay_heatmap(img_bgr, prob_sup, alpha=gate_overlay_alpha),
        )
    if not export_fig6b_only:
        _save_dpc_flow_overview(
            out_dir,
            max_side,
            img_bgr,
            raw_mask_prob,
            prior_mask_prob,
            occ_map,
            gate_map,
            residual_update,
            fused_mask_prob,
            overlay,
        )

    with open(os.path.join(out_dir, "affine_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"query_idx={q_idx}\n")
        f.write(f"prior_path={str(getattr(cfg.MODEL.MASK_FORMER, 'PRIOR_PATH', ''))}\n")
        f.write(f"selected_bank_idx={selected_bank_idx}\n")
        if prior_bank_weight_results is not None:
            bank_weights = prior_bank_weight_results[0, q_idx].detach().float().cpu().numpy().tolist()
            f.write(f"prior_bank_weights={bank_weights}\n")
        if align_params_raw is not None:
            f.write(f"align_params_raw={align_params_raw.tolist()}\n")
        if align_params_effective is not None:
            for k, v in align_params_effective.items():
                f.write(f"{k}={v}\n")
        if affine_matrix is not None:
            f.write(f"affine_matrix={affine_matrix.tolist()}\n")

    if hi_count > 0:
        gate_hi_mean = float(gate_map[hi > 0].mean())
        prob_unc_hi_mean = float(prob_unc[hi > 0].mean())
        prob_sup_hi_mean = float(prob_sup[hi > 0].mean())
    else:
        gate_hi_mean = float("nan")
        prob_unc_hi_mean = float("nan")
        prob_sup_hi_mean = float("nan")
    prob_unc_mean = float(prob_unc.mean())
    prob_sup_mean = float(prob_sup.mean())
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"image={os.path.abspath(image_path)}\n")
        f.write(f"config={os.path.abspath(config_file)}\n")
        f.write(f"weights={os.path.abspath(weights)}\n")
        f.write(f"score_thr={score_thr}\n")
        f.write(f"query_idx={q_idx}\n")
        f.write(f"instance_score={best_score}\n")
        f.write(f"gate_mean={gate_mean}\n")
        f.write(f"gate_highlight_mean={gate_hi_mean}\n")
        f.write("sampling_prob_uncertainty_semantics=norm(uncertainty_map)*highlight_mask\n")
        f.write("sampling_prob_support_semantics=norm(gate_map*abs(prior_prob-raw_prob))*highlight_mask\n")
        f.write(f"prob_unc_mean={prob_unc_mean}\n")
        f.write(f"prob_unc_highlight_mean={prob_unc_hi_mean}\n")
        f.write(f"prob_sup_mean={prob_sup_mean}\n")
        f.write(f"prob_sup_highlight_mean={prob_sup_hi_mean}\n")
        f.write("pred_prior_gates_semantics=effective_gate_after_occluder_suppression\n")
        f.write(f"occluder_mean={occ_mean}\n")
        f.write("prior_occluder_png_semantics=normalized_for_visual_contrast\n")
        f.write("prior_occluder_abs_png_semantics=absolute_occluder_probability\n")
        f.write(f"residual_update_mean={float(np.mean(residual_update))}\n")
        f.write(f"residual_update_absmax={float(np.max(np.abs(residual_update)))}\n")
        f.write(f"selected_bank_idx={selected_bank_idx}\n")
        if prior_bank_weight_results is not None:
            bank_weights = prior_bank_weight_results[0, q_idx].detach().float().cpu().numpy().tolist()
            f.write(f"prior_bank_weights={bank_weights}\n")
        if align_params_raw is not None:
            f.write(f"align_params={align_params_raw.tolist()}\n")
        if affine_matrix is not None:
            f.write(f"affine_matrix={affine_matrix.tolist()}\n")
        f.write(f"prior_path={str(getattr(cfg.MODEL.MASK_FORMER, 'PRIOR_PATH', ''))}\n")

    return {
        "image": os.path.abspath(image_path),
        "out_dir": os.path.abspath(out_dir),
        "query_idx": int(q_idx),
        "instance_score": float(best_score),
        "gate_mean": float(gate_mean),
        "highlight_pixels": int(hi_count),
        "gate_highlight_mean": float(gate_hi_mean),
        "prob_unc_mean": float(prob_unc_mean),
        "prob_unc_highlight_mean": float(prob_unc_hi_mean),
        "prob_sup_mean": float(prob_sup_mean),
        "prob_sup_highlight_mean": float(prob_sup_hi_mean),
        "occluder_mean": float(occ_mean),
    }


def main() -> None:
    args = parse_args()
    if bool(args.export_fig6a_only) and bool(args.export_fig6b_only):
        raise ValueError("--export-fig6a-only 与 --export-fig6b-only 不能同时设置。")
    _ensure_dir(args.out_dir)
    image_paths = _resolve_image_paths(args)
    if not image_paths:
        raise ValueError("未解析到任何输入图片。请提供 --image / --images / --image-list / --dataset-root。")

    cfg = _build_cfg(
        os.path.abspath(args.config_file),
        os.path.abspath(args.weights),
        float(args.score_thr),
        str(args.prior_path).strip(),
        int(args.num_classes),
    )
    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(cfg.MODEL.WEIGHTS)

    multi = len(image_paths) > 1
    rows: List[dict] = []
    for i, image_path in enumerate(image_paths):
        if multi:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            case_dir = os.path.join(args.out_dir, f"{i:03d}_{stem}")
        else:
            case_dir = args.out_dir
        row = _run_one_image(
            predictor=predictor,
            cfg=cfg,
            config_file=str(args.config_file),
            weights=str(args.weights),
            image_path=image_path,
            out_dir=case_dir,
            score_thr=float(args.score_thr),
            max_side=int(max(64, args.max_side)),
            gate_overlay_alpha=float(args.gate_overlay_alpha),
            highlight_v_thr=int(args.highlight_v_thr),
            highlight_s_max=int(args.highlight_s_max),
            export_fig6a_only=bool(args.export_fig6a_only),
            export_fig6b_only=bool(args.export_fig6b_only),
        )
        rows.append(row)
        print(f"[{i+1}/{len(image_paths)}] done: {row['out_dir']}")

    summary_csv = os.path.join(args.out_dir, "gate_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "image",
            "out_dir",
            "query_idx",
            "instance_score",
            "gate_mean",
            "highlight_pixels",
            "gate_highlight_mean",
            "prob_unc_mean",
            "prob_unc_highlight_mean",
            "prob_sup_mean",
            "prob_sup_highlight_mean",
            "occluder_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"saved summary: {os.path.abspath(summary_csv)}")
    print(f"done. saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

