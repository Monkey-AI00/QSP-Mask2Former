#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 M2F-QSP（Mask2Former + Query-Aligned Shape Prior）的中间产物：

- Input Image
- Raw Mask Logits (pred_masks_raw)
- Aligned Prior (pred_prior_masks)
- Prior Gate (pred_prior_gates，当前为 spatial gate map)
- Occluder Suppression (pred_prior_occluders，可视化线缆/遮挡抑制区域)
- Fused Mask Logits (pred_masks)
- Final Output Overlay

说明：
- 该脚本不会走 MaskFormer 推理后的封装结果，而是手动执行
  backbone -> sem_seg_head，直接拿 raw outputs 进行可视化。
- 会基于最终实例推理得分，选取“最高分实例对应的 query”来展示中间产物。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    gray = _to_u8_heat(arr01)
    colored = _colormap_jet(gray)
    colored = _resize_to(colored, (max_side, max_side))
    _imwrite(os.path.join(out_dir, name), colored)


def _save_tensor_gray(name: str, arr01: np.ndarray, out_dir: str, max_side: int) -> None:
    gray = _to_u8_heat(arr01)
    gray = _resize_to(gray, (max_side, max_side))
    _imwrite(os.path.join(out_dir, name), gray)


def _save_binary_mask(name: str, mask01: np.ndarray, out_dir: str, max_side: int) -> None:
    mask_u8 = ((mask01 > 0.5).astype(np.uint8) * 255)
    mask_u8 = _resize_to(mask_u8, (max_side, max_side))
    _imwrite(os.path.join(out_dir, name), mask_u8)


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
    p.add_argument("--image", required=True, help="single image path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--prior-path", default="", help="override MODEL.MASK_FORMER.PRIOR_PATH")
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--max-side", type=int, default=256, help="visualization canvas size for small tensors")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)

    cfg = _build_cfg(
        os.path.abspath(args.config_file),
        os.path.abspath(args.weights),
        float(args.score_thr),
        str(args.prior_path).strip(),
        int(args.num_classes),
    )
    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(cfg.MODEL.WEIGHTS)
    model = predictor.model
    model.eval()

    img_bgr = _imread_bgr(args.image)
    _imwrite(os.path.join(args.out_dir, "input_image.png"), img_bgr)

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
            raw_mask_prob = fused_mask_prob

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

    max_side = int(max(64, args.max_side))
    _save_tensor_heat("raw_mask.png", raw_mask_prob, args.out_dir, max_side)
    _save_binary_mask("raw_mask_prediction.png", raw_mask_prob, args.out_dir, max_side)
    _save_tensor_heat("mask_features.png", mask_features_map, args.out_dir, max_side)
    _save_tensor_heat("aligned_prior.png", prior_mask_prob, args.out_dir, max_side)
    _save_tensor_heat("prior_gate.png", gate_map, args.out_dir, max_side)
    _save_tensor_heat("prior_occluder.png", occ_map, args.out_dir, max_side)
    _save_tensor_heat("fused_mask.png", fused_mask_prob, args.out_dir, max_side)
    if selected_prior is not None:
        _save_tensor_gray("selected_prior.png", np.clip(selected_prior, 0.0, 1.0), args.out_dir, max_side)
    if warped_grid is not None:
        _save_tensor_gray("affine_grid.png", warped_grid, args.out_dir, max_side)

    _imwrite(os.path.join(args.out_dir, "final_mask.png"), (final_mask * 255).astype(np.uint8))
    overlay = _overlay_mask(img_bgr, final_mask.astype(np.float32), color_bgr=(0, 255, 0), alpha=0.45)
    _imwrite(os.path.join(args.out_dir, "final_overlay.png"), overlay)

    with open(os.path.join(args.out_dir, "affine_params.txt"), "w", encoding="utf-8") as f:
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

    with open(os.path.join(args.out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"image={os.path.abspath(args.image)}\n")
        f.write(f"config={os.path.abspath(args.config_file)}\n")
        f.write(f"weights={os.path.abspath(args.weights)}\n")
        f.write(f"score_thr={args.score_thr}\n")
        f.write(f"query_idx={q_idx}\n")
        f.write(f"instance_score={best_score}\n")
        f.write(f"gate_mean={gate_mean}\n")
        f.write(f"occluder_mean={occ_mean}\n")
        f.write(f"selected_bank_idx={selected_bank_idx}\n")
        if prior_bank_weight_results is not None:
            bank_weights = prior_bank_weight_results[0, q_idx].detach().float().cpu().numpy().tolist()
            f.write(f"prior_bank_weights={bank_weights}\n")
        if align_params_raw is not None:
            f.write(f"align_params={align_params_raw.tolist()}\n")
        if affine_matrix is not None:
            f.write(f"affine_matrix={affine_matrix.tolist()}\n")
        f.write(f"prior_path={str(getattr(cfg.MODEL.MASK_FORMER, 'PRIOR_PATH', ''))}\n")

    print(f"done. saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

