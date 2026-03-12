#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 ShapePriorAdapter + PointRend 的中间产物（与项目代码一致，不用“临时贴图”）：
- Input Image（可选：先注入 synthetic highlight）
- Aligned Prior（rotated_prior）
- Gate Map（GatedFusion 的 gate_map）
- Refined Feature（融合后的 ROI 特征图，可视化 channel-mean）
- Coarse Mask（coarse_head 输出）
- Upsample（PointRend subdivision 第一次 upsample 后的 mask_logits）
- Final Output（pred_masks + overlay）

用法示例：
  PYTHONPATH=... python visualize_shape_prior_flow.py \
    --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_plug.yaml \
    --weights output/plug_pointrend_ft/model_final.pth \
    --image /path/to/xxx.png \
    --out-dir ./output/shape_flow_vis \
    --shape-prior-npy /home/user/sjw/Yolo_pointrend/detectron2/plug_canonical_prior.npy
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# -------------------------------
# 控制台降噪：屏蔽与可视化结果无关的 warning
# -------------------------------
# fvcore/detectron2 内部 torch.load 的 FutureWarning（不影响可视化）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*",
)
# torch.meshgrid indexing 参数提示（不影响可视化）
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torch\.meshgrid.*indexing argument.*",
)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config

# trigger registry for ShapeAwareCoarseMaskHead
import custom_heads  # noqa: F401

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from PIL import Image


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-8:
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
    rgb = (jet(gray_u8.astype(np.float32) / 255.0)[..., :3] * 255.0).astype(np.uint8)  # (H,W,3) RGB
    return rgb


def _resize_to(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = int(size[0]), int(size[1])
    if _HAS_CV2:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)  # type: ignore[attr-defined]
    pil = Image.fromarray(img)
    pil2 = pil.resize((w, h), resample=Image.NEAREST)
    return np.array(pil2)


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
    Image.fromarray(img).save(path)


def _overlay_mask(bgr: np.ndarray, mask01: np.ndarray, color_bgr=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    m = (mask01 > 0.5).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError("mask must be HxW")
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:, :] = color_bgr
    out[m > 0] = (out[m > 0] * (1 - alpha) + overlay[m > 0] * alpha).astype(np.uint8)
    return out


def build_cfg(config_file: str, weights: str, score_thr: float) -> "object":
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_MASK_HEAD.NAME = "ShapeAwareCoarseMaskHead"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thr)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True, help="single image path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--shape-prior-npy", default="", help="override SHAPE_PRIOR_PATH")
    p.add_argument("--max-side", type=int, default=256, help="visualization canvas size for small tensors")
    p.add_argument(
        "--no-save-input",
        action="store_true",
        help="不保存 input_raw/input_highlight/input_image（当目录里已有人手准备好的输入图时使用）。",
    )
    # optional synthetic highlight for the input image (for the 5-step visualization list)
    p.add_argument("--highlight-severity", type=float, default=0.0, help=">0 时对输入图像注入合成高光（0~1）")
    p.add_argument("--highlight-spots", type=int, nargs=2, default=[1, 3])
    p.add_argument("--highlight-sigma", type=int, nargs=2, default=[30, 80])
    p.add_argument("--highlight-intensity", type=int, nargs=2, default=[150, 255])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)

    if str(args.shape_prior_npy).strip():
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(str(args.shape_prior_npy).strip())

    # enable debug taps in ShapeAwareCoarseMaskHead
    os.environ["SHAPE_PRIOR_DEBUG"] = "1"

    cfg = build_cfg(os.path.abspath(args.config_file), os.path.abspath(args.weights), args.score_thr)
    predictor = DefaultPredictor(cfg)
    # ensure weights loaded (DefaultPredictor does, but keep explicit in case of changes)
    DetectionCheckpointer(predictor.model).load(cfg.MODEL.WEIGHTS)

    img_bgr_raw = _imread_bgr(args.image)
    img_bgr = img_bgr_raw

    # optional: inject synthetic highlight to input (no object-focus by default)
    sev = float(max(0.0, min(1.0, float(getattr(args, "highlight_severity", 0.0)))))
    if sev > 0:
        from highlight_mapper import HighlightAugConfig, apply_synthetic_highlight

        i0, i1 = int(args.highlight_intensity[0]), int(args.highlight_intensity[1])
        i0s = int(round(i0 * sev))
        i1s = int(round(i1 * sev))
        i0s = max(0, min(255, i0s))
        i1s = max(0, min(255, i1s))
        if i1s < i0s:
            i0s, i1s = i1s, i0s

        hcfg = HighlightAugConfig(
            prob=1.0,
            spots_range=(int(args.highlight_spots[0]), int(args.highlight_spots[1])),
            sigma_range=(int(args.highlight_sigma[0]), int(args.highlight_sigma[1])),
            intensity_range=(i0s, i1s),
            focus_on_object=False,
            clip_to_object=False,
            object_mask_dilate=0,
            object_mask_feather=0,
        )
        rgb = img_bgr_raw[:, :, ::-1].copy()
        rgb2 = apply_synthetic_highlight(rgb, hcfg, dataset_dict=None)
        img_bgr = rgb2[:, :, ::-1].copy()
    else:
        pass

    # Save inputs only if requested (some users already have input_raw/highlight prepared)
    if not bool(getattr(args, "no_save_input", False)):
        _imwrite(os.path.join(args.out_dir, "input_raw.png"), img_bgr_raw)
        if sev > 0:
            _imwrite(os.path.join(args.out_dir, "input_highlight.png"), img_bgr)
        # always save the actual model input
        _imwrite(os.path.join(args.out_dir, "input_image.png"), img_bgr)

    outputs = predictor(img_bgr)
    inst = outputs.get("instances", None)
    if inst is None or len(inst) == 0:
        raise RuntimeError("no instances predicted (try lowering --score-thr)")

    # pick top-scoring instance
    idx = int(inst.scores.argmax().item()) if hasattr(inst, "scores") else 0
    pred_mask = inst.pred_masks[idx].to("cpu").numpy().astype(np.uint8)

    # pull debug cache from mask head
    mh = predictor.model.roi_heads.mask_head
    cache = getattr(mh, "shape_debug_cache", {}) or {}

    max_side = int(max(64, args.max_side))

    def save_tensor_heat(name: str, arr01: np.ndarray) -> None:
        gray = _to_u8_heat(arr01)
        colored = _colormap_jet(gray)
        colored = _resize_to(colored, (max_side, max_side))
        _imwrite(os.path.join(args.out_dir, name), colored)

    # Aligned Prior
    rp = cache.get("rotated_prior", None)
    if rp is not None:
        rp_np = rp.numpy() if hasattr(rp, "numpy") else np.asarray(rp)
        rp_np = np.squeeze(rp_np)
        save_tensor_heat("aligned_prior.png", rp_np)

    # Gate Map (0~1)
    gm = cache.get("gate_map", None)
    if gm is not None:
        gm_np = gm.numpy() if hasattr(gm, "numpy") else np.asarray(gm)
        gm_np = np.squeeze(gm_np)
        save_tensor_heat("gate_map.png", gm_np.astype(np.float32))

    # Refined feature (channel mean, normalized for visualization)
    rf = cache.get("refined_feature_mean", None)
    if rf is not None:
        rf_np = rf.numpy() if hasattr(rf, "numpy") else np.asarray(rf)
        rf_np = np.squeeze(rf_np)
        save_tensor_heat("refined_feature_mean.png", _norm01(rf_np))

    # Coarse Mask (sigmoid)
    cm = cache.get("coarse_mask_logits", None)
    if cm is not None:
        cm_np = cm.numpy() if hasattr(cm, "numpy") else np.asarray(cm)
        cm_np = np.squeeze(cm_np)
        cm_prob = _sigmoid(cm_np)
        save_tensor_heat("coarse_mask.png", cm_prob)

    # Upsample stages (mask_logits after each subdivision upsample)
    k = 1
    while True:
        up = cache.get(f"mask_logits_upsample{k}", None)
        if up is None:
            break
        up_np = up.numpy() if hasattr(up, "numpy") else np.asarray(up)
        up_np = np.squeeze(up_np)
        up_prob = _sigmoid(up_np)
        save_tensor_heat(f"upsample_step{k}.png", up_prob)
        k += 1

    # Final mask logits (if cached)
    fin = cache.get("mask_logits_final", None)
    if fin is not None:
        fin_np = fin.numpy() if hasattr(fin, "numpy") else np.asarray(fin)
        fin_np = np.squeeze(fin_np)
        fin_prob = _sigmoid(fin_np)
        save_tensor_heat("mask_logits_final.png", fin_prob)

    # Final output mask + overlay
    _imwrite(os.path.join(args.out_dir, "final_mask.png"), (pred_mask * 255).astype(np.uint8))
    overlay = _overlay_mask(img_bgr, pred_mask.astype(np.float32), color_bgr=(0, 255, 0), alpha=0.45)
    _imwrite(os.path.join(args.out_dir, "final_overlay.png"), overlay)

    # save some metadata
    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"image={os.path.abspath(args.image)}\n")
        f.write(f"config={os.path.abspath(args.config_file)}\n")
        f.write(f"weights={os.path.abspath(args.weights)}\n")
        f.write(f"score_thr={args.score_thr}\n")
        if "pred_angle" in cache:
            try:
                ang = float(cache["pred_angle"].numpy().reshape(-1)[0])
            except Exception:
                ang = None
            f.write(f"pred_angle_rad={ang}\n")

    print(f"done. saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()


