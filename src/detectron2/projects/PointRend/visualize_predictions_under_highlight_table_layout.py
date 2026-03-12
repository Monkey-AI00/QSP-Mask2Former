#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格排版的定性对比图（每行 5 列）：
  col1: Input（高反光）
  col2: GroundTruth
  col3: BASE (no prior) [可选]
  col4: Mask R-CNN [可选]
  col5: Mask2Former
  col6: Mask2Former + SDF
  col7: SPG-PointRend（ShapeAwareCoarseMaskHead）

并在“高反光区域内”的错误位置自动画红色虚线框（以 XOR(GT,Pred) ∩ Highlight 为主）。

示例：
  PYTHONPATH=/path/to/detectron2:/path/to/detectron2/projects/PointRend:/path/to/Mask2Former \
  python -u visualize_predictions_under_highlight_table_layout.py \
    --dataset-root /home/user/sjw/Yolo_pointrend/detectron2/plug_train1_highlight_eval \
    --json-file plug_train.json \
    --out-dir ./output/pred_vis_table1 \
    --mask2former-root /home/user/sjw/Yolo_pointrend/Mask2Former \
    --config-mask2former /home/user/sjw/Yolo_pointrend/detectron2/projects/PointRend/configs/InstanceSegmentation/mask2former_R50_plug.yaml \
    --weights-mask2former /path/to/m2f_base/model_final.pth \
    --weights-mask2former-sdf /path/to/m2f_sdf/model_final.pth \
    --config-pointrend /home/user/sjw/Yolo_pointrend/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_plug.yaml \
    --weights-spg /path/to/spg_pointrend/model_final.pth \
    --shape-prior-npy /home/user/sjw/Yolo_pointrend/detectron2/plug_canonical_prior.npy \
    --num-rows 4 --scan-max 200 --score-thr 0.5
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config

# 触发注册：ShapeAwareCoarseMaskHead（SPG-PointRend）
import custom_heads  # noqa: F401

from highlight_mapper import _ann_to_mask  # type: ignore
from train_plug import register_plug_dataset


warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.load.*weights_only=False.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*torch\.meshgrid.*indexing argument.*")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Table-layout qualitative comparison under highlight.")
    p.add_argument("--dataset-root", required=True, help="强曝光评测集目录（图片+json）")
    p.add_argument(
        "--clean-root",
        default="",
        help="可选：对应的 clean 数据集目录（与 dataset-root 共享相同 file_name 相对路径）。提供后可用于去掉强光周围光晕。",
    )
    p.add_argument("--dataset-name", default="plug_highlight_eval_table_vis", help="注册到 detectron2 的数据集名称")
    p.add_argument("--json-file", default="plug_train.json", help="COCO json 文件名（相对 dataset-root）")
    p.add_argument("--out-dir", default="./output/pred_vis_table1", help="输出目录")
    p.add_argument("--seed", type=int, default=0)

    # selection: pick images with strong highlight
    p.add_argument("--num-rows", type=int, default=4, help="输出多少行（多少张图）")
    p.add_argument("--scan-max", type=int, default=200, help="最多扫描多少张图用于挑“高反光”样本")
    p.add_argument("--highlight-v-thr", type=int, default=245, help="HSV 的 V 阈值（高亮）")
    p.add_argument("--highlight-s-max", type=int, default=70, help="HSV 的 S 上限（偏白高亮）")
    p.add_argument("--highlight-dilate", type=int, default=7, help="高亮区域膨胀像素（扩大关注区域）")

    # visualization
    p.add_argument("--cell-width", type=int, default=520, help="每个格子的宽度（像素），高度按比例缩放")
    p.add_argument("--box-topk", type=int, default=1, help="每个方法最多画几个红框")
    p.add_argument("--box-min-area", type=int, default=120, help="红框最小面积阈值（像素）")
    p.add_argument("--box-pad", type=int, default=8, help="红框外扩像素")
    p.add_argument(
        "--pred-erode",
        type=int,
        default=0,
        help="可选：叠加可视化前先腐蚀预测 mask（像素，建议 2~4）以减弱边缘淡色光晕；不影响红框定位。",
    )
    p.add_argument("--score-thr", type=float, default=0.5, help="可视化过滤分数阈值")

    # Mask2Former
    p.add_argument("--mask2former-root", required=True, help="Mask2Former 项目根目录")
    p.add_argument("--config-mask2former", required=True, help="Mask2Former config yaml")
    p.add_argument("--weights-mask2former", required=True, help="Mask2Former 权重")
    p.add_argument("--weights-mask2former-sdf", required=True, help="Mask2Former+SDF 权重")

    # SPG-PointRend
    p.add_argument("--config-pointrend", required=True, help="PointRend plug config yaml")
    p.add_argument("--weights-base", default="", help="BASE(no prior) PointRendMaskHead 权重（可选；提供后会插入一列）")
    p.add_argument("--weights-spg", required=True, help="SPG-PointRend 权重（ShapeAwareCoarseMaskHead）")
    p.add_argument("--shape-prior-npy", default="", help="shape prior .npy（用于 ShapeAwareCoarseMaskHead）")

    # Mask R-CNN (optional)
    p.add_argument("--config-maskrcnn", default="", help="Mask R-CNN config yaml（可选；提供 weights 后必须提供）")
    p.add_argument("--weights-maskrcnn", default="", help="Mask R-CNN 权重（可选；提供后会插入一列）")

    return p.parse_args()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _title_bar(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    h = out.shape[0]
    w = out.shape[1]
    cv2.rectangle(out, (0, 0), (w - 1, min(34, h - 1)), (0, 0, 0), thickness=-1)
    cv2.putText(out, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _resize_to_width(img_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    scale = float(width) / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (width, nh), interpolation=cv2.INTER_AREA)


def _stack_row(imgs: List[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    outs = []
    for im in imgs:
        if im.shape[0] == h:
            outs.append(im)
        else:
            pad = h - im.shape[0]
            outs.append(cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return np.concatenate(outs, axis=1)


def _stack_grid(rows: List[np.ndarray]) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    outs = []
    for r in rows:
        if r.shape[1] == w:
            outs.append(r)
        else:
            pad = w - r.shape[1]
            outs.append(cv2.copyMakeBorder(r, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return np.concatenate(outs, axis=0)


def _overlay_mask(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    *,
    color_bgr: Tuple[int, int, int],
    alpha: float = 0.35,
    outline: bool = True,
) -> np.ndarray:
    out = img_bgr.copy()
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return out
    color = np.array(color_bgr, dtype=np.uint8)[None, None, :]
    out[m > 0] = (
        out[m > 0].astype(np.float32) * (1.0 - float(alpha)) + color.astype(np.float32) * float(alpha)
    ).astype(np.uint8)
    if outline:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color_bgr, thickness=2, lineType=cv2.LINE_AA)
    return out


def _erode_mask(mask01: np.ndarray, erode_px: int) -> np.ndarray:
    """
    对二值 mask 做形态学腐蚀，用于“显示时”收紧边界，减弱半透明叠加造成的淡色光晕。
    """
    ep = int(max(0, erode_px))
    if ep <= 0:
        return (mask01 > 0).astype(np.uint8)
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return m
    kernel = np.ones((3, 3), dtype=np.uint8)
    m2 = cv2.erode(m, kernel, iterations=ep)
    return (m2 > 0).astype(np.uint8)


def _highlight_mask(img_bgr: np.ndarray, v_thr: int, s_max: int, dilate_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hi = (v >= int(v_thr)) & (s <= int(s_max))
    m = hi.astype(np.uint8)
    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)
    return (m > 0).astype(np.uint8)


def _highlight_score(img_bgr: np.ndarray, v_thr: int, s_max: int) -> float:
    m = _highlight_mask(img_bgr, v_thr=v_thr, s_max=s_max, dilate_px=0)
    return float(m.mean())


def _draw_dashed_line(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color, thickness: int, dash: int, gap: int) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length <= 0:
        return
    for i in range(0, length, dash + gap):
        t1 = i / length
        t2 = min(i + dash, length) / length
        xa = int(round(x1 + (x2 - x1) * t1))
        ya = int(round(y1 + (y2 - y1) * t1))
        xb = int(round(x1 + (x2 - x1) * t2))
        yb = int(round(y1 + (y2 - y1) * t2))
        cv2.line(img, (xa, ya), (xb, yb), color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, color=(0, 0, 255), thickness: int = 2) -> None:
    dash, gap = 10, 6
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash, gap)
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash, gap)
    _draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness, dash, gap)
    _draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness, dash, gap)


def _draw_solid_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, color=(0, 0, 255), thickness: int = 2) -> None:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness, lineType=cv2.LINE_AA)


def _boxes_from_error(
    *,
    gt01: np.ndarray,
    pr01: np.ndarray,
    hi01: np.ndarray,
    topk: int,
    min_area: int,
    pad: int,
) -> List[Tuple[int, int, int, int]]:
    gt = (gt01 > 0).astype(np.uint8)
    pr = (pr01 > 0).astype(np.uint8)
    hi = (hi01 > 0).astype(np.uint8)
    err = cv2.bitwise_xor(gt, pr)
    err = cv2.bitwise_and(err, hi)
    if int(err.sum()) == 0:
        return []
    # connected components -> boxes
    num, labels, stats, _ = cv2.connectedComponentsWithStats(err, connectivity=8)
    boxes: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i].tolist()
        if area < int(min_area):
            continue
        boxes.append((area, x, y, x + w - 1, y + h - 1))
    boxes.sort(reverse=True, key=lambda t: t[0])
    out: List[Tuple[int, int, int, int]] = []
    H, W = gt.shape[:2]
    for area, x1, y1, x2, y2 in boxes[: int(topk)]:
        x1 = max(0, x1 - int(pad))
        y1 = max(0, y1 - int(pad))
        x2 = min(W - 1, x2 + int(pad))
        y2 = min(H - 1, y2 + int(pad))
        out.append((x1, y1, x2, y2))
    return out


def _predict_best_mask(predictor: DefaultPredictor, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], object]:
    out = predictor(img_bgr)
    inst = out.get("instances", None)
    if inst is None or len(inst) == 0:
        return None, inst
    inst_cpu = inst.to("cpu")
    if hasattr(inst_cpu, "scores"):
        idx = int(inst_cpu.scores.argmax().item())
    else:
        idx = 0
    if not hasattr(inst_cpu, "pred_masks"):
        return None, inst_cpu
    m = inst_cpu.pred_masks[idx].numpy().astype(np.uint8)
    return m, inst_cpu


def _safe_imread(p: str) -> Optional[np.ndarray]:
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def _load_clean_match(*, clean_root: str, highlight_root: str, highlight_path: str) -> Optional[np.ndarray]:
    """
    尝试从 clean_root 里找到与 highlight_path 对应的原图：
      clean_path = clean_root / relpath(highlight_path, highlight_root)
    """
    if not str(clean_root).strip():
        return None
    try:
        rel = os.path.relpath(highlight_path, highlight_root)
    except Exception:
        rel = os.path.basename(highlight_path)
    cand = os.path.join(clean_root, rel)
    if os.path.isfile(cand):
        return _safe_imread(cand)
    # fallback: basename match
    cand2 = os.path.join(clean_root, os.path.basename(highlight_path))
    if os.path.isfile(cand2):
        return _safe_imread(cand2)
    return None


def _remove_halo_with_clean(*, img_highlight_bgr: np.ndarray, img_clean_bgr: np.ndarray, obj_mask01: np.ndarray) -> np.ndarray:
    """
    去光晕：把“强光引入的增量”严格裁剪在目标内部。
      delta = highlight - clean
      out = clean + delta * obj_mask
    这样能去掉目标外部的光晕/泛白，同时保留目标内部强光。
    """
    if img_clean_bgr.shape[:2] != img_highlight_bgr.shape[:2]:
        img_clean_bgr = cv2.resize(img_clean_bgr, (img_highlight_bgr.shape[1], img_highlight_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    m = (obj_mask01 > 0).astype(np.float32)[..., None]
    hi = img_highlight_bgr.astype(np.float32)
    cl = img_clean_bgr.astype(np.float32)
    delta = hi - cl
    out = cl + delta * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_pointrend_predictor(
    *,
    config_file: str,
    weights: str,
    mask_head_name: str,
    num_classes: int,
    score_thr: float,
) -> DefaultPredictor:
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_MASK_HEAD.NAME = mask_head_name
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = int(num_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thr)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(weights)
    predictor.model.eval()
    return predictor


def _build_mask2former_predictor(
    *,
    mask2former_root: str,
    config_file: str,
    weights: str,
    num_classes: int,
    score_thr: float,
) -> DefaultPredictor:
    m2f_root = os.path.abspath(str(mask2former_root))
    if m2f_root not in sys.path:
        sys.path.insert(0, m2f_root)
    # trigger registration
    import mask2former as _mask2former  # noqa: F401  # type: ignore

    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config  # type: ignore

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    # instance-only
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = float(score_thr)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(weights)
    predictor.model.eval()
    return predictor


def _build_maskrcnn_predictor(
    *,
    config_file: str,
    weights: str,
    num_classes: int,
    score_thr: float,
) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thr)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(weights)
    predictor.model.eval()
    return predictor


def _gt_mask_from_dict(d: dict, h: int, w: int) -> Optional[np.ndarray]:
    annos = d.get("annotations", [])
    if not annos:
        return None
    ann = annos[0]
    m = _ann_to_mask(ann, h=h, w=w)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    if str(args.shape_prior_npy).strip():
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(str(args.shape_prior_npy).strip())

    dataset_name, num_classes = register_plug_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        json_filename=args.json_file,
    )
    metadata = MetadataCatalog.get(dataset_name)
    dicts = DatasetCatalog.get(dataset_name)
    if not dicts:
        raise RuntimeError("dataset is empty")

    # scan for highlighty images
    cand = list(dicts)
    random.shuffle(cand)
    cand = cand[: max(1, int(args.scan_max))]
    scored: List[Tuple[float, dict]] = []
    for d in cand:
        img = cv2.imread(d["file_name"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        scored.append((_highlight_score(img, v_thr=args.highlight_v_thr, s_max=args.highlight_s_max), d))
    scored.sort(reverse=True, key=lambda t: t[0])
    picks = [d for _, d in scored[: max(1, int(args.num_rows))]]
    if not picks:
        raise RuntimeError("no valid images found for visualization")

    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)

    # predictors
    base_pred: Optional[DefaultPredictor] = None
    if str(getattr(args, "weights_base", "")).strip():
        base_pred = _build_pointrend_predictor(
            config_file=os.path.abspath(args.config_pointrend),
            weights=os.path.abspath(args.weights_base),
            mask_head_name="PointRendMaskHead",
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )

    maskrcnn_pred: Optional[DefaultPredictor] = None
    if str(getattr(args, "weights_maskrcnn", "")).strip():
        if not str(getattr(args, "config_maskrcnn", "")).strip():
            raise ValueError("提供 --weights-maskrcnn 时必须同时提供 --config-maskrcnn")
        maskrcnn_pred = _build_maskrcnn_predictor(
            config_file=os.path.abspath(str(args.config_maskrcnn).strip()),
            weights=os.path.abspath(str(args.weights_maskrcnn).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )

    m2f_pred = _build_mask2former_predictor(
        mask2former_root=args.mask2former_root,
        config_file=os.path.abspath(args.config_mask2former),
        weights=os.path.abspath(args.weights_mask2former),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    m2f_sdf_pred = _build_mask2former_predictor(
        mask2former_root=args.mask2former_root,
        config_file=os.path.abspath(args.config_mask2former),
        weights=os.path.abspath(args.weights_mask2former_sdf),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    spg_pred = _build_pointrend_predictor(
        config_file=os.path.abspath(args.config_pointrend),
        weights=os.path.abspath(args.weights_spg),
        mask_head_name="ShapeAwareCoarseMaskHead",
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )

    rows_img: List[np.ndarray] = []
    for i, d in enumerate(picks):
        img_path = d["file_name"]
        img_bgr = _safe_imread(img_path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        gt = _gt_mask_from_dict(d, h=h, w=w)
        if gt is None:
            continue

        # 输入图：如果提供 clean-root，则把强光“增量”裁剪到目标内部，去掉周围光晕
        input_bgr = img_bgr
        if str(getattr(args, "clean_root", "")).strip():
            clean_bgr = _load_clean_match(clean_root=str(args.clean_root).strip(), highlight_root=str(args.dataset_root).strip(), highlight_path=img_path)
            if clean_bgr is not None:
                input_bgr = _remove_halo_with_clean(img_highlight_bgr=img_bgr, img_clean_bgr=clean_bgr, obj_mask01=gt)

        hi = _highlight_mask(img_bgr, v_thr=args.highlight_v_thr, s_max=args.highlight_s_max, dilate_px=args.highlight_dilate)

        base_mask: Optional[np.ndarray] = None
        if base_pred is not None:
            base_mask, _ = _predict_best_mask(base_pred, img_bgr)
        maskrcnn_mask: Optional[np.ndarray] = None
        if maskrcnn_pred is not None:
            maskrcnn_mask, _ = _predict_best_mask(maskrcnn_pred, img_bgr)
        m2f_mask, _ = _predict_best_mask(m2f_pred, img_bgr)
        m2f_sdf_mask, _ = _predict_best_mask(m2f_sdf_pred, img_bgr)
        spg_mask, _ = _predict_best_mask(spg_pred, img_bgr)

        # fill empty with zeros for downstream ops
        if base_pred is not None and base_mask is None:
            base_mask = np.zeros_like(gt)
        if maskrcnn_pred is not None and maskrcnn_mask is None:
            maskrcnn_mask = np.zeros_like(gt)
        if m2f_mask is None:
            m2f_mask = np.zeros_like(gt)
        if m2f_sdf_mask is None:
            m2f_sdf_mask = np.zeros_like(gt)
        if spg_mask is None:
            spg_mask = np.zeros_like(gt)

        # visualization-only erosion (does not change error boxes)
        ep = int(getattr(args, "pred_erode", 0))
        base_mask_vis = _erode_mask(base_mask, ep) if (base_mask is not None) else None
        maskrcnn_mask_vis = _erode_mask(maskrcnn_mask, ep) if (maskrcnn_mask is not None) else None
        m2f_mask_vis = _erode_mask(m2f_mask, ep)
        m2f_sdf_mask_vis = _erode_mask(m2f_sdf_mask, ep)
        spg_mask_vis = _erode_mask(spg_mask, ep)

        # visual cells
        input_vis = _title_bar(input_bgr, "Input (highlight)")
        gt_vis = _title_bar(_overlay_mask(input_bgr, gt, color_bgr=(0, 255, 0), alpha=0.35, outline=True), "GroundTruth")

        base_vis: Optional[np.ndarray] = None
        if base_pred is not None and base_mask_vis is not None:
            base_vis = _overlay_mask(input_bgr, base_mask_vis, color_bgr=(255, 170, 120), alpha=0.35, outline=True)
            for x1, y1, x2, y2 in _boxes_from_error(
                gt01=gt, pr01=base_mask, hi01=hi, topk=args.box_topk, min_area=args.box_min_area, pad=args.box_pad
            ):
                _draw_solid_rect(base_vis, x1, y1, x2, y2, color=(0, 0, 255), thickness=2)
            base_vis = _title_bar(base_vis, "BASE (no prior)")

        maskrcnn_vis: Optional[np.ndarray] = None
        if maskrcnn_pred is not None and maskrcnn_mask_vis is not None:
            maskrcnn_vis = _overlay_mask(input_bgr, maskrcnn_mask_vis, color_bgr=(120, 200, 255), alpha=0.35, outline=True)
            for x1, y1, x2, y2 in _boxes_from_error(
                gt01=gt, pr01=maskrcnn_mask, hi01=hi, topk=args.box_topk, min_area=args.box_min_area, pad=args.box_pad
            ):
                _draw_dashed_rect(maskrcnn_vis, x1, y1, x2, y2, color=(0, 0, 255), thickness=2)
            maskrcnn_vis = _title_bar(maskrcnn_vis, "Mask R-CNN")

        m2f_vis = _overlay_mask(input_bgr, m2f_mask_vis, color_bgr=(200, 120, 255), alpha=0.35, outline=True)
        for x1, y1, x2, y2 in _boxes_from_error(
            gt01=gt, pr01=m2f_mask, hi01=hi, topk=args.box_topk, min_area=args.box_min_area, pad=args.box_pad
        ):
            _draw_dashed_rect(m2f_vis, x1, y1, x2, y2, color=(0, 0, 255), thickness=2)
        m2f_vis = _title_bar(m2f_vis, "Mask2Former")

        sdf_vis = _overlay_mask(input_bgr, m2f_sdf_mask_vis, color_bgr=(255, 200, 80), alpha=0.35, outline=True)
        for x1, y1, x2, y2 in _boxes_from_error(
            gt01=gt, pr01=m2f_sdf_mask, hi01=hi, topk=args.box_topk, min_area=args.box_min_area, pad=args.box_pad
        ):
            _draw_solid_rect(sdf_vis, x1, y1, x2, y2, color=(0, 0, 255), thickness=2)
        sdf_vis = _title_bar(sdf_vis, "Mask2Former + SDF")

        spg_vis = _overlay_mask(input_bgr, spg_mask_vis, color_bgr=(80, 255, 150), alpha=0.35, outline=True)
        for x1, y1, x2, y2 in _boxes_from_error(
            gt01=gt, pr01=spg_mask, hi01=hi, topk=args.box_topk, min_area=args.box_min_area, pad=args.box_pad
        ):
            _draw_solid_rect(spg_vis, x1, y1, x2, y2, color=(0, 0, 255), thickness=2)
        spg_vis = _title_bar(spg_vis, "SPG-PointRend (ours)")

        cells: List[np.ndarray] = [input_vis, gt_vis]
        if base_vis is not None:
            cells.append(base_vis)
        if maskrcnn_vis is not None:
            cells.append(maskrcnn_vis)
        cells.extend([m2f_vis, sdf_vis, spg_vis])
        cells = [_resize_to_width(c, int(args.cell_width)) for c in cells]
        row = _stack_row(cells)
        rows_img.append(row)

        # per-row also save
        stem = f"{i:02d}_" + os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_row.png"), row)

    grid = _stack_grid(rows_img)
    out_path = os.path.join(out_dir, "table_layout_grid.png")
    cv2.imwrite(out_path, grid)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

