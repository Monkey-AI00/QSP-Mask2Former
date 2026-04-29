#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定性验证：强曝光下“形状先验”是否有效。

做法：
- 在强曝光评测集（像素已被增强，但标注不变）上抽样若干图片
- 画出 GT mask（绿色）
- 分别用：
  1) 原版 PointRend（PointRendMaskHead）
  2) ShapePrior 版（ShapeAwareCoarseMaskHead）
  在同一张强曝光图上画预测 mask
- 保存并排对比图到 out_dir

示例：
  PYTHONPATH=... conda run -n pointrend python visualize_predictions_under_highlight.py \
    --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_plug.yaml \
    --dataset-root /home/users1/sjw/cursor/workspace/datasets/plug_train1_highlight_eval \
    --json-file plug_train.json \
    --weights-base /path/to/base/model_final.pth \
    --weights-prior /path/to/prior/model_final.pth \
    --shape-prior-npy /home/users1/sjw/cursor/workspace/outputs/plug_prior/plug_canonical_prior.npy \
    --out-dir /home/users1/sjw/cursor/workspace/outputs/output/pred_vis_highlight \
    --num 20 --seed 0
"""

from __future__ import annotations

import os
# 避免 matplotlib 默认缓存目录不可写导致的警告/性能问题
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import warnings

# 屏蔽一些与结果无关的噪声警告（不影响训练/推理正确性）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torch\.meshgrid.*indexing argument.*",
)

import argparse
import random
from typing import List, Tuple

import cv2
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as d2_utils
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import ColorMode, Visualizer

# 触发注册：ShapeAwareCoarseMaskHead
import custom_heads  # noqa: F401

# 复用你的数据集注册逻辑（会过滤掉空类别 / background）
from train_plug import register_plug_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative comparison under synthetic highlight.")
    p.add_argument("--config-file", required=True, help="PointRend plug 配置文件路径")
    p.add_argument("--dataset-root", required=True, help="强曝光评测集目录（图片+json）")
    p.add_argument("--dataset-name", default="plug_highlight_eval_vis", help="注册到 detectron2 的数据集名称")
    p.add_argument("--json-file", default="plug_train.json", help="COCO json 文件名（相对 dataset-root）")
    p.add_argument("--weights-base", required=True, help="原版 PointRend 权重（不带 shape prior）")
    p.add_argument("--weights-prior", required=True, help="带 shape prior 的权重（ShapeAwareCoarseMaskHead）")
    p.add_argument("--shape-prior-npy", default="", help="Phase-1 输出的 .npy（留空则用默认路径或环境变量）")
    p.add_argument("--out-dir", default="./output/pred_vis_highlight", help="输出目录")
    p.add_argument("--num", type=int, default=20, help="抽样图片数量")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--score-thr", type=float, default=0.5, help="可视化过滤分数阈值")
    return p.parse_args()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _build_predictor(
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
    cfg.MODEL.DEVICE = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or True) else "cpu"
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    # 显式 load 一次（DefaultPredictor 内部也会 load，但这里确保日志更直观）
    DetectionCheckpointer(predictor.model).load(weights)
    predictor.model.eval()
    return predictor


def _viz_instances(img_bgr: np.ndarray, metadata, instances, title: str) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    v = Visualizer(img_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    v = v.draw_instance_predictions(instances.to("cpu"))
    out = v.get_image()
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # title bar
    cv2.rectangle(out_bgr, (0, 0), (min(520, out_bgr.shape[1] - 1), 32), (0, 0, 0), thickness=-1)
    cv2.putText(out_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out_bgr


def _viz_gt(img_bgr: np.ndarray, metadata, dataset_dict, title: str) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    v = Visualizer(img_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    v = v.draw_dataset_dict(dataset_dict)
    out = v.get_image()
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (0, 0), (min(520, out_bgr.shape[1] - 1), 32), (0, 0, 0), thickness=-1)
    cv2.putText(out_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out_bgr


def _viz_raw(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out_bgr = img_bgr.copy()
    cv2.rectangle(out_bgr, (0, 0), (min(520, out_bgr.shape[1] - 1), 32), (0, 0, 0), thickness=-1)
    cv2.putText(out_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out_bgr


def _stack_horiz(imgs: List[np.ndarray]) -> np.ndarray:
    # pad to same height
    h = max(im.shape[0] for im in imgs)
    outs = []
    for im in imgs:
        if im.shape[0] == h:
            outs.append(im)
            continue
        pad = h - im.shape[0]
        outs.append(cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return np.concatenate(outs, axis=1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # 设置 shape prior 路径（ShapeAwareCoarseMaskHead 会从环境变量读取）
    if args.shape_prior_npy.strip():
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(args.shape_prior_npy.strip())

    dataset_name, num_classes = register_plug_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        json_filename=args.json_file,
    )
    metadata = MetadataCatalog.get(dataset_name)
    dicts = DatasetCatalog.get(dataset_name)
    if not dicts:
        raise RuntimeError("dataset is empty")

    picks = random.sample(dicts, k=min(int(args.num), len(dicts)))
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)

    # 两个 predictor
    base_pred = _build_predictor(
        config_file=args.config_file,
        weights=os.path.abspath(args.weights_base),
        mask_head_name="PointRendMaskHead",
        num_classes=num_classes,
        score_thr=args.score_thr,
    )
    prior_pred = _build_predictor(
        config_file=args.config_file,
        weights=os.path.abspath(args.weights_prior),
        mask_head_name="ShapeAwareCoarseMaskHead",
        num_classes=num_classes,
        score_thr=args.score_thr,
    )

    print(f"out_dir: {out_dir}")
    print(f"num_samples: {len(picks)}")

    for i, d in enumerate(picks):
        img_path = d["file_name"]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[skip] failed to read: {img_path}")
            continue

        # predictor 输入默认是 BGR uint8（detectron2 默认 FORMAT=BGR）
        base_out = base_pred(img_bgr)
        prior_out = prior_pred(img_bgr)

        raw_vis = _viz_raw(img_bgr, "RAW highlighted image")
        gt_vis = _viz_gt(img_bgr, metadata, d, "GT overlay")
        base_vis = _viz_instances(img_bgr, metadata, base_out["instances"], "BASE pred (no prior)")
        prior_vis = _viz_instances(img_bgr, metadata, prior_out["instances"], "PRIOR pred (shape prior)")

        side = _stack_horiz([raw_vis, gt_vis, base_vis, prior_vis])

        stem = f"{i:03d}_" + os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.png")
        cv2.imwrite(out_path, side)
        print(f"[{i+1}/{len(picks)}] saved: {out_path}")


if __name__ == "__main__":
    main()


