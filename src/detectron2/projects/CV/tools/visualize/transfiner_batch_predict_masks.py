#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 transfiner 环境中批量推理掩码，供主工程可视化脚本以子进程调用。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import cv2
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch predict masks using transfiner environment.")
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image-list", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--score-thr", type=float, default=0.5)
    return p.parse_args()


def _read_lines(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(os.path.abspath(s))
    return out


def _build_predictor(config_file: str, weights: str, score_thr: float) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.abspath(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thr)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(cfg.MODEL.WEIGHTS)
    predictor.model.eval()
    return predictor


def _predict_best_mask(predictor: DefaultPredictor, img_bgr: np.ndarray) -> np.ndarray:
    out = predictor(img_bgr)
    inst = out.get("instances", None)
    if inst is None or len(inst) == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    inst_cpu = inst.to("cpu")
    if not hasattr(inst_cpu, "pred_masks"):
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    if hasattr(inst_cpu, "scores"):
        idx = int(inst_cpu.scores.argmax().item())
    else:
        idx = 0
    return inst_cpu.pred_masks[idx].numpy().astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_paths = _read_lines(args.image_list)
    if not image_paths:
        raise RuntimeError("empty image-list")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_json)), "masks")
    os.makedirs(out_dir, exist_ok=True)
    predictor = _build_predictor(args.config_file, args.weights, args.score_thr)
    records = []
    for i, p in enumerate(image_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        m = _predict_best_mask(predictor, img)
        mp = os.path.join(out_dir, f"{i:06d}.png")
        cv2.imwrite(mp, (m > 0).astype(np.uint8) * 255)
        records.append({"image_path": p, "mask_path": mp})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)
    print(f"[transfiner_batch] wrote {len(records)} masks")


if __name__ == "__main__":
    main()

