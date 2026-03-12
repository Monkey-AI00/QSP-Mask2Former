#!/usr/bin/env python3
"""
YOLO 推理脚本：输出检测框中心点像素坐标 (u, v)
"""

from ultralytics import YOLO
import os
import argparse
from typing import Union

import cv2
import numpy as np

# 允许从任意工作目录运行该脚本：把脚本所在目录加入 sys.path，方便本地导入同目录模块
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from yolo_utils import infer_detections, load_yolo


def predict_uv(weights: str, source: str, conf: float = 0.25, device: Union[str, int] = 0):
    """
    使用 YOLO 推理并输出像素坐标 (u, v)。

    (u, v) 定义为检测框中心点的像素坐标：
      u = x_center, v = y_center
    坐标原点在图像左上角，u 向右，v 向下。
    """
    weights = str(weights)
    source = str(source)

    if not os.path.exists(weights):
        raise FileNotFoundError(f"未找到权重文件: {weights}")
    if not os.path.exists(source):
        raise FileNotFoundError(f"未找到输入图片/目录: {source}")

    model = load_yolo(weights)

    img = cv2.imread(source)
    if img is None:
        raise ValueError(f"无法读取图片: {source}（当前脚本只支持单张图片路径）")

    dets = infer_detections(model, img, conf=conf, device=device)
    print(f"\n=== {source} ===")
    if len(dets) == 0:
        print("未检测到目标")
        return

    for i, d in enumerate(dets):
        u, v = d.uv
        print(f"[{i}] cls={d.cls} conf={d.conf:.3f}  (u,v)=({u:.1f},{v:.1f})  int=({int(round(u))},{int(round(v))})")


def main():
    parser = argparse.ArgumentParser(description="YOLO 推理输出 (u,v) 像素坐标")
    parser.add_argument("--weights", required=True, help="推理用权重路径，如 runs/detect/xxx/weights/best.pt")
    parser.add_argument("--source", required=True, help="推理输入图片/目录路径")
    parser.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值")
    parser.add_argument("--device", default=0, help="推理设备（0/1/... 或 'cpu'）")
    args = parser.parse_args()

    predict_uv(weights=args.weights, source=args.source, conf=args.conf, device=args.device)


if __name__ == "__main__":
    main()


