from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
from ultralytics import YOLO


@dataclass
class YoloDetection:
    cls: int
    conf: float
    xyxy: np.ndarray  # (4,) float32/float64

    @property
    def uv(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy.tolist()
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def load_yolo(weights: str) -> YOLO:
    return YOLO(str(weights))


def infer_detections(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    device: Union[str, int] = 0,
    classes: Optional[List[int]] = None,
) -> List[YoloDetection]:
    """
    对单张 BGR 图做推理，返回检测框列表（按置信度从高到低）。
    """
    results = model.predict(source=image_bgr, conf=conf, device=device, verbose=False, classes=classes)
    if not results:
        return []

    r: Any = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    dets = [YoloDetection(cls=int(c), conf=float(p), xyxy=xyxy[i]) for i, (c, p) in enumerate(zip(clss, confs))]
    dets.sort(key=lambda d: d.conf, reverse=True)
    return dets


