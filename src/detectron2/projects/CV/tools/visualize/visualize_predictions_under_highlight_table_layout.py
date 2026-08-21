#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D.1 多方法误差对比图（XOR: FP/FN）：

每个方法输出两列：Pred Overlay + XOR Error（FP/FN 固定配色）。
目标方法集合：
- Input
- GroundTruth
- Mask R-CNN
- MaskTransfiner
- PointRend(Base)
- SPG-PointRend
- M2F+SDF
- M2F+Geo.Loss
- QSP-Mask2Former（ours）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config

# 触发注册：ShapeAwareCoarseMaskHead（SPG-PointRend）
try:
    import custom_heads  # noqa: F401
except ModuleNotFoundError:
    from train import custom_heads  # noqa: F401

try:
    from train_plug import register_plug_dataset
except ModuleNotFoundError:
    from train.train_plug import register_plug_dataset

try:
    from highlight_mapper import _ann_to_mask  # type: ignore
except ModuleNotFoundError:
    # 本地兜底：兼容没有 highlight_mapper.py 的项目布局
    def _ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
        seg = ann.get("segmentation", None)
        out = np.zeros((h, w), dtype=np.uint8)
        if isinstance(seg, list) and len(seg) > 0:
            # polygon 格式：[[x1,y1,x2,y2,...], ...]
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillPoly(out, [pts.astype(np.int32)], color=1)
            if np.any(out > 0):
                return out

        # 非 polygon（如 RLE）或异常时回退 bbox
        bbox = ann.get("bbox", None)
        if isinstance(bbox, list) and len(bbox) >= 4:
            x, y, bw, bh = [float(v) for v in bbox[:4]]
            x1 = int(max(0, min(w - 1, round(x))))
            y1 = int(max(0, min(h - 1, round(y))))
            x2 = int(max(0, min(w - 1, round(x + bw - 1))))
            y2 = int(max(0, min(h - 1, round(y + bh - 1))))
            if x2 >= x1 and y2 >= y1:
                out[y1 : y2 + 1, x1 : x2 + 1] = 1
        return out


warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.load.*weights_only=False.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*torch\.meshgrid.*indexing argument.*")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _natural_sort_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


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
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="按数据集原始顺序稳定选择图片：不随机打乱，也不按高光分数重排；便于与无GT脚本按索引对齐。",
    )
    p.add_argument(
        "--index-range",
        default="",
        help="可选：只可视化指定序号范围（如 '15-20' 或 '15:20'，均为闭区间）。",
    )
    p.add_argument(
        "--index-base",
        type=int,
        default=1,
        choices=(0, 1),
        help="--index-range 的序号基准：1 表示第 1 张=索引1（默认）；0 表示第 0 张=索引0。",
    )
    p.add_argument("--highlight-v-thr", type=int, default=245, help="HSV 的 V 阈值（高亮）")
    p.add_argument("--highlight-s-max", type=int, default=70, help="HSV 的 S 上限（偏白高亮）")
    p.add_argument("--highlight-dilate", type=int, default=7, help="高亮区域膨胀像素（扩大关注区域）")

    # visualization
    p.add_argument("--cell-width", type=int, default=520, help="每个格子的宽度（像素），高度按比例缩放")
    p.add_argument(
        "--no-header",
        "--hide-header",
        dest="no_header",
        action="store_true",
        help="去掉图片上方的全局列图注；默认保留图注。",
    )
    p.add_argument(
        "--no-row-tags",
        "--hide-row-tags",
        dest="no_row_tags",
        action="store_true",
        help="去掉每行左上角的 S1/S2/... 标签；默认保留标签。",
    )
    p.add_argument(
        "--pred-erode",
        type=int,
        default=0,
        help="可选：叠加可视化前先腐蚀预测 mask（像素，建议 2~4）以减弱边缘淡色光晕。",
    )
    p.add_argument("--score-thr", type=float, default=0.5, help="可视化过滤分数阈值")
    p.add_argument(
        "--mode",
        choices=["pred_only", "pred_xor"],
        default="pred_xor",
        help="可视化模式：pred_only=旧版纯预测图；pred_xor=新版预测+XOR误差图。",
    )
    p.add_argument("--xor-fp-color", default="0,0,255", help="FP 颜色(B,G,R)")
    p.add_argument("--xor-fn-color", default="255,0,0", help="FN 颜色(B,G,R)")
    p.add_argument("--xor-alpha", type=float, default=0.75, help="FP/FN 叠加透明度")
    p.add_argument("--xor-bg-dim", type=float, default=0.6, help="XOR 背景亮度系数（0~1）")

    # Mask2Former
    p.add_argument("--mask2former-root", required=True, help="Mask2Former 项目根目录")
    p.add_argument("--config-mask2former", required=True, help="Mask2Former config yaml")
    p.add_argument("--weights-mask2former", default="", help="Mask2Former 基线权重（可选）")
    p.add_argument("--weights-mask2former-sdf", default="", help="M2F+SDF 权重（可选）")
    p.add_argument("--weights-mask2former-geoloss", required=True, help="M2F+Geo.Loss 权重")
    p.add_argument("--config-mask2former-qsp", default="", help="Mask2Former+QSP config yaml（可选）")
    p.add_argument("--weights-mask2former-qsp", required=True, help="Mask2Former+QSP 权重")
    p.add_argument(
        "--prior-path-mask2former-qsp",
        default="",
        help="可选：覆盖 Mask2Former+QSP 配置里的 PRIOR_PATH，用于快速切换 prior。",
    )

    # PointRend Base
    p.add_argument("--config-pointrend", required=True, help="PointRend plug config yaml")
    p.add_argument("--weights-base", required=True, help="PointRend(Base) 权重")
    p.add_argument("--weights-spg", default="", help="SPG-PointRend 权重（ShapeAwareCoarseMaskHead，可选）")
    p.add_argument("--shape-prior-npy", default="", help="shape prior .npy（可选，供 SPG 和 QSP 覆盖 PRIOR_PATH）")

    # Mask R-CNN
    p.add_argument("--config-maskrcnn", required=True, help="Mask R-CNN config yaml")
    p.add_argument("--weights-maskrcnn", required=True, help="Mask R-CNN 权重")

    # MaskTransfiner（跨仓库子进程）
    p.add_argument("--transfiner-root", default="", help="transfiner 项目根目录（可选）")
    p.add_argument("--config-transfiner", default="", help="transfiner config yaml（可选）")
    p.add_argument("--weights-transfiner", default="", help="transfiner 权重（可选）")

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


def _make_header_row(column_titles: List[str], cell_width: int, header_h: int = 54) -> np.ndarray:
    w = int(cell_width) * len(column_titles)
    h = int(header_h)
    header = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.line(header, (0, h - 1), (w - 1, h - 1), (165, 165, 165), 2, cv2.LINE_AA)
    for i, title in enumerate(column_titles):
        x0 = i * int(cell_width)
        cv2.line(header, (x0, 0), (x0, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        tx = x0 + max(6, (int(cell_width) - tw) // 2)
        ty = max(th + 6, (h + th) // 2 - 4)
        cv2.putText(header, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(header, (w - 1, 0), (w - 1, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    return header


def _add_row_tag(row_img: np.ndarray, tag: str) -> np.ndarray:
    out = row_img.copy()
    cv2.rectangle(out, (6, 6), (44, 28), (248, 248, 248), thickness=-1)
    cv2.rectangle(out, (6, 6), (44, 28), (178, 178, 178), thickness=1)
    cv2.putText(out, tag, (13, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    return out


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


def _parse_bgr_triplet(s: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError(f"invalid BGR color: {s}")
    vals = [int(max(0, min(255, int(x)))) for x in parts]
    return int(vals[0]), int(vals[1]), int(vals[2])


def _xor_vis(
    img_bgr: np.ndarray,
    gt01: np.ndarray,
    pr01: np.ndarray,
    *,
    fp_color: Tuple[int, int, int],
    fn_color: Tuple[int, int, int],
    alpha: float,
    bg_dim: float,
) -> Tuple[np.ndarray, int, int]:
    gt = (gt01 > 0)
    pr = (pr01 > 0)
    fp = pr & (~gt)
    fn = (~pr) & gt
    vis = np.clip(img_bgr.astype(np.float32) * float(max(0.0, min(1.0, bg_dim))), 0, 255).astype(np.uint8)
    a = float(max(0.0, min(1.0, alpha)))
    if np.any(fp):
        c = np.array(fp_color, dtype=np.float32)
        vis[fp] = (vis[fp].astype(np.float32) * (1.0 - a) + c * a).astype(np.uint8)
    if np.any(fn):
        c = np.array(fn_color, dtype=np.float32)
        vis[fn] = (vis[fn].astype(np.float32) * (1.0 - a) + c * a).astype(np.uint8)
    return vis, int(fp.sum()), int(fn.sum())


def _predict_transfiner_masks_subprocess(
    *,
    transfiner_root: str,
    config_file: str,
    weights: str,
    image_paths: List[str],
    score_thr: float,
) -> Dict[str, np.ndarray]:
    """
    使用子进程在 transfiner 环境中批量推理，返回 {abs_image_path: mask01}。
    """
    helper_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfiner_batch_predict_masks.py")
    if not os.path.isfile(helper_py):
        raise FileNotFoundError(f"helper script not found: {helper_py}")
    tmp_dir = tempfile.mkdtemp(prefix="transfiner_pred_")
    list_file = os.path.join(tmp_dir, "images.txt")
    out_json = os.path.join(tmp_dir, "result.json")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(os.path.abspath(str(p)) + "\n")
    env = os.environ.copy()
    tf_root = os.path.abspath(str(transfiner_root))
    env["PYTHONPATH"] = tf_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    cmd = [
        sys.executable,
        "-u",
        helper_py,
        "--config-file",
        os.path.abspath(str(config_file)),
        "--weights",
        os.path.abspath(str(weights)),
        "--image-list",
        list_file,
        "--output-json",
        out_json,
        "--score-thr",
        str(float(score_thr)),
    ]
    cp = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            "transfiner batch predict failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}\n"
        )
    with open(out_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, np.ndarray] = {}
    for r in obj.get("records", []):
        ip = os.path.abspath(str(r.get("image_path", "")))
        mp = str(r.get("mask_path", "")).strip()
        if not ip or not mp or (not os.path.isfile(mp)):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[ip] = (m > 0).astype(np.uint8)
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
    a = a.strip()
    b = b.strip()
    if (not a) or (not b):
        raise ValueError("index-range missing start/end")
    ia = int(a)
    ib = int(b)
    if ib < ia:
        ia, ib = ib, ia
    return ia, ib


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
    prior_path_override: str = "",
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
    if str(prior_path_override).strip():
        cfg.MODEL.MASK_FORMER.PRIOR_ON = True
        cfg.MODEL.MASK_FORMER.PRIOR_PATH = str(prior_path_override).strip()
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
    cand = sorted(list(dicts), key=lambda x: _natural_sort_key(str(x.get("file_name", ""))))
    cand = cand[: max(1, int(args.scan_max))]
    if bool(getattr(args, "no_shuffle", False)):
        picks = list(cand)
    else:
        random.shuffle(cand)
        scored: List[Tuple[float, dict]] = []
        for d in cand:
            img = cv2.imread(d["file_name"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            scored.append((_highlight_score(img, v_thr=args.highlight_v_thr, s_max=args.highlight_s_max), d))
        scored.sort(reverse=True, key=lambda t: t[0])
        picks = [d for _, d in scored]

    if str(getattr(args, "index_range", "")).strip():
        a, b = _parse_index_range(str(args.index_range).strip())
        base = int(getattr(args, "index_base", 1))
        if base == 1:
            a0 = max(0, a - 1)
            b0 = max(0, b - 1)
        else:
            a0 = max(0, a)
            b0 = max(0, b)
        picks = picks[a0 : b0 + 1]
    else:
        picks = picks[: max(1, int(args.num_rows))]
    if not picks:
        raise RuntimeError("no valid images found for visualization")

    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)
    fp_color = _parse_bgr_triplet(args.xor_fp_color)
    fn_color = _parse_bgr_triplet(args.xor_fn_color)

    # predictors
    maskrcnn_pred = _build_maskrcnn_predictor(
        config_file=os.path.abspath(str(args.config_maskrcnn).strip()),
        weights=os.path.abspath(str(args.weights_maskrcnn).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    base_pred = _build_pointrend_predictor(
        config_file=os.path.abspath(args.config_pointrend),
        weights=os.path.abspath(args.weights_base),
        mask_head_name="PointRendMaskHead",
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    m2f_geoloss_pred = _build_mask2former_predictor(
        mask2former_root=args.mask2former_root,
        config_file=os.path.abspath(args.config_mask2former),
        weights=os.path.abspath(args.weights_mask2former_geoloss),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
    )
    m2f_base_pred = None
    if str(getattr(args, "weights_mask2former", "")).strip():
        m2f_base_pred = _build_mask2former_predictor(
            mask2former_root=args.mask2former_root,
            config_file=os.path.abspath(args.config_mask2former),
            weights=os.path.abspath(str(args.weights_mask2former).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )
    m2f_sdf_pred = None
    if str(getattr(args, "weights_mask2former_sdf", "")).strip():
        m2f_sdf_pred = _build_mask2former_predictor(
            mask2former_root=args.mask2former_root,
            config_file=os.path.abspath(args.config_mask2former),
            weights=os.path.abspath(str(args.weights_mask2former_sdf).strip()),
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )
    config_mask2former_qsp = str(getattr(args, "config_mask2former_qsp", "")).strip() or str(
        getattr(args, "config_mask2former", "")
    ).strip()
    if not config_mask2former_qsp:
        raise ValueError("提供 --weights-mask2former-qsp 时必须同时提供 --config-mask2former-qsp 或 --config-mask2former")
    shape_prior_npy = str(getattr(args, "shape_prior_npy", "")).strip()
    if shape_prior_npy:
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(shape_prior_npy)
    qsp_prior_override = str(getattr(args, "prior_path_mask2former_qsp", "")).strip() or shape_prior_npy
    m2f_qsp_pred = _build_mask2former_predictor(
        mask2former_root=args.mask2former_root,
        config_file=os.path.abspath(config_mask2former_qsp),
        weights=os.path.abspath(str(args.weights_mask2former_qsp).strip()),
        num_classes=num_classes,
        score_thr=float(args.score_thr),
        prior_path_override=os.path.abspath(qsp_prior_override) if qsp_prior_override else "",
    )
    spg_pred = None
    if str(getattr(args, "weights_spg", "")).strip():
        spg_pred = _build_pointrend_predictor(
            config_file=os.path.abspath(args.config_pointrend),
            weights=os.path.abspath(str(args.weights_spg).strip()),
            mask_head_name="ShapeAwareCoarseMaskHead",
            num_classes=num_classes,
            score_thr=float(args.score_thr),
        )

    tf_root = str(getattr(args, "transfiner_root", "")).strip()
    tf_cfg = str(getattr(args, "config_transfiner", "")).strip()
    tf_w = str(getattr(args, "weights_transfiner", "")).strip()
    has_tf = bool(tf_root or tf_cfg or tf_w)
    if has_tf and not (tf_root and tf_cfg and tf_w):
        raise ValueError("启用 MaskTransfiner 时必须同时提供 --transfiner-root --config-transfiner --weights-transfiner")
    transfiner_mask_map: Dict[str, np.ndarray] = {}
    if has_tf:
        pick_paths = [os.path.abspath(str(d.get("file_name", ""))) for d in picks if str(d.get("file_name", "")).strip()]
        transfiner_mask_map = _predict_transfiner_masks_subprocess(
            transfiner_root=tf_root,
            config_file=tf_cfg,
            weights=tf_w,
            image_paths=pick_paths,
            score_thr=float(args.score_thr),
        )

    rows_img: List[np.ndarray] = []
    column_titles: List[str] = []
    error_rows: List[dict] = []
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

        base_mask, _ = _predict_best_mask(base_pred, img_bgr)
        maskrcnn_mask, _ = _predict_best_mask(maskrcnn_pred, img_bgr)
        m2f_geoloss_mask, _ = _predict_best_mask(m2f_geoloss_pred, img_bgr)
        m2f_base_mask = None
        if m2f_base_pred is not None:
            m2f_base_mask, _ = _predict_best_mask(m2f_base_pred, img_bgr)
        m2f_sdf_mask = None
        if m2f_sdf_pred is not None:
            m2f_sdf_mask, _ = _predict_best_mask(m2f_sdf_pred, img_bgr)
        m2f_qsp_mask, _ = _predict_best_mask(m2f_qsp_pred, img_bgr)
        spg_mask = None
        if spg_pred is not None:
            spg_mask, _ = _predict_best_mask(spg_pred, img_bgr)
        tf_mask = transfiner_mask_map.get(os.path.abspath(img_path), None) if has_tf else None

        # fill empty with zeros for downstream ops
        if base_mask is None:
            base_mask = np.zeros_like(gt)
        if maskrcnn_mask is None:
            maskrcnn_mask = np.zeros_like(gt)
        if m2f_geoloss_mask is None:
            m2f_geoloss_mask = np.zeros_like(gt)
        if m2f_base_pred is not None and m2f_base_mask is None:
            m2f_base_mask = np.zeros_like(gt)
        if m2f_sdf_pred is not None and m2f_sdf_mask is None:
            m2f_sdf_mask = np.zeros_like(gt)
        if m2f_qsp_mask is None:
            m2f_qsp_mask = np.zeros_like(gt)
        if spg_pred is not None and spg_mask is None:
            spg_mask = np.zeros_like(gt)
        if has_tf and (tf_mask is None):
            tf_mask = np.zeros_like(gt)
        if has_tf and tf_mask is not None and tf_mask.shape != gt.shape:
            tf_mask = cv2.resize(tf_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # visualization-only erosion
        ep = int(getattr(args, "pred_erode", 0))
        base_mask_vis = _erode_mask(base_mask, ep)
        maskrcnn_mask_vis = _erode_mask(maskrcnn_mask, ep)
        m2f_geoloss_mask_vis = _erode_mask(m2f_geoloss_mask, ep)
        m2f_base_mask_vis = _erode_mask(m2f_base_mask, ep) if (m2f_base_mask is not None) else None
        m2f_sdf_mask_vis = _erode_mask(m2f_sdf_mask, ep) if (m2f_sdf_mask is not None) else None
        m2f_qsp_mask_vis = _erode_mask(m2f_qsp_mask, ep)
        spg_mask_vis = _erode_mask(spg_mask, ep) if (spg_mask is not None) else None
        tf_mask_vis = _erode_mask(tf_mask, ep) if (tf_mask is not None) else None

        # visual cells
        input_vis = input_bgr.copy()
        gt_vis = _overlay_mask(input_bgr, gt, color_bgr=(0, 255, 0), alpha=0.35, outline=True)

        method_masks: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]] = [
            ("Mask R-CNN", maskrcnn_mask, maskrcnn_mask_vis, (120, 200, 255)),
            ("PointRend(Base)", base_mask, base_mask_vis, (255, 170, 120)),
            ("M2F+Geo.Loss", m2f_geoloss_mask, m2f_geoloss_mask_vis, (255, 200, 80)),
            ("QSP-Mask2Former (ours)", m2f_qsp_mask, m2f_qsp_mask_vis, (80, 210, 160)),
        ]
        if spg_mask is not None and spg_mask_vis is not None:
            method_masks.insert(2, ("SPG-PointRend", spg_mask, spg_mask_vis, (110, 215, 120)))
        if m2f_base_mask is not None and m2f_base_mask_vis is not None:
            method_masks.insert(-2, ("Mask2Former", m2f_base_mask, m2f_base_mask_vis, (190, 120, 235)))
        if m2f_sdf_mask is not None and m2f_sdf_mask_vis is not None:
            method_masks.insert(-1, ("M2F+SDF", m2f_sdf_mask, m2f_sdf_mask_vis, (235, 150, 210)))
        if has_tf and tf_mask is not None and tf_mask_vis is not None:
            method_masks.insert(1, ("MaskTransfiner", tf_mask, tf_mask_vis, (200, 120, 255)))

        cells: List[np.ndarray] = [input_vis, gt_vis]
        local_titles: List[str] = ["Raw", "GT"]
        for method_name, raw_mask, vis_mask, color_bgr in method_masks:
            pred_vis = _overlay_mask(input_bgr, vis_mask, color_bgr=color_bgr, alpha=0.35, outline=True)
            if str(args.mode) == "pred_only":
                cells.append(pred_vis)
                local_titles.append(method_name)
            else:
                xor_img, fp_px, fn_px = _xor_vis(
                    input_bgr,
                    gt,
                    raw_mask,
                    fp_color=fp_color,
                    fn_color=fn_color,
                    alpha=float(args.xor_alpha),
                    bg_dim=float(args.xor_bg_dim),
                )
                xor_vis = xor_img
                cells.extend([pred_vis, xor_vis])
                local_titles.extend([f"{method_name} Pred", f"{method_name} XOR"])
                error_rows.append(
                    {
                        "image_path": str(img_path),
                        "method": method_name,
                        "fp_pixels": int(fp_px),
                        "fn_pixels": int(fn_px),
                        "xor_pixels": int(fp_px + fn_px),
                        "gt_pixels": int((gt > 0).sum()),
                        "pred_pixels": int((raw_mask > 0).sum()),
                    }
                )

        if not column_titles:
            column_titles = list(local_titles)

        cells = [_resize_to_width(c, int(args.cell_width)) for c in cells]
        row = _stack_row(cells)
        if not bool(args.no_row_tags):
            row = _add_row_tag(row, f"S{i + 1}")
        rows_img.append(row)

        # per-row also save
        stem = f"{i:02d}_" + os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_row.png"), row)

    grid_body = _stack_grid(rows_img)
    if bool(args.no_header):
        grid = grid_body
    else:
        header = _make_header_row(column_titles, int(args.cell_width), header_h=54)
        grid = np.concatenate([header, grid_body], axis=0)
    if str(args.mode) == "pred_only":
        out_path = os.path.join(out_dir, "table_layout_grid.png")
    else:
        out_path = os.path.join(out_dir, "table_layout_grid_xor.png")
    cv2.imwrite(out_path, grid)
    print(f"saved: {out_path}")

    if str(args.mode) == "pred_xor":
        stats_csv = os.path.join(out_dir, "per_image_error_stats.csv")
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["image_path", "method", "fp_pixels", "fn_pixels", "xor_pixels", "gt_pixels", "pred_pixels"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in error_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"saved: {stats_csv}")


if __name__ == "__main__":
    main()

