#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无需 GT/标注的强反光定性对比可视化（表格排版）：

- 自动从 dataset-root 扫描图片，并按“高反光得分”挑选 topK
- 对每张图跑多个模型推理（可选），将预测 mask 叠加到输入图上
  输出：
    - 每张图一行：<stem>_row.png
    - 总表格：table_layout_grid.png

与 visualize_predictions_under_highlight_table_layout.py 的区别：
- 不注册 detectron2 数据集、不读取 COCO json、不需要任何标注
- 仅做推理可视化（没有 GT 列，也不画 GT-vs-Pred 错误框）
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config

# 触发注册：ShapeAwareCoarseMaskHead（如果用户传了 SPG 权重）
try:
    import custom_heads  # noqa: F401
except Exception:
    custom_heads = None  # type: ignore


warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.load.*weights_only=False.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*torch\.meshgrid.*indexing argument.*")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _natural_sort_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference-only table-layout qualitative comparison under highlight.")
    p.add_argument("--dataset-root", required=True, help="图片目录（递归扫描常见图片后缀）")
    p.add_argument(
        "--clean-root",
        default="",
        help="可选：对应的 clean 图片目录（与 dataset-root 共享相对路径）。用于仅显示时的去光晕。",
    )
    p.add_argument("--out-dir", default="./output/pred_vis_table_infer", help="输出目录")
    p.add_argument("--seed", type=int, default=0)

    # explicit image selection (optional)
    p.add_argument(
        "--image-list",
        default="",
        help="可选：一个 txt 文件，每行一个图片路径（可为绝对路径，或相对于 --dataset-root 的相对路径）。"
        "提供后将忽略 scan-max 与强光打分挑选逻辑，按该列表顺序可视化。",
    )
    p.add_argument(
        "--images",
        default="",
        help="可选：用逗号分隔的图片路径列表（绝对路径或相对 --dataset-root）。提供后同样会忽略强光挑选逻辑。",
    )
    p.add_argument(
        "--index-range",
        default="",
        help="可选：只可视化指定序号范围（如 '15-20' 或 '15:20'，均为闭区间）。"
        "序号基准由 --index-base 控制；会在最终 picks 上做切片（保持原有顺序）。",
    )
    p.add_argument(
        "--index-base",
        type=int,
        default=1,
        choices=(0, 1),
        help="--index-range 的序号基准：1 表示第 1 张=索引1（默认，更符合人类习惯）；0 表示第 0 张=索引0。",
    )

    # selection: pick images with strong highlight
    p.add_argument("--num-rows", type=int, default=4, help="输出多少行（多少张图）")
    p.add_argument("--scan-max", type=int, default=200, help="最多扫描多少张图用于挑“高反光”样本")
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="按文件自然顺序稳定选择图片：不随机打乱，也不按高光分数重排；此时 --index-range 直接对应文件顺序。",
    )
    p.add_argument("--highlight-v-thr", type=int, default=245, help="HSV 的 V 阈值（高亮）")
    p.add_argument("--highlight-s-max", type=int, default=70, help="HSV 的 S 上限（偏白高亮）")
    p.add_argument("--highlight-dilate", type=int, default=7, help="高亮区域膨胀像素（扩大关注区域）")

    # visualization
    p.add_argument("--cell-width", type=int, default=520, help="每个格子的宽度（像素），高度按比例缩放")
    p.add_argument(
        "--pred-erode",
        type=int,
        default=0,
        help="可选：叠加可视化前先腐蚀预测 mask（像素，建议 2~4）以减弱边缘淡色光晕。",
    )
    p.add_argument("--score-thr", type=float, default=0.5, help="可视化过滤分数阈值（取最高分实例）")
    p.add_argument(
        "--show-highlight-col",
        action="store_true",
        help="额外插入一列 HighLightMask（由阈值自动计算，不需要标注）",
    )

    # Mask2Former (optional)
    p.add_argument("--mask2former-root", default="", help="Mask2Former 项目根目录（可选）")
    p.add_argument("--config-mask2former", default="", help="Mask2Former config yaml（可选）")
    p.add_argument("--weights-mask2former", default="", help="Mask2Former 权重（可选）")
    p.add_argument("--weights-mask2former-sdf", default="", help="Mask2Former+SDF 权重（可选）")
    p.add_argument("--config-mask2former-qsp", default="", help="Mask2Former+QSP config yaml（可选）")
    p.add_argument("--weights-mask2former-qsp", default="", help="Mask2Former+QSP 权重（可选）")
    p.add_argument(
        "--prior-path-mask2former-qsp",
        default="",
        help="可选：覆盖 Mask2Former+QSP 配置里的 PRIOR_PATH，用于快速切换 prior。",
    )

    # PointRend (optional)
    p.add_argument("--config-pointrend", default="", help="PointRend config yaml（可选）")
    p.add_argument("--weights-base", default="", help="BASE(no prior) PointRendMaskHead 权重（可选）")
    p.add_argument("--weights-spg", default="", help="SPG-PointRend 权重（ShapeAwareCoarseMaskHead，可选）")
    p.add_argument("--shape-prior-npy", default="", help="shape prior .npy（用于 ShapeAwareCoarseMaskHead，可选）")

    # Mask R-CNN (optional)
    p.add_argument("--config-maskrcnn", default="", help="Mask R-CNN config yaml（可选）")
    p.add_argument("--weights-maskrcnn", default="", help="Mask R-CNN 权重（可选）")

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
    _h, s, v = cv2.split(hsv)
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


def _safe_imread(p: str) -> Optional[np.ndarray]:
    return cv2.imread(p, cv2.IMREAD_COLOR)


def _load_clean_match(*, clean_root: str, highlight_root: str, highlight_path: str) -> Optional[np.ndarray]:
    if not str(clean_root).strip():
        return None
    try:
        rel = os.path.relpath(highlight_path, highlight_root)
    except Exception:
        rel = os.path.basename(highlight_path)
    cand = os.path.join(clean_root, rel)
    if os.path.isfile(cand):
        return _safe_imread(cand)
    cand2 = os.path.join(clean_root, os.path.basename(highlight_path))
    if os.path.isfile(cand2):
        return _safe_imread(cand2)
    return None


def _remove_halo_with_clean_and_hi(*, img_highlight_bgr: np.ndarray, img_clean_bgr: np.ndarray, hi01: np.ndarray) -> np.ndarray:
    """
    无 GT 的“近似去光晕”：把 highlight-clean 的增量裁剪在 high-light 区域内。
      delta = highlight - clean
      out = clean + delta * hi_mask
    """
    if img_clean_bgr.shape[:2] != img_highlight_bgr.shape[:2]:
        img_clean_bgr = cv2.resize(
            img_clean_bgr, (img_highlight_bgr.shape[1], img_highlight_bgr.shape[0]), interpolation=cv2.INTER_AREA
        )
    m = (hi01 > 0).astype(np.float32)[..., None]
    hi = img_highlight_bgr.astype(np.float32)
    cl = img_clean_bgr.astype(np.float32)
    delta = hi - cl
    out = cl + delta * m
    return np.clip(out, 0, 255).astype(np.uint8)


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


def _iter_images(root: str) -> Iterable[str]:
    r = os.path.abspath(str(root))
    for dp, _dns, fns in os.walk(r):
        _dns.sort(key=_natural_sort_key)
        fns.sort(key=_natural_sort_key)
        for fn in fns:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(dp, fn)


def _normalize_image_path(p: str, dataset_root: str) -> str:
    pp = str(p).strip()
    if not pp:
        return ""
    if os.path.isabs(pp):
        return os.path.abspath(pp)
    return os.path.abspath(os.path.join(dataset_root, pp))


def _read_image_list_file(list_path: str, dataset_root: str) -> List[str]:
    lp = os.path.abspath(str(list_path))
    if not os.path.isfile(lp):
        raise FileNotFoundError(f"--image-list not found: {lp}")
    out: List[str] = []
    with open(lp, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            out.append(_normalize_image_path(s, dataset_root))
    return out


def _parse_index_range(s: str) -> Tuple[int, int]:
    """
    Parse 'a-b' or 'a:b' into (a, b) inclusive.
    """
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


@dataclass(frozen=True)
class _Method:
    key: str
    title: str
    predictor: DefaultPredictor
    color_bgr: Tuple[int, int, int]


def _ordered_methods(methods: List[_Method]) -> List[_Method]:
    """
    Paper-friendly ordering:
    1) traditional baseline
    2) PointRend family
    3) Mask2Former family
    4) our strongest variant last
    """
    order = {
        "maskrcnn": 10,
        "base": 20,
        "spg": 30,
        "m2f": 40,
        "m2f_sdf": 50,
        "m2f_qsp": 60,
    }
    return sorted(methods, key=lambda m: (order.get(m.key, 999), m.title))


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    dataset_root = os.path.abspath(str(args.dataset_root))
    out_dir = os.path.abspath(str(args.out_dir))
    _ensure_dir(out_dir)

    if str(args.shape_prior_npy).strip():
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(str(args.shape_prior_npy).strip())

    # 选择要可视化的图片：优先用显式列表，否则走“强光打分挑选”
    picks: List[str] = []
    if str(getattr(args, "images", "")).strip():
        raw = [s.strip() for s in str(args.images).split(",")]
        picks = [_normalize_image_path(s, dataset_root) for s in raw if str(s).strip()]
    elif str(getattr(args, "image_list", "")).strip():
        picks = _read_image_list_file(str(args.image_list).strip(), dataset_root)
    else:
        # 收集候选图片（最多 scan-max）
        all_imgs = list(_iter_images(dataset_root))
        if not all_imgs:
            raise RuntimeError(f"no images found under: {dataset_root}")
        cand = all_imgs[: max(1, int(args.scan_max))]
        if bool(getattr(args, "no_shuffle", False)):
            # Stable mode: keep natural file order so index-range matches directory ordering.
            picks = cand
        else:
            random.shuffle(cand)
            scored: List[Tuple[float, str]] = []
            for p in cand:
                im = _safe_imread(p)
                if im is None:
                    continue
                scored.append((_highlight_score(im, v_thr=args.highlight_v_thr, s_max=args.highlight_s_max), p))
            scored.sort(reverse=True, key=lambda t: t[0])
            # 注意：这里不要按 num-rows 截断。后面会统一应用 index-range 或 num-rows。
            picks = [p for _s, p in scored]

    # 清理：只保留存在的文件（保持顺序）
    picks = [p for p in picks if p and os.path.isfile(p)]

    if not picks:
        raise RuntimeError("no valid images found for visualization")

    # 可选：按序号范围切片（闭区间）
    if str(getattr(args, "index_range", "")).strip():
        a, b = _parse_index_range(str(args.index_range).strip())
        base = int(getattr(args, "index_base", 1))
        if base not in (0, 1):
            base = 1
        if base == 1:
            a0 = max(0, a - 1)
            b0 = max(0, b - 1)
        else:
            a0 = max(0, a)
            b0 = max(0, b)
        # Python slice end is exclusive, so +1 for inclusive range
        picks = picks[a0 : b0 + 1]
        if not picks:
            raise RuntimeError("index-range selected 0 images (check range/base and available picks)")
    else:
        # 为了与原行为一致：最多输出 num-rows 张（显式列表也会保序截断）
        if int(args.num_rows) > 0:
            picks = picks[: int(args.num_rows)]

    # 组装 predictors（全部可选）
    methods: List[_Method] = []

    # Mask R-CNN
    if str(getattr(args, "weights_maskrcnn", "")).strip():
        if not str(getattr(args, "config_maskrcnn", "")).strip():
            raise ValueError("提供 --weights-maskrcnn 时必须同时提供 --config-maskrcnn")
        pred = _build_maskrcnn_predictor(
            config_file=os.path.abspath(str(args.config_maskrcnn).strip()),
            weights=os.path.abspath(str(args.weights_maskrcnn).strip()),
            num_classes=1,  # plug 默认 1；如需自动多类，建议改成命令行传入
            score_thr=float(args.score_thr),
        )
        methods.append(_Method("maskrcnn", "Mask R-CNN", pred, (90, 170, 230)))

    # PointRend base/spg
    if str(getattr(args, "weights_base", "")).strip() or str(getattr(args, "weights_spg", "")).strip():
        if not str(getattr(args, "config_pointrend", "")).strip():
            raise ValueError("提供 PointRend 权重时必须同时提供 --config-pointrend")

    if str(getattr(args, "weights_base", "")).strip():
        pred = _build_pointrend_predictor(
            config_file=os.path.abspath(str(args.config_pointrend).strip()),
            weights=os.path.abspath(str(args.weights_base).strip()),
            mask_head_name="PointRendMaskHead",
            num_classes=1,
            score_thr=float(args.score_thr),
        )
        methods.append(_Method("base", "PointRend", pred, (110, 185, 245)))

    if str(getattr(args, "weights_spg", "")).strip():
        pred = _build_pointrend_predictor(
            config_file=os.path.abspath(str(args.config_pointrend).strip()),
            weights=os.path.abspath(str(args.weights_spg).strip()),
            mask_head_name="ShapeAwareCoarseMaskHead",
            num_classes=1,
            score_thr=float(args.score_thr),
        )
        methods.append(_Method("spg", "SPG-PointRend", pred, (110, 215, 120)))

    # Mask2Former base/sdf
    if (
        str(getattr(args, "weights_mask2former", "")).strip()
        or str(getattr(args, "weights_mask2former_sdf", "")).strip()
        or str(getattr(args, "weights_mask2former_qsp", "")).strip()
    ):
        if not str(getattr(args, "mask2former_root", "")).strip():
            raise ValueError("提供 Mask2Former 权重时必须同时提供 --mask2former-root")
        if (
            not str(getattr(args, "config_mask2former", "")).strip()
            and not str(getattr(args, "config_mask2former_qsp", "")).strip()
        ):
            raise ValueError("提供 Mask2Former 权重时必须同时提供 --config-mask2former 或 --config-mask2former-qsp")

    if str(getattr(args, "weights_mask2former", "")).strip():
        pred = _build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root).strip(),
            config_file=os.path.abspath(str(args.config_mask2former).strip()),
            weights=os.path.abspath(str(args.weights_mask2former).strip()),
            num_classes=1,
            score_thr=float(args.score_thr),
        )
        methods.append(_Method("m2f", "Mask2Former", pred, (190, 120, 235)))

    if str(getattr(args, "weights_mask2former_sdf", "")).strip():
        pred = _build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root).strip(),
            config_file=os.path.abspath(str(args.config_mask2former).strip()),
            weights=os.path.abspath(str(args.weights_mask2former_sdf).strip()),
            num_classes=1,
            score_thr=float(args.score_thr),
        )
        methods.append(_Method("m2f_sdf", "Mask2Former + SDF", pred, (235, 150, 210)))

    if str(getattr(args, "weights_mask2former_qsp", "")).strip():
        config_mask2former_qsp = str(getattr(args, "config_mask2former_qsp", "")).strip() or str(
            getattr(args, "config_mask2former", "")
        ).strip()
        if not config_mask2former_qsp:
            raise ValueError("提供 --weights-mask2former-qsp 时必须同时提供 --config-mask2former-qsp 或 --config-mask2former")
        pred = _build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root).strip(),
            config_file=os.path.abspath(config_mask2former_qsp),
            weights=os.path.abspath(str(args.weights_mask2former_qsp).strip()),
            num_classes=1,
            score_thr=float(args.score_thr),
            prior_path_override=os.path.abspath(str(args.prior_path_mask2former_qsp).strip())
            if str(getattr(args, "prior_path_mask2former_qsp", "")).strip()
            else "",
        )
        methods.append(_Method("m2f_qsp", "Mask2Former + QSP", pred, (80, 210, 160)))

    if not methods:
        raise ValueError("未提供任何模型权重；请至少提供一组 weights（如 --weights-spg 或 --weights-mask2former）。")
    methods = _ordered_methods(methods)

    rows_img: List[np.ndarray] = []
    for i, img_path in enumerate(picks):
        img_bgr = _safe_imread(img_path)
        if img_bgr is None:
            continue

        # highlight mask
        hi = _highlight_mask(
            img_bgr, v_thr=args.highlight_v_thr, s_max=args.highlight_s_max, dilate_px=args.highlight_dilate
        )

        # 输入图：如果提供 clean-root，则用 hi 区域做近似“去光晕”（仅用于显示）
        input_bgr = img_bgr
        if str(getattr(args, "clean_root", "")).strip():
            clean_bgr = _load_clean_match(
                clean_root=str(args.clean_root).strip(),
                highlight_root=dataset_root,
                highlight_path=img_path,
            )
            if clean_bgr is not None:
                input_bgr = _remove_halo_with_clean_and_hi(img_highlight_bgr=img_bgr, img_clean_bgr=clean_bgr, hi01=hi)

        cells: List[np.ndarray] = []
        input_vis = _title_bar(input_bgr, "Input (highlight)")
        cells.append(input_vis)

        if bool(getattr(args, "show_highlight_col", False)):
            hi_vis = _overlay_mask(input_bgr, hi, color_bgr=(0, 0, 255), alpha=0.25, outline=True)
            hi_vis = _title_bar(hi_vis, "HighlightMask (auto)")
            cells.append(hi_vis)

        # predictions
        ep = int(getattr(args, "pred_erode", 0))
        for m in methods:
            pr, _inst = _predict_best_mask(m.predictor, img_bgr)
            if pr is None:
                pr = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            pr_vis = _erode_mask(pr, ep)
            vis = _overlay_mask(input_bgr, pr_vis, color_bgr=m.color_bgr, alpha=0.35, outline=True)
            vis = _title_bar(vis, m.title)
            cells.append(vis)

        cells = [_resize_to_width(c, int(args.cell_width)) for c in cells]
        row = _stack_row(cells)
        rows_img.append(row)

        stem = f"{i:02d}_" + os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_row.png"), row)

    if not rows_img:
        raise RuntimeError("no rows generated (all images failed to read?)")

    grid = _stack_grid(rows_img)
    out_path = os.path.join(out_dir, "table_layout_grid.png")
    cv2.imwrite(out_path, grid)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

