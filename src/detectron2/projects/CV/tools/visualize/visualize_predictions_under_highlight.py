#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定性验证：强曝光下“形状先验”是否有效。

默认输出四栏论文版式（全局列标题，不显示左上角 S1/S2/... 行标签）：
1) Input Image（原始高光输入）
2) Baseline Mask（二值 mask）
3) Shape Prior Prediction（彩色分割区域）
4) Final Mask（二值 mask）

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

指定图片示例：
  --index-range 31-40 --index-base 1 --no-shuffle
  --images "0001.png,0008.png"
  --image-list /path/to/image_list.txt
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
import sys
from typing import List, Tuple

import cv2
import numpy as np

# 允许从任意工作目录启动时导入 CV/train 下的自定义模块。
_THIS_FILE = os.path.abspath(__file__)
_CV_ROOT = os.path.abspath(os.path.join(os.path.dirname(_THIS_FILE), "..", ".."))
_TRAIN_DIR = os.path.join(_CV_ROOT, "train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

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
    p = argparse.ArgumentParser(description="Four-column qualitative comparison under synthetic highlight.")
    p.add_argument("--config-file", required=True, help="PointRend plug 配置文件路径")
    p.add_argument("--dataset-root", required=True, help="强曝光评测集目录（图片+json）")
    p.add_argument("--dataset-name", default="plug_highlight_eval_vis", help="注册到 detectron2 的数据集名称")
    p.add_argument("--json-file", default="plug_train.json", help="COCO json 文件名（相对 dataset-root）")
    p.add_argument("--weights-base", required=True, help="原版 PointRend 权重（不带 shape prior）")
    p.add_argument("--weights-prior", required=True, help="带 shape prior 的权重（ShapeAwareCoarseMaskHead）")
    p.add_argument("--shape-prior-npy", default="", help="Phase-1 输出的 .npy（留空则用默认路径或环境变量）")
    p.add_argument("--mask2former-root", default="", help="Mask2Former 项目根目录（可选；也可通过 PYTHONPATH 导入）")
    p.add_argument(
        "--config-mask2former-qsp",
        default=os.path.join(_CV_ROOT, "configs", "InstanceSegmentation", "mask2former_R50_plug_qsp_aug.yaml"),
        help="Mask2Former+QSP config；默认使用 CV/configs/InstanceSegmentation/mask2former_R50_plug_qsp_aug.yaml",
    )
    p.add_argument(
        "--weights-mask2former-qsp",
        default="",
        help="Mask2Former+QSP 权重（可选；提供后额外生成 QSP 四栏结果）",
    )
    p.add_argument(
        "--prior-path-mask2former-qsp",
        default="",
        help="Mask2Former+QSP 先验路径（可选；覆盖 QSP config 中的 PRIOR_PATH）",
    )
    p.add_argument("--out-dir", default="./output/pred_vis_highlight", help="输出目录")
    p.add_argument("--num", type=int, default=20, help="抽样图片数量")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument(
        "--images",
        default="",
        help="可选：逗号分隔的图片路径或文件名；相对路径相对于 --dataset-root。",
    )
    p.add_argument(
        "--image-list",
        default="",
        help="可选：txt 图片列表，每行一个绝对路径或相对于 --dataset-root 的路径。",
    )
    p.add_argument(
        "--index-range",
        default="",
        help="可选：按数据集顺序选择闭区间，如 31-40 或 31:40；默认序号从1开始。",
    )
    p.add_argument(
        "--index-base",
        type=int,
        default=1,
        choices=(0, 1),
        help="--index-range 的序号基准：1表示第1张图，0表示第0张图。",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不随机抽样，按数据集注册顺序选择图片；与 --index-range 配合使用。",
    )
    p.add_argument("--score-thr", type=float, default=0.5, help="可视化过滤分数阈值")
    p.add_argument("--cell-width", type=int, default=520, help="四栏表格中每个单元格的宽度")
    p.add_argument("--header-height", type=int, default=72, help="全局列标题高度")
    p.add_argument("--guidance-alpha", type=float, default=0.42, help="Shape Prior Guidance 灰色叠加透明度")
    p.add_argument("--max-failure-boxes", type=int, default=2, help="每个样本最多绘制的红色失败区域框数量")
    return p.parse_args()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_index_range(value: str) -> Tuple[int, int]:
    """解析闭区间序号，支持 ``31-40``、``31:40`` 或单个序号 ``31``。"""
    text = str(value).strip()
    if not text:
        raise ValueError("index-range 不能为空")
    if "-" in text:
        start, end = text.split("-", 1)
    elif ":" in text:
        start, end = text.split(":", 1)
    else:
        start = end = text
    a, b = int(start.strip()), int(end.strip())
    if b < a:
        a, b = b, a
    return a, b


def _read_image_list(path: str) -> List[str]:
    list_path = os.path.abspath(str(path))
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"--image-list 不存在: {list_path}")
    values: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                values.append(item)
    return values


def _select_dataset_dicts(
    dataset_dicts: List[dict],
    dataset_root: str,
    args: argparse.Namespace,
) -> List[dict]:
    """按照显式路径、图片列表或数据集序号选择 COCO 数据集记录。"""
    root = os.path.abspath(str(dataset_root))
    image_specs: List[str] = []
    if str(args.images).strip():
        image_specs.extend(s.strip() for s in str(args.images).split(",") if s.strip())
    if str(args.image_list).strip():
        image_specs.extend(_read_image_list(str(args.image_list).strip()))

    if image_specs and str(args.index_range).strip():
        raise ValueError("--images/--image-list 与 --index-range 不能同时使用")
    if image_specs and args.no_shuffle:
        # 显式列表天然按用户顺序输出，保留该参数不报错以兼容批处理命令。
        pass

    records = list(dataset_dicts)
    if image_specs:
        by_abs_path = {}
        by_rel_path = {}
        by_basename = {}
        for record in records:
            raw_file_name = str(record.get("file_name", "")).strip()
            if not raw_file_name:
                continue
            file_name = os.path.abspath(
                raw_file_name if os.path.isabs(raw_file_name) else os.path.join(root, raw_file_name)
            )
            rel_name = os.path.normpath(os.path.relpath(file_name, root))
            base_name = os.path.basename(file_name)
            by_abs_path.setdefault(file_name, []).append(record)
            by_rel_path.setdefault(rel_name, []).append(record)
            by_basename.setdefault(base_name, []).append(record)

        selected: List[dict] = []
        seen = set()
        missing: List[str] = []
        for spec in image_specs:
            candidate = os.path.abspath(spec if os.path.isabs(spec) else os.path.join(root, spec))
            rel_name = os.path.normpath(os.path.relpath(candidate, root))
            matches = by_abs_path.get(candidate, [])
            if not matches:
                matches = by_rel_path.get(os.path.normpath(spec), [])
            if not matches:
                matches = by_rel_path.get(rel_name, [])
            if not matches and os.path.basename(spec):
                matches = by_basename.get(os.path.basename(spec), [])
            if not matches:
                missing.append(spec)
                continue
            if len(matches) > 1:
                raise ValueError(f"图片文件名不唯一，请使用相对或绝对路径: {spec}")
            record = matches[0]
            record_file_name = str(record["file_name"]).strip()
            key = os.path.abspath(
                record_file_name if os.path.isabs(record_file_name) else os.path.join(root, record_file_name)
            )
            if key not in seen:
                selected.append(record)
                seen.add(key)

        if missing:
            available = [str(d.get("file_name", "")) for d in records[:5]]
            raise FileNotFoundError(
                "以下图片不在注册后的数据集中: "
                + ", ".join(missing)
                + f"\n示例可用路径: {available}"
            )
        if not selected:
            raise RuntimeError("显式图片列表没有选择到有效图片")
        return selected

    if str(args.index_range).strip():
        start, end = _parse_index_range(str(args.index_range))
        if int(args.index_base) == 1:
            start, end = start - 1, end - 1
        start = max(0, start)
        end = max(0, end)
        selected = records[start : end + 1]
        if not selected:
            raise RuntimeError(
                f"--index-range {args.index_range} 未选中图片；数据集有效图片数为 {len(records)}"
            )
        return selected

    if args.no_shuffle:
        return records[: max(0, int(args.num))]
    return random.sample(records, k=min(int(args.num), len(records)))


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


def _build_mask2former_qsp_predictor(
    *,
    mask2former_root: str,
    config_file: str,
    weights: str,
    prior_path: str,
    num_classes: int,
    score_thr: float,
) -> DefaultPredictor:
    """构建 Mask2Former+QSP predictor，并支持命令行覆盖先验路径。"""
    root = os.path.abspath(str(mask2former_root).strip()) if str(mask2former_root).strip() else ""
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import mask2former as _mask2former  # noqa: F401  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "无法导入 mask2former。请设置 --mask2former-root，或将 Mask2Former 根目录加入 PYTHONPATH。"
        ) from exc

    from detectron2.projects.deeplab import add_deeplab_config  # type: ignore[import-not-found]
    from mask2former import add_maskformer2_config  # type: ignore[import-not-found]

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(os.path.abspath(str(config_file)))
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.abspath(str(weights))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = float(score_thr)
    if str(prior_path).strip():
        cfg.MODEL.MASK_FORMER.PRIOR_ON = True
        cfg.MODEL.MASK_FORMER.PRIOR_PATH = os.path.abspath(str(prior_path).strip())
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(os.path.abspath(str(weights)))
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


def _make_header_row(
    column_titles: List[Tuple[str, str]],
    cell_width: int,
    header_h: int = 72,
) -> np.ndarray:
    """参考 visualize_predictions_table_layout_direct.py 的全局列标题风格。"""
    w = int(cell_width) * len(column_titles)
    h = int(header_h)
    header = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.line(header, (0, h - 1), (w - 1, h - 1), (165, 165, 165), 2, cv2.LINE_AA)
    for i, (title, subtitle) in enumerate(column_titles):
        x0 = i * int(cell_width)
        x1 = x0 + int(cell_width)
        cv2.line(header, (x0, 0), (x0, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        tx = x0 + max(6, (int(cell_width) - tw) // 2)
        cv2.putText(header, title, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
        (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        sx = x0 + max(6, (int(cell_width) - sw) // 2)
        cv2.putText(header, subtitle, (sx, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (75, 75, 75), 1, cv2.LINE_AA)
        cv2.line(header, (x0, 0), (x0, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
        if i == len(column_titles) - 1:
            cv2.line(header, (x1 - 1, 0), (x1 - 1, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    return header


def _add_row_tag(row_img: np.ndarray, tag: str) -> np.ndarray:
    """参考 direct table layout：左上角添加 S1/S2... 行标签。"""
    out = row_img.copy()
    cv2.rectangle(out, (6, 6), (44, 28), (248, 248, 248), thickness=-1)
    cv2.rectangle(out, (6, 6), (44, 28), (178, 178, 178), thickness=1)
    cv2.putText(out, tag, (13, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def _fit_cell(img: np.ndarray, cell_width: int, cell_height: int) -> np.ndarray:
    """等比例缩放并居中填充到固定单元格，保证四栏严格对齐。"""
    h, w = img.shape[:2]
    scale = min(float(cell_width) / max(w, 1), float(cell_height) / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    x = (cell_width - nw) // 2
    y = (cell_height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def _instances_union_mask(instances, height: int, width: int) -> np.ndarray:
    """把预测实例合并为单张二值 mask；没有预测时返回全黑。"""
    if instances is None or len(instances) == 0 or not instances.has("pred_masks"):
        return np.zeros((height, width), dtype=np.uint8)
    masks = instances.pred_masks.to("cpu").numpy().astype(bool)
    if masks.ndim == 3:
        return np.any(masks, axis=0).astype(np.uint8)
    return np.zeros((height, width), dtype=np.uint8)


def _binary_mask_view(mask01: np.ndarray) -> np.ndarray:
    """示例图中的黑底白色 mask。"""
    m = (mask01 > 0).astype(np.uint8)
    return np.repeat((m * 255)[:, :, None], 3, axis=2)


def _overlay_gray_mask(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    alpha: float = 0.42,
) -> np.ndarray:
    """Shape Prior Guidance：在输入图上以半透明灰色显示引导区域。"""
    out = img_bgr.copy().astype(np.float32)
    m = np.clip(mask01.astype(np.float32), 0.0, 1.0)
    color = np.full_like(out, 220.0)
    a = np.clip(m * float(alpha), 0.0, 1.0)[..., None]
    out = out * (1.0 - a) + color * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_color_mask(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float = 0.42,
) -> np.ndarray:
    """在原图上以指定颜色填充分割区域，并绘制同色轮廓。"""
    out = img_bgr.copy().astype(np.float32)
    mask = (mask01 > 0).astype(np.uint8)
    color = np.asarray(color_bgr, dtype=np.float32)
    if np.any(mask):
        out[mask > 0] = out[mask > 0] * (1.0 - float(alpha)) + color * float(alpha)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out_u8, contours, -1, tuple(int(v) for v in color_bgr), 2, cv2.LINE_AA)
    return out_u8


def _gt_mask_from_dict(dataset_dict, height: int, width: int) -> np.ndarray:
    """从 COCO annotation 生成 GT mask；失败时返回空 mask。"""
    try:
        anns = dataset_dict.get("annotations", [])
        inst = d2_utils.annotations_to_instances(anns, (height, width), mask_format="bitmask")
        if inst.has("gt_masks"):
            masks = inst.gt_masks.tensor.numpy().astype(bool)
            if masks.ndim == 3:
                return np.any(masks, axis=0).astype(np.uint8)
    except Exception:
        pass
    return np.zeros((height, width), dtype=np.uint8)


def _failure_boxes(
    gt_mask: np.ndarray,
    baseline_mask: np.ndarray,
    final_mask: np.ndarray,
    max_boxes: int = 2,
) -> List[Tuple[int, int, int, int]]:
    """
    生成示例图中的红色失败区域框：
    优先使用 GT 与 baseline/final 的差异；若没有 GT，则使用 baseline/final 差异。
    """
    if int(gt_mask.sum()) > 0:
        diff = ((gt_mask > 0) ^ (baseline_mask > 0)) | ((gt_mask > 0) ^ (final_mask > 0))
    else:
        diff = (baseline_mask > 0) ^ (final_mask > 0)
    diff_u8 = diff.astype(np.uint8)
    if int(diff_u8.sum()) == 0:
        return []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(diff_u8, connectivity=8)
    candidates = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 24:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        candidates.append((area, x, y, w, h))
    candidates.sort(reverse=True)
    if not candidates:
        ys, xs = np.where(diff_u8 > 0)
        return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]
    boxes: List[Tuple[int, int, int, int]] = []
    pad = 8
    height, width = diff_u8.shape
    for _, x, y, w, h in candidates[: int(max_boxes)]:
        boxes.append(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(width - 1, x + w + pad),
                min(height - 1, y + h + pad),
            )
        )
    return boxes


def _draw_failure_boxes(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = img_bgr.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _aligned_prior_image(
    img_bgr: np.ndarray,
    prior_instances,
    shape_debug_cache,
) -> np.ndarray:
    """
    将 ShapePriorAdapter 的 rotated_prior 映射回 top instance box。
    没有 debug cache 时退化为 prior 最终预测 mask。
    """
    cache = shape_debug_cache or {}
    rotated_prior = cache.get("rotated_prior", None)
    if rotated_prior is None or prior_instances is None or len(prior_instances) == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)
    try:
        prior_np = rotated_prior.detach().cpu().numpy() if hasattr(rotated_prior, "detach") else np.asarray(rotated_prior)
        prior_np = np.squeeze(prior_np).astype(np.float32)
        if prior_np.ndim != 2:
            return np.zeros(img_bgr.shape[:2], dtype=np.float32)
        idx = int(prior_instances.scores.argmax().item()) if prior_instances.has("scores") else 0
        box = prior_instances.pred_boxes.tensor[idx].detach().cpu().numpy().tolist()
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, min(img_bgr.shape[1] - 1, x0))
        y0 = max(0, min(img_bgr.shape[0] - 1, y0))
        x1 = max(x0 + 1, min(img_bgr.shape[1], x1))
        y1 = max(y0 + 1, min(img_bgr.shape[0], y1))
        resized = cv2.resize(prior_np, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        out = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        out[y0:y1, x0:x1] = np.clip(resized, 0.0, 1.0)
        return out
    except Exception:
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)


def _stack_grid(rows: List[np.ndarray]) -> np.ndarray:
    if not rows:
        raise RuntimeError("no rows to stack")
    width = max(r.shape[1] for r in rows)
    normalized = []
    for row in rows:
        if row.shape[1] == width:
            normalized.append(row)
        else:
            normalized.append(
                cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            )
    return np.concatenate(normalized, axis=0)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # 设置 shape prior 路径（ShapeAwareCoarseMaskHead 会从环境变量读取）
    if args.shape_prior_npy.strip():
        os.environ["SHAPE_PRIOR_PATH"] = os.path.abspath(args.shape_prior_npy.strip())
    # 让 ShapeAwareCoarseMaskHead 缓存 rotated_prior / gate_map 等中间结果，
    # 用于第三栏 Shape Prior Guidance。
    os.environ["SHAPE_PRIOR_DEBUG"] = "1"

    dataset_name, num_classes = register_plug_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        json_filename=args.json_file,
    )
    metadata = MetadataCatalog.get(dataset_name)
    dicts = DatasetCatalog.get(dataset_name)
    if not dicts:
        raise RuntimeError("dataset is empty")

    picks = _select_dataset_dicts(dicts, args.dataset_root, args)
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
    qsp_pred = None
    if str(args.weights_mask2former_qsp).strip():
        qsp_config = os.path.abspath(str(args.config_mask2former_qsp).strip())
        if not os.path.isfile(qsp_config):
            raise FileNotFoundError(f"QSP config 不存在: {qsp_config}")
        qsp_prior = str(args.prior_path_mask2former_qsp).strip()
        if qsp_prior and not os.path.isfile(os.path.abspath(qsp_prior)):
            raise FileNotFoundError(f"QSP prior 不存在: {os.path.abspath(qsp_prior)}")
        qsp_pred = _build_mask2former_qsp_predictor(
            mask2former_root=str(args.mask2former_root),
            config_file=qsp_config,
            weights=os.path.abspath(str(args.weights_mask2former_qsp).strip()),
            prior_path=qsp_prior,
            num_classes=num_classes,
            score_thr=args.score_thr,
        )
        print(f"qsp_config: {qsp_config}")
        print(f"qsp_weights: {os.path.abspath(str(args.weights_mask2former_qsp).strip())}")
        if qsp_prior:
            print(f"qsp_prior: {os.path.abspath(qsp_prior)}")

    print(f"out_dir: {out_dir}")
    print(f"num_samples: {len(picks)}")
    for selected_idx, selected_dict in enumerate(picks):
        print(f"selected[{selected_idx}]: {selected_dict['file_name']}")

    cell_width = max(160, int(args.cell_width))
    cell_height = max(120, int(round(cell_width * 0.75)))
    column_titles = [
        ("1) Input Image", "(Original Highlighted Input)"),
        ("2) Baseline Mask", "(Binary Mask / Broken Boundary)"),
        ("3) Shape Prior Prediction", "(Colored Segmentation)"),
        ("4) Final Mask", "(Binary Mask / Corrected Boundary)"),
    ]
    rows_img: List[np.ndarray] = []
    qsp_rows_img: List[np.ndarray] = []

    for i, d in enumerate(picks):
        img_path = d["file_name"]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[skip] failed to read: {img_path}")
            continue

        # predictor 输入默认是 BGR uint8（detectron2 默认 FORMAT=BGR）
        base_out = base_pred(img_bgr)
        prior_out = prior_pred(img_bgr)
        qsp_out = qsp_pred(img_bgr) if qsp_pred is not None else None

        height, width = img_bgr.shape[:2]
        gt_mask = _gt_mask_from_dict(d, height, width)
        baseline_mask = _instances_union_mask(base_out["instances"], height, width)
        final_mask = _instances_union_mask(prior_out["instances"], height, width)
        qsp_mask = (
            _instances_union_mask(qsp_out["instances"], height, width)
            if qsp_out is not None
            else None
        )
        # 第1栏：直接显示评测集中的原始高光输入，不叠加任何预测结果。
        input_panel = img_bgr.copy()

        # 第2栏：基线模型的标准黑底白色二值 mask。
        baseline_mask_panel = _binary_mask_view(baseline_mask)

        # 第3栏：保留之前脚本的彩色形状先验预测区域风格，不绘制红色框。
        guidance_panel = _overlay_color_mask(
            img_bgr,
            final_mask,
            color_bgr=(80, 210, 160),
            alpha=0.42,
        )

        # 第4栏：带先验模型的标准黑底白色最终二值 mask。
        final_panel = _binary_mask_view(final_mask)
        qsp_mask_panel = _binary_mask_view(qsp_mask) if qsp_mask is not None else None
        qsp_overlay_panel = (
            _overlay_color_mask(
                img_bgr,
                qsp_mask,
                color_bgr=(80, 170, 235),
                alpha=0.42,
            )
            if qsp_mask is not None
            else None
        )

        cells = [
            _fit_cell(input_panel, cell_width, cell_height),
            _fit_cell(baseline_mask_panel, cell_width, cell_height),
            _fit_cell(guidance_panel, cell_width, cell_height),
            _fit_cell(final_panel, cell_width, cell_height),
        ]
        row = np.concatenate(cells, axis=1)
        rows_img.append(row)

        stem = f"{i:03d}_" + os.path.splitext(os.path.basename(img_path))[0]
        row_path = os.path.join(out_dir, f"{stem}_four_column.png")
        cv2.imwrite(row_path, row)
        # 保留原脚本的无后缀兼容文件名，但内容更新为四栏版式。
        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), row)

        # 同时保存四个单栏结果，方便后续论文排版。
        cv2.imwrite(os.path.join(out_dir, f"{stem}_input_highlight.png"), input_panel)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_baseline_mask.png"), baseline_mask_panel)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_shape_prior_overlay.png"), guidance_panel)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_final_mask.png"), final_panel)
        if qsp_mask_panel is not None and qsp_overlay_panel is not None:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_qsp_mask.png"), qsp_mask_panel)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_qsp_overlay.png"), qsp_overlay_panel)
            qsp_cells = [
                _fit_cell(input_panel, cell_width, cell_height),
                _fit_cell(baseline_mask_panel, cell_width, cell_height),
                _fit_cell(qsp_overlay_panel, cell_width, cell_height),
                _fit_cell(qsp_mask_panel, cell_width, cell_height),
            ]
            qsp_row = np.concatenate(qsp_cells, axis=1)
            qsp_rows_img.append(qsp_row)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_four_column_qsp.png"), qsp_row)
        print(f"[{i+1}/{len(picks)}] saved: {row_path}")

    if not rows_img:
        raise RuntimeError("no valid rows were generated")

    header = _make_header_row(
        column_titles,
        cell_width=cell_width,
        header_h=max(60, int(args.header_height)),
    )
    grid_body = _stack_grid(rows_img)
    grid = np.concatenate([header, grid_body], axis=0)
    grid_path = os.path.join(out_dir, "four_column_grid.png")
    cv2.imwrite(grid_path, grid)
    print(f"saved: {grid_path}")
    if qsp_rows_img:
        # QSP 专用图不添加顶部列标题，便于直接用于论文排版。
        qsp_grid = _stack_grid(qsp_rows_img)
        qsp_grid_path = os.path.join(out_dir, "four_column_qsp_grid.png")
        cv2.imwrite(qsp_grid_path, qsp_grid)
        print(f"saved: {qsp_grid_path}")


if __name__ == "__main__":
    main()


