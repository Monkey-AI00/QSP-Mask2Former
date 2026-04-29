#!/usr/bin/env python3
"""
Mech-Eye 实时预览 + 实例分割 + 同帧生成 3D 点云（PLY）

当前脚本同时支持两套推理后端：
1) PointRend / PointRend + Shape Prior
2) Mask2Former / Mask2Former + QSP

目标（同一帧闭环，避免错位）：
1) Mech-Eye: capture_2d_and_3d() -> color + depth
2) 实例分割模型对 color 推理，生成二值 mask（前景=255，背景=0）
3) Mech-Eye SDK: get_point_cloud_after_mapping(depth, mask, color, intrinsics, points_xyz_bgr)

交互：
- 'q'：退出
- 's'：保存当前帧的 color/mask/ply（并可选 cleargrasp depth completion）

注意：
- 运行需要 GUI（cv2.imshow）。无 GUI 请改用 --save_dir 离线输出。
- 常见需要 sudo（SDK 写 /var/log 权限 + 设备访问权限）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import cv2
import numpy as np
import torch

# 添加 detectron2 到路径（脚本位于 detectron2/projects/PointRend 下）
_D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _D2_ROOT not in sys.path:
    sys.path.insert(0, _D2_ROOT)
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_D2_ROOT, "..", ".."))
_DEFAULT_MASK2FORMER_ROOT = os.path.abspath(os.path.join(_WORKSPACE_ROOT, "..", "Mask2Former"))
_DEFAULT_POINTREND_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "configs", "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_plug.yaml")
)
_DEFAULT_MASK2FORMER_BASE_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "configs", "InstanceSegmentation", "mask2former_R50_plug.yaml")
)
_DEFAULT_MASK2FORMER_QSP_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "configs", "InstanceSegmentation", "mask2former_R50_plug_qsp_aug.yaml")
)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import ColorMode, Visualizer

# 触发注册：ShapeAwareCoarseMaskHead（当构建 prior predictor 时需要）
import custom_heads  # noqa: F401

# Mech-Eye（运行时需要 MechEyeAPI）
if TYPE_CHECKING:  # pragma: no cover
    from mecheye.area_scan_3d_camera import (  # type: ignore
        Camera,
        CameraIntrinsics,
        FileFormat_PLY,
        Frame2DAnd3D,
        PointCloudEdgePreservation,
        PointCloudNoiseRemoval,
        PointCloudOutlierRemoval,
        PointCloudSurfaceSmoothing,
        Scanning3DExposureSequence,
        TexturedPointCloud,
        get_point_cloud_after_mapping,
    )
    from mecheye.shared import GrayScale2DImage, show_error  # type: ignore
else:
    from mecheye.area_scan_3d_camera import *  # type: ignore # noqa: F401,F403
    from mecheye.shared import *  # type: ignore # noqa: F401,F403

try:  # pragma: no cover
    from mecheye.area_scan_3d_camera_utils import print_camera_info as _print_camera_info  # type: ignore
except Exception:  # pragma: no cover
    _print_camera_info = None


def _print_camera_info_fallback(ci: Any) -> None:
    ip = getattr(ci, "ip_address", None)
    sn = getattr(ci, "serial_number", None)
    model = getattr(ci, "model", None)
    fw = getattr(ci, "firmware_version", None)
    print(f"  ip_address={ip} serial_number={sn} model={model} firmware={fw}")


def discover_and_print_cameras() -> list:
    print("Discovering all available cameras...")
    camera_infos = Camera.discover_cameras()
    if len(camera_infos) == 0:
        print("No cameras found.")
        return []
    for i in range(len(camera_infos)):
        print(f"Camera index: {i}")
        ci = camera_infos[i]
        if _print_camera_info is not None:
            _print_camera_info(ci)
        else:
            _print_camera_info_fallback(ci)
    return camera_infos


def connect_camera(camera: "Camera", *, ip: str = "", serial: str = "", index: int = -1) -> bool:
    if ip:
        print(f"Connecting by IP: {ip} ...")
        st = camera.connect(ip)
        if not st.is_ok():
            show_error(st)
            return False
        print("Connected to the camera successfully.")
        return True

    camera_infos = discover_and_print_cameras()
    if not camera_infos:
        return False

    if serial:
        serial = str(serial).strip()
        for i, ci in enumerate(camera_infos):
            if str(getattr(ci, "serial_number", "")).strip() == serial:
                print(f"Connecting by serial_number={serial} (index={i}) ...")
                st = camera.connect(ci)
                if not st.is_ok():
                    show_error(st)
                    return False
                print("Connected to the camera successfully.")
                return True
        print(f"未在 discover_cameras() 列表中找到 serial_number={serial}")
        return False

    if index < 0:
        index = 0
    if not (0 <= int(index) < len(camera_infos)):
        print(f"--index 越界：{index}，有效范围 [0, {len(camera_infos)-1}]")
        return False

    print(f"Connecting by index: {index} ...")
    st = camera.connect(camera_infos[int(index)])
    if not st.is_ok():
        show_error(st)
        return False
    print("Connected to the camera successfully.")
    return True


def _add_mask2former_to_syspath(mask2former_root: str) -> str:
    root = os.path.abspath(str(mask2former_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _require_existing_file(path: str, label: str) -> str:
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def build_pointrend_predictor(
    *,
    config_file: str,
    weights: str,
    mask_head_name: str,
    score_thresh: float,
    device: str,
    num_classes: int = 1,
) -> DefaultPredictor:
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_MASK_HEAD.NAME = str(mask_head_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device
    # 与 inference_plug.py 保持一致：ROI_HEADS 和 POINT_HEAD 类别数必须一致
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = int(num_classes)
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    # 显式 load 一次（DefaultPredictor 内部也会 load，但这里确保日志更直观/与 visualize 脚本一致）
    DetectionCheckpointer(predictor.model).load(weights)
    predictor.model.eval()
    return predictor


def build_mask2former_predictor(
    *,
    mask2former_root: str,
    config_file: str,
    weights: str,
    score_thresh: float,
    device: str,
    num_classes: int = 1,
    prior_path_override: str = "",
) -> DefaultPredictor:
    _add_mask2former_to_syspath(mask2former_root)

    # 显式 import 以触发 Mask2Former 注册（meta arch / dataset mapper / decoder 等）。
    import mask2former as _mask2former  # noqa: F401  # type: ignore

    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config  # type: ignore

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = float(score_thresh)
    if str(prior_path_override).strip():
        cfg.MODEL.MASK_FORMER.PRIOR_ON = True
        cfg.MODEL.MASK_FORMER.PRIOR_PATH = str(prior_path_override).strip()
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(weights)
    predictor.model.eval()
    return predictor


def instances_to_binary_mask_u8(
    instances,
    *,
    mask_mode: Literal["union", "maxscore"] = "union",
) -> np.ndarray:
    """
    复用 inference_plug.py 逻辑：
    - union：所有实例并集
    - maxscore：只保留最高分实例
    输出：H×W 的 uint8，前景=255，背景=0
    """
    if not hasattr(instances, "pred_masks") or len(instances) == 0:
        # instances 可能为空；但需要输出同尺寸 mask
        # 这里假设 instances.image_size 可用（detectron2 Instances 通常有）
        h, w = getattr(instances, "image_size", (0, 0))
        if h == 0 or w == 0:
            raise ValueError("instances 为空且无法确定 image_size，请确认输入图像尺寸。")
        return np.zeros((h, w), dtype=np.uint8)

    pred_masks = instances.pred_masks
    pred_masks_np = pred_masks.numpy() if hasattr(pred_masks, "numpy") else np.asarray(pred_masks)
    if pred_masks_np.ndim != 3:
        raise ValueError(f"pred_masks 维度异常，期望 (N,H,W)，实际为 {pred_masks_np.shape}")

    if mask_mode == "maxscore":
        scores = instances.scores.numpy() if hasattr(instances, "scores") else None
        best_idx = int(np.argmax(scores)) if scores is not None and scores.size > 0 else 0
        binary_mask = pred_masks_np[best_idx]
    else:
        binary_mask = np.any(pred_masks_np, axis=0)

    return (binary_mask.astype(np.uint8) * 255)


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    a,b: [x1,y1,x2,y2]
    """
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def instances_to_binary_mask_u8_pc(
    instances,
    *,
    pc_mask_mode: Literal["union", "maxscore", "iou"] = "iou",
    iou_thresh: float = 0.1,
    join_dilate: int = 25,
) -> np.ndarray:
    """
    点云用 mask 的更稳健版本：
    - union：所有实例并集（可能把绳子/背景也并进来）
    - maxscore：只取最高分实例（可能导致插头被拆成多块时不完整）
    - iou：以最高分实例为锚点，把与其 bbox IoU>=阈值 的实例并起来（推荐）
    """
    if not hasattr(instances, "pred_masks") or len(instances) == 0:
        h, w = getattr(instances, "image_size", (0, 0))
        if h == 0 or w == 0:
            raise ValueError("instances 为空且无法确定 image_size，请确认输入图像尺寸。")
        return np.zeros((h, w), dtype=np.uint8)

    if pc_mask_mode in ("union", "maxscore"):
        return instances_to_binary_mask_u8(instances, mask_mode=pc_mask_mode)  # type: ignore[arg-type]

    # iou 模式：以最高分实例为锚点，合并“同一目标的碎片”
    scores = instances.scores.numpy() if hasattr(instances, "scores") else None
    best_idx = int(np.argmax(scores)) if scores is not None and scores.size > 0 else 0

    boxes = None
    if hasattr(instances, "pred_boxes"):
        try:
            boxes = instances.pred_boxes.tensor.numpy()
        except Exception:
            boxes = None
    if boxes is None or boxes.shape[0] == 0:
        # 没有 bbox 就退化成 union
        return instances_to_binary_mask_u8(instances, mask_mode="union")

    best_box = boxes[best_idx]
    pm = instances.pred_masks
    pm_np = pm.numpy() if hasattr(pm, "numpy") else np.asarray(pm)
    if pm_np.ndim != 3:
        raise ValueError(f"pred_masks 维度异常，期望 (N,H,W)，实际为 {pm_np.shape}")

    anchor = pm_np[best_idx].astype(np.uint8)
    if join_dilate and int(join_dilate) > 0:
        k = int(join_dilate)
        # kernel 尽量取奇数，保持中心对齐
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        anchor_d = cv2.dilate(anchor, kernel, iterations=1)
    else:
        anchor_d = anchor

    keep: list[int] = []
    thr = float(iou_thresh)
    for i in range(int(boxes.shape[0])):
        iou = _bbox_iou_xyxy(best_box, boxes[i])
        # 像素级“接触/重叠”判定：锚点(膨胀后)与候选实例是否有交集
        try:
            touch = bool(np.any(anchor_d & pm_np[i].astype(np.uint8)))
        except Exception:
            touch = False
        if iou >= thr or touch:
            keep.append(i)
    if not keep:
        keep = [best_idx]

    binary = np.any(pm_np[np.array(keep, dtype=np.int64)], axis=0)
    return (binary.astype(np.uint8) * 255)


def _mask_stats(mask_u8: np.ndarray) -> str:
    """
    打印 mask 的前景占比与前景 bbox，快速判断是否“选反/选错”。
    """
    if mask_u8 is None or mask_u8.size == 0:
        return "mask=empty"
    m = mask_u8.astype(np.uint8, copy=False)
    fg = m > 0
    total = int(m.shape[0] * m.shape[1])
    fg_cnt = int(np.count_nonzero(fg))
    if fg_cnt == 0:
        return f"mask_fg=0/{total} (0.000) bbox=None"
    ys, xs = np.where(fg)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    ratio = float(fg_cnt) / float(total)
    return f"mask_fg={fg_cnt}/{total} ({ratio:.3f}) bbox=({x1},{y1})-({x2},{y2})"


def _mask_fg_ratio(mask_u8: np.ndarray) -> float:
    if mask_u8 is None or mask_u8.size == 0:
        return 0.0
    m = mask_u8.astype(np.uint8, copy=False)
    total = float(m.shape[0] * m.shape[1])
    if total <= 0:
        return 0.0
    fg = float(np.count_nonzero(m > 0))
    return fg / total


def _overlay_binary_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    color_bgr: tuple[int, int, int] = (80, 210, 160),
    alpha: float = 0.35,
    outline: bool = True,
) -> np.ndarray:
    out = img_bgr.copy()
    if mask_u8 is None or mask_u8.size == 0:
        return out
    m = (mask_u8 > 0).astype(np.uint8)
    if not np.any(m):
        return out
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out_f = out.astype(np.float32)
    sel = m.astype(bool)
    out_f[sel] = out_f[sel] * (1.0 - float(alpha)) + color * float(alpha)
    out = np.clip(out_f, 0, 255).astype(np.uint8)
    if outline:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color_bgr), 2, cv2.LINE_AA)
    return out


def _build_output_masks(
    instances,
    *,
    mask_mode: str,
    pc_mask_mode: str,
    pc_iou_thresh: float,
    pc_join_dilate: int,
    mask_close: int,
    mask_dilate: int,
    mask_erode: int,
    invert_mask: bool,
    auto_invert_mask: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    mask_u8_vis = instances_to_binary_mask_u8(instances, mask_mode=mask_mode)  # type: ignore[arg-type]
    mask_u8_vis = _morph_mask(mask_u8_vis, close=mask_close, dilate=mask_dilate, erode=mask_erode)
    mask_u8_pc = instances_to_binary_mask_u8_pc(
        instances,
        pc_mask_mode=pc_mask_mode,  # type: ignore[arg-type]
        iou_thresh=pc_iou_thresh,
        join_dilate=pc_join_dilate,
    )
    mask_u8_pc = _morph_mask(mask_u8_pc, close=mask_close, dilate=mask_dilate, erode=mask_erode)

    invert_applied = False
    if invert_mask:
        invert_applied = True
    elif auto_invert_mask:
        r0 = _mask_fg_ratio(mask_u8_pc)
        r1 = _mask_fg_ratio(255 - mask_u8_pc)
        invert_applied = r1 < r0

    if invert_applied:
        mask_u8_vis = 255 - mask_u8_vis
        mask_u8_pc = 255 - mask_u8_pc
    return mask_u8_vis, mask_u8_pc, invert_applied


def mask_u8_to_sdk_grayscale2d(mask_u8: np.ndarray) -> "GrayScale2DImage":
    """
    PointRend 二值 mask 语义：前景=255，背景=0
    Mech-Eye SDK 示例语义：保留=0，剔除=255
    因此需要把前景(255)映射为 0；背景(0)映射为 255。
    """
    if mask_u8.ndim != 2:
        raise ValueError(f"mask_u8 必须是 2D，实际 shape={mask_u8.shape}")
    h, w = mask_u8.shape[:2]
    sdk_mask = GrayScale2DImage()
    sdk_mask.resize(int(w), int(h))

    # 逐像素写入（SDK 结构），优先正确性；如后续要加速可改 ROI/采样
    for i in range(int(h)):
        row = mask_u8[i]
        for j in range(int(w)):
            is_fg = int(row[j]) > 0
            sdk_mask.at(i, j).gray = 0 if is_fg else 255
    return sdk_mask


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return []
    return [float(p) for p in parts]


def _apply_mecheye_params(
    camera: "Camera",
    *,
    exposure_sequence: list[float] | None = None,
    pc_surface_smoothing: str = "",
    pc_noise_removal: str = "",
    pc_outlier_removal: str = "",
    pc_edge_preservation: str = "",
    save_to_device: bool = False,
) -> None:
    """
    采集前设置相机参数（HDR/点云后处理）。
    """
    current_user_set = camera.current_user_set()

    # HDR：多曝光（本质是 3D 扫描曝光序列）
    if exposure_sequence:
        show_error(current_user_set.set_float_array_value(Scanning3DExposureSequence.name, exposure_sequence))

    # 点云后处理（SDK 内置）
    def _set_level(param_cls, level: str):
        if not level:
            return
        key = f"Value_{level.capitalize()}"
        v = getattr(param_cls, key, None)
        if v is None:
            raise ValueError(f"不支持 {param_cls.name} 的 level={level}，期望 off/weak/normal/strong")
        show_error(current_user_set.set_enum_value(param_cls.name, v))

    def _set_edge(level: str):
        if not level:
            return
        key = f"Value_{level.capitalize()}"
        v = getattr(PointCloudEdgePreservation, key, None)
        if v is None:
            raise ValueError("不支持 PointCloudEdgePreservation 的 level，期望 sharp/normal/smooth")
        show_error(current_user_set.set_enum_value(PointCloudEdgePreservation.name, v))

    _set_level(PointCloudSurfaceSmoothing, pc_surface_smoothing)
    _set_level(PointCloudNoiseRemoval, pc_noise_removal)
    _set_level(PointCloudOutlierRemoval, pc_outlier_removal)
    _set_edge(pc_edge_preservation)

    if save_to_device:
        show_error(current_user_set.save_all_parameters_to_device())


def _morph_mask(mask_u8: np.ndarray, *, close: int = 0, dilate: int = 0, erode: int = 0) -> np.ndarray:
    """
    对二值 mask 做形态学处理（提高边缘连续性/填小孔）。
    """
    m = mask_u8
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"mask_u8 必须是 2D，实际 shape={m.shape}")
    if close and close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k, iterations=1)
    if erode and erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        m = cv2.erode(m, k, iterations=1)
    return m


def _render_depth_data(depth: np.ndarray) -> np.ndarray:
    """
    参考官方 render_depth_map.py：把深度渲染成 Jet 彩色，0/无效为黑色。
    """
    if depth is None or depth.size == 0:
        return np.array([])
    d = depth.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((*d.shape[:2], 3), dtype=np.uint8)
    minv = float(np.min(d[valid]))
    maxv = float(np.max(d[valid]))
    if np.isclose(maxv - minv, 0):
        depth8 = np.clip(d, 0, 255).astype(np.uint8)
    else:
        depth8 = cv2.convertScaleAbs(d, alpha=(255.0 / (minv - maxv)), beta=((maxv * 255.0) / (maxv - minv) + 1))
    colored = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
    colored[~valid] = (0, 0, 0)
    return colored


def _depth_to_png_u16(depth_np: np.ndarray) -> np.ndarray:
    """
    将深度图转换为适合 PNG 保存的 uint16 单通道。
    - 若输入为 uint16：直接返回
    - 若输入为 uint8：提升到 uint16（保持数值）
    - 若输入为 float：四舍五入到 uint16，超出范围会 clip

    说明：PNG 支持 16-bit 灰度图，适合保存“深度数值”而不是 8-bit 可视化。
    """
    if depth_np is None or np.asarray(depth_np).size == 0:
        return np.array([], dtype=np.uint16)
    d = np.asarray(depth_np)
    if d.dtype == np.uint16:
        return d
    if d.dtype == np.uint8:
        return d.astype(np.uint16)
    df = d.astype(np.float64, copy=False)
    df = np.nan_to_num(df, nan=0.0, posinf=0.0, neginf=0.0)
    df = np.clip(np.rint(df), 0.0, 65535.0)
    return df.astype(np.uint16)


def _camera_matrix_to_dict(cm: Any) -> dict:
    return {
        "fx": float(getattr(cm, "fx")),
        "fy": float(getattr(cm, "fy")),
        "cx": float(getattr(cm, "cx")),
        "cy": float(getattr(cm, "cy")),
    }


def _intrinsics_to_dict(intr: Any) -> dict:
    """
    把 Mech-Eye CameraIntrinsics 序列化为 json 友好的 dict。
    目前重投影验证只强依赖 (fx,fy,cx,cy)。
    """
    out: dict[str, Any] = {}
    # 常见结构：intr.color.camera_matrix / intr.depth.camera_matrix
    for name in ("color", "depth"):
        sub = getattr(intr, name, None)
        if sub is None:
            continue
        cm = getattr(sub, "camera_matrix", None)
        if cm is None:
            continue
        out[name] = {"camera_matrix": _camera_matrix_to_dict(cm)}
    return out


def _add_cleargrasp_to_syspath() -> str:
    """
    让我们可以在同一 python 进程里直接 import ClearGrasp：
    - cleargrasp 根目录: detectron2/cleargrasp
    """
    cg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../cleargrasp"))
    if cg_root not in sys.path:
        sys.path.insert(0, cg_root)
    return cg_root


def _depth_u16_to_m(depth_u16: np.ndarray, *, unit: Literal["mm", "m"] = "mm") -> np.ndarray:
    d = np.asarray(depth_u16)
    if d.dtype != np.uint16:
        d = d.astype(np.uint16, copy=False)
    out = d.astype(np.float32)
    if str(unit) == "mm":
        out = out / 1000.0
    return out


def _depth_m_to_u16(depth_m: np.ndarray, *, unit: Literal["mm", "m"] = "mm") -> np.ndarray:
    d = np.asarray(depth_m).astype(np.float32, copy=False)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if str(unit) == "mm":
        d = d * 1000.0
    d = np.clip(np.rint(d), 0.0, 65535.0)
    return d.astype(np.uint16)


def _get_depth_k_from_mecheye_intrinsics(intrinsics: Any) -> tuple[float, float, float, float]:
    """
    从 Mech-Eye CameraIntrinsics 中取 depth 相机内参 (fx, fy, cx, cy)。
    """
    depth = getattr(intrinsics, "depth", None)
    if depth is None:
        raise ValueError("intrinsics.depth 不存在")
    cm = getattr(depth, "camera_matrix", None)
    if cm is None:
        raise ValueError("intrinsics.depth.camera_matrix 不存在")
    return float(getattr(cm, "fx")), float(getattr(cm, "fy")), float(getattr(cm, "cx")), float(getattr(cm, "cy"))


def _scale_k_for_resize(
    fx: float, fy: float, cx: float, cy: float, *, in_w: int, in_h: int, out_w: int, out_h: int
) -> tuple[float, float, float, float]:
    sx = float(out_w) / float(in_w)
    sy = float(out_h) / float(in_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def _backproject_masked_xyzrgb(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        stride = 1
    d = depth_m.astype(np.float32, copy=False)
    m = mask_u8.astype(np.uint8, copy=False)
    valid = np.isfinite(d) & (d > 0) & (m > 0)
    if stride > 1:
        ss = np.zeros_like(valid, dtype=bool)
        ss[::stride, ::stride] = True
        valid = valid & ss
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    z = d[ys, xs]
    x = ((xs.astype(np.float32) - float(cx)) * z) / float(fx)
    y = ((ys.astype(np.float32) - float(cy)) * z) / float(fy)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    bgr = color_bgr[ys, xs].astype(np.uint8, copy=False)
    rgb = bgr[:, ::-1]
    return xyz, rgb


def _write_ply_xyzrgb_ascii(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = int(xyz.shape[0])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")


def main():
    parser = argparse.ArgumentParser(description="Mech-Eye Live -> Instance Segmentation -> Mask -> PointCloud")
    parser.add_argument(
        "--config-file",
        default=_DEFAULT_POINTREND_CONFIG,
        help="共享配置文件。对 PointRend 保持兼容；对 Mask2Former 若未显式提供 base/prior config，将回退到内置默认配置。",
    )
    parser.add_argument("--config-file-base", default="", help="base 模型配置文件（可选）")
    parser.add_argument("--config-file-prior", default="", help="prior/QSP 模型配置文件（可选）")
    parser.add_argument(
        "--model-family",
        choices=["pointrend", "mask2former"],
        default="pointrend",
        help="选择实时推理后端：pointrend 或 mask2former。",
    )
    parser.add_argument(
        "--mask2former-root",
        default=_DEFAULT_MASK2FORMER_ROOT,
        help="Mask2Former 仓库根目录（包含 mask2former/ 包）。仅 model-family=mask2former 时使用。",
    )

    # 对齐 visualize_predictions_under_highlight.py 的参数命名
    parser.add_argument("--weights-base", default="", help="base 模型权重（PointRend 原版或 Mask2Former 原版）")
    parser.add_argument("--weights-prior", default="", help="prior 模型权重（ShapePrior 或 Mask2Former+QSP）")
    # 兼容旧参数：--weights（等价于 --weights-prior）
    parser.add_argument("--weights", default="", help="(兼容) 等价于 --weights-prior")

    parser.add_argument("--score-thr", type=float, default=0.5, help="分数阈值（对齐 visualize 的 --score-thr）")
    # 兼容旧参数：--score-thresh
    parser.add_argument("--score-thresh", type=float, default=None, help="(兼容) 若提供则覆盖 --score-thr")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=1, help="类别数（你的 plug=1）")

    # Shape Prior
    parser.add_argument(
        "--shape-prior-npy",
        default="",
        help="形状先验 .npy 路径。PointRend prior 会写入 SHAPE_PRIOR_PATH；Mask2Former+QSP 会覆盖 MODEL.MASK_FORMER.PRIOR_PATH。",
    )
    parser.add_argument(
        "--mode",
        choices=["prior", "base", "both"],
        default="prior",
        help="实时推理模式：prior=只跑形状先验模型；base=只跑原版模型；both=两者都跑并并排显示。",
    )
    parser.add_argument(
        "--pc-from",
        choices=["prior", "base"],
        default="prior",
        help="生成点云/保存 mask 时使用哪个模型的输出（默认 prior）。当 mode=both 时很有用。",
    )

    parser.add_argument("--mask-mode", choices=["union", "maxscore"], default="union")
    parser.add_argument(
        "--pc-mask-mode",
        choices=["union", "maxscore", "iou"],
        default="iou",
        help="用于生成点云/cleargrasp 输出点云的 mask 策略。推荐 iou：以最高分实例为锚点合并相邻碎片，避免 union 把无关实例并进来。",
    )
    parser.add_argument("--pc-iou-thresh", type=float, default=0.1, help="pc-mask-mode=iou 时的 bbox IoU 阈值（0~1，建议 0.05~0.2）")
    parser.add_argument(
        "--pc-join-dilate",
        type=int,
        default=25,
        help="pc-mask-mode=iou 时，锚点实例 mask 的膨胀核大小（像素）。用于把相邻碎片并入同一目标。0 表示关闭。建议 11~51。",
    )

    parser.add_argument("--ip", default="", help="Mech-Eye 相机 IP（推荐，例如 169.254.5.157）")
    parser.add_argument("--serial", default="", help="Mech-Eye 序列号（可选）")
    parser.add_argument("--index", type=int, default=-1, help="discover 列表索引（可选）")
    parser.add_argument("--discover", action="store_true", help="仅扫描相机列表然后退出")

    parser.add_argument("--win", default="Mech-Eye Live Segmentation", help="窗口标题")
    parser.add_argument("--wait", type=int, default=1, help="cv2.waitKey 毫秒")
    parser.add_argument("--infer-every", type=int, default=1, help="每 N 帧跑一次分割（减轻算力压力）")
    parser.add_argument("--show-masks-only", action="store_true", help="只叠加掩码（不画 bbox/label）")

    parser.add_argument("--save-dir", default="", help="按 's' 保存输出到该目录（color/mask/ply）")
    parser.add_argument("--no-gui", action="store_true", help="不打开窗口（适合无 GUI，只保存）")
    parser.add_argument("--capture-only", action="store_true", help="仅采集相机帧，不构建分割模型。")
    parser.add_argument("--max-captures", type=int, default=0, help="自动采集 N 帧后退出（0 表示不限制）。")
    parser.add_argument("--save-depth", action="store_true", help="按 's' 额外保存 raw depth 的彩色渲染图（便于快速查看）")
    parser.add_argument("--show-depth", action="store_true", help="实时可视化 raw depth：显示 raw depth 的彩色渲染图")
    parser.add_argument("--save-raw-ply", action="store_true", help="按 's' 同时保存未加 mask 的原始点云（用于对照是否采集侧已碎）")
    parser.add_argument("--save-intrinsics", action="store_true", help="按 's' 同时保存相机内参 intrinsics_*.json（用于重投影验证）")

    # ClearGrasp（depth completion）——已集成到同一 python 环境，按 's' 直接调用
    parser.add_argument("--cleargrasp", action="store_true", help="按 's' 直接调用 ClearGrasp 补全 depth，并输出 completed depth/ply（仅插头区域）")
    parser.add_argument("--cleargrasp-normals-weights", default="", help="ClearGrasp normals checkpoint_normals.pth")
    parser.add_argument("--cleargrasp-outlines-weights", default="", help="ClearGrasp outlines checkpoint_outlines.pth")
    parser.add_argument("--cleargrasp-depth2depth-exe", default="", help="depth2depth 可执行文件路径")
    parser.add_argument("--cleargrasp-out-w", type=int, default=256, help="ClearGrasp/depth2depth 输出宽（越小越快）")
    parser.add_argument("--cleargrasp-out-h", type=int, default=144, help="ClearGrasp/depth2depth 输出高（越小越快）")
    parser.add_argument("--cleargrasp-inertia", type=float, default=1000.0)
    parser.add_argument("--cleargrasp-smoothness", type=float, default=0.0001)
    parser.add_argument("--cleargrasp-tangent", type=float, default=1.0)
    parser.add_argument("--cleargrasp-depth-unit", choices=["mm", "m"], default="mm", help="Mech-Eye raw depth 的单位（默认 mm）")
    parser.add_argument("--cleargrasp-ply-stride", type=int, default=1, help="completed 点云像素步长降采样")
    parser.add_argument(
        "--cleargrasp-fill-thresh",
        type=float,
        default=0.0,
        help="把深度 <= 该阈值(米) 的像素也视为“空洞”并用补全结果替换。0 表示仅填 depth<=0 的洞。建议 0~0.01。",
    )
    parser.add_argument("--cleargrasp-filter-d", type=int, default=0, help="对补全深度做双边滤波：d>0 开启（建议 3~7）")
    parser.add_argument("--cleargrasp-filter-sigma-color", type=float, default=5.0)
    parser.add_argument("--cleargrasp-filter-sigma-space", type=float, default=10.0)

    # 采集/点云质量相关（对应你截图里的 HDR + Post-processing 建议）
    parser.add_argument("--exposure-seq", default="", help="HDR 多曝光序列（3D），例如 '5,10' 或 '3,6,12'；为空则不修改")
    parser.add_argument("--pc-smoothing", choices=["", "off", "weak", "normal", "strong"], default="", help="点云表面平滑")
    parser.add_argument("--pc-noise", choices=["", "off", "weak", "normal", "strong"], default="", help="点云噪声去除")
    parser.add_argument("--pc-outlier", choices=["", "off", "weak", "normal", "strong"], default="", help="点云离群点去除")
    parser.add_argument("--pc-edge", choices=["", "sharp", "normal", "smooth"], default="", help="边缘保持（Sharp/Normal/Smooth）")
    parser.add_argument("--save-userset", action="store_true", help="把上述参数保存到相机当前 user set（下次也生效）")

    # mask 形态学（补边/填小孔，可能让映射点更连续）
    parser.add_argument("--mask-close", type=int, default=0, help="mask 闭运算核大小（建议 5~15）")
    parser.add_argument("--mask-dilate", type=int, default=0, help="mask 膨胀核大小（建议 3~9）")
    parser.add_argument("--mask-erode", type=int, default=0, help="mask 腐蚀核大小（建议 3~9）")
    parser.add_argument("--invert-mask", action="store_true", help="强制反转二值 mask（当发现插头在 mask 里是黑色/被排除时使用）")
    parser.add_argument(
        "--auto-invert-mask",
        action="store_true",
        help="自动选择 mask 方向：比较原 mask 与反转 mask 的前景占比，取更小的一侧作为“目标前景”（更像只保留插头）。",
    )
    args = parser.parse_args()

    # 阈值兼容：若用户传了 --score-thresh，则覆盖 --score-thr
    score_thr = float(args.score_thr) if getattr(args, "score_thr", None) is not None else 0.5
    if args.score_thresh is not None:
        score_thr = float(args.score_thresh)

    # 兼容：若传了 --weights 但没传 --weights-prior，则认为它是 prior 权重
    if str(args.weights).strip() and not str(args.weights_prior).strip():
        args.weights_prior = str(args.weights).strip()

    mode = str(args.mode).strip().lower()
    pc_from = str(args.pc_from).strip().lower()
    model_family = str(args.model_family).strip().lower()

    shared_config = str(getattr(args, "config_file", "")).strip()
    if shared_config and os.path.abspath(shared_config) == _DEFAULT_POINTREND_CONFIG:
        shared_config_for_m2f = ""
    else:
        shared_config_for_m2f = shared_config

    if model_family == "mask2former":
        base_config_file = _require_existing_file(
            str(args.config_file_base).strip() or shared_config_for_m2f or _DEFAULT_MASK2FORMER_BASE_CONFIG,
            "Mask2Former base config",
        )
        prior_config_file = _require_existing_file(
            str(args.config_file_prior).strip() or shared_config_for_m2f or _DEFAULT_MASK2FORMER_QSP_CONFIG,
            "Mask2Former prior config",
        )
        family_prefix = "mask2former"
        base_title = "BASE (Mask2Former)"
        prior_title = "PRIOR (Mask2Former+QSP)"
    else:
        base_config_file = _require_existing_file(shared_config or _DEFAULT_POINTREND_CONFIG, "PointRend config")
        prior_config_file = base_config_file
        family_prefix = "pointrend"
        base_title = "BASE (PointRend)"
        prior_title = "PRIOR (PointRend+ShapePrior)"

    base_pred: Optional[DefaultPredictor] = None
    prior_pred: Optional[DefaultPredictor] = None

    if not bool(args.capture_only):
        prior_path_override = str(args.shape_prior_npy).strip()
        if prior_path_override:
            prior_path_override = _require_existing_file(prior_path_override, "prior npy")

        # prior 模式需要设置 prior 路径
        if mode in ("prior", "both") or pc_from == "prior":
            if model_family == "mask2former":
                if prior_path_override:
                    print(f"[mask2former][qsp] PRIOR_PATH override={prior_path_override}")
                else:
                    print(f"[mask2former][qsp] PRIOR_PATH follows config: {prior_config_file}")
            else:
                if prior_path_override:
                    os.environ["SHAPE_PRIOR_PATH"] = prior_path_override
                # 如果用户没传 shape_prior_npy，则保留环境变量已有值；否则走 custom_heads 默认路径
                env_prior = os.environ.get("SHAPE_PRIOR_PATH", "").strip()
                default_prior_fn = getattr(custom_heads, "_default_prior_path", None)
                default_prior = default_prior_fn() if callable(default_prior_fn) else ""
                final_prior = env_prior or default_prior
                print(f"[shape_prior] SHAPE_PRIOR_PATH={final_prior}")
                if final_prior:
                    os.environ["SHAPE_PRIOR_PATH"] = final_prior

        # 构建 base / prior predictor
        if mode in ("base", "both") or pc_from == "base":
            if not str(args.weights_base).strip():
                raise ValueError("mode=base/both 或 pc-from=base 时必须提供 --weights-base")
            base_weights = _require_existing_file(str(args.weights_base).strip(), "base weights")
            if model_family == "mask2former":
                base_pred = build_mask2former_predictor(
                    mask2former_root=str(args.mask2former_root),
                    config_file=base_config_file,
                    weights=base_weights,
                    num_classes=int(args.num_classes),
                    score_thresh=float(score_thr),
                    device=str(args.device),
                )
            else:
                base_pred = build_pointrend_predictor(
                    config_file=base_config_file,
                    weights=base_weights,
                    mask_head_name="PointRendMaskHead",
                    num_classes=int(args.num_classes),
                    score_thresh=float(score_thr),
                    device=str(args.device),
                )

        if mode in ("prior", "both") or pc_from == "prior":
            if not str(args.weights_prior).strip():
                raise ValueError("mode=prior/both 或 pc-from=prior 时必须提供 --weights-prior（或用兼容参数 --weights）")
            prior_weights = _require_existing_file(str(args.weights_prior).strip(), "prior weights")
            if model_family == "mask2former":
                prior_pred = build_mask2former_predictor(
                    mask2former_root=str(args.mask2former_root),
                    config_file=prior_config_file,
                    weights=prior_weights,
                    num_classes=int(args.num_classes),
                    score_thresh=float(score_thr),
                    device=str(args.device),
                    prior_path_override=prior_path_override,
                )
            else:
                prior_pred = build_pointrend_predictor(
                    config_file=prior_config_file,
                    weights=prior_weights,
                    mask_head_name="ShapeAwareCoarseMaskHead",
                    num_classes=int(args.num_classes),
                    score_thresh=float(score_thr),
                    device=str(args.device),
                )

        if base_pred is None and prior_pred is None:
            raise RuntimeError("未构建任何 predictor：请检查 --mode/--pc-from 与权重参数是否匹配。")

        # 可视化用的 metadata（避免 metadata=None 导致 draw_instance_predictions 取类名时报错）
        meta_name = f"__mecheye_live_{family_prefix}__"
        metadata = MetadataCatalog.get(meta_name)
        if not getattr(metadata, "thing_classes", None) or len(getattr(metadata, "thing_classes", [])) != int(args.num_classes):
            metadata.thing_classes = [f"cls{i}" for i in range(int(args.num_classes))]
    else:
        print("[capture-only] 跳过分割模型加载，仅进行相机采集。")

    def _add_titlebar(img_bgr: np.ndarray, title: str) -> np.ndarray:
        out_bgr = img_bgr.copy()
        cv2.rectangle(out_bgr, (0, 0), (min(520, out_bgr.shape[1] - 1), 32), (0, 0, 0), thickness=-1)
        cv2.putText(out_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out_bgr

    def _viz_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, title: str) -> np.ndarray:
        vis_bgr = _overlay_binary_mask(img_bgr, mask_u8, color_bgr=(80, 210, 160), alpha=0.35, outline=True)
        return _add_titlebar(vis_bgr, title)

    def _stack_horiz(imgs: list[np.ndarray]) -> np.ndarray:
        if not imgs:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        h = max(im.shape[0] for im in imgs)
        outs: list[np.ndarray] = []
        for im in imgs:
            if im.shape[0] == h:
                outs.append(im)
                continue
            pad = h - im.shape[0]
            outs.append(cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        return np.concatenate(outs, axis=1)

    camera = Camera()
    if args.discover:
        discover_and_print_cameras()
        return
    if not connect_camera(camera, ip=str(args.ip).strip(), serial=str(args.serial).strip(), index=int(args.index)):
        return

    save_dir = str(args.save_dir).strip()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 采集前设置 HDR + 点云后处理（优先从源头减少空洞）
    exp_seq = _parse_float_list(args.exposure_seq)
    _apply_mecheye_params(
        camera,
        exposure_sequence=exp_seq if exp_seq else None,
        pc_surface_smoothing=str(args.pc_smoothing),
        pc_noise_removal=str(args.pc_noise),
        pc_outlier_removal=str(args.pc_outlier),
        pc_edge_preservation=str(args.pc_edge),
        save_to_device=bool(args.save_userset),
    )

    # 内参只需获取一次
    intrinsics = CameraIntrinsics()
    show_error(camera.get_camera_intrinsics(intrinsics))

    # ClearGrasp：初始化一次（避免按 s 每次都重新加载权重）
    depthcomplete = None
    if bool(args.cleargrasp):
        if not str(args.cleargrasp_normals_weights).strip() or not str(args.cleargrasp_outlines_weights).strip() or not str(args.cleargrasp_depth2depth_exe).strip():
            print("[cleargrasp] 缺少参数：--cleargrasp-normals-weights / --cleargrasp-outlines-weights / --cleargrasp-depth2depth-exe")
        else:
            try:
                _add_cleargrasp_to_syspath()
                from api import depth_completion_api  # type: ignore

                # 取 depth 内参，并按 ClearGrasp 输出分辨率缩放
                fx, fy, cx, cy = _get_depth_k_from_mecheye_intrinsics(intrinsics)
                out_w, out_h = int(args.cleargrasp_out_w), int(args.cleargrasp_out_h)
                depthcomplete = {
                    "api": depth_completion_api,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "out_w": out_w,
                    "out_h": out_h,
                }
                print("[cleargrasp] 已启用（将在按 's' 时首次构建 depth completion 模型并运行）")
            except Exception as e:
                print(f"[cleargrasp] 初始化失败：{e}")
                depthcomplete = None

    frame_all = Frame2DAnd3D()
    frame_idx = 0
    last_vis: Optional[np.ndarray] = None
    last_color_bgr: Optional[np.ndarray] = None
    last_depth = None
    last_color_sdk = None
    last_depth_render: Optional[np.ndarray] = None

    print("开始实时预览：'s' 保存当前帧 (color/mask/ply)，'q' 退出。")
    try:
        while True:
            frame_idx += 1
            st = camera.capture_2d_and_3d(frame_all)
            if not st.is_ok():
                show_error(st)
                continue

            color_sdk = frame_all.frame_2d().get_color_image()
            depth = frame_all.frame_3d().get_depth_map()
            color_bgr = color_sdk.data()  # numpy BGR

            do_infer = (int(args.infer_every) <= 1) or (frame_idx % int(args.infer_every) == 0)
            if bool(args.capture_only):
                last_vis = color_bgr
                last_color_bgr = color_bgr
                last_depth = depth
                last_color_sdk = color_sdk
                if bool(args.show_depth) and not bool(args.no_gui):
                    depth_np = depth.data()
                    last_depth_render = _render_depth_data(depth_np)
            elif do_infer:
                panels: list[np.ndarray] = []

                if mode in ("base", "both") and base_pred is not None:
                    base_out = base_pred(color_bgr)
                    base_inst = base_out["instances"].to("cpu")
                    _base_mask_vis, base_mask_pc, _ = _build_output_masks(
                        base_inst,
                        mask_mode=str(args.mask_mode),
                        pc_mask_mode=str(args.pc_mask_mode),
                        pc_iou_thresh=float(args.pc_iou_thresh),
                        pc_join_dilate=int(args.pc_join_dilate),
                        mask_close=int(args.mask_close),
                        mask_dilate=int(args.mask_dilate),
                        mask_erode=int(args.mask_erode),
                        invert_mask=bool(args.invert_mask),
                        auto_invert_mask=bool(args.auto_invert_mask),
                    )
                    panels.append(_viz_mask(img_bgr=color_bgr, mask_u8=base_mask_pc, title=f"{base_title} | PC mask"))

                if mode in ("prior", "both") and prior_pred is not None:
                    prior_out = prior_pred(color_bgr)
                    prior_inst = prior_out["instances"].to("cpu")
                    _prior_mask_vis, prior_mask_pc, _ = _build_output_masks(
                        prior_inst,
                        mask_mode=str(args.mask_mode),
                        pc_mask_mode=str(args.pc_mask_mode),
                        pc_iou_thresh=float(args.pc_iou_thresh),
                        pc_join_dilate=int(args.pc_join_dilate),
                        mask_close=int(args.mask_close),
                        mask_dilate=int(args.mask_dilate),
                        mask_erode=int(args.mask_erode),
                        invert_mask=bool(args.invert_mask),
                        auto_invert_mask=bool(args.auto_invert_mask),
                    )
                    panels.append(_viz_mask(img_bgr=color_bgr, mask_u8=prior_mask_pc, title=f"{prior_title} | PC mask"))

                if not panels:
                    last_vis = color_bgr
                elif len(panels) == 1:
                    last_vis = panels[0]
                else:
                    raw_panel = _add_titlebar(color_bgr, "RAW")
                    last_vis = _stack_horiz([raw_panel] + panels)

                # 保存最近一帧的 raw 数据引用（用于按 's' 同帧导出）
                last_color_bgr = color_bgr
                last_depth = depth
                last_color_sdk = color_sdk

                # Depth 可视化：raw depth
                if bool(args.show_depth) and not bool(args.no_gui):
                    depth_np = depth.data()
                    last_depth_render = _render_depth_data(depth_np)
            else:
                # 不推理时直接显示原图
                last_vis = color_bgr
                last_color_bgr = color_bgr
                last_depth = depth
                last_color_sdk = color_sdk

            if not bool(args.no_gui):
                cv2.imshow(str(args.win), last_vis if last_vis is not None else color_bgr)
                if bool(args.show_depth):
                    if last_depth_render is not None and last_depth_render.size > 0:
                        cv2.imshow(str(args.win) + " | depth_raw_render", last_depth_render)
                key = cv2.waitKey(int(args.wait)) & 0xFF
            else:
                key = 255

            if key == ord("q"):
                break

            if int(args.max_captures) > 0 and frame_idx >= int(args.max_captures):
                print(f"[capture] reached max captures={int(args.max_captures)}, auto stop.")
                break

            if key == ord("s"):
                if not save_dir:
                    print("未设置 --save-dir，无法保存。")
                    continue
                # 关键：按 's' 时强制对“当前帧”做一次推理，保证 mask 与 depth 同帧对应
                base_inst = None
                prior_inst = None
                if base_pred is not None:
                    base_inst = base_pred(color_bgr)["instances"].to("cpu")
                if prior_pred is not None:
                    prior_inst = prior_pred(color_bgr)["instances"].to("cpu")

                inst_for_pc = prior_inst if pc_from == "prior" else base_inst
                if inst_for_pc is None:
                    raise RuntimeError(f"pc-from={pc_from} 但对应 predictor/instances 不存在，请检查 --mode/--pc-from/权重参数。")

                # mask 用途分离：
                # - mask_u8_vis：用于保存/调试（受 --mask-mode 控制）
                # - mask_u8_pc：用于点云导出与 cleargrasp 点云过滤（受 --pc-mask-mode 控制）
                mask_u8_vis, mask_u8_pc, invert_applied = _build_output_masks(
                    inst_for_pc,
                    mask_mode=str(args.mask_mode),
                    pc_mask_mode=str(args.pc_mask_mode),
                    pc_iou_thresh=float(args.pc_iou_thresh),
                    pc_join_dilate=int(args.pc_join_dilate),
                    mask_close=int(args.mask_close),
                    mask_dilate=int(args.mask_dilate),
                    mask_erode=int(args.mask_erode),
                    invert_mask=bool(args.invert_mask),
                    auto_invert_mask=bool(args.auto_invert_mask),
                )
                if bool(args.auto_invert_mask):
                    r0 = _mask_fg_ratio(mask_u8_pc if not invert_applied else 255 - mask_u8_pc)
                    r1 = _mask_fg_ratio(255 - (mask_u8_pc if not invert_applied else 255 - mask_u8_pc))
                    print(f"[mask][auto-invert] ratio_raw={r0:.3f} ratio_inv={r1:.3f} use={'invert' if invert_applied else 'raw'}")

                print("[mask][vis] " + _mask_stats(mask_u8_vis))
                print("[mask][pc ] " + _mask_stats(mask_u8_pc))

                # 保存可视化：直接显示最终用于点云的 mask，避免和点云过滤逻辑不一致
                vis_bgr = _viz_mask(img_bgr=color_bgr, mask_u8=mask_u8_pc, title=f"PC_FROM={pc_from.upper()} | PC mask")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                color_path = Path(save_dir) / f"mecheye_color_{ts}.png"
                mask_path = Path(save_dir) / f"{family_prefix}_{pc_from}_mask_{ts}.png"
                vis_path = Path(save_dir) / f"{family_prefix}_{pc_from}_vis_{ts}.png"
                ply_path = Path(save_dir) / f"{pc_from}_mask_pointcloud_{ts}.ply"
                raw_ply_path = Path(save_dir) / f"raw_pointcloud_{ts}.ply"
                depth_raw_path = Path(save_dir) / f"mecheye_depth_raw_{ts}.png"
                depth_raw_render_path = Path(save_dir) / f"mecheye_depth_raw_render_{ts}.png"
                intr_path = Path(save_dir) / f"intrinsics_{ts}.json"
                depth_completed_path = Path(save_dir) / f"mecheye_depth_completed_{ts}.png"
                ply_completed_path = Path(save_dir) / f"mecheye_pointcloud_completed_{ts}.ply"

                cv2.imwrite(str(color_path), color_bgr)
                # 保存用于“点云/cleargrasp”的 mask（更符合你要的“只包含插头点云”）
                cv2.imwrite(str(mask_path), mask_u8_pc)
                cv2.imwrite(str(vis_path), vis_bgr)

                # 如果同时跑了 base/prior，也把另一个的可视化/二值 mask 一并保存，便于离线对照
                try:
                    if base_inst is not None and pc_from != "base":
                        _m_base_vis, m_base, _base_invert = _build_output_masks(
                            base_inst,
                            mask_mode=str(args.mask_mode),
                            pc_mask_mode=str(args.pc_mask_mode),
                            pc_iou_thresh=float(args.pc_iou_thresh),
                            pc_join_dilate=int(args.pc_join_dilate),
                            mask_close=int(args.mask_close),
                            mask_dilate=int(args.mask_dilate),
                            mask_erode=int(args.mask_erode),
                            invert_mask=bool(args.invert_mask),
                            auto_invert_mask=bool(args.auto_invert_mask),
                        )
                        cv2.imwrite(str(Path(save_dir) / f"{family_prefix}_base_mask_{ts}.png"), m_base)
                        cv2.imwrite(
                            str(Path(save_dir) / f"{family_prefix}_base_vis_{ts}.png"),
                            _viz_mask(color_bgr, m_base, f"{base_title} | PC mask"),
                        )
                    if prior_inst is not None and pc_from != "prior":
                        _m_prior_vis, m_prior, _prior_invert = _build_output_masks(
                            prior_inst,
                            mask_mode=str(args.mask_mode),
                            pc_mask_mode=str(args.pc_mask_mode),
                            pc_iou_thresh=float(args.pc_iou_thresh),
                            pc_join_dilate=int(args.pc_join_dilate),
                            mask_close=int(args.mask_close),
                            mask_dilate=int(args.mask_dilate),
                            mask_erode=int(args.mask_erode),
                            invert_mask=bool(args.invert_mask),
                            auto_invert_mask=bool(args.auto_invert_mask),
                        )
                        cv2.imwrite(str(Path(save_dir) / f"{family_prefix}_prior_mask_{ts}.png"), m_prior)
                        cv2.imwrite(
                            str(Path(save_dir) / f"{family_prefix}_prior_vis_{ts}.png"),
                            _viz_mask(color_bgr, m_prior, f"{prior_title} | PC mask"),
                        )
                except Exception as _e:
                    # 不影响主流程：仅用于额外对照输出
                    pass

                # 按 's'：默认只保存 raw depth（16-bit PNG）
                depth_np = depth.data()
                cv2.imwrite(str(depth_raw_path), _depth_to_png_u16(depth_np))
                if bool(args.save_depth):
                    rendered = _render_depth_data(depth_np)
                    if rendered.size > 0:
                        cv2.imwrite(str(depth_raw_render_path), rendered)

                # intrinsics：如果用户开启了 --save-intrinsics 或 --cleargrasp，则保存一份 json（ClearGrasp 需要）
                if bool(args.save_intrinsics) or bool(args.cleargrasp):
                    intr_dict = _intrinsics_to_dict(intrinsics)
                    intr_path.write_text(json.dumps(intr_dict, indent=2), encoding="utf-8")
                    print(f"[intr] saved: {intr_path}")

                # ClearGrasp depth completion（同一 python 环境直接调用）
                if bool(args.cleargrasp):
                    if depthcomplete is None:
                        print("[cleargrasp] 未初始化，跳过。")
                    else:
                        try:
                            # 仅首次按 's' 时构建模型（加载权重，耗时较长）
                            if "model" not in depthcomplete:
                                H, W = depth_np.shape[:2]
                                fx0, fy0, cx0, cy0 = (
                                    float(depthcomplete["fx"]),
                                    float(depthcomplete["fy"]),
                                    float(depthcomplete["cx"]),
                                    float(depthcomplete["cy"]),
                                )
                                out_w, out_h = int(depthcomplete["out_w"]), int(depthcomplete["out_h"])
                                fx2, fy2, cx2, cy2 = _scale_k_for_resize(
                                    fx0, fy0, cx0, cy0, in_w=W, in_h=H, out_w=out_w, out_h=out_h
                                )

                                api = depthcomplete["api"]
                                depthcomplete["model"] = api.DepthToDepthCompletion(
                                    normalsWeightsFile=str(args.cleargrasp_normals_weights),
                                    outlinesWeightsFile=str(args.cleargrasp_outlines_weights),
                                    masksWeightsFile="",
                                    normalsModel="drn",
                                    outlinesModel="drn",
                                    depth2depthExecutable=str(args.cleargrasp_depth2depth_exe),
                                    outputImgHeight=out_h,
                                    outputImgWidth=out_w,
                                    fx=fx2,
                                    fy=fy2,
                                    cx=cx2,
                                    cy=cy2,
                                    filter_d=int(args.cleargrasp_filter_d),
                                    filter_sigmaColor=float(args.cleargrasp_filter_sigma_color),
                                    filter_sigmaSpace=float(args.cleargrasp_filter_sigma_space),
                                    normalsInferenceHeight=out_h,
                                    normalsInferenceWidth=out_w,
                                    outlinesInferenceHeight=out_h,
                                    outlinesInferenceWidth=out_w,
                                    min_depth=0.0,
                                    max_depth=3.0,
                                )
                                print("[cleargrasp] model loaded.")

                            # depth: uint16 -> meters float32
                            depth_m = _depth_u16_to_m(_depth_to_png_u16(depth_np), unit=str(args.cleargrasp_depth_unit))  # type: ignore[arg-type]
                            rgb_rgb = color_bgr[:, :, ::-1]  # BGR->RGB

                            out_depth_m_small, _ = depthcomplete["model"].depth_completion(
                                rgb_rgb,
                                depth_m,
                                inertia_weight=float(args.cleargrasp_inertia),
                                smoothness_weight=float(args.cleargrasp_smoothness),
                                tangent_weight=float(args.cleargrasp_tangent),
                                mode_modify_input_depth="",
                            )

                            # 输出回原分辨率，仅填补原始空洞（保持观测深度）
                            out_depth_m_up = cv2.resize(
                                out_depth_m_small, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST
                            ).astype(np.float32, copy=False)
                            completed_m = depth_m.copy()
                            # 连接处缺点常见原因：深度并非严格为 0，而是极小/异常值；可用 fill-thresh 把它们也当洞来填
                            thr = float(args.cleargrasp_fill_thresh)
                            holes = (~np.isfinite(completed_m)) | (completed_m <= max(0.0, thr))
                            completed_m[holes] = out_depth_m_up[holes]

                            # 保存 completed depth（与 raw depth 同单位的 uint16 PNG）
                            cv2.imwrite(
                                str(depth_completed_path), _depth_m_to_u16(completed_m, unit=str(args.cleargrasp_depth_unit))
                            )  # type: ignore[arg-type]

                            # 只输出插头区域点云：使用 mask_u8_pc（已受 --invert-mask/--auto-invert-mask 影响）
                            fx, fy, cx, cy = _get_depth_k_from_mecheye_intrinsics(intrinsics)
                            xyz, rgb = _backproject_masked_xyzrgb(
                                completed_m,
                                color_bgr,
                                mask_u8_pc,
                                fx=fx,
                                fy=fy,
                                cx=cx,
                                cy=cy,
                                stride=int(args.cleargrasp_ply_stride),
                            )
                            _write_ply_xyzrgb_ascii(ply_completed_path, xyz, rgb)
                            print(
                                f"[cleargrasp] saved: {depth_completed_path.name}, {ply_completed_path.name} (points={int(xyz.shape[0])})"
                            )
                        except Exception as e:
                            print(f"[cleargrasp] 失败：{e}")

                # 同帧生成点云（SDK mapping）
                sdk_mask = mask_u8_to_sdk_grayscale2d(mask_u8_pc)
                points_xyz_bgr = TexturedPointCloud()
                show_error(get_point_cloud_after_mapping(depth, sdk_mask, color_sdk, intrinsics, points_xyz_bgr))
                show_error(Frame2DAnd3D.save_point_cloud(points_xyz_bgr, FileFormat_PLY, str(ply_path)), f"保存点云到: {ply_path}")

                if bool(args.save_raw_ply):
                    # 对照：保存同一帧的“原始点云”（未加 mask）
                    show_error(frame_all.save_textured_point_cloud(FileFormat_PLY, str(raw_ply_path)), f"保存原始点云到: {raw_ply_path}")

                extra = ""
                if bool(args.cleargrasp):
                    extra = f", {depth_completed_path.name}, {ply_completed_path.name}"
                print(f"✓ 保存: {color_path.name}, {mask_path.name}, {depth_raw_path.name}, {ply_path.name}{extra}")

    finally:
        camera.disconnect()
        if not bool(args.no_gui):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    main()


