#!/usr/bin/env python3
"""
把你当前项目保存出来的 RGB + raw depth 接入 ClearGrasp（depth completion），并输出：
1) depth_completed（16-bit PNG，单位为 4000 * depth(m)，与 ClearGrasp/depth2depth 约定一致）
2) point cloud（PLY，可选）

设计目标：先跑通“离线一帧”的最短集成链路，再考虑集成到实时/按 s 的流程里。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import cv2
import numpy as np


def _add_cleargrasp_to_syspath() -> None:
    # 本项目内 cleargrasp 位于 detectron2/cleargrasp
    # 让我们可以 `from api import depth_completion_api`
    cg_root = os.path.join(os.path.dirname(__file__), "../../cleargrasp")
    cg_root = os.path.abspath(cg_root)
    if cg_root not in sys.path:
        sys.path.insert(0, cg_root)


def _load_intrinsics_k(intrinsics_json: str, *, camera: Literal["depth", "color"] = "depth") -> Tuple[float, float, float, float]:
    d = json.loads(Path(intrinsics_json).read_text(encoding="utf-8"))
    if camera not in d:
        raise KeyError(f"intrinsics json 不包含 '{camera}'。可用 keys: {list(d.keys())}")
    cm = d[camera]["camera_matrix"]
    return float(cm["fx"]), float(cm["fy"]), float(cm["cx"]), float(cm["cy"])


def _scale_intrinsics_for_resize(
    fx: float, fy: float, cx: float, cy: float, *, in_w: int, in_h: int, out_w: int, out_h: int
) -> Tuple[float, float, float, float]:
    sx = float(out_w) / float(in_w)
    sy = float(out_h) / float(in_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def _depth_u16_to_m(depth_u16: np.ndarray, *, unit: Literal["mm", "m"] = "mm") -> np.ndarray:
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"depth 必须是 uint16 PNG（cv2.IMREAD_UNCHANGED）。实际 dtype={depth_u16.dtype}")
    d = depth_u16.astype(np.float32)
    if unit == "mm":
        return d / 1000.0
    return d


def _depth_m_to_scaled_png_u16(depth_m: np.ndarray) -> np.ndarray:
    # 与 ClearGrasp utils.scale_depth 一致：uint16 = clip(depth_m, 0..floor(65535/4000)) * 4000
    if depth_m.dtype != np.float32:
        depth_m = depth_m.astype(np.float32)
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_m = np.clip(depth_m, 0.0, np.floor(65535.0 / 4000.0))
    return (depth_m * 4000.0).astype(np.uint16)


def _backproject_to_xyzrgb(
    depth_m: np.ndarray,
    rgb_bgr: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mask_u8: Optional[np.ndarray] = None,
    keep: Literal["fg", "bg"] = "fg",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        stride = 1
    d = depth_m.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)

    if mask_u8 is not None and mask_u8.size > 0:
        m = mask_u8.astype(np.uint8, copy=False)
        if m.shape[:2] != d.shape[:2]:
            m = cv2.resize(m, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
        if str(keep) == "fg":
            valid = valid & (m > 0)
        else:
            valid = valid & (m == 0)

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
    bgr = rgb_bgr[ys, xs].astype(np.uint8, copy=False)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="ClearGrasp depth completion for saved RGB+Depth, then optional PLY export")
    ap.add_argument("--rgb", required=True, help="RGB 图路径（你的 mecheye_color_*.png）")
    ap.add_argument("--depth", required=True, help="raw depth PNG（你的 mecheye_depth_raw_*.png，16-bit）")
    ap.add_argument("--intrinsics", required=True, help="intrinsics_*.json（mecheye_live_pointrend_pointcloud.py --save-intrinsics）")
    ap.add_argument("--intrinsics-camera", choices=["depth", "color"], default="depth", help="用哪一路内参（推荐 depth）")
    ap.add_argument("--depth-unit", choices=["mm", "m"], default="mm", help="raw depth PNG 的单位（默认 mm）")

    ap.add_argument("--normals-weights", required=True, help="ClearGrasp surface_normals checkpoint_normals.pth")
    ap.add_argument("--outlines-weights", required=True, help="ClearGrasp outlines checkpoint_outlines.pth")
    ap.add_argument("--depth2depth-exe", required=True, help="depth2depth 可执行文件路径")

    ap.add_argument("--out-w", type=int, default=256, help="ClearGrasp/depth2depth 输出宽（越小越快）")
    ap.add_argument("--out-h", type=int, default=144, help="ClearGrasp/depth2depth 输出高（越小越快）")
    ap.add_argument("--inertia", type=float, default=1000.0)
    ap.add_argument("--smoothness", type=float, default=0.0001)
    ap.add_argument("--tangent", type=float, default=1.0)

    ap.add_argument("--out-depth", default="", help="输出 completed depth 的 PNG 路径（默认：<depth>_completed.png）")
    ap.add_argument("--out-ply", default="", help="输出点云 PLY 路径（可选）")
    ap.add_argument("--ply-stride", type=int, default=1, help="点云像素步长降采样")
    ap.add_argument("--mask", default="", help="可选：二值 mask PNG（白/255=插头）。提供后输出点云将按 mask 过滤，仅保留指定区域。")
    ap.add_argument("--invert-mask", action="store_true", help="反转 mask 语义：保留 mask==0 的区域（当插头在 mask 里是黑色时用）")
    args = ap.parse_args()

    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)
    intr_path = Path(args.intrinsics)
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)
    if not intr_path.exists():
        raise FileNotFoundError(intr_path)

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"无法读取 rgb: {rgb_path}")
    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise ValueError(f"无法读取 depth: {depth_path}")
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"depth 必须是 16-bit PNG。实际 dtype={depth_u16.dtype}")
    if depth_u16.shape[:2] != rgb_bgr.shape[:2]:
        raise ValueError(f"rgb/depth 尺寸不一致：rgb={rgb_bgr.shape[:2]} depth={depth_u16.shape[:2]}")

    mask_u8: Optional[np.ndarray] = None
    if str(args.mask).strip():
        mp = Path(str(args.mask).strip())
        if not mp.exists():
            raise FileNotFoundError(mp)
        mask_u8 = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise ValueError(f"无法读取 mask: {mp}")
        if mask_u8.shape[:2] != depth_u16.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (depth_u16.shape[1], depth_u16.shape[0]), interpolation=cv2.INTER_NEAREST)

    H, W = depth_u16.shape[:2]
    fx, fy, cx, cy = _load_intrinsics_k(str(intr_path), camera=str(args.intrinsics_camera))

    # 将 raw depth 转为 meters float32（ClearGrasp 约定）
    depth_m = _depth_u16_to_m(depth_u16, unit=str(args.depth_unit))

    # ClearGrasp 内部会把输入 resize 到 out_w/out_h，所以需要把内参按 resize 同比例缩放
    out_w, out_h = int(args.out_w), int(args.out_h)
    fx2, fy2, cx2, cy2 = _scale_intrinsics_for_resize(fx, fy, cx, cy, in_w=W, in_h=H, out_w=out_w, out_h=out_h)

    _add_cleargrasp_to_syspath()
    from api import depth_completion_api  # type: ignore

    depthcomplete = depth_completion_api.DepthToDepthCompletion(
        normalsWeightsFile=str(args.normals_weights),
        outlinesWeightsFile=str(args.outlines_weights),
        masksWeightsFile="",
        normalsModel="drn",
        outlinesModel="drn",
        depth2depthExecutable=str(args.depth2depth_exe),
        outputImgHeight=out_h,
        outputImgWidth=out_w,
        fx=fx2,
        fy=fy2,
        cx=cx2,
        cy=cy2,
        # 关闭双边滤波，先拿最直接的输出；需要的话再开
        filter_d=0,
        filter_sigmaColor=0,
        filter_sigmaSpace=0,
        normalsInferenceHeight=out_h,
        normalsInferenceWidth=out_w,
        outlinesInferenceHeight=out_h,
        outlinesInferenceWidth=out_w,
        min_depth=0.0,
        max_depth=3.0,
    )

    # 输入 RGB 要给 uint8 RGB（ClearGrasp 用 ToTensor，不强制，但保持一致）
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    out_depth_m, _ = depthcomplete.depth_completion(
        rgb_rgb,
        depth_m,
        inertia_weight=float(args.inertia),
        smoothness_weight=float(args.smoothness),
        tangent_weight=float(args.tangent),
        mode_modify_input_depth="",
    )

    # 输出回原分辨率（只填补原始空洞，尽量不改动已有观测深度）
    out_depth_m_up = cv2.resize(out_depth_m, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    completed_m = depth_m.copy()
    holes = completed_m <= 0
    completed_m[holes] = out_depth_m_up[holes]

    out_depth_path = str(args.out_depth).strip()
    if not out_depth_path:
        out_depth_path = str(depth_path.with_name(depth_path.stem + "_completed.png"))
    out_depth_path_p = Path(out_depth_path)
    cv2.imwrite(str(out_depth_path_p), _depth_m_to_scaled_png_u16(completed_m))
    print(f"[out] depth_completed (16-bit, 4000*meters): {out_depth_path_p}")

    out_ply = str(args.out_ply).strip()
    if out_ply:
        keep = "bg" if bool(args.invert_mask) else "fg"
        xyz, rgb = _backproject_to_xyzrgb(
            completed_m,
            rgb_bgr,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            mask_u8=mask_u8,
            keep=keep,  # type: ignore[arg-type]
            stride=int(args.ply_stride),
        )
        out_ply_p = Path(out_ply)
        _write_ply_xyzrgb_ascii(out_ply_p, xyz, rgb)
        print(f"[out] ply: {out_ply_p} (points={int(xyz.shape[0])})")


if __name__ == "__main__":
    main()


