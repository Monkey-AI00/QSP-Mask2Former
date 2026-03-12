#!/usr/bin/env python3
"""
重投影验证：检验 3D 点云与 2D mask 的一致性

输入：
- --pcd: 点云 .ply（Mech-Eye 通常单位 mm）
- --mask: 2D mask PNG（前景=255，背景=0）
- --intrinsics: 保存的 intrinsics_*.json（由 mecheye_live_pointrend_pointcloud.py --save-intrinsics 生成）
- --image: (可选) 原始 color 图，用于输出叠加可视化

输出：
- 统计：投影到图像范围内的点中，有多少落在 mask 前景里
- 可选输出 overlay 图：mask 内点=绿，mask 外点=红

说明：
这里只做 pinhole 投影（u=fx*x/z+cx, v=fy*y/z+cy），忽略畸变。
在大多数验证场景足够定位 3D-2D 是否对齐/是否反了。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_intrinsics_k(path: str, camera: str) -> tuple[float, float, float, float]:
    """
    读取 intrinsics_*.json，兼容只包含 depth 的情况。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))

    # 优先按用户指定 camera；如果缺失则回退
    if camera not in d:
        # 常见：只保存了 depth
        if "depth" in d:
            camera = "depth"
        elif "color" in d:
            camera = "color"
        else:
            raise KeyError(f"intrinsics json 不包含 'color' 或 'depth'。可用 keys: {list(d.keys())}")

    cm = d[camera]["camera_matrix"]
    return float(cm["fx"]), float(cm["fy"]), float(cm["cx"]), float(cm["cy"])


def _project(points: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    points: Nx3 (same unit as fx/fy/cx/cy assume pixel-based intrinsics; points in camera coordinates)
    returns: u,v,valid_mask (z>0)
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    valid = z > 1e-6
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return u, v, valid


def main():
    ap = argparse.ArgumentParser(description="3D->2D 重投影验证（点云 vs mask）")
    ap.add_argument("--pcd", required=True, help="点云 .ply 路径")
    ap.add_argument("--mask", required=True, help="二值 mask PNG（前景=255）")
    ap.add_argument("--intrinsics", required=True, help="intrinsics_*.json 路径")
    ap.add_argument("--camera", choices=["color", "depth"], default="color", help="使用哪一路内参做投影（mask 来自 color 时选 color）")
    ap.add_argument("--unit", choices=["mm", "m"], default="mm", help="点云单位（Mech-Eye 通常为 mm）")

    ap.add_argument("--image", default="", help="可选：原始 color 图，用于生成 overlay")
    ap.add_argument("--output", default="", help="可选：overlay 输出路径；默认 <mask_stem>_reproj.png")
    ap.add_argument("--max-points", type=int, default=200000, help="最多抽样多少点用于绘图/统计（避免太慢）")
    ap.add_argument("--point-size", type=int, default=1, help="overlay 点大小（像素）")
    ap.add_argument("--invert-mask", action="store_true", help="把 mask 反转后再做一致性统计/绘图（前景<->背景）")
    ap.add_argument("--auto-invert", action="store_true", help="同时评估原 mask 与反转 mask，自动选择 ratio 更高的那一个绘图")
    args = ap.parse_args()

    import open3d as o3d
    import cv2

    pcd_path = Path(args.pcd)
    mask_path = Path(args.mask)
    intr_path = Path(args.intrinsics)
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    if not intr_path.exists():
        raise FileNotFoundError(intr_path)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 mask: {mask_path}")
    H, W = mask.shape[:2]

    fx, fy, cx, cy = _load_intrinsics_k(str(intr_path), str(args.camera))

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise ValueError("点云为空")

    pts = np.asarray(pcd.points).astype(np.float64)
    # 单位：mm -> m（投影用的是比例 x/z，不影响；但保留一致性）
    if str(args.unit) == "mm":
        pts = pts * 0.001

    # 抽样（避免 100w 点太慢）
    N = pts.shape[0]
    maxn = int(args.max_points)
    if maxn > 0 and N > maxn:
        idx = np.random.default_rng(0).choice(N, size=maxn, replace=False)
        pts_s = pts[idx]
    else:
        pts_s = pts

    u, v, valid = _project(pts_s, fx, fy, cx, cy)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_img = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui2 = ui[in_img]
    vi2 = vi[in_img]
    in_mask_raw = mask[vi2, ui2] > 0
    in_mask_inv = (mask[vi2, ui2] == 0)

    denom = int(ui2.size)
    num_raw = int(np.count_nonzero(in_mask_raw))
    ratio_raw = float(num_raw) / float(denom) if denom > 0 else 0.0
    num_inv = int(np.count_nonzero(in_mask_inv))
    ratio_inv = float(num_inv) / float(denom) if denom > 0 else 0.0

    # 选择使用哪种 mask 语义绘图/统计
    use_inv = bool(args.invert_mask)
    if bool(args.auto_invert):
        use_inv = ratio_inv > ratio_raw

    in_mask = in_mask_inv if use_inv else in_mask_raw
    num = int(np.count_nonzero(in_mask))
    ratio = float(num) / float(denom) if denom > 0 else 0.0

    print(
        f"[reproj] in_image_points={denom} "
        f"raw: in_mask_points={num_raw} ratio={ratio_raw:.3f} | "
        f"invert: in_mask_points={num_inv} ratio={ratio_inv:.3f} | "
        f"use={'invert' if use_inv else 'raw'}"
    )

    # 可视化 overlay
    if str(args.image).strip():
        img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取 image: {args.image}")
        if img.shape[0] != H or img.shape[1] != W:
            raise ValueError(f"image 与 mask 尺寸不一致：image={img.shape[:2]} mask={(H,W)}")
        canvas = img.copy()
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    ps = max(1, int(args.point_size))
    # mask 外：红；mask 内：绿
    for x, y, ok in zip(ui2.tolist(), vi2.tolist(), in_mask.tolist()):
        c = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(canvas, (int(x), int(y)), ps, c, -1)

    out = str(args.output).strip()
    if not out:
        suffix = "_reproj_inv.png" if use_inv else "_reproj.png"
        out = str(mask_path.with_name(mask_path.stem + suffix))
    cv2.imwrite(out, canvas)
    print(f"[out] saved overlay: {out}")


if __name__ == "__main__":
    main()


