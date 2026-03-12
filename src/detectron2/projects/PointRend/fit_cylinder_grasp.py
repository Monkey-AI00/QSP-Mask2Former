#!/usr/bin/env python3
"""
圆柱 RANSAC 拟合（用于插头抓取姿态稳定化）

背景：
你的点云表面有空洞（白色螺母缺失、红色壳体斑驳），这类缺失很难“补满”且会导致姿态抖动。
更稳的路线是：对插头头部近似为圆柱体，使用 RANSAC 在有空洞情况下拟合圆柱轴线与半径，
从而得到稳定的抓取位姿（Axis + Radius）。

输入：
- 一个已经尽量裁剪到插头主体的点云 .ply（建议先用 postprocess_pointcloud.py 去噪/聚类）
  注意：你仓库里的 Mech-Eye 点云通常单位为 mm（ply 头部有注释）。

输出：
- 终端打印：轴线方向、轴线上一点、半径、内点数量等
- 可选保存 json：轴线与抓取位姿（单位米）
- 可选可视化：点云 + 轴线（需要 GUI）

实现说明：
Open3D 主线版本没有稳定的 `segment_cylinder` API（部分版本/分支可能有）。
因此这里实现一个“基于法向的圆柱 RANSAC”：
- 先估计点法向
- RANSAC 随机采样若干点，用它们的法向统计得到候选轴向（轴向应与法向近似垂直）
- 在候选轴向下，将点投影到垂直于轴的平面，做 2D 圆拟合得到轴线位置与半径
- 通过距离阈值筛内点，选最优模型

适用：圆柱表面缺失 30~60% 仍可稳拟合（只要剩余点覆盖一定角度范围）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class CylinderModel:
    axis_point_m: np.ndarray  # (3,) point on axis (meters)
    axis_dir: np.ndarray  # (3,) unit vector
    radius_m: float
    inlier_idx: np.ndarray  # (K,)
    rms_m: float


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def _fit_circle_2d_taubin(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    2D 圆拟合（Taubin/代数拟合的一种简化实现）。
    返回：(center(2,), radius)
    """
    x = xy[:, 0]
    y = xy[:, 1]
    x_m = x.mean()
    y_m = y.mean()
    u = x - x_m
    v = y - y_m

    Suu = np.sum(u * u)
    Suv = np.sum(u * v)
    Svv = np.sum(v * v)
    Suuu = np.sum(u * u * u)
    Svvv = np.sum(v * v * v)
    Suvv = np.sum(u * v * v)
    Svuu = np.sum(v * u * u)

    A = np.array([[Suu, Suv], [Suv, Svv]], dtype=np.float64)
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu], dtype=np.float64)

    # 退化情况：点接近共线
    det = float(np.linalg.det(A))
    if abs(det) < 1e-12:
        c = np.array([x_m, y_m], dtype=np.float64)
        r = float(np.sqrt(np.mean((x - x_m) ** 2 + (y - y_m) ** 2)))
        return c, r

    uc, vc = np.linalg.solve(A, b)
    cx = x_m + uc
    cy = y_m + vc
    r = float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))
    return np.array([cx, cy], dtype=np.float64), r


def _axis_basis(axis_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定轴向 w（单位向量），构造正交基 (u,v) 使得 u,v ⟂ w 且 u ⟂ v。
    """
    w = _unit(axis_dir)
    # 选一个不共线的参考向量
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, w))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(np.cross(w, a))
    v = _unit(np.cross(w, u))
    return u, v


def _project_to_plane(points: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """
    投影到垂直于 axis_dir 的平面坐标系 (u,v) 上，返回 Nx2。
    """
    u, v = _axis_basis(axis_dir)
    d = points - axis_point.reshape(1, 3)
    return np.stack([d @ u, d @ v], axis=1)


def ransac_fit_cylinder(
    points_m: np.ndarray,
    normals: np.ndarray,
    *,
    radius_min_m: float,
    radius_max_m: float,
    distance_thresh_m: float,
    iters: int = 2000,
    min_inliers: int = 500,
    seed: int = 0,
) -> Optional[CylinderModel]:
    """
    基于法向的圆柱 RANSAC。
    """
    rng = np.random.default_rng(int(seed))
    N = points_m.shape[0]
    if N < max(200, min_inliers):
        return None

    best: Optional[CylinderModel] = None

    # 预过滤：法向要有效
    nn = np.linalg.norm(normals, axis=1)
    good = nn > 1e-6
    P = points_m[good]
    Nn = normals[good] / nn[good][:, None]
    if P.shape[0] < max(200, min_inliers):
        return None

    M = P.shape[0]

    for _ in range(int(iters)):
        # 采样若干点的法向，轴向应与法向近似垂直，因此轴向可以取法向的“最小主方向”
        k = 50 if M >= 50 else M
        idx = rng.choice(M, size=k, replace=False)
        ns = Nn[idx]  # (k,3)

        # 轴向 w：使得 sum((n·w)^2) 最小 => w 为 ns 协方差的最小特征向量
        C = ns.T @ ns
        w_vals, w_vecs = np.linalg.eigh(C)
        axis_dir = _unit(w_vecs[:, int(np.argmin(w_vals))])
        if float(np.linalg.norm(axis_dir)) < 1e-6:
            continue

        # 轴线点：暂用点云质心在该轴上的投影（可稳定）
        centroid = P.mean(axis=0)
        # axis_point = centroid 本身就可以，只影响平面投影的原点
        axis_point = centroid

        xy = _project_to_plane(P, axis_point, axis_dir)
        c2, r = _fit_circle_2d_taubin(xy)
        if not (radius_min_m <= r <= radius_max_m):
            continue

        # 由 2D 圆心恢复 3D 轴线点（在轴的垂直平面内的偏移）
        u, v = _axis_basis(axis_dir)
        axis_point2 = axis_point + c2[0] * u + c2[1] * v

        # 计算点到圆柱面的距离：到轴线距离 - r
        d = P - axis_point2.reshape(1, 3)
        # 到轴线的垂直距离
        cross = np.cross(d, axis_dir.reshape(1, 3))
        dist_axis = np.linalg.norm(cross, axis=1)
        resid = np.abs(dist_axis - float(r))
        inliers = np.where(resid < float(distance_thresh_m))[0]

        if inliers.size < int(min_inliers):
            continue

        rms = float(np.sqrt(np.mean(resid[inliers] ** 2)))
        if best is None or inliers.size > best.inlier_idx.size or (inliers.size == best.inlier_idx.size and rms < best.rms_m):
            best = CylinderModel(
                axis_point_m=axis_point2.astype(np.float64),
                axis_dir=axis_dir.astype(np.float64),
                radius_m=float(r),
                inlier_idx=inliers.astype(np.int64),
                rms_m=rms,
            )

    return best


def main():
    ap = argparse.ArgumentParser(description="RANSAC 拟合圆柱（插头抓取姿态）")
    ap.add_argument("--input", required=True, help="输入点云 .ply（建议已去噪/裁剪到插头主体）")
    ap.add_argument("--unit", choices=["mm", "m"], default="mm", help="输入点云单位（Mech-Eye 通常为 mm）")
    ap.add_argument("--voxel", type=float, default=1.0, help="下采样体素（单位同输入；mm 建议 0.5~2）")
    ap.add_argument("--normal-radius", type=float, default=10.0, help="法向估计搜索半径（单位同输入；mm 建议 5~20）")

    ap.add_argument("--radius-min", type=float, default=20.0, help="圆柱半径下界（单位同输入；例如 20mm）")
    ap.add_argument("--radius-max", type=float, default=40.0, help="圆柱半径上界（单位同输入；例如 40mm）")
    ap.add_argument("--dist-thresh", type=float, default=5.0, help="内点距离阈值（单位同输入；mm 建议 2~8）")
    ap.add_argument("--iters", type=int, default=2000, help="RANSAC 迭代次数")
    ap.add_argument("--min-inliers", type=int, default=1000, help="最少内点数（太小容易拟合到噪声）")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--output",
        default="",
        help="输出拟合后的点云（圆柱内点）.ply；不填则默认生成 <input_stem>_cyl_inliers.ply",
    )
    ap.add_argument("--save-json", default="", help="保存拟合结果到 json（单位米）")
    ap.add_argument("--vis", action="store_true", help="可视化（Open3D）")
    args = ap.parse_args()

    import open3d as o3d

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(p)

    pcd = o3d.io.read_point_cloud(str(p))
    if pcd.is_empty():
        raise ValueError("点云为空")

    # 下采样（输入单位）
    if float(args.voxel) > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel))

    # 法向估计（输入单位）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(args.normal_radius), max_nn=60))
    pcd.normalize_normals()

    pts = np.asarray(pcd.points).astype(np.float64)
    nrm = np.asarray(pcd.normals).astype(np.float64)

    scale = 0.001 if str(args.unit) == "mm" else 1.0
    pts_m = pts * scale
    nrm_m = nrm  # 法向无单位

    rmin = float(args.radius_min) * scale
    rmax = float(args.radius_max) * scale
    dth = float(args.dist_thresh) * scale

    model = ransac_fit_cylinder(
        pts_m,
        nrm_m,
        radius_min_m=rmin,
        radius_max_m=rmax,
        distance_thresh_m=dth,
        iters=int(args.iters),
        min_inliers=int(args.min_inliers),
        seed=int(args.seed),
    )
    if model is None:
        raise RuntimeError("RANSAC 拟合失败：请检查点云是否已裁剪到圆柱表面、或放宽 radius/dist_thresh/min_inliers。")

    print(
        f"[cyl] inliers={model.inlier_idx.size}/{pts_m.shape[0]}  "
        f"radius={model.radius_m*1000:.2f}mm  rms={model.rms_m*1000:.2f}mm"
    )
    ax = model.axis_dir
    ap0 = model.axis_point_m
    print(f"[cyl] axis_dir={ax.tolist()}")
    print(f"[cyl] axis_point(m)={ap0.tolist()}")

    # 输出一个简单抓取姿态示例：抓取点取轴线点；抓取方向取与轴线垂直的任意方向（这里用 u 基向量）
    u, v = _axis_basis(model.axis_dir)
    grasp = {
        "axis_point_m": ap0.tolist(),
        "axis_dir": ax.tolist(),
        "radius_m": float(model.radius_m),
        "rms_m": float(model.rms_m),
        "grasp_point_m": ap0.tolist(),
        "grasp_approach_dir": u.tolist(),  # 你可按机器人坐标系约定进一步约束
        "inliers": int(model.inlier_idx.size),
    }

    if str(args.save_json).strip():
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(grasp, indent=2), encoding="utf-8")
        print(f"[out] saved json: {out}")

    # 导出：圆柱内点点云（拟合后的点云）
    out_ply = str(args.output).strip()
    if not out_ply:
        out_ply = str(p.with_name(p.stem + "_cyl_inliers.ply"))
    in_pcd_full = o3d.geometry.PointCloud(pcd)
    in_pcd_full = in_pcd_full.select_by_index(model.inlier_idx.tolist())
    o3d.io.write_point_cloud(out_ply, in_pcd_full, write_ascii=False, compressed=False, print_progress=False)
    print(f"[out] saved inlier ply: {out_ply}")

    if bool(args.vis):
        # 可视化：原点云淡色，内点红色，轴线画出来
        pcd_vis = o3d.geometry.PointCloud(pcd)
        pcd_vis.paint_uniform_color([0.6, 0.6, 0.6])

        in_pcd = o3d.geometry.PointCloud(in_pcd_full)
        in_pcd.paint_uniform_color([1.0, 0.1, 0.1])

        # 轴线用 line set 表示（取一个可视化长度）
        L = 0.2  # meters
        p1 = ap0 - 0.5 * L * ax
        p2 = ap0 + 0.5 * L * ax
        # 转回输入单位，便于与点云叠加
        p1_u = (p1 / scale).astype(np.float64)
        p2_u = (p2 / scale).astype(np.float64)
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([p1_u, p2_u]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line.colors = o3d.utility.Vector3dVector([[0.1, 1.0, 0.1]])

        o3d.visualization.draw_geometries([pcd_vis, in_pcd, line], window_name="Cylinder RANSAC (inliers red, axis green)")


if __name__ == "__main__":
    main()


