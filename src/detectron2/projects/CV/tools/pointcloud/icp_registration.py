#!/usr/bin/env python3
"""
相机残缺点云（source）与 CAD 完整点云（target）ICP 配准

实现思路（对应你截图的“核心逻辑”）：
1) 加载 source/target 点云（PLY）
2) 预处理：体素下采样 + 法向估计（点到面 ICP 需要法向）
3) 粗配准：FPFH + RANSAC（避免两云初始距离大导致直接 ICP 飞掉）
4) 精配准：ICP（推荐 point-to-plane）
5) 输出：4x4 变换矩阵（把 source 变换到 target 坐标系）

依赖：
  pip install open3d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np


def _import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未找到 open3d。请先安装：pip install open3d\n"
            f"原始错误: {e}"
        )


def _load_point_cloud(o3d, path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"点云为空或读取失败：{path}")
    return pcd


def _print_bounds(o3d, name: str, pcd) -> dict:
    aabb = pcd.get_axis_aligned_bounding_box()
    min_b = np.asarray(aabb.min_bound)
    max_b = np.asarray(aabb.max_bound)
    extent = max_b - min_b
    diag = float(np.linalg.norm(extent))
    center = np.asarray(aabb.get_center())
    n = int(len(pcd.points))
    print(f"[{name}] points={n}")
    print(f"[{name}] min={min_b} max={max_b}")
    print(f"[{name}] extent={extent} diag={diag:.6f} center={center}")
    return {"min": min_b, "max": max_b, "extent": extent, "diag": diag, "center": center}


def _scale_pcd(pcd, scale: float):
    if float(scale) == 1.0:
        return pcd
    pcd.scale(float(scale), center=(0.0, 0.0, 0.0))
    return pcd


def _translate_pcd(pcd, t: np.ndarray):
    if np.allclose(t, 0):
        return pcd
    pcd.translate(t.astype(np.float64), relative=True)
    return pcd


def _preprocess(o3d, pcd, voxel: float) -> Tuple[object, object]:
    """
    返回：
      - pcd_down: 下采样点云（带法向）
      - fpfh: FPFH 特征（粗配准用）
    """
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
    )
    pcd_down.normalize_normals()

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return pcd_down, fpfh


def _sorted_eigvecs_from_cov(cov: np.ndarray) -> np.ndarray:
    """
    返回按特征值从大到小排序后的特征向量矩阵 V（3x3），列向量为主轴方向。
    """
    w, v = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(w)[::-1]
    v = v[:, order]
    # 数值稳定：正交化（理论上 v 已正交）
    # 保证右手系：det>0；否则翻转最后一轴
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1.0
    return v


def _pca_basis_from_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 10:
        raise ValueError("点数过少，无法做 PCA")
    mu = pts.mean(axis=0, keepdims=True)
    x = pts - mu
    cov = (x.T @ x) / max(1, (x.shape[0] - 1))
    return _sorted_eigvecs_from_cov(cov)


def _pca_main_axis_from_points(pts: np.ndarray) -> np.ndarray:
    """
    返回 PCA 第一主轴（单位向量）。
    """
    V = _pca_basis_from_points(pts)
    axis = V[:, 0]
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (axis / n).astype(np.float64)


def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    x, y, z = a
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _candidate_pca_rotations(Vs: np.ndarray, Vt: np.ndarray) -> List[np.ndarray]:
    """
    PCA 主轴存在符号/轴交换不确定性。这里生成一组候选旋转矩阵 R，
    使得 R * Vs ≈ Vt（列向量为轴）。

    - 3! 轴排列 * 2^3 符号翻转，共 48 个；过滤 det>0。
    """
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    signs = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]
    Rs: List[np.ndarray] = []
    for p in perms:
        P = np.eye(3)[:, list(p)]
        for s in signs:
            S = np.diag(np.array(s, dtype=np.float64))
            Vsp = Vs @ P @ S
            R = Vt @ Vsp.T
            if np.linalg.det(R) < 0:
                continue
            Rs.append(R)
    # 去重（数值上可能有重复）
    uniq: List[np.ndarray] = []
    for R in Rs:
        if not any(np.allclose(R, U, atol=1e-6) for U in uniq):
            uniq.append(R)
    return uniq


def _make_T(R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _score_result(res) -> Tuple[float, float]:
    # fitness 越大越好，rmse 越小越好；用于排序
    return (float(res.fitness), -float(res.inlier_rmse))


def _clone_or_copy(o3d, pcd):
    # Open3D 0.18+ 有 clone；老版本用 copy.deepcopy 兜底
    if hasattr(pcd, "clone"):
        return pcd.clone()
    import copy

    return copy.deepcopy(pcd)


def _vis_pair(o3d, src, tgt, T=np.eye(4), title: str = ""):
    src_vis = _clone_or_copy(o3d, src)
    tgt_vis = _clone_or_copy(o3d, tgt)
    src_vis.paint_uniform_color([1.0, 0.85, 0.0])  # 黄：相机点云
    tgt_vis.paint_uniform_color([0.0, 0.8, 0.0])   # 绿：CAD 点云
    src_vis.transform(np.asarray(T, dtype=np.float64))
    if title:
        print(f"[VIS] {title}")
    o3d.visualization.draw_geometries([src_vis, tgt_vis])


def _expand_aabb(aabb, margin: float):
    min_b = np.asarray(aabb.min_bound, dtype=np.float64) - float(margin)
    max_b = np.asarray(aabb.max_bound, dtype=np.float64) + float(margin)
    return type(aabb)(min_b, max_b)


def _crop_target_near_source(o3d, target, source, T_source_to_target: np.ndarray, margin: float):
    """
    将 target 裁剪到“source 变换后”的包围盒附近，减少 CAD 远处区域对 ICP 的拉偏。
    """
    src_tmp = _clone_or_copy(o3d, source)
    src_tmp.transform(np.asarray(T_source_to_target, dtype=np.float64))
    aabb = src_tmp.get_axis_aligned_bounding_box()
    aabb2 = _expand_aabb(aabb, margin=float(margin))
    tgt_crop = target.crop(aabb2)
    return tgt_crop


def _estimate_normals(o3d, pcd, radius: float):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=50)
    )
    pcd.normalize_normals()
    return pcd


def _orient_normals_source_towards_camera(o3d, pcd, camera_location: np.ndarray) -> None:
    """
    将 source 法线统一朝向“相机位置”（适合单侧扫描的表面，避免平均法线方向随机）。
    """
    if hasattr(pcd, "orient_normals_towards_camera_location"):
        pcd.orient_normals_towards_camera_location(camera_location=np.asarray(camera_location, dtype=np.float64))


def _orient_normals_target_consistent(o3d, pcd, k: int) -> None:
    """
    将 target 法线做一致化（闭合/完整模型更适合）。
    """
    if int(k) <= 0:
        return
    if hasattr(pcd, "orient_normals_consistent_tangent_plane"):
        pcd.orient_normals_consistent_tangent_plane(int(k))


def _mean_normal(pcd) -> np.ndarray:
    n = np.asarray(pcd.normals) if hasattr(pcd, "normals") else None
    if n is None or n.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    m = np.mean(n, axis=0)
    norm = float(np.linalg.norm(m))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (m / norm).astype(np.float64)


def _apply_anti_sink_after_init(
    o3d,
    src_for_icp,
    *,
    T_init: np.ndarray,
    voxel: float,
    offset_m: float,
    flip: bool,
    normal_radius_mult: float,
    target_consistent_normals_k: int,
) -> np.ndarray:
    """
    在已确定的 T_init 姿态下执行 Anti-Sink：
    - 先把 source 变换到 target 坐标系（用 T_init）
    - 在该姿态下估计/定向法线，得到 mean normal
    - 沿 mean normal 推出 offset_m
    - 用左乘把该平移并入 T_init：T_init' = Trans(shift) @ T_init

    返回更新后的 T_init。
    """
    if float(offset_m) <= 0:
        return np.asarray(T_init, dtype=np.float64)

    T_init = np.asarray(T_init, dtype=np.float64)
    src_aligned = _clone_or_copy(o3d, src_for_icp)
    src_aligned.transform(T_init)

    # 用临时下采样点云估计 mean normal（更稳）
    v_tmp = float(voxel)
    src_tmp = src_aligned.voxel_down_sample(v_tmp)
    r = float(normal_radius_mult) * float(v_tmp)
    _estimate_normals(o3d, src_tmp, radius=r)
    # 单侧表面：尽量把法线统一朝向“相机位置”。这里用原点作为近似参考。
    _orient_normals_source_towards_camera(o3d, src_tmp, camera_location=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    # （可选）对闭合目标做一致化法线：这里没有 target 点云对象，保留参数仅做记录兼容
    _ = target_consistent_normals_k

    mn = _mean_normal(src_tmp)
    if bool(flip):
        mn = -mn
    shift = mn * float(offset_m)

    T_shift = np.eye(4, dtype=np.float64)
    T_shift[:3, 3] = shift
    T_new = T_shift @ T_init

    print(f"[anti-sink@post-init] mean_normal={mn} offset={float(offset_m):.6f} shift={shift}")
    return T_new


def _yaw_search_after_antisink(
    o3d,
    src_down,
    tgt_down,
    *,
    T_init: np.ndarray,
    axis: np.ndarray,
    steps: int,
    thresh: float,
    iters: int,
) -> np.ndarray:
    """
    在 Anti-Sink 之后做“绕主轴的旋转搜索”（只改旋转，不主动改平移；平移由 T_init/后续 ICP 负责）。
    每个角度用少量迭代的点到面 ICP 打分，选最优的 T。
    """
    steps = int(steps)
    if steps <= 1:
        return np.asarray(T_init, dtype=np.float64)

    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    reg = o3d.pipelines.registration
    criteria = reg.ICPConvergenceCriteria(max_iteration=int(iters))
    est = reg.TransformationEstimationPointToPlane()

    best_T = np.asarray(T_init, dtype=np.float64)
    best_fit = -1.0
    best_rmse = 1e9
    best_deg = 0.0

    for i in range(steps):
        ang = 2.0 * np.pi * float(i) / float(steps)
        R_delta = _axis_angle_to_R(axis, ang)
        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, :3] = R_delta

        # 在 target 坐标系绕主轴旋转：左乘
        T_try = T_delta @ best_T if i == 0 else T_delta @ np.asarray(T_init, dtype=np.float64)

        res = reg.registration_icp(
            src_down,
            tgt_down,
            max_correspondence_distance=float(thresh),
            init=T_try,
            estimation_method=est,
            criteria=criteria,
        )
        fit = float(res.fitness)
        rmse = float(res.inlier_rmse)
        if (fit > best_fit) or (np.isclose(fit, best_fit) and rmse < best_rmse):
            best_fit = fit
            best_rmse = rmse
            best_T = np.asarray(res.transformation)
            best_deg = float(i) * 360.0 / float(steps)

    print(f"[yaw-search] steps={steps} best_deg={best_deg:.1f} fitness={best_fit:.4f} rmse={best_rmse:.6f}")
    print("[yaw-search] T_init=")
    print(best_T)
    return best_T


def _icp_once(o3d, src, tgt, *, init_T: np.ndarray, thresh: float, kind: str, max_iter: int, robust: Optional[str], robust_scale: float):
    if kind == "point_to_plane":
        # Open3D 的鲁棒核 API 在不同版本/CPU-CUDA 后端里可能不存在；这里做兼容降级
        if robust and robust != "none":
            reg = o3d.pipelines.registration
            has_kernel = hasattr(reg, "RobustKernel") and hasattr(reg, "RobustKernelType")
            if has_kernel:
                try:
                    if robust == "huber":
                        kernel = reg.RobustKernel(reg.RobustKernelType.Huber, float(robust_scale))
                    elif robust == "cauchy":
                        kernel = reg.RobustKernel(reg.RobustKernelType.Cauchy, float(robust_scale))
                    else:
                        raise ValueError("--robust 仅支持 none/huber/cauchy")
                    estimator = reg.TransformationEstimationPointToPlane(kernel)
                except Exception:
                    # 任何不兼容都降级
                    print("[warn] 当前 Open3D 后端不支持 RobustKernel，已自动降级为普通 point-to-plane ICP。")
                    estimator = reg.TransformationEstimationPointToPlane()
            else:
                print("[warn] 当前 Open3D 后端缺少 RobustKernelType/RobustKernel，已自动降级为普通 point-to-plane ICP。")
                estimator = reg.TransformationEstimationPointToPlane()
        else:
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
    return o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=float(thresh),
        init=np.asarray(init_T, dtype=np.float64),
        estimation_method=estimator,
        criteria=criteria,
    )


def _build_trimmed_correspondences(
    o3d,
    src_transformed,
    tgt,
    *,
    thresh: float,
    trim_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于最近邻建立对应，并做 trimmed（只保留最近的一部分对应点）。

    返回：
      src_idx (K,), tgt_idx (K,)
    """
    if not (0.0 < float(trim_ratio) <= 1.0):
        raise ValueError("--trim-ratio 必须在 (0, 1] 内")

    src_pts = np.asarray(src_transformed.points)
    if src_pts.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    kdt = o3d.geometry.KDTreeFlann(tgt)
    thresh2 = float(thresh) ** 2
    pairs: List[Tuple[int, int]] = []
    d2_list: List[float] = []

    for i in range(src_pts.shape[0]):
        _, idx, d2 = kdt.search_knn_vector_3d(src_pts[i], 1)
        if not idx:
            continue
        d2v = float(d2[0])
        if d2v <= thresh2:
            pairs.append((i, int(idx[0])))
            d2_list.append(d2v)

    if not pairs:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    d2_arr = np.asarray(d2_list, dtype=np.float64)
    order = np.argsort(d2_arr)  # 最近的在前
    keep = max(3, int(np.ceil(len(order) * float(trim_ratio))))
    order = order[:keep]

    src_idx = np.fromiter((pairs[j][0] for j in order), dtype=np.int64, count=keep)
    tgt_idx = np.fromiter((pairs[j][1] for j in order), dtype=np.int64, count=keep)
    return src_idx, tgt_idx


def _trimmed_icp(
    o3d,
    src,
    tgt,
    *,
    init_T: np.ndarray,
    thresh: float,
    kind: str,
    max_iter: int,
    trim_ratio: float,
    min_corr: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Trimmed ICP：只使用最近的 trim_ratio 部分对应点来更新位姿。

    返回：
      T, fitness, inlier_rmse
    """
    T = np.asarray(init_T, dtype=np.float64)
    reg = o3d.pipelines.registration
    if kind == "point_to_plane":
        estimator = reg.TransformationEstimationPointToPlane()
    else:
        estimator = reg.TransformationEstimationPointToPoint()

    last_rmse = 0.0
    last_fit = 0.0
    prev_rmse = None

    for _ in range(int(max_iter)):
        src_t = _clone_or_copy(o3d, src)
        src_t.transform(T)

        src_idx, tgt_idx = _build_trimmed_correspondences(
            o3d, src_t, tgt, thresh=float(thresh), trim_ratio=float(trim_ratio)
        )
        if src_idx.size < int(min_corr):
            break

        corres_np = np.stack([src_idx, tgt_idx], axis=1).astype(np.int32)
        corres = o3d.utility.Vector2iVector(corres_np)

        # 估计增量变换：把 src_t 对齐到 tgt
        deltaT = np.asarray(estimator.compute_transformation(src_t, tgt, corres), dtype=np.float64)
        T = deltaT @ T

        # 简单指标（基于 trimmed 对应）
        src_pts = np.asarray(src_t.points)[src_idx]
        tgt_pts = np.asarray(tgt.points)[tgt_idx]
        d2 = np.sum((src_pts - tgt_pts) ** 2, axis=1)
        last_rmse = float(np.sqrt(np.mean(d2))) if d2.size else 0.0
        last_fit = float(src_idx.size) / float(max(1, len(src.points)))

        if prev_rmse is not None and abs(prev_rmse - last_rmse) < 1e-6:
            break
        prev_rmse = last_rmse

    return T, last_fit, last_rmse


def _multiscale_icp(
    o3d,
    src_full,
    tgt_full,
    *,
    init_T: np.ndarray,
    voxel: float,
    kind: str,
    robust: Optional[str],
    robust_scale: float,
    levels: int,
    iters: int,
    icp_mult: float,
    trim_ratio: float,
    trim_min_corr: int,
    icp_thresh_min: float,
) -> Tuple[np.ndarray, object]:
    """
    多尺度 ICP：voxel -> voxel/2 -> voxel/4 ... 逐级细化
    返回最终变换与最后一次 ICP 结果。
    """
    T = np.asarray(init_T, dtype=np.float64)
    last_res = None
    prev_T = T.copy()
    prev_res = None
    for li in range(int(levels)):
        v = float(voxel) / (2.0 ** li)
        if v <= 0:
            break
        thresh = max(float(icp_mult) * v, float(icp_thresh_min))
        src = src_full.voxel_down_sample(v)
        tgt = tgt_full.voxel_down_sample(v)
        _estimate_normals(o3d, src, radius=v * 2.5)
        _estimate_normals(o3d, tgt, radius=v * 2.5)
        if float(trim_ratio) < 1.0:
            T, fit, rmse = _trimmed_icp(
                o3d,
                src,
                tgt,
                init_T=T,
                thresh=thresh,
                kind=kind,
                max_iter=int(iters),
                trim_ratio=float(trim_ratio),
                min_corr=int(trim_min_corr),
            )

            class _Res:
                def __init__(self, transformation, fitness, inlier_rmse):
                    self.transformation = transformation
                    self.fitness = fitness
                    self.inlier_rmse = inlier_rmse

            last_res = _Res(T, fit, rmse)
            print(
                f"[multi-scale][trimmed] level={li} voxel={v:.6f} thresh={thresh:.6f} trim={trim_ratio:.2f} "
                f"fitness={fit:.4f} rmse={rmse:.6f}"
            )
        else:
            last_res = _icp_once(
                o3d,
                src,
                tgt,
                init_T=T,
                thresh=thresh,
                kind=kind,
                max_iter=int(iters),
                robust=robust,
                robust_scale=float(robust_scale),
            )
            T = np.asarray(last_res.transformation)
            print(
                f"[multi-scale] level={li} voxel={v:.6f} thresh={thresh:.6f} "
                f"fitness={last_res.fitness:.4f} rmse={last_res.inlier_rmse:.6f}"
            )

        # 如果更细一层导致对应点骤减（常见于残缺点云 + 阈值过严），就保留上一层结果并停止细化
        cur_fit = float(getattr(last_res, "fitness", 0.0)) if last_res is not None else 0.0
        if li > 0 and cur_fit <= 1e-6:
            print("[multi-scale] fitness 接近 0，停止细化并保留上一层结果。")
            T = prev_T
            last_res = prev_res if prev_res is not None else last_res
            break

        prev_T = np.asarray(T, dtype=np.float64).copy()
        prev_res = last_res
    return T, last_res


def main() -> int:
    parser = argparse.ArgumentParser(description="ICP 点云配准（相机残缺 -> CAD 完整）")
    parser.add_argument("--source", required=True, help="相机点云（残缺）PLY 路径")
    parser.add_argument("--target", required=True, help="CAD 点云（完整）PLY 路径")
    parser.add_argument("--voxel", type=float, default=0.0025, help="体素尺寸（单位与点云一致；m 级建议 0.002~0.01）")
    parser.add_argument("--source-scale", type=float, default=1.0, help="source 缩放系数（常见 mm->m 用 0.001）")
    parser.add_argument("--target-scale", type=float, default=1.0, help="target 缩放系数（常见 mm->m 用 0.001）")
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="自动按 bbox 对角线把 source 缩放到与 target 尺度一致（优先解决单位不一致）",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="粗/精配准前先做“质心/中心对齐”（平移到原点），提高初始化成功率；最终输出矩阵会还原回原坐标系",
    )
    parser.add_argument(
        "--anti-sink-offset-m",
        type=float,
        default=0.0,
        help="抗‘陷入内部’偏移距离（米）。在 --center 后，沿 source 平均法线方向将 source 推出（例如 0.01~0.02）",
    )
    parser.add_argument(
        "--anti-sink-flip",
        action="store_true",
        help="反向偏移（当发现被推反了/更陷入时使用）",
    )
    parser.add_argument(
        "--normal-radius-mult",
        type=float,
        default=4.0,
        help="估计法线的半径倍率：radius = normal_radius_mult * voxel（用于 anti-sink 的 mean normal）",
    )
    parser.add_argument(
        "--target-consistent-normals-k",
        type=int,
        default=0,
        help="对 target 做一致法线（orient_normals_consistent_tangent_plane）的 kNN 参数。0=关闭；推荐 50~200",
    )
    parser.add_argument("--ransac-mult", type=float, default=1.5, help="RANSAC 距离阈值 = ransac_mult * voxel")
    parser.add_argument("--icp-mult", type=float, default=0.6, help="ICP 距离阈值 = icp_mult * voxel")
    parser.add_argument("--max-ransac", type=int, default=100000, help="RANSAC 最大迭代次数")
    parser.add_argument("--max-icp", type=int, default=80, help="ICP 最大迭代次数")
    parser.add_argument("--icp", choices=["point_to_plane", "point_to_point"], default="point_to_plane", help="ICP 类型")
    parser.add_argument(
        "--init",
        choices=["auto", "ransac", "pca"],
        default="auto",
        help="初始化方式：ransac=仅用 FPFH+RANSAC；pca=仅用 PCA 主轴对齐；auto=RANSAC 失败则回退 PCA（推荐，适合圆柱类物体）",
    )
    parser.add_argument(
        "--ransac-min-fitness",
        type=float,
        default=0.05,
        help="当 init=auto 时，RANSAC fitness 低于该阈值则认为失败并回退 PCA",
    )
    parser.add_argument(
        "--yaw-steps",
        type=int,
        default=8,
        help="PCA 初始化后，绕目标主轴再做多起点搜索的角度个数（0=不搜索；圆柱体建议 4~16）",
    )
    parser.add_argument(
        "--yaw-search-steps",
        type=int,
        default=0,
        help="在 anti-sink@post-init 之后做绕主轴旋转搜索的步数（推荐 12 或 16）。0=关闭",
    )
    parser.add_argument(
        "--yaw-search-iters",
        type=int,
        default=25,
        help="yaw-search 每个角度的快速 ICP 迭代次数（点到面）",
    )
    parser.add_argument(
        "--yaw-search-thresh-mult",
        type=float,
        default=6.0,
        help="yaw-search 的阈值倍率：thresh = yaw_search_thresh_mult * voxel（建议 4~12）",
    )
    parser.add_argument(
        "--crop-target",
        type=float,
        default=6.0,
        help="在最终 ICP 前将 target 裁剪到 source(变换后) AABB 附近，margin = crop_target * voxel；0=不裁剪",
    )
    parser.add_argument(
        "--multiscale-levels",
        type=int,
        default=3,
        help="多尺度 ICP 层数（3 表示 voxel, voxel/2, voxel/4），1=只做单尺度",
    )
    parser.add_argument(
        "--multiscale-iters",
        type=int,
        default=60,
        help="每个尺度的 ICP 最大迭代次数",
    )
    parser.add_argument(
        "--icp-thresh-min",
        type=float,
        default=0.00075,
        help="多尺度 ICP 的对应阈值下限（米）。避免最后一层阈值过小导致对应点骤减；建议 0.0005~0.002",
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=1.0,
        help="Trimmed ICP：每次迭代只保留最近的前 trim_ratio 部分对应点（0.5~0.9 常用）；1.0=关闭 trimmed",
    )
    parser.add_argument(
        "--trim-min-corr",
        type=int,
        default=80,
        help="Trimmed ICP 最少对应点数，低于该值则停止迭代",
    )
    parser.add_argument(
        "--robust",
        choices=["none", "huber", "cauchy"],
        default="huber",
        help="point-to-plane ICP 的鲁棒核（抑制离群点）。point_to_point 时忽略",
    )
    parser.add_argument(
        "--robust-scale",
        type=float,
        default=0.01,
        help="鲁棒核尺度（单位同点云；m 级可从 0.005~0.02 调）",
    )
    parser.add_argument(
        "--vis-raw",
        action="store_true",
        help="配准前先显示一次“将进入配准的输入叠加”（已应用 scale/auto-scale/center）。关闭窗口后继续执行 RANSAC/ICP",
    )
    parser.add_argument("--no-vis", action="store_true", help="不弹窗可视化")
    parser.add_argument("--save", default="", help="保存结果到 json（包含 4x4 矩阵与评估指标）")
    args = parser.parse_args()

    o3d = _import_open3d()

    source_path = Path(args.source).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()
    src = _load_point_cloud(o3d, source_path)
    tgt = _load_point_cloud(o3d, target_path)

    # 先打印边界框，定位“单位不一致 / 读错文件 / 点云为空”等问题
    print("=== 原始点云边界框（未缩放）===")
    b_src0 = _print_bounds(o3d, "source", src)
    b_tgt0 = _print_bounds(o3d, "target", tgt)
    if b_src0["diag"] > 0 and b_tgt0["diag"] > 0:
        ratio = max(b_src0["diag"], b_tgt0["diag"]) / max(1e-12, min(b_src0["diag"], b_tgt0["diag"]))
        print(f"[scale-check] diag ratio={ratio:.3f}（>~50 通常就是 mm/m 单位不一致）")

    # 手动缩放（或自动缩放）
    src = _clone_or_copy(o3d, src)
    tgt = _clone_or_copy(o3d, tgt)
    _scale_pcd(src, float(args.source_scale))
    _scale_pcd(tgt, float(args.target_scale))

    if args.auto_scale:
        # 按 bbox 对角线把 source 缩放到 target（用于快速修正单位不一致）
        b_src = _print_bounds(o3d, "source(after user scale)", src)
        b_tgt = _print_bounds(o3d, "target(after user scale)", tgt)
        if b_src["diag"] > 0 and b_tgt["diag"] > 0:
            s = float(b_tgt["diag"] / b_src["diag"])
            _scale_pcd(src, s)
            print(f"[auto-scale] 额外对 source 施加 scale={s:.6f}（使两者 bbox 对角线接近）")

    print("=== 用于配准的点云边界框（缩放后）===")
    b_src = _print_bounds(o3d, "source*", src)
    b_tgt = _print_bounds(o3d, "target*", tgt)

    # 用于最终可视化的“缩放后原始坐标系”点云副本（避免 --center 改写原地坐标导致显示错位）
    src_vis_base = _clone_or_copy(o3d, src)
    tgt_vis_base = _clone_or_copy(o3d, tgt)

    voxel = float(args.voxel)
    if voxel <= 0:
        raise ValueError("--voxel 必须 > 0")
    # ICP 阈值（用于 PCA 初始化阶段的快速打分 & 最终 ICP）
    icp_thresh = float(args.icp_mult) * voxel

    print(f"加载完成：source={len(src.points)} points, target={len(tgt.points)} points")
    print(f"预处理：voxel={voxel}")

    # 可选：质心对齐（只影响初始化，不改变最终输出矩阵的物理意义）
    if args.center:
        cs = np.asarray(src.get_center(), dtype=np.float64)
        ct = np.asarray(tgt.get_center(), dtype=np.float64)
        _translate_pcd(src, -cs)
        _translate_pcd(tgt, -ct)
        print(f"[center] translate source by {-cs}, target by {-ct}")
    else:
        cs = np.zeros(3, dtype=np.float64)
        ct = np.zeros(3, dtype=np.float64)

    if args.vis_raw and not args.no_vis:
        # 重要：Open3D 的 draw_geometries 是阻塞的；用户需要关闭窗口才能继续 RANSAC/ICP。
        print("[VIS] 将显示配准前叠加（已应用 scale/auto-scale/center）。请关闭窗口以继续...")
        _vis_pair(o3d, src, tgt, np.eye(4), title="input overlay (before registration)")

    src_down, src_fpfh = _preprocess(o3d, src, voxel)
    tgt_down, tgt_fpfh = _preprocess(o3d, tgt, voxel)
    print(f"下采样后：source={len(src_down.points)} points, target={len(tgt_down.points)} points")

    # 3) 初始化（RANSAC 或 PCA）
    ransac_thresh = float(args.ransac_mult) * voxel
    result_ransac = None
    T_init = np.eye(4, dtype=np.float64)

    def _run_ransac() -> Tuple[np.ndarray, object]:
        print(f"粗配准（RANSAC）：distance_threshold={ransac_thresh}")
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=ransac_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_thresh),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(int(args.max_ransac), 500),
        )
        T = np.asarray(res.transformation)
        print("RANSAC 结果：")
        print(f"  fitness={res.fitness:.4f} inlier_rmse={res.inlier_rmse:.6f}")
        print("  T_init=")
        print(T)
        return T, res

    def _run_pca_init() -> np.ndarray:
        # PCA 主轴对齐：对圆柱类/特征少的物体，比 FPFH 稳定得多
        pts_s = np.asarray(src_down.points)
        pts_t = np.asarray(tgt_down.points)
        Vs = _pca_basis_from_points(pts_s)
        Vt = _pca_basis_from_points(pts_t)
        cand_R = _candidate_pca_rotations(Vs, Vt)
        if not cand_R:
            print("[PCA] 未生成候选旋转，回退单位矩阵")
            return np.eye(4, dtype=np.float64)

        # 额外：绕目标第一主轴（最大方差方向）做多起点搜索，解决圆柱体“绕轴不唯一”的问题
        yaw_steps = max(0, int(args.yaw_steps))
        axis = Vt[:, 0]  # 目标主轴
        if yaw_steps <= 1:
            yaw_angles = [0.0]
        else:
            yaw_angles = [2.0 * np.pi * k / yaw_steps for k in range(yaw_steps)]

        best_T = None
        best_score = None
        best_desc = ""
        # 为了快速选最优初始化，用点到点 ICP 跑少量迭代做打分（不追求最终精度）
        quick_crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25)
        quick_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # 使用与最终 ICP 相同量纲的阈值（与 voxel 成比例）
        quick_thresh = max(1e-6, float(icp_thresh))

        print(f"[PCA] 候选旋转={len(cand_R)}，yaw_steps={yaw_steps}（绕目标主轴）")
        for i, R0 in enumerate(cand_R):
            for j, ang in enumerate(yaw_angles):
                Ry = _axis_angle_to_R(axis, float(ang))
                R = Ry @ R0
                T0 = _make_T(R)
                res = o3d.pipelines.registration.registration_icp(
                    src_down,
                    tgt_down,
                    max_correspondence_distance=quick_thresh,
                    init=T0,
                    estimation_method=quick_est,
                    criteria=quick_crit,
                )
                sc = _score_result(res)
                if best_score is None or sc > best_score:
                    best_score = sc
                    best_T = np.asarray(res.transformation)
                    best_desc = f"R{i}/yaw{j}"
        if best_T is None:
            return np.eye(4, dtype=np.float64)
        print(f"[PCA] 选择最佳初始化: {best_desc} (fitness={best_score[0]:.4f}, rmse={-best_score[1]:.6f})")
        print("[PCA] T_init=")
        print(best_T)
        return best_T

    # 决策
    if args.init in ("auto", "ransac"):
        T_r, result_ransac = _run_ransac()
        if args.init == "ransac":
            T_init = T_r
        else:
            if float(result_ransac.fitness) >= float(args.ransac_min_fitness):
                T_init = T_r
            else:
                print(f"[auto-init] RANSAC fitness={result_ransac.fitness:.4f} < {args.ransac_min_fitness}，回退 PCA 初始化")
                T_init = _run_pca_init()
    else:
        print("[init] 使用 PCA 初始化（跳过 RANSAC）")
        T_init = _run_pca_init()

    # 图示问题：Shell Centroid Bias（半壳质心偏移导致“陷入内部”）
    # 关键修正：Anti-Sink 必须放在 T_init 确定之后、最终 ICP 之前，
    # 否则 _run_pca_init() 内部用于打分的“快速 ICP（允许平移）”会把偏移抵消掉。
    if float(args.anti_sink_offset_m) > 0:
        if not args.center:
            print("[warn] anti-sink 建议配合 --center 使用（更稳定）。")
        T_init = _apply_anti_sink_after_init(
            o3d,
            src_for_icp=src,
            T_init=T_init,
            voxel=float(voxel),
            offset_m=float(args.anti_sink_offset_m),
            flip=bool(args.anti_sink_flip),
            normal_radius_mult=float(args.normal_radius_mult),
            target_consistent_normals_k=int(args.target_consistent_normals_k),
        )

    # 4) 精配准：ICP
    print(f"精配准（ICP）：type={args.icp} distance_threshold={icp_thresh} max_iter={args.max_icp}")

    # 先在“工作坐标系”（可能 center 后）里做裁剪与多尺度 ICP
    src_for_icp = src
    tgt_for_icp = tgt
    if float(args.crop_target) > 0:
        margin = float(args.crop_target) * float(voxel)
        tgt_for_icp = _crop_target_near_source(o3d, tgt_for_icp, src_for_icp, T_init, margin=margin)
        print(f"[crop] target 裁剪后点数: {len(tgt_for_icp.points)} (margin={margin})")

    # ====== 关键修正：Anti-Sink 之后再做旋转角度搜索（解决“两个尖尖没对上”）======
    if int(args.yaw_search_steps) > 1:
        # 用 target 下采样点云的 PCA 主轴作为旋转轴（在当前工作坐标系）
        axis = _pca_main_axis_from_points(np.asarray(tgt_down.points))
        thresh = float(args.yaw_search_thresh_mult) * float(voxel)
        T_init = _yaw_search_after_antisink(
            o3d,
            src_down,
            tgt_down,
            T_init=T_init,
            axis=axis,
            steps=int(args.yaw_search_steps),
            thresh=float(thresh),
            iters=int(args.yaw_search_iters),
        )

    # 多尺度 ICP（更稳，更接近“中心”）
    T_centered, last_res = _multiscale_icp(
        o3d,
        src_for_icp,
        tgt_for_icp,
        init_T=T_init,
        voxel=float(voxel),
        kind=str(args.icp),
        robust=str(args.robust),
        robust_scale=float(args.robust_scale),
        levels=int(args.multiscale_levels),
        iters=int(args.multiscale_iters),
        icp_mult=float(args.icp_mult),
        trim_ratio=float(args.trim_ratio),
        trim_min_corr=int(args.trim_min_corr),
        icp_thresh_min=float(args.icp_thresh_min),
    )
    result_icp = last_res
    # 如果启用了 --center，需要把“中心化坐标系”的解还原到原坐标系：
    # T = T_ct * T_centered * T_cs, 其中 T_cs: translate by -cs, T_ct: translate by +ct
    if args.center:
        T_cs = np.eye(4, dtype=np.float64)
        T_cs[:3, 3] = -cs
        T_ct = np.eye(4, dtype=np.float64)
        T_ct[:3, 3] = ct
        T_final = T_ct @ T_centered @ T_cs
    else:
        T_final = T_centered
    print("ICP 结果：")
    if result_icp is not None:
        print(f"  fitness={result_icp.fitness:.4f} inlier_rmse={result_icp.inlier_rmse:.6f}")
    print("  T_final (source->target)=")
    print(T_final)

    if not args.no_vis:
        # 注意：当启用 --center 时，src/tgt 已被平移到中心化坐标系；
        # 此时若用还原到原坐标系的 T_final 去变换中心化点云，会造成“看起来完全没对齐”的假象。
        # 解决：显示时用“缩放后原始坐标系”的点云副本 + T_final。
        if args.center:
            _vis_pair(o3d, src_vis_base, tgt_vis_base, T_final, title="aligned overlay (after ICP)")
        else:
            _vis_pair(o3d, src, tgt, T_final, title="aligned overlay (after ICP)")

    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(source_path),
            "target": str(target_path),
            "voxel": voxel,
            "source_scale": float(args.source_scale),
            "target_scale": float(args.target_scale),
            "auto_scale": bool(args.auto_scale),
            "center": bool(args.center),
            "anti_sink_offset_m": float(args.anti_sink_offset_m),
            "anti_sink_flip": bool(args.anti_sink_flip),
            "normal_radius_mult": float(args.normal_radius_mult),
            "target_consistent_normals_k": int(args.target_consistent_normals_k),
            "ransac_threshold": ransac_thresh,
            "icp_threshold": icp_thresh,
            "crop_target": float(args.crop_target),
            "multiscale_levels": int(args.multiscale_levels),
            "multiscale_iters": int(args.multiscale_iters),
            "icp_thresh_min": float(args.icp_thresh_min),
            "trim_ratio": float(args.trim_ratio),
            "trim_min_corr": int(args.trim_min_corr),
            "robust": str(args.robust),
            "robust_scale": float(args.robust_scale),
            "init": str(args.init),
            "ransac_min_fitness": float(args.ransac_min_fitness),
            "yaw_steps": int(args.yaw_steps),
            "ransac": None
            if result_ransac is None
            else {"fitness": float(result_ransac.fitness), "inlier_rmse": float(result_ransac.inlier_rmse)},
            "icp": {"fitness": float(result_icp.fitness), "inlier_rmse": float(result_icp.inlier_rmse)},
            "T_source_to_target": T_final.tolist(),
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"✓ 已保存结果: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


