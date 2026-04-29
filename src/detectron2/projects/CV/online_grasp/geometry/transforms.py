"""Geometry transform helpers migrated from legacy script."""

from __future__ import annotations

import numpy as np


def _parse_csv_floats(text: str, n: int) -> np.ndarray:
    arr = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(arr) != n:
        raise ValueError(f"期望 {n} 个数值，实际 {len(arr)}: {text}")
    return np.asarray(arr, dtype=np.float64)


def _parse_csv_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    return vals


def _as_T_4x4(vals: list[float] | None) -> np.ndarray:
    if vals is None:
        return np.eye(4, dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    # 兼容两种常见输入格式：
    # 1) 16 个一维数（行优先）
    # 2) 标准 4x4 嵌套数组
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"4x4 矩阵参数格式错误，期望 4x4 或 16 元素，实际 shape={arr.shape}")


def _pca_basis(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 10:
        return np.eye(3, dtype=np.float64)
    c = np.mean(pts, axis=0, keepdims=True)
    x = pts - c
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    v = v[:, order]
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1.0
    return v


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return x / n


def _rot_to_euler_xyz_deg(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> XYZ 欧拉角(roll, pitch, yaw)，约定 R = Rz @ Ry @ Rx。"""
    sp = -float(R[2, 0])
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    rpy = np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64))
    return float(rpy[0]), float(rpy[1]), float(rpy[2])


def _make_T_from_xyz_m(xyz_m: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return T


def _euler_xyz_deg_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Dobot CR16A: GetPose 返回姿态按 XYZ 欧拉角解释（R = Rz @ Ry @ Rx）。"""
    roll = np.deg2rad(float(rx_deg))
    pitch = np.deg2rad(float(ry_deg))
    yaw = np.deg2rad(float(rz_deg))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _pose_mm_deg_to_T(pose_mm_deg: np.ndarray) -> np.ndarray:
    """Dobot CR16A GetPose 的 [X,Y,Z,Rx,Ry,Rz] -> T_base_to_flange，姿态角采用 XYZ 欧拉角。"""
    p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _euler_xyz_deg_to_rot(p[3], p[4], p[5])
    T[:3, 3] = p[:3] * 0.001
    return T

__all__ = [
    "_as_T_4x4",
    "_parse_csv_floats",
    "_parse_csv_float_list",
    "_pca_basis",
    "_unit",
    "_rot_to_euler_xyz_deg",
    "_euler_xyz_deg_to_rot",
    "_pose_mm_deg_to_T",
    "_make_T_from_xyz_m",
]

