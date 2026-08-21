from typing import Any

import numpy as np

from validation import validate_intrinsics, validate_transform


def sample_depth(depth_map: np.ndarray, u: float, v: float) -> float:
    h, w = depth_map.shape[:2]
    x = int(np.clip(round(float(u)), 0, w - 1))
    y = int(np.clip(round(float(v)), 0, h - 1))
    depth = float(depth_map[y, x])
    return depth / 1000.0 if depth > 10.0 else depth


def pixel_to_camera(u: float, v: float, depth: float, intrinsics: dict[str, Any]) -> list[float]:
    camera = validate_intrinsics(intrinsics)
    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    z = float(depth)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("depth must be a positive finite value")
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return [x, y, z]


def project_camera(point_xyz: list[float] | np.ndarray, intrinsics: dict[str, Any]) -> tuple[float, float] | None:
    camera = validate_intrinsics(intrinsics)
    point = np.asarray(point_xyz, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    u = camera["fx"] * point[0] / point[2] + camera["cx"]
    v = camera["fy"] * point[1] / point[2] + camera["cy"]
    return float(u), float(v)


def make_transform(rotation: list[list[float]] | np.ndarray, translation: list[float] | np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return validate_transform(transform, "transform")


def transform_point(matrix: list[list[float]] | np.ndarray, point_xyz: list[float] | np.ndarray) -> np.ndarray:
    transform = validate_transform(matrix, "transform")
    point = np.asarray(point_xyz, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must be a finite xyz vector")
    return (transform @ np.append(point, 1.0))[:3]


def invert_transform(matrix: list[list[float]] | np.ndarray) -> np.ndarray:
    transform = validate_transform(matrix, "transform")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return validate_transform(inverse, "inverse transform")


def camera_to_world(point_cam: list[float], extrinsic=None) -> list[float]:
    transform = np.eye(4, dtype=float) if extrinsic is None else extrinsic
    return transform_point(transform, point_cam).tolist()
