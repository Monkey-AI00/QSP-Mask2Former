from typing import Any

import numpy as np


def sample_depth(depth_map: np.ndarray, u: float, v: float) -> float:
    h, w = depth_map.shape[:2]
    x = int(np.clip(round(float(u)), 0, w - 1))
    y = int(np.clip(round(float(v)), 0, h - 1))
    depth = float(depth_map[y, x])
    return depth / 1000.0 if depth > 10.0 else depth


def pixel_to_camera(u: float, v: float, depth: float, intrinsics: dict[str, Any]) -> list[float]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    z = float(depth)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return [x, y, z]


def camera_to_world(point_cam: list[float], extrinsic=None) -> list[float]:
    return list(point_cam)
