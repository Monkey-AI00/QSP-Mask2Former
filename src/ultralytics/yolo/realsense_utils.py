from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:  # pragma: no cover
    rs = None  # type: ignore


@dataclass
class RSIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # (N,)

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _require_rs():
    if rs is None:
        raise ImportError("未安装 pyrealsense2，请先安装并确保在对应 conda 环境内运行。")


def start_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    align_to_color: bool = True,
):
    """
    启动 RealSense pipeline，并返回 (pipeline, profile, align_or_None)。
    """
    _require_rs()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color) if align_to_color else None
    return pipeline, profile, align


def get_color_intrinsics(profile) -> RSIntrinsics:
    """
    获取 color stream 内参（推荐：YOLO 用 color，且 depth 已对齐到 color）。
    """
    _require_rs()
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    dist = np.array(list(intr.coeffs), dtype=np.float64)
    return RSIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        dist_coeffs=dist,
    )


def frames_to_aligned_color_depth(frames, align) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """
    输入 wait_for_frames() 的 frames，输出 (color_bgr, depth_frame)。
    depth_frame 若对齐到 color，则可直接用 (u,v) 取深度。
    """
    _require_rs()
    if align is not None:
        frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    color = np.asanyarray(color_frame.get_data())  # BGR
    return color, depth_frame


def depth_at_uv(depth_frame, u: float, v: float) -> float:
    """
    获取像素点的深度（米）。若为 0 表示无效/盲区。
    """
    _require_rs()
    return float(depth_frame.get_distance(int(round(u)), int(round(v))))


def deproject_uv_depth_to_cam_xyz(intr: RSIntrinsics, u: float, v: float, z_m: float) -> np.ndarray:
    """
    反投影：由 (u,v,z) 得到相机坐标系下 3D 点 (x,y,z)，单位米。
    使用 RealSense 的 rs2_deproject_pixel_to_point 保持一致性。
    """
    _require_rs()
    # 构造与 rs.intrinsics 兼容的结构
    _intr = rs.intrinsics()
    _intr.width = intr.width
    _intr.height = intr.height
    _intr.ppx = intr.cx
    _intr.ppy = intr.cy
    _intr.fx = intr.fx
    _intr.fy = intr.fy
    # 畸变模型与系数：这里不强依赖，常见情况下对齐后的彩色点使用近似也可
    _intr.model = rs.distortion.none
    _intr.coeffs = [0, 0, 0, 0, 0]

    x, y, z = rs.rs2_deproject_pixel_to_point(_intr, [float(u), float(v)], float(z_m))
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


