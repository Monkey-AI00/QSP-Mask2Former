from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore


def _require_cv2():
    if cv2 is None:
        raise ImportError("未安装 opencv-contrib-python（需要包含 cv2.aruco）。")
    if not hasattr(cv2, "aruco"):
        raise ImportError("当前 OpenCV 未包含 aruco 模块，请安装 opencv-contrib-python。")


@dataclass
class TagPose:
    tag_id: int
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,)  单位：米；含义：X_cam = R * X_tag + t
    corners: np.ndarray  # (4,2)

    def R(self) -> np.ndarray:
        _require_cv2()
        R, _ = cv2.Rodrigues(self.rvec.reshape(3, 1))
        return R

    def cam_to_tag(self, X_cam: np.ndarray) -> np.ndarray:
        """
        把相机坐标系下点 X_cam 变换到 tag 坐标系：
        已知 X_cam = R * X_tag + t  =>  X_tag = R^T * (X_cam - t)
        """
        R = self.R()
        return R.T @ (X_cam.reshape(3) - self.tvec.reshape(3))


def _get_apriltag_dict(dict_name: str):
    _require_cv2()
    name = dict_name.lower()
    mapping = {
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "apriltag_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    }
    if name not in mapping:
        raise ValueError(f"不支持的 AprilTag 字典 {dict_name}，可选：{list(mapping.keys())}")
    return cv2.aruco.getPredefinedDictionary(mapping[name])


def detect_apriltag_poses(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_m: float,
    dict_name: str = "apriltag_36h11",
) -> List[TagPose]:
    """
    检测 AprilTag 并估计位姿。
    返回 TagPose 列表（不保证顺序）。
    """
    _require_cv2()
    if tag_size_m <= 0:
        raise ValueError("tag_size_m 必须为正数（单位：米）。")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = _get_apriltag_dict(dict_name)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return []

    # estimatePoseSingleMarkers：输入 corners (N,1,4,2)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners_list, tag_size_m, camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64)
    )
    out: List[TagPose] = []
    for i, tag_id in enumerate(ids.flatten().tolist()):
        corners = np.array(corners_list[i], dtype=np.float64).reshape(4, 2)
        out.append(
            TagPose(
                tag_id=int(tag_id),
                rvec=np.array(rvecs[i], dtype=np.float64).reshape(3),
                tvec=np.array(tvecs[i], dtype=np.float64).reshape(3),
                corners=corners,
            )
        )
    return out


def draw_tag_poses(
    image_bgr: np.ndarray,
    poses: List[TagPose],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_len_m: float = 0.05,
) -> np.ndarray:
    _require_cv2()
    vis = image_bgr.copy()
    for p in poses:
        cv2.aruco.drawDetectedMarkers(vis, [p.corners.reshape(1, 4, 2)], np.array([[p.tag_id]], dtype=np.int32))
        cv2.drawFrameAxes(
            vis, camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64), p.rvec.reshape(3, 1), p.tvec.reshape(3, 1), axis_len_m
        )
    return vis


