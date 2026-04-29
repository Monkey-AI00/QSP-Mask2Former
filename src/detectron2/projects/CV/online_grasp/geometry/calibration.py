"""Calibration loading helpers migrated from legacy script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from online_grasp.config.defaults import _DEFAULT_T_CAM_TO_FLANGE
from online_grasp.geometry.transforms import _as_T_4x4
from pose_from_icp import _load_base_cam_T


def _load_T_from_json(path: str, keys: tuple[str, ...]) -> np.ndarray:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    for k in keys:
        if k in data:
            return _as_T_4x4(data[k])
    raise KeyError(f"json 中未找到任一键: {keys}")


def _load_cam_to_flange(args) -> np.ndarray:
    """
    Eye-in-Hand 已知标定语义：T_cam_to_flange（相机坐标系 -> 法兰中心坐标系）。
    """
    if args.T_cam_flange is not None:
        return _as_T_4x4(args.T_cam_flange)
    if str(args.cam_flange_json).strip():
        data = json.loads(Path(args.cam_flange_json).expanduser().resolve().read_text(encoding="utf-8"))
        for k in ("T_cam_to_flange", "T_camera_to_flange", "T_cam_flange", "T_cam2flange"):
            if k in data:
                return _as_T_4x4(data[k])
        raise KeyError("cam-flange json 中未找到 T_cam_to_flange/T_camera_to_flange/T_cam_flange/T_cam2flange")
    return np.asarray(_DEFAULT_T_CAM_TO_FLANGE, dtype=np.float64)


def _load_fixed_base_to_camera(args) -> np.ndarray:
    if args.T_base_cam is not None:
        return _as_T_4x4(args.T_base_cam)
    if str(args.base_cam_json).strip():
        return _load_base_cam_T(Path(args.base_cam_json).expanduser().resolve())
    raise ValueError("Eye-to-Hand 模式需要 --base-cam-json 或 --T-base-cam")


def _load_region_to_grasp(args) -> np.ndarray:
    """
    固定抓取参考变换。
    约定按当前脚本记号右乘使用：
        T_grasp_cam = T_region_cam @ T_region_to_grasp
    因此该矩阵应理解为“抓取参考系相对于局部模板系的固定关系”。
    """
    if args.T_grasp_region_to_grasp is not None:
        print("[grasp-ref] 使用命令行提供的固定变换 T_region_to_grasp")
        return _as_T_4x4(args.T_grasp_region_to_grasp)
    if str(args.grasp_region_to_grasp_json).strip():
        T = _load_T_from_json(
            args.grasp_region_to_grasp_json,
            (
                "T_region_to_grasp",
                "T_grasp_region",
                "T_grasp_reference",
                "T_region_to_grasp_ref",
            ),
        )
        print("[grasp-ref] 使用 JSON 提供的固定变换 T_region_to_grasp")
        return T
    print("[grasp-ref] 未提供固定变换，默认 T_region_to_grasp = I（等价旧行为）")
    return np.eye(4, dtype=np.float64)


def _load_region_in_obj(args) -> np.ndarray:
    """
    FoundationPose 专用：加载 T_region_in_obj（局部模板系在整物体坐标系下的位姿）。
    语义：p_obj = T_region_in_obj @ p_region
    用途：T_region_cam = T_obj_cam @ T_region_in_obj
    """
    if getattr(args, "T_fp_region_in_obj", None) is not None:
        print("[fp-ref] 使用命令行提供的 T_region_in_obj")
        return _as_T_4x4(args.T_fp_region_in_obj)
    fp_json = str(getattr(args, "fp_region_in_obj_json", "") or "").strip()
    if fp_json:
        T = _load_T_from_json(
            fp_json,
            ("T_region_in_obj", "T_region_to_obj", "T_region_obj"),
        )
        print("[fp-ref] 使用 JSON 提供的 T_region_in_obj")
        return T
    print("[fp-ref] 未提供 T_region_in_obj，默认 I（mesh 坐标系 = region 坐标系）")
    return np.eye(4, dtype=np.float64)

__all__ = [
    "_load_T_from_json",
    "_load_cam_to_flange",
    "_load_fixed_base_to_camera",
    "_load_region_to_grasp",
    "_load_region_in_obj",
]

