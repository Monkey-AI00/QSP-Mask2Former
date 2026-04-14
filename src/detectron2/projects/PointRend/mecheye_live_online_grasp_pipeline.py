#!/usr/bin/env python3
"""
机械臂在线抓取闭环：
同帧采集 -> 实例分割(mask_pc) -> 目标点云 -> 点云预处理 -> 粗配准+ICP -> 抓取位姿 -> 机械臂执行

说明：
- 复用 `mecheye_live_pointrend_pointcloud_shape_prior.py` 的相机、分割、mask 形态学、深度补全工具函数。
- 默认不保存中间文件，避免“先保存再离线处理”；保留必要可视化与日志。
- 支持 `pc_mask_mode=iou` 自动合并同一目标碎块，提升目标点云完整性。
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

# 将 detectron2 根目录加入路径（当前脚本位于 detectron2/projects/PointRend）
_D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _D2_ROOT not in sys.path:
    sys.path.insert(0, _D2_ROOT)

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils  # noqa: E402
from pose_from_icp import _load_base_cam_T  # noqa: E402
from postprocess_pointcloud import post_process_point_cloud  # noqa: E402

_DEFAULT_T_CAM_TO_FLANGE = np.array(
    [
        [0.992923, 0.004783, -0.118660, 0.023239],
        [-0.003195, 0.999903, 0.013572, -0.112770],
        [0.118713, -0.013096, 0.992842, 0.118154],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:
        raise RuntimeError(f"未找到 open3d，请先安装: pip install open3d; 原始错误: {e}")


def _parse_csv_floats(text: str, n: int) -> np.ndarray:
    arr = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(arr) != n:
        raise ValueError(f"期望 {n} 个数值，实际 {len(arr)}: {text}")
    return np.asarray(arr, dtype=np.float64)


def _parse_csv_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    return vals


def _load_T_from_json(path: str, keys: tuple[str, ...]) -> np.ndarray:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    for k in keys:
        if k in data:
            return _as_T_4x4(data[k])
    raise KeyError(f"json 中未找到任一键: {keys}")


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


def _rot_to_euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> ZYX 欧拉角(roll, pitch, yaw)，与 Dobot CR16A 的 GetPose 解释保持一致。"""
    sy = -float(R[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    if abs(abs(sy) - 1.0) < 1e-8:
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


def _euler_zyx_deg_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Dobot CR16A: GetPose 返回姿态按 ZYX 欧拉角解释。"""
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
    """Dobot CR16A GetPose 的 [X,Y,Z,Rx,Ry,Rz] -> T_base_to_flange，姿态角采用 ZYX 欧拉角。"""
    p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _euler_zyx_deg_to_rot(p[3], p[4], p[5])
    T[:3, 3] = p[:3] * 0.001
    return T


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


def _load_ik_candidates(args) -> list:
    """
    加载候选法兰姿态集合（ICP 稳定定位 + 候选姿态试 IK 模式）。
    返回 list[list[float]]，每个元素为 [rx, ry, rz] (deg, ZYX 欧拉角)。
    来源优先级：--ik-candidate-rpy-json > --ik-candidate-rpy-list
    """
    candidates: list[list[float]] = []
    source_type = str(getattr(args, "ik_candidate_source_type", "euler_rpy")).strip().lower()
    print(f"[ik-cand][type] source_type={source_type}")
    if source_type == "joint_guess":
        raise ValueError(
            "ik_candidate_source_type=joint_guess 不被接受："
            "当前候选接口仅支持 TCP 姿态角 RX/RY/RZ，不支持 J4/J5/J6。"
        )
    if source_type != "euler_rpy":
        raise ValueError(f"未知 ik_candidate_source_type: {source_type}")
    print("[ik-cand][warn] ik-candidate-rpy-* 语义为 TCP orientation (RX/RY/RZ)，不是关节角 J4/J5/J6")
    json_path = str(getattr(args, "ik_candidate_rpy_json", "") or "").strip()
    if json_path:
        data = json.loads(Path(json_path).expanduser().resolve().read_text(encoding="utf-8"))
        raw = data.get("candidates", data.get("ik_candidates", data.get("rpy_list", [])))
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                candidates.append([float(x) for x in item])
        if candidates:
            print(f"[ik-cand] 从 JSON 加载 {len(candidates)} 个候选法兰姿态")
            return candidates
    rpy_list = str(getattr(args, "ik_candidate_rpy_list", "") or "").strip()
    if rpy_list:
        for part in rpy_list.split(";"):
            vals = [float(x) for x in part.strip().split(",") if x.strip()]
            if len(vals) == 3:
                candidates.append(vals)
        if candidates:
            print(f"[ik-cand] 从命令行加载 {len(candidates)} 个候选法兰姿态")
            return candidates
    return candidates


def _load_predictor(args):
    score_thr = float(args.score_thr)
    model_family = str(args.model_family).strip().lower()
    prior_path_override = str(args.shape_prior_npy).strip()
    if prior_path_override:
        prior_path_override = live_utils._require_existing_file(prior_path_override, "prior npy")

    if model_family == "mask2former":
        config_prior = live_utils._require_existing_file(
            str(args.config_file_prior).strip() or live_utils._DEFAULT_MASK2FORMER_QSP_CONFIG,
            "Mask2Former prior config",
        )
        weights_prior = live_utils._require_existing_file(str(args.weights_prior).strip(), "prior weights")
        predictor = live_utils.build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root),
            config_file=config_prior,
            weights=weights_prior,
            score_thresh=score_thr,
            device=str(args.device),
            num_classes=int(args.num_classes),
            prior_path_override=prior_path_override,
        )
    else:
        config = live_utils._require_existing_file(
            str(args.config_file).strip() or live_utils._DEFAULT_POINTREND_CONFIG,
            "PointRend config",
        )
        weights_prior = live_utils._require_existing_file(str(args.weights_prior).strip(), "prior weights")
        if prior_path_override:
            os.environ["SHAPE_PRIOR_PATH"] = str(prior_path_override)
        predictor = live_utils.build_pointrend_predictor(
            config_file=config,
            weights=weights_prior,
            mask_head_name="ShapeAwareCoarseMaskHead",
            score_thresh=score_thr,
            device=str(args.device),
            num_classes=int(args.num_classes),
        )
    return predictor


class DobotPoseExecutor:
    def __init__(
        self,
        ip: str,
        user: int,
        tool: int,
        a: int,
        v: int,
        cp: int,
        *,
        gripper_enable: bool = False,
        gripper_port: str = "/dev/ttyUSB0",
        gripper_baudrate: int = 115200,
        gripper_open_position: int = 900,
        gripper_init_timeout: float = 5.0,
    ):
        tcp_root = Path(__file__).resolve().parents[2] / "TCP-IP-Python-V4-main" / "TCP-IP-Python-V4-main"
        if str(tcp_root) not in sys.path:
            sys.path.insert(0, str(tcp_root))
        from move import DobotMove, GripperController, parse_ints  # type: ignore

        self._DobotMove = DobotMove
        self._GripperController = GripperController
        self._parse_ints = parse_ints
        self.bot = self._DobotMove(str(ip))
        self.bot.start()
        self.user = int(user)
        self.tool = int(tool)
        self.a = int(a)
        self.v = int(v)
        self.cp = int(cp)
        self._float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
        self._gripper_enable = bool(gripper_enable)
        self._gripper_port = str(gripper_port)
        self._gripper_baudrate = int(gripper_baudrate)
        self._gripper_open_position = int(gripper_open_position)
        self._gripper_init_timeout = float(gripper_init_timeout)
        self._gripper = None
        self._gripper_opened_once = False

    def _send_and_wait(self, resp: str, step_name: str) -> None:
        nums = self._parse_ints(resp)
        if len(nums) < 2 or int(nums[0]) != 0:
            raise RuntimeError(f"{step_name} 下发失败: {resp}")
        cmd_id = int(nums[1])
        ok = self.bot.wait_done(cmd_id, timeout_s=120.0)
        if not ok:
            raise TimeoutError(f"{step_name} wait_done 超时")

    def movj_pose(self, pose_mm_deg: np.ndarray, step_name: str) -> None:
        x, y, z, rx, ry, rz = [float(v) for v in pose_mm_deg.tolist()]
        resp = self.bot.dashboard.MovJ(
            x,
            y,
            z,
            rx,
            ry,
            rz,
            0,
            user=self.user,
            tool=self.tool,
            a=self.a,
            v=self.v,
            cp=self.cp,
        )
        self._send_and_wait(resp, step_name)

    def movl_pose(self, pose_mm_deg: np.ndarray, step_name: str) -> None:
        x, y, z, rx, ry, rz = [float(v) for v in pose_mm_deg.tolist()]
        resp = self.bot.dashboard.MovL(
            x,
            y,
            z,
            rx,
            ry,
            rz,
            0,
            user=self.user,
            tool=self.tool,
            a=self.a,
            v=self.v,
            cp=self.cp,
        )
        self._send_and_wait(resp, step_name)

    def solve_ik_to_joint_deg(self, pose_mm_deg: np.ndarray) -> np.ndarray:
        """
        先用控制器逆解笛卡尔位姿，再转为关节角执行，避免直接笛卡尔指令触发关节限位。
        Dobot 返回字符串格式可能随固件有差异，这里按“首个 ErrorID + 后续关节数值”做稳健解析。
        """
        x, y, z, rx, ry, rz = [float(v) for v in np.asarray(pose_mm_deg, dtype=np.float64).reshape(6).tolist()]
        resp = self.bot.dashboard.InverseSolution(
            x,
            y,
            z,
            rx,
            ry,
            rz,
            user=self.user,
            tool=self.tool,
            isJoint=1,
        )
        vals = [float(v) for v in self._float_re.findall(str(resp))]
        if len(vals) < 7:
            raise RuntimeError(f"InverseSolution 返回异常: {resp}")
        err = int(vals[0])
        if err != 0:
            raise RuntimeError(f"InverseSolution ErrorID={err}, raw={resp}")
        joints = np.asarray(vals[1:7], dtype=np.float64)
        return joints

    def movj_joint(self, joint_deg: np.ndarray, step_name: str) -> None:
        j1, j2, j3, j4, j5, j6 = [float(v) for v in np.asarray(joint_deg, dtype=np.float64).reshape(6).tolist()]
        resp = self.bot.dashboard.MovJ(
            j1,
            j2,
            j3,
            j4,
            j5,
            j6,
            1,
            user=self.user,
            tool=self.tool,
            a=self.a,
            v=self.v,
            cp=self.cp,
        )
        self._send_and_wait(resp, step_name)

    def get_current_pose_mm_deg(self) -> np.ndarray:
        resp = self.bot.dashboard.GetPose()
        vals = [float(x) for x in self._float_re.findall(str(resp))]
        if len(vals) < 7:
            raise RuntimeError(f"GetPose 返回异常: {resp}")
        err = int(vals[0])
        if err != 0:
            raise RuntimeError(f"GetPose ErrorID={err}, raw={resp}")
        return np.asarray(vals[1:7], dtype=np.float64)

    def get_T_base_flange(self) -> np.ndarray:
        pose = self.get_current_pose_mm_deg()
        return _pose_mm_deg_to_T(pose)

    def init_and_open_gripper_once(self) -> None:
        if not self._gripper_enable:
            return
        if self._gripper is None:
            self._gripper = self._GripperController(
                port=self._gripper_port,
                baudrate=self._gripper_baudrate,
                init_timeout_s=self._gripper_init_timeout,
            )
        if self._gripper_opened_once:
            return
        print("[gripper] 在识别初始位执行夹爪初始化并张开")
        self._gripper.connect_and_initialize()
        self._gripper.open_gripper(self._gripper_open_position)
        self._gripper_opened_once = True

    def close(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.release()
            except Exception:
                pass
        try:
            self.bot.close()
        except Exception:
            pass


class OnlineGraspPipeline:
    def __init__(self, args):
        self.args = args
        self.o3d = _import_open3d()
        self.predictor = _load_predictor(args)
        self.eye_in_hand = bool(args.eye_in_hand)
        self.T_cam_to_flange: Optional[np.ndarray] = None
        self.T_base_cam_fixed: Optional[np.ndarray] = None
        self.executor: Optional[DobotPoseExecutor] = None
        self.depthcomplete = None
        self._frame_idx = 0
        self.T_region_to_grasp = _load_region_to_grasp(args)

        self._pose_backend: str = str(getattr(args, "pose_backend", "icp")).strip().lower()

        # ---- ICP 稳定定位 + 候选 IK 状态变量 ----
        self._ik_candidates: list = []
        self._last_T_source_to_target: Optional[np.ndarray] = None
        self._consecutive_icp_failures: int = 0
        self._icp_fallback_after: int = int(getattr(args, "icp_fallback_after", 5))
        self._stable_history: list = []
        self._stable_frames_required: int = int(getattr(args, "stable_frames", 3))
        self._stable_pos_thr_mm: float = float(getattr(args, "stable_pos_thr_mm", 5.0))
        self._stable_rot_thr_deg: float = float(getattr(args, "stable_rot_thr_deg", 10.0))
        self._last_icp_pose_source: str = "unknown"
        self._last_icp_quality: dict[str, Any] = {}
        self._fixed_template_rpy_cache: Optional[np.ndarray] = None

        # ---- ICP 后端专属初始化 ----
        self.reg_target_name = "grasp_region_cad(from_cad_ply)"
        self.T_source_to_target_manual_init: Optional[np.ndarray] = None
        self.use_local_region_template = True
        self.source_buffer = deque(maxlen=max(1, int(args.fusion_frames)))

        if self._pose_backend == "icp":
            cad_ply_str = str(getattr(args, "cad_ply", "") or "").strip()
            if not cad_ply_str:
                raise ValueError("ICP 模式需要 --cad-ply 参数")
            cad_path = Path(cad_ply_str).expanduser().resolve()
            if not cad_path.exists():
                raise FileNotFoundError(f"CAD 点云不存在: {cad_path}")
            self.cad_pcd = self.o3d.io.read_point_cloud(str(cad_path))
            if len(self.cad_pcd.points) == 0:
                raise RuntimeError("CAD 点云为空")
            self.cad_pcd = self.cad_pcd.voxel_down_sample(float(args.icp_voxel))
            self.cad_pcd.estimate_normals(
                search_param=self.o3d.geometry.KDTreeSearchParamHybrid(radius=float(args.icp_voxel) * 2.5, max_nn=40)
            )
            self.cad_pcd.normalize_normals()
            self.reg_target_pcd = self.cad_pcd
            print("[reg] 局部模板模式：target 直接使用 --cad-ply (handle_target.ply)")

            sample_only = bool(args.handle_target_sample)
            if args.grasp_region_init_T is not None:
                self.T_source_to_target_manual_init = _as_T_4x4(args.grasp_region_init_T)
                print("[reg] 使用命令行提供的局部模板初始位姿 T_source_to_target")
            elif str(args.grasp_region_init_json).strip():
                self.T_source_to_target_manual_init = _load_T_from_json(
                    args.grasp_region_init_json,
                    ("T_source_to_target", "T_init", "T_source_to_grasp_region_cad"),
                )
                print("[reg] 使用 JSON 提供的局部模板初始位姿 T_source_to_target")
            elif not sample_only:
                raise ValueError("局部模板模式必须提供固定初值：--grasp-region-init-T 或 --grasp-region-init-json")
            else:
                print("[sample] 当前为单帧样本导出模式，允许不提供 --grasp-region-init-*")

        # ---- FoundationPose bridge 初始化 ----
        elif self._pose_backend == "foundationpose":
            self._init_fp_bridge(args)
        else:
            raise ValueError(f"未知 pose_backend: {self._pose_backend}")

        # ---- 加载候选法兰姿态（仅 ICP 模式） ----
        if self._pose_backend == "icp":
            self._ik_candidates = _load_ik_candidates(args)
            if self._ik_candidates:
                for ci, crpy in enumerate(self._ik_candidates):
                    print(f"  [ik-cand][#{ci}] rpy(deg)={crpy}")
            else:
                print("[ik-cand] 未提供候选法兰姿态，将使用 ICP 直接输出的完整位姿（旧行为）")

        # 相机
        self.camera = live_utils.Camera()
        if not live_utils.connect_camera(
            self.camera,
            ip=str(args.ip).strip(),
            serial=str(args.serial).strip(),
            index=int(args.index),
        ):
            raise RuntimeError("连接相机失败")

        exp_seq = live_utils._parse_float_list(args.exposure_seq)
        live_utils._apply_mecheye_params(
            self.camera,
            exposure_sequence=exp_seq if exp_seq else None,
            pc_surface_smoothing=str(args.pc_smoothing),
            pc_noise_removal=str(args.pc_noise),
            pc_outlier_removal=str(args.pc_outlier),
            pc_edge_preservation=str(args.pc_edge),
            save_to_device=bool(args.save_userset),
        )
        self.intrinsics = live_utils.CameraIntrinsics()
        live_utils.show_error(self.camera.get_camera_intrinsics(self.intrinsics))
        _fx, _fy, _cx, _cy = live_utils._get_depth_k_from_mecheye_intrinsics(self.intrinsics)
        self._K_3x3 = np.array([[_fx, 0.0, _cx], [0.0, _fy, _cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # 导出样本调试模式下，不需要机器人连接/执行
        sample_only = bool(args.handle_target_sample) if self._pose_backend == "icp" else False
        need_robot = (bool(args.auto_execute) or bool(self.eye_in_hand)) and (not sample_only)
        if need_robot:
            self.executor = DobotPoseExecutor(
                ip=str(args.robot_ip).strip(),
                user=int(args.robot_user),
                tool=int(args.robot_tool),
                a=int(args.robot_a),
                v=int(args.robot_v),
                cp=int(args.robot_cp),
                gripper_enable=bool(args.gripper_enable),
                gripper_port=str(args.gripper_port),
                gripper_baudrate=int(args.gripper_baudrate),
                gripper_open_position=int(args.gripper_open_position),
                gripper_init_timeout=float(args.gripper_init_timeout),
            )
        if self._pose_backend == "icp" and self._ik_candidates and self.executor is None:
            print("[ik-cand][warn] 提供了候选法兰姿态但当前未连接机器人，IK 测试将全部跳过")

        if self.eye_in_hand and not sample_only:
            self.T_cam_to_flange = _load_cam_to_flange(args)
            print("[calib] mode=eye-in-hand, 已知 T_cam_to_flange，每帧计算 T_base_cam(t)=T_base_flange(t)@inv(T_cam_to_flange)")
        elif not self.eye_in_hand:
            self.T_base_cam_fixed = _load_fixed_base_to_camera(args)
            print("[calib] mode=eye-to-hand, 使用固定 T_base_cam")
        else:
            print("[sample] handle-target-sample 模式：跳过机器人连接与手眼链路计算")

        if bool(args.cleargrasp):
            self._init_cleargrasp()

    def move_robot_to_recognition_pose(self) -> None:
        if bool(self.args.handle_target_sample):
            return
        if not bool(self.args.plan_before_recog):
            return
        if self.executor is None:
            raise RuntimeError("开启 --plan-before-recog 时需要可用机器人连接（请检查 --robot-ip）")
        p1_joint = _parse_csv_floats(self.args.p1, 6)
        print(f"[robot-plan] 识别前先运动到固定关节位 p1={p1_joint.tolist()}")
        self.executor.movj_joint(p1_joint, "pre_recog_p1_movj")
        self.executor.init_and_open_gripper_once()
        print("[robot-plan] 已到达固定识别位，开始视觉识别")

    def _init_cleargrasp(self):
        if (
            not str(self.args.cleargrasp_normals_weights).strip()
            or not str(self.args.cleargrasp_outlines_weights).strip()
            or not str(self.args.cleargrasp_depth2depth_exe).strip()
        ):
            print("[cleargrasp] 缺少权重/可执行文件，跳过")
            return
        live_utils._add_cleargrasp_to_syspath()
        from api import depth_completion_api  # type: ignore

        fx, fy, cx, cy = live_utils._get_depth_k_from_mecheye_intrinsics(self.intrinsics)
        self.depthcomplete = {
            "api": depth_completion_api,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "out_w": int(self.args.cleargrasp_out_w),
            "out_h": int(self.args.cleargrasp_out_h),
        }
        print("[cleargrasp] 已启用")

    def infer_mask_from_frame(self, color_bgr: np.ndarray) -> np.ndarray:
        out = self.predictor(color_bgr)
        inst = out["instances"].to("cpu")
        _, mask_pc, invert_applied = live_utils._build_output_masks(
            inst,
            mask_mode=str(self.args.mask_mode),
            pc_mask_mode=str(self.args.pc_mask_mode),
            pc_iou_thresh=float(self.args.pc_iou_thresh),
            pc_join_dilate=int(self.args.pc_join_dilate),
            mask_close=int(self.args.mask_close),
            mask_dilate=int(self.args.mask_dilate),
            mask_erode=int(self.args.mask_erode),
            invert_mask=bool(self.args.invert_mask),
            auto_invert_mask=bool(self.args.auto_invert_mask),
        )
        print("[mask][pc ] " + live_utils._mask_stats(mask_pc) + f" invert={invert_applied}")
        return mask_pc

    def _maybe_complete_depth(self, depth_np: np.ndarray, color_bgr: np.ndarray) -> np.ndarray:
        if not bool(self.args.cleargrasp) or self.depthcomplete is None:
            return live_utils._depth_u16_to_m(
                live_utils._depth_to_png_u16(depth_np),
                unit=str(self.args.depth_unit),
            )

        if "model" not in self.depthcomplete:
            H, W = depth_np.shape[:2]
            fx0, fy0, cx0, cy0 = (
                float(self.depthcomplete["fx"]),
                float(self.depthcomplete["fy"]),
                float(self.depthcomplete["cx"]),
                float(self.depthcomplete["cy"]),
            )
            out_w, out_h = int(self.depthcomplete["out_w"]), int(self.depthcomplete["out_h"])
            fx2, fy2, cx2, cy2 = live_utils._scale_k_for_resize(fx0, fy0, cx0, cy0, in_w=W, in_h=H, out_w=out_w, out_h=out_h)
            api = self.depthcomplete["api"]
            self.depthcomplete["model"] = api.DepthToDepthCompletion(
                normalsWeightsFile=str(self.args.cleargrasp_normals_weights),
                outlinesWeightsFile=str(self.args.cleargrasp_outlines_weights),
                masksWeightsFile="",
                normalsModel="drn",
                outlinesModel="drn",
                depth2depthExecutable=str(self.args.cleargrasp_depth2depth_exe),
                outputImgHeight=out_h,
                outputImgWidth=out_w,
                fx=fx2,
                fy=fy2,
                cx=cx2,
                cy=cy2,
                filter_d=int(self.args.cleargrasp_filter_d),
                filter_sigmaColor=float(self.args.cleargrasp_filter_sigma_color),
                filter_sigmaSpace=float(self.args.cleargrasp_filter_sigma_space),
                normalsInferenceHeight=out_h,
                normalsInferenceWidth=out_w,
                outlinesInferenceHeight=out_h,
                outlinesInferenceWidth=out_w,
                min_depth=0.0,
                max_depth=3.0,
            )
            print("[cleargrasp] model loaded")

        depth_m = live_utils._depth_u16_to_m(
            live_utils._depth_to_png_u16(depth_np),
            unit=str(self.args.depth_unit),
        )
        rgb = color_bgr[:, :, ::-1]
        out_small, _ = self.depthcomplete["model"].depth_completion(
            rgb,
            depth_m,
            inertia_weight=float(self.args.cleargrasp_inertia),
            smoothness_weight=float(self.args.cleargrasp_smoothness),
            tangent_weight=float(self.args.cleargrasp_tangent),
            mode_modify_input_depth="",
        )
        out_up = cv2.resize(out_small, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
        completed = depth_m.copy()
        thr = float(self.args.cleargrasp_fill_thresh)
        holes = (~np.isfinite(completed)) | (completed <= max(0.0, thr))
        completed[holes] = out_up[holes]
        return completed

    def build_target_pointcloud(self, depth_obj, mask_pc: np.ndarray, color_bgr: np.ndarray):
        depth_np = depth_obj.data()
        depth_m = self._maybe_complete_depth(depth_np, color_bgr)
        fx, fy, cx, cy = live_utils._get_depth_k_from_mecheye_intrinsics(self.intrinsics)
        xyz_m, rgb = live_utils._backproject_masked_xyzrgb(
            depth_m,
            color_bgr,
            mask_pc,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=max(1, int(self.args.pc_stride)),
        )
        pcd = self.o3d.geometry.PointCloud()
        if xyz_m.shape[0] == 0:
            return pcd
        pcd.points = self.o3d.utility.Vector3dVector(xyz_m.astype(np.float64))
        pcd.colors = self.o3d.utility.Vector3dVector((rgb.astype(np.float64) / 255.0))
        return pcd

    def preprocess_target_pointcloud(self, pcd):
        if len(pcd.points) < int(self.args.min_points):
            return pcd
        if bool(self.args.grasp_region_preserve) and bool(self.use_local_region_template):
            # 局部抓取区域模式：保留区域结构，避免 DBSCAN largest-cluster 破坏局部面片几何
            p = pcd
            if float(self.args.pp_voxel) > 0:
                p = p.voxel_down_sample(float(self.args.pp_voxel))
            _, ind = p.remove_statistical_outlier(
                nb_neighbors=int(self.args.pp_sor_nb),
                std_ratio=float(self.args.pp_sor_std),
            )
            p = p.select_by_index(ind)
            if int(self.args.pp_ror_nb) > 0 and float(self.args.pp_ror_radius) > 0:
                _, ind = p.remove_radius_outlier(
                    nb_points=int(self.args.pp_ror_nb),
                    radius=float(self.args.pp_ror_radius),
                )
                p = p.select_by_index(ind)
            print(f"[pp][local] preserve grasp region -> points={len(p.points)}")
            return p
        pcd2 = post_process_point_cloud(
            pcd,
            voxel_size=float(self.args.pp_voxel),
            drop_zero_xyz=True,
            drop_zero_eps=1e-9,
            sor_nb_neighbors=int(self.args.pp_sor_nb),
            sor_std_ratio=float(self.args.pp_sor_std),
            ror_nb_points=int(self.args.pp_ror_nb),
            ror_radius=float(self.args.pp_ror_radius),
            dbscan_eps=float(self.args.pp_dbscan_eps),
            dbscan_min_points=int(self.args.pp_dbscan_min_points),
            keep_top_k=int(self.args.pp_keep_top_k),
            cluster_select=str(self.args.pp_cluster_select),
        )
        return pcd2

    def _clone_pcd(self, pcd):
        if hasattr(pcd, "clone"):
            return pcd.clone()
        return copy.deepcopy(pcd)

    def _pcd_stats_str(self, pcd, name: str) -> str:
        n = int(len(pcd.points))
        if n <= 0:
            return f"{name}: points=0"
        pts = np.asarray(pcd.points, dtype=np.float64)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        ext = mx - mn
        return (
            f"{name}: points={n} "
            f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
            f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}) "
            f"extent=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f})"
        )

    def _pcd_extent(self, pcd) -> np.ndarray:
        if len(pcd.points) <= 0:
            return np.zeros(3, dtype=np.float64)
        pts = np.asarray(pcd.points, dtype=np.float64)
        return pts.max(axis=0) - pts.min(axis=0)

    def _update_and_fuse_source(self, pcd):
        if len(pcd.points) > 0:
            src_to_buffer = self._clone_pcd(pcd)
            if bool(self.args.fusion_after_coarse_align):
                src_to_buffer = self._coarse_align_for_fusion(src_to_buffer)
            self.source_buffer.append(src_to_buffer)
        valid = [x for x in self.source_buffer if len(x.points) > 0]
        required = int(self.args.fusion_frames)
        if len(valid) < required:
            raise RuntimeError(f"SKIP: 融合有效帧未达完整窗口: {len(valid)}/{required}")
        fused = self.o3d.geometry.PointCloud()
        for x in valid:
            fused += x
        fused = fused.voxel_down_sample(float(self.args.fusion_voxel))
        if len(fused.points) < int(self.args.fused_min_points):
            raise RuntimeError(f"融合点数不足: {len(fused.points)} < {int(self.args.fused_min_points)}")
        print(
            f"[fusion] frames={len(valid)}/{int(self.args.fusion_frames)} "
            f"fused_points={len(fused.points)} voxel={float(self.args.fusion_voxel):.4f}"
        )
        return fused, len(valid)

    def _coarse_align_for_fusion(self, source_pcd):
        """
        可选：先将当前帧 source 粗对齐到 target 邻域，再写入融合缓存，降低相机姿态变化带来的融合糊化。
        """
        src = source_pcd.voxel_down_sample(float(self.args.icp_voxel))
        if len(src.points) < int(self.args.min_points):
            raise RuntimeError(f"SKIP: fusion_after_coarse_align 点数不足: {len(src.points)}")
        target = self.reg_target_pcd
        reg = self.o3d.pipelines.registration
        coarse_dist = float(self.args.icp_voxel) * float(self.args.coarse_icp_dist_mult)
        eval_dist = max(coarse_dist * 1.2, coarse_dist)
        init_candidates = self._build_init_candidates(src, target)
        scored = self._score_init_candidates(src, target, init_candidates, eval_dist=eval_dist)
        best = self._select_best_init(scored)
        if best is None or float(best["fitness"]) < float(self.args.icp_coarse_fitness_thr):
            raise RuntimeError(
                f"SKIP: fusion_after_coarse_align init 失败，best_fitness="
                f"{0.0 if best is None else float(best['fitness']):.4f}"
            )
        coarse = reg.registration_icp(
            src,
            target,
            max_correspondence_distance=coarse_dist,
            init=np.asarray(best["T"], dtype=np.float64),
            estimation_method=reg.TransformationEstimationPointToPoint(),
            criteria=reg.ICPConvergenceCriteria(max_iteration=max(10, int(self.args.max_icp_stage1) // 2)),
        )
        if float(coarse.fitness) < float(self.args.icp_stage1_fitness_thr):
            raise RuntimeError(
                f"SKIP: fusion_after_coarse_align coarse 失败: "
                f"{float(coarse.fitness):.4f} < {float(self.args.icp_stage1_fitness_thr):.4f}"
            )
        src_aligned = self._clone_pcd(source_pcd)
        src_aligned.transform(np.asarray(coarse.transformation, dtype=np.float64))
        print(
            f"[fusion] fusion_after_coarse_align=True coarse_fit={float(coarse.fitness):.4f} "
            f"rmse={float(coarse.inlier_rmse):.6f}"
        )
        return src_aligned

    def _check_extent_consistency(self, source_pcd, target_pcd) -> None:
        es = self._pcd_extent(source_pcd)
        et = self._pcd_extent(target_pcd)
        if np.any(es <= 1e-9) or np.any(et <= 1e-9):
            raise RuntimeError(f"SKIP: bbox extent 异常: source={es.tolist()} target={et.tolist()}")
        ratio = es / et
        rmin = float(np.min(ratio))
        rmax = float(np.max(ratio))
        print(f"[bbox] source_extent={es.tolist()} target_extent={et.tolist()}")
        print(f"[bbox] source/target extent ratio={ratio.tolist()} min={rmin:.3f} max={rmax:.3f}")
        if rmin < float(self.args.bbox_ratio_min) or rmax > float(self.args.bbox_ratio_max):
            raise RuntimeError(
                f"SKIP: bbox 尺寸不一致，ratio_min={rmin:.3f}, ratio_max={rmax:.3f}, "
                f"阈值=[{float(self.args.bbox_ratio_min):.3f},{float(self.args.bbox_ratio_max):.3f}]"
            )

    def _make_center_align_init(self, source_pcd, target_pcd) -> np.ndarray:
        pts_s = np.asarray(source_pcd.points, dtype=np.float64)
        pts_t = np.asarray(target_pcd.points, dtype=np.float64)
        cs = pts_s.mean(axis=0)
        ct = pts_t.mean(axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = ct - cs
        return T

    def _make_pca_align_init(self, source_pcd, target_pcd) -> np.ndarray:
        pts_s = np.asarray(source_pcd.points, dtype=np.float64)
        pts_t = np.asarray(target_pcd.points, dtype=np.float64)
        Rs = _pca_basis(pts_s)
        Rt = _pca_basis(pts_t)
        R = Rt @ Rs.T
        cs = pts_s.mean(axis=0)
        ct = pts_t.mean(axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = ct - (R @ cs)
        return T

    def _expand_rotation_flip_candidates(self, T_base: np.ndarray) -> list[dict[str, Any]]:
        def _rot_axis_180(axis: np.ndarray) -> np.ndarray:
            a = _unit(np.asarray(axis, dtype=np.float64))
            return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(a, a)

        R0 = np.asarray(T_base[:3, :3], dtype=np.float64)
        t0 = np.asarray(T_base[:3, 3], dtype=np.float64)
        thickness_axis = _unit(R0[:, 2])
        main_axis = _unit(R0[:, 0])
        cands = [{"name": "pca", "R_delta": np.eye(3, dtype=np.float64)}]
        cands.append({"name": "pca_flip_thickness", "R_delta": _rot_axis_180(thickness_axis)})
        cands.append({"name": "pca_flip_main", "R_delta": _rot_axis_180(main_axis)})
        cands.append({"name": "pca_flip_combo", "R_delta": _rot_axis_180(main_axis) @ _rot_axis_180(thickness_axis)})
        out = []
        for c in cands:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(c["R_delta"], dtype=np.float64) @ R0
            T[:3, 3] = t0
            out.append({"name": c["name"], "T": T})
        return out

    def _build_online_init_candidates(self, source_pcd, target_pcd) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        center_T = self._make_center_align_init(source_pcd, target_pcd)
        candidates.append({"name": "online_center_align", "T": center_T})
        pca_T = self._make_pca_align_init(source_pcd, target_pcd)
        for c in self._expand_rotation_flip_candidates(pca_T):
            candidates.append({"name": f"online_{c['name']}", "T": c["T"]})
        max_n = max(1, int(self.args.init_candidate_max))
        return candidates[:max_n]

    def _score_init_candidates(
        self,
        source_pcd,
        target_pcd,
        init_candidates: list[dict[str, Any]],
        *,
        eval_dist: float,
    ) -> list[dict[str, Any]]:
        reg = self.o3d.pipelines.registration
        scored: list[dict[str, Any]] = []
        for i, cand in enumerate(init_candidates):
            T = np.asarray(cand["T"], dtype=np.float64)
            ev = reg.evaluate_registration(
                source_pcd,
                target_pcd,
                max_correspondence_distance=float(eval_dist),
                transformation=T,
            )
            item = {
                "idx": i,
                "name": str(cand["name"]),
                "T": T,
                "fitness": float(ev.fitness),
                "rmse": float(ev.inlier_rmse),
                "dist": float(eval_dist),
            }
            scored.append(item)
            print(
                f"[icp][cand] #{i} src={item['name']} "
                f"fitness={item['fitness']:.4f} rmse={item['rmse']:.6f} dist={item['dist']:.5f}"
            )
        return scored

    def _select_best_init(self, scored_candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not scored_candidates:
            return None
        best = max(scored_candidates, key=lambda x: x["fitness"])
        print(
            f"[icp][init] best_candidate=#{int(best['idx'])} src={best['name']} "
            f"fitness={float(best['fitness']):.4f} rmse={float(best['rmse']):.6f}"
        )
        return best

    def _build_init_candidates(self, source_pcd, target_pcd) -> list[dict[str, Any]]:
        cands: list[dict[str, Any]] = []
        if (
            self._last_T_source_to_target is not None
            and self._consecutive_icp_failures < int(self.args.icp_fallback_after)
        ):
            cands.append(
                {
                    "name": "last_success",
                    "T": np.asarray(self._last_T_source_to_target, dtype=np.float64).copy(),
                }
            )
            print(
                f"[icp][init] priority=last_success "
                f"(failures={self._consecutive_icp_failures}/{int(self.args.icp_fallback_after)})"
            )
        if bool(self.args.online_init_enable):
            cands.extend(self._build_online_init_candidates(source_pcd, target_pcd))
        if self.T_source_to_target_manual_init is not None:
            cands.append(
                {
                    "name": "manual_fixed",
                    "T": np.asarray(self.T_source_to_target_manual_init, dtype=np.float64),
                }
            )
        return cands

    def _transform_rot_deg(self, R: np.ndarray) -> float:
        r = np.asarray(R, dtype=np.float64).reshape(3, 3)
        cos_theta = (np.trace(r) - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_theta)))

    def _compute_transform_delta(self, T_delta: np.ndarray) -> tuple[float, float]:
        t = np.asarray(T_delta[:3, 3], dtype=np.float64)
        trans_mm = float(np.linalg.norm(t) * 1000.0)
        rot_deg = self._transform_rot_deg(np.asarray(T_delta[:3, :3], dtype=np.float64))
        return trans_mm, rot_deg

    def _should_fallback_to_coarse(
        self,
        coarse_fitness: float,
        refine_fitness: float,
        residual_trans_mm: float,
        residual_rot_deg: float,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        cfit = max(1e-9, float(coarse_fitness))
        ratio = float(refine_fitness) / cfit
        if ratio < float(self.args.refine_min_fitness_ratio):
            reasons.append(f"fitness_ratio={ratio:.3f}<{float(self.args.refine_min_fitness_ratio):.3f}")
        if residual_trans_mm > float(self.args.refine_max_residual_trans_mm):
            reasons.append(
                f"residual_trans_mm={residual_trans_mm:.2f}>{float(self.args.refine_max_residual_trans_mm):.2f}"
            )
        if residual_rot_deg > float(self.args.refine_max_residual_rot_deg):
            reasons.append(
                f"residual_rot_deg={residual_rot_deg:.2f}>{float(self.args.refine_max_residual_rot_deg):.2f}"
            )
        return (len(reasons) > 0), reasons

    def _select_final_icp_transform(
        self,
        T_coarse: np.ndarray,
        T_refine: np.ndarray,
        *,
        coarse_fitness: float,
        coarse_rmse: float,
        refine_fitness: float,
        refine_rmse: float,
        residual_T: np.ndarray,
    ) -> tuple[np.ndarray, str, list[str]]:
        residual_trans_mm, residual_rot_deg = self._compute_transform_delta(residual_T)
        print(
            f"[icp][quality] coarse_fit={float(coarse_fitness):.4f} coarse_rmse={float(coarse_rmse):.6f} "
            f"refine_fit={float(refine_fitness):.4f} refine_rmse={float(refine_rmse):.6f} "
            f"residual_trans_mm={residual_trans_mm:.2f} residual_rot_deg={residual_rot_deg:.2f}"
        )
        self._last_icp_quality = {
            "coarse_fitness": float(coarse_fitness),
            "coarse_rmse": float(coarse_rmse),
            "refine_fitness": float(refine_fitness),
            "refine_rmse": float(refine_rmse),
            "residual_trans_mm": float(residual_trans_mm),
            "residual_rot_deg": float(residual_rot_deg),
        }
        if not bool(self.args.refine_fallback_enable):
            return np.asarray(T_refine, dtype=np.float64), "refine", []
        do_fb, reasons = self._should_fallback_to_coarse(
            float(coarse_fitness),
            float(refine_fitness),
            float(residual_trans_mm),
            float(residual_rot_deg),
        )
        if do_fb:
            print(f"[icp][fallback] use=coarse reason={' | '.join(reasons)}")
            return np.asarray(T_coarse, dtype=np.float64), "coarse", reasons
        return np.asarray(T_refine, dtype=np.float64), "refine", []

    def _thin_source_main_surface(self, pcd, *, axis_mode_override: Optional[str] = None):
        if not bool(self.args.surface_thin_enable):
            return pcd
        if len(pcd.points) <= 0:
            raise RuntimeError("SKIP: 主曲面提纯输入为空")

        pts = np.asarray(pcd.points, dtype=np.float64)
        center = pts.mean(axis=0)
        axis_mode = str(axis_mode_override or self.args.surface_thin_axis).strip().lower()
        if axis_mode == "auto":
            basis = _pca_basis(pts)
            axis = basis[:, 2]  # 最小方差方向视作厚度方向
        elif axis_mode == "x":
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif axis_mode == "y":
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif axis_mode == "z":
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError(f"未知 surface_thin_axis: {axis_mode}")
        axis = _unit(axis)

        proj = (pts - center) @ axis
        mid = float(np.median(proj))
        band_m = max(1e-4, float(self.args.surface_thin_band_mm) * 0.001)
        half = band_m * 0.5
        keep_mask = np.abs(proj - mid) <= half
        keep_idx = np.flatnonzero(keep_mask)
        kept = int(keep_idx.size)
        total = int(pts.shape[0])
        keep_ratio = kept / max(1, total)
        print(
            f"[surface-thin] axis={axis_mode} axis_vec={axis.tolist()} "
            f"band_mm={float(self.args.surface_thin_band_mm):.2f} keep={kept}/{total} ({keep_ratio:.3f})"
        )
        if kept < int(self.args.surface_thin_min_points):
            raise RuntimeError(
                f"SKIP: 主曲面提纯后点数不足: {kept} < {int(self.args.surface_thin_min_points)}"
            )
        return pcd.select_by_index(keep_idx.tolist())

    def _visualize_icp_overlay(
        self,
        raw_source_pcd,
        source_pcd,
        target_pcd,
        T_source_to_cad: np.ndarray,
        *,
        stage1_fitness: float,
        stage1_rmse: float,
        coarse_fitness: float,
        fine_fitness: float,
        rmse: float,
    ) -> None:
        """
        可视化 ICP 配准叠加：
        - 橙色：原始融合 source（变换后）
        - 黄色：主曲面提纯后的 source（变换后）
        - 绿色：handle_target.ply 局部模板点云（target）
        """
        raw_vis = self._clone_pcd(raw_source_pcd)
        src_vis = self._clone_pcd(source_pcd)
        tgt_vis = self._clone_pcd(target_pcd)
        raw_vis.paint_uniform_color([1.0, 0.45, 0.0])
        src_vis.paint_uniform_color([1.0, 0.85, 0.0])
        tgt_vis.paint_uniform_color([0.0, 0.8, 0.0])
        raw_vis.transform(np.asarray(T_source_to_cad, dtype=np.float64))
        src_vis.transform(np.asarray(T_source_to_cad, dtype=np.float64))
        win_name = (
            f"ICP overlay | orange=raw_fused_source | yellow=surface_thinned_source | green={self.reg_target_name} "
            f"| frame={self._frame_idx} raw={len(raw_source_pcd.points)} thin={len(source_pcd.points)} tgt={len(target_pcd.points)} "
            f"| stage1_fit={stage1_fitness:.3f} stage1_rmse={stage1_rmse:.4f} "
            f"| init_fit={coarse_fitness:.3f} stage2_fit={fine_fitness:.3f} stage2_rmse={rmse:.4f}"
        )
        self.o3d.visualization.draw_geometries([raw_vis, src_vis, tgt_vis], window_name=win_name)

    def _compute_T_base_cam_dynamic(self) -> np.ndarray:
        if self.eye_in_hand:
            if self.executor is None or self.T_cam_to_flange is None:
                raise RuntimeError("Eye-in-Hand 模式未初始化 robot executor 或 T_cam_to_flange")
            # 已知：T_cam_to_flange（相机->法兰），实时读取：T_base_to_flange
            # 目标：T_base_to_camera = T_base_to_flange @ T_flange_to_camera
            T_base_flange = self.executor.get_T_base_flange()
            T_flange_cam = np.linalg.inv(np.asarray(self.T_cam_to_flange, dtype=np.float64))
            return np.asarray(T_base_flange, dtype=np.float64) @ T_flange_cam
        if self.T_base_cam_fixed is None:
            raise RuntimeError("固定外参 T_base_cam 未设置")
        return np.asarray(self.T_base_cam_fixed, dtype=np.float64)

    def estimate_pose_and_grasp(self, fused_source_pcd, T_base_cam: np.ndarray, *, fused_frames: int):
        # source=融合后的实时局部抓取区域点云, target=局部 CAD 模板(handle_target)
        raw_source = fused_source_pcd.voxel_down_sample(float(self.args.icp_voxel))
        if len(raw_source.points) < int(self.args.min_points):
            raise RuntimeError(f"SKIP: 点云点数不足: {len(raw_source.points)}")
        target = self.reg_target_pcd
        print(
            "[icp][pair] "
            f"source=fused_live_grasp_region, target={self.reg_target_name}, "
            f"use_local_template={self.use_local_region_template}, fused_frames={int(fused_frames)}"
        )
        print("[icp][pair] " + self._pcd_stats_str(raw_source, "source_raw_fused"))
        print("[icp][pair] " + self._pcd_stats_str(target, "target"))
        self._check_extent_consistency(raw_source, target)

        reg = self.o3d.pipelines.registration
        coarse_dist = float(self.args.icp_voxel) * float(self.args.coarse_icp_dist_mult)
        refine_dist = float(self.args.icp_voxel) * float(self.args.refine_icp_dist_mult)
        eval_dist = max(coarse_dist * 1.2, coarse_dist)

        # stage0: last_success > online candidates > manual_fixed
        init_candidates = self._build_init_candidates(raw_source, target)
        if not init_candidates:
            raise RuntimeError("SKIP: 缺少可用初始位姿（last_success/online/manual 均不可用）")
        scored = self._score_init_candidates(raw_source, target, init_candidates, eval_dist=eval_dist)
        non_manual = [x for x in scored if str(x["name"]) != "manual_fixed"]
        thr_stage0 = float(self.args.icp_coarse_fitness_thr)
        best_non_manual = self._select_best_init(non_manual) if non_manual else None
        if best_non_manual is not None and float(best_non_manual["fitness"]) >= thr_stage0:
            best_init = best_non_manual
        else:
            best_init = self._select_best_init(scored)
            if best_init is not None and str(best_init["name"]) == "manual_fixed":
                print("[fallback] online/track 初值不可用，回退 manual_fixed")
        if best_init is None:
            raise RuntimeError("SKIP: all init candidates failed: no valid candidate")
        if float(best_init["fitness"]) < thr_stage0:
            raise RuntimeError(
                f"SKIP: all init candidates failed: best_fitness={float(best_init['fitness']):.4f} "
                f"< thr={thr_stage0:.4f}"
            )
        T_init = np.asarray(best_init["T"], dtype=np.float64)
        print("[icp][init] T_source_to_target=" + np.array2string(T_init, precision=5, suppress_small=True))

        # coarse ICP: 大距离 p2p，把初值拉进正确 basin
        coarse = reg.registration_icp(
            raw_source,
            target,
            max_correspondence_distance=coarse_dist,
            init=T_init,
            estimation_method=reg.TransformationEstimationPointToPoint(),
            criteria=reg.ICPConvergenceCriteria(max_iteration=int(self.args.max_icp_stage1)),
        )
        print(
            f"[icp][coarse] fitness={float(coarse.fitness):.4f} rmse={float(coarse.inlier_rmse):.6f} "
            f"dist={coarse_dist:.5f}"
        )
        print(
            "[icp][coarse] T_source_to_target="
            + np.array2string(np.asarray(coarse.transformation, dtype=np.float64), precision=5, suppress_small=True)
        )
        if float(coarse.fitness) < float(self.args.icp_stage1_fitness_thr):
            raise RuntimeError(
                f"SKIP: ICP coarse fitness 过低: {float(coarse.fitness):.4f} < {float(self.args.icp_stage1_fitness_thr):.4f}"
            )
        if float(coarse.inlier_rmse) > float(self.args.icp_stage1_rmse_thr):
            raise RuntimeError(
                f"SKIP: ICP coarse rmse 过高: {float(coarse.inlier_rmse):.6f} > {float(self.args.icp_stage1_rmse_thr):.6f}"
            )

        # coarse 后再做主曲面提纯，减少未对齐时薄片方向不稳定
        source_coarse_aligned = self._clone_pcd(raw_source)
        source_coarse_aligned.transform(np.asarray(coarse.transformation, dtype=np.float64))
        axis_mode = str(self.args.surface_thin_axis).strip().lower()
        source_refine_aligned = self._thin_source_main_surface(source_coarse_aligned, axis_mode_override=axis_mode)
        if len(source_refine_aligned.points) < int(self.args.min_points):
            raise RuntimeError(
                f"SKIP: 主曲面提纯后点数不足: {len(source_refine_aligned.points)} < {int(self.args.min_points)}"
            )
        print("[icp][pair] " + self._pcd_stats_str(source_refine_aligned, "source_coarse_aligned_surface_thinned"))
        self._check_extent_consistency(source_refine_aligned, target)
        source_refine_aligned.estimate_normals(
            search_param=self.o3d.geometry.KDTreeSearchParamHybrid(radius=float(self.args.icp_voxel) * 2.5, max_nn=40)
        )
        source_refine_aligned.normalize_normals()

        # refine ICP: 小距离 p2pl，毫米级精配准
        refine = reg.registration_icp(
            source_refine_aligned,
            target,
            max_correspondence_distance=refine_dist,
            init=np.eye(4, dtype=np.float64),
            estimation_method=reg.TransformationEstimationPointToPlane(),
            criteria=reg.ICPConvergenceCriteria(max_iteration=int(self.args.max_icp_stage2)),
        )
        print(
            f"[icp][refine] fitness={float(refine.fitness):.4f} rmse={float(refine.inlier_rmse):.6f} "
            f"dist={refine_dist:.5f}"
        )
        print(
            "[icp][refine] T_residual="
            + np.array2string(np.asarray(refine.transformation, dtype=np.float64), precision=5, suppress_small=True)
        )

        T_source_to_region = np.asarray(refine.transformation, dtype=np.float64) @ np.asarray(coarse.transformation, dtype=np.float64)
        print(
            "[icp][refine] T_source_to_target_final="
            + np.array2string(np.asarray(T_source_to_region, dtype=np.float64), precision=5, suppress_small=True)
        )

        if bool(self.args.vis_icp):
            every = max(1, int(self.args.vis_icp_every))
            should_vis = (self._frame_idx % every == 0)
            if bool(self.args.vis_icp_only_fail):
                should_vis = should_vis and (float(refine.fitness) <= float(self.args.vis_icp_fail_thr))
            if should_vis:
                source_refine_raw = self._clone_pcd(source_refine_aligned)
                source_refine_raw.transform(np.linalg.inv(np.asarray(coarse.transformation, dtype=np.float64)))
                print("[icp][vis] 打开 ICP 叠加窗口（关闭窗口后主循环继续）")
                self._visualize_icp_overlay(
                    raw_source,
                    source_refine_raw,
                    target,
                    np.asarray(T_source_to_region, dtype=np.float64),
                    stage1_fitness=float(coarse.fitness),
                    stage1_rmse=float(coarse.inlier_rmse),
                    coarse_fitness=float(best_init["fitness"]),
                    fine_fitness=float(refine.fitness),
                    rmse=float(refine.inlier_rmse),
                )

        T_final, final_src, fb_reasons = self._select_final_icp_transform(
            np.asarray(coarse.transformation, dtype=np.float64),
            np.asarray(T_source_to_region, dtype=np.float64),
            coarse_fitness=float(coarse.fitness),
            coarse_rmse=float(coarse.inlier_rmse),
            refine_fitness=float(refine.fitness),
            refine_rmse=float(refine.inlier_rmse),
            residual_T=np.asarray(refine.transformation, dtype=np.float64),
        )
        if final_src == "refine":
            if float(refine.fitness) < float(self.args.icp_fine_fitness_thr):
                if bool(self.args.keep_coarse_track_on_refine_fail):
                    self._last_T_source_to_target = np.asarray(coarse.transformation, dtype=np.float64).copy()
                    print("[fallback] refine 失败，使用 coarse 结果作为下一帧 warm start")
                raise RuntimeError(
                    f"SKIP: ICP refine fitness 过低: {float(refine.fitness):.4f} < {float(self.args.icp_fine_fitness_thr):.4f}"
                )
            if float(refine.inlier_rmse) > float(self.args.icp_rmse_thr):
                if bool(self.args.keep_coarse_track_on_refine_fail):
                    self._last_T_source_to_target = np.asarray(coarse.transformation, dtype=np.float64).copy()
                    print("[fallback] refine rmse 失败，使用 coarse 结果作为下一帧 warm start")
                raise RuntimeError(
                    f"SKIP: ICP refine rmse 过高: {float(refine.inlier_rmse):.6f} > {float(self.args.icp_rmse_thr):.6f}"
                )
        elif final_src == "coarse":
            if (float(coarse.fitness) < float(self.args.icp_stage1_fitness_thr)) or (
                float(coarse.inlier_rmse) > float(self.args.icp_stage1_rmse_thr)
            ):
                raise RuntimeError(
                    "SKIP: refine 回退到 coarse 后质量仍不满足阈值"
                )
            print(f"[icp][fallback] final_pose_source=coarse reasons={' | '.join(fb_reasons) if fb_reasons else 'none'}")

        T_source_to_region = np.asarray(T_final, dtype=np.float64)
        self._last_icp_pose_source = str(final_src)
        print(f"[icp][quality] final_pose_source={self._last_icp_pose_source}")
        self._last_T_source_to_target = T_source_to_region.copy()
        self._consecutive_icp_failures = 0
        T_region_cam = np.linalg.inv(T_source_to_region)  # region(local template) -> camera
        T_region_to_grasp = np.asarray(self.T_region_to_grasp, dtype=np.float64)
        T_grasp_cam = T_region_cam @ T_region_to_grasp
        T_grasp_base = np.asarray(T_base_cam, dtype=np.float64) @ T_grasp_cam

        pregrasp_xyz = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        T_base_pregrasp = T_grasp_base @ _make_T_from_xyz_m(pregrasp_xyz)
        T_base_grasp = T_grasp_base @ _make_T_from_xyz_m(grasp_xyz)

        print("[grasp-ref] T_region_to_camera=" + np.array2string(T_region_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_region_to_grasp=" + np.array2string(T_region_to_grasp, precision=5, suppress_small=True))
        print("[grasp-ref] T_grasp_to_camera=" + np.array2string(T_grasp_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_grasp_to_base=" + np.array2string(T_grasp_base, precision=5, suppress_small=True))
        print("[grasp-ref] T_base_pregrasp=" + np.array2string(T_base_pregrasp, precision=5, suppress_small=True))
        print("[grasp-ref] T_base_grasp=" + np.array2string(T_base_grasp, precision=5, suppress_small=True))
        return T_base_pregrasp, T_base_grasp, T_region_cam, T_grasp_cam, T_grasp_base

    def _T_to_robot_pose_mm_deg(self, T: np.ndarray) -> np.ndarray:
        xyz_m = np.asarray(T[:3, 3], dtype=np.float64)
        r, p, y = _rot_to_euler_zyx_deg(np.asarray(T[:3, :3], dtype=np.float64))
        xyz_mm = xyz_m * 1000.0
        return np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], r, p, y], dtype=np.float64)

    def execute_robot_plan(
        self,
        T_base_pregrasp: np.ndarray,
        T_base_grasp: np.ndarray,
        *,
        T_region_cam: Optional[np.ndarray] = None,
        T_grasp_cam: Optional[np.ndarray] = None,
        T_grasp_base: Optional[np.ndarray] = None,
    ):
        pose_pre_raw = self._T_to_robot_pose_mm_deg(T_base_pregrasp)
        pose_grasp_raw = self._T_to_robot_pose_mm_deg(T_base_grasp)
        pose_pre = np.asarray(pose_pre_raw, dtype=np.float64).copy()
        pose_grasp = np.asarray(pose_grasp_raw, dtype=np.float64).copy()

        use_fixed_rpy = bool(self.args.debug_use_fixed_rpy)
        fixed_rpy = None
        if use_fixed_rpy:
            fixed_rpy = _parse_csv_floats(self.args.debug_fixed_rpy, 3)
            pose_pre[3:] = fixed_rpy
            pose_grasp[3:] = fixed_rpy
            print("[robot][debug] 固定姿态 IK 诊断模式已开启（保持 xyz，强制替换 rpy）")
            print(f"[robot][debug] debug_fixed_rpy(deg)={fixed_rpy.tolist()}")

        if T_region_cam is not None:
            print("[robot][debug] T_region_to_camera=" + np.array2string(np.asarray(T_region_cam), precision=5, suppress_small=True))
        print("[robot][debug] T_region_to_grasp=" + np.array2string(np.asarray(self.T_region_to_grasp), precision=5, suppress_small=True))
        if T_grasp_cam is not None:
            print("[robot][debug] T_grasp_to_camera=" + np.array2string(np.asarray(T_grasp_cam), precision=5, suppress_small=True))
        if T_grasp_base is not None:
            print("[robot][debug] T_grasp_to_base=" + np.array2string(np.asarray(T_grasp_base), precision=5, suppress_small=True))
        print(f"[robot][debug] raw_pregrasp(mm,deg)={pose_pre_raw.tolist()}")
        print(f"[robot][debug] raw_grasp(mm,deg)={pose_grasp_raw.tolist()}")
        if use_fixed_rpy:
            print(f"[robot][debug] fixed_rpy_pregrasp(mm,deg)={pose_pre.tolist()}")
            print(f"[robot][debug] fixed_rpy_grasp(mm,deg)={pose_grasp.tolist()}")
        print(f"[robot] pregrasp(mm,deg)={pose_pre.tolist()}")
        print(f"[robot] grasp(mm,deg)={pose_grasp.tolist()}")

        if self.executor is None:
            print("[robot] auto_execute=False，仅输出位姿")
            return
        # 统一改为：笛卡尔 -> 逆解关节 -> 关节 MovJ，避免笛卡尔轨迹在某些分支触发关节限位。
        try:
            j_pre = self.executor.solve_ik_to_joint_deg(pose_pre)
            j_grasp = self.executor.solve_ik_to_joint_deg(pose_grasp)
        except Exception:
            if use_fixed_rpy:
                print("[robot][diag] 固定姿态 IK 仍失败：当前位置本身可能不可达（或 user/tool/TCP 设置不一致）")
            raise

        if use_fixed_rpy:
            print("[robot][diag] 固定姿态 IK 成功：当前位置大概率可达，问题主要在原始抓取姿态定义")
        print(f"[robot] pregrasp_joint(deg)={j_pre.tolist()}")
        print(f"[robot] grasp_joint(deg)={j_grasp.tolist()}")
        self.executor.movj_joint(j_pre, "pregrasp_movj_joint")
        self.executor.movj_joint(j_grasp, "grasp_movj_joint")
        print("[robot] 执行完成")

    # ---------------------------------------------------------------- #
    #  ICP 稳定定位 + 候选法兰姿态试 IK                                    #
    # ---------------------------------------------------------------- #

    def _format_pose_mm_deg(self, pose_mm_deg: np.ndarray) -> str:
        p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
        return f"[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f},{p[3]:.2f},{p[4]:.2f},{p[5]:.2f}]"

    def _format_pose_full_precision(self, pose_mm_deg: np.ndarray) -> str:
        p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
        return f"[{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{p[3]:.6f},{p[4]:.6f},{p[5]:.6f}]"

    def _get_fixed_template_rpy(self) -> np.ndarray:
        if self._fixed_template_rpy_cache is not None:
            return np.asarray(self._fixed_template_rpy_cache, dtype=np.float64).reshape(3)
        source = str(getattr(self.args, "fixed_template_source", "cli")).strip().lower()
        if source == "cli":
            rpy = _parse_csv_floats(str(self.args.fixed_template_rpy), 3)
        elif source == "current_robot_pose":
            if self.executor is None:
                raise RuntimeError("runtime模板姿态来源为 current_robot_pose，但当前无可用机器人连接")
            pose_now = self.executor.get_current_pose_mm_deg()
            rpy = np.asarray(pose_now[3:6], dtype=np.float64)
        elif source == "json":
            cfg_path = str(getattr(self.args, "fixed_template_json", "") or "").strip()
            if not cfg_path:
                raise ValueError("fixed_template_source=json 时必须提供 --fixed-template-json")
            data = json.loads(Path(cfg_path).expanduser().resolve().read_text(encoding="utf-8"))
            raw = data.get("fixed_template_rpy", data.get("template_rpy", data.get("rpy", None)))
            if raw is None:
                raise KeyError("fixed-template json 未找到 fixed_template_rpy/template_rpy/rpy")
            rpy = np.asarray(raw, dtype=np.float64).reshape(-1)
            if rpy.size != 3:
                raise ValueError(f"fixed-template json rpy 维度错误: {rpy}")
        else:
            raise ValueError(f"未知 fixed_template_source: {source}")
        self._fixed_template_rpy_cache = np.asarray(rpy, dtype=np.float64).reshape(3)
        print(f"[ik-template] source={source}")
        print(f"[ik-template] rpy={self._fixed_template_rpy_cache.tolist()}")
        return np.asarray(self._fixed_template_rpy_cache, dtype=np.float64).reshape(3)

    def _build_fixed_template_pose(self, grasp_pos_base_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rpy = self._get_fixed_template_rpy()
        grasp_pos_m = np.asarray(grasp_pos_base_mm, dtype=np.float64).reshape(3) / 1000.0
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _euler_zyx_deg_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        T[:3, 3] = grasp_pos_m
        pregrasp_xyz_m = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz_m = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        T_pre = T @ _make_T_from_xyz_m(pregrasp_xyz_m)
        T_grasp = T @ _make_T_from_xyz_m(grasp_xyz_m)
        pose_pre = self._T_to_robot_pose_mm_deg(T_pre)
        pose_grasp = self._T_to_robot_pose_mm_deg(T_grasp)
        return pose_pre, pose_grasp

    def _run_runtime_probe_and_execute(self, pose_pre: np.ndarray, pose_grasp: np.ndarray) -> bool:
        print(f"[runtime][pre] rounded={self._format_pose_mm_deg(pose_pre)}")
        print(f"[runtime][pre] full={self._format_pose_full_precision(pose_pre)}")
        probe_pre = self._probe_pose_reachability_by_motion(pose_pre, "runtime_pregrasp")
        print(
            f"[runtime][probe] stage=pre ok={probe_pre.get('ok', None)} "
            f"ran={probe_pre.get('ran', False)} mode={probe_pre.get('mode', '')}"
        )
        if (not bool(probe_pre.get("ran", False))) or (not bool(probe_pre.get("ok", False))):
            print("[runtime][probe] pregrasp probe failed, skip current frame")
            return False
        do_grasp_probe = bool(getattr(self.args, "runtime_probe_grasp", False))
        if do_grasp_probe:
            print(f"[runtime][grasp] rounded={self._format_pose_mm_deg(pose_grasp)}")
            print(f"[runtime][grasp] full={self._format_pose_full_precision(pose_grasp)}")
            probe_grasp = self._probe_pose_reachability_by_motion(pose_grasp, "runtime_grasp")
            print(
                f"[runtime][probe] stage=grasp ok={probe_grasp.get('ok', None)} "
                f"ran={probe_grasp.get('ran', False)} mode={probe_grasp.get('mode', '')}"
            )
            if (not bool(probe_grasp.get("ran", False))) or (not bool(probe_grasp.get("ok", False))):
                print("[runtime][probe] grasp probe failed, skip current frame")
                return False
        if self.executor is None:
            print("[runtime][exec] auto_execute=False，仅输出位姿")
            return True
        if not bool(self.args.auto_execute):
            print("[runtime][exec] auto_execute=False，仅输出位姿")
            return True
        if bool(self.args.prefer_api_ik_joint_execution):
            pre_api = self._ik_try_pose_with_reason(pose_pre)
            grasp_api = self._ik_try_pose_with_reason(pose_grasp)
            if bool(pre_api.get("ok", False)) and bool(grasp_api.get("ok", False)):
                print("[runtime][exec] exec_mode=joint_solution")
                self.executor.movj_joint(np.asarray(pre_api["joint"], dtype=np.float64), "runtime_pregrasp_movj_joint")
                self.executor.movj_joint(np.asarray(grasp_api["joint"], dtype=np.float64), "runtime_grasp_movj_joint")
                print("[runtime][exec] done")
                return True
        print("[runtime][exec] exec_mode=direct_pose_fallback")
        self.executor.movj_pose(np.asarray(pose_pre, dtype=np.float64), "runtime_pregrasp_movj_pose")
        self.executor.movj_pose(np.asarray(pose_grasp, dtype=np.float64), "runtime_grasp_movj_pose")
        print("[runtime][exec] done")
        return True

    def _ik_try_pose_with_reason(self, pose_mm_deg: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "joint": None, "err": "", "err_code": None, "raw": ""}
        if self.executor is None:
            out["err"] = "no_executor"
            return out
        try:
            j = self.executor.solve_ik_to_joint_deg(np.asarray(pose_mm_deg, dtype=np.float64))
            out["ok"] = True
            out["joint"] = j
            return out
        except Exception as e:
            msg = str(e)
            out["raw"] = msg
            out["err"] = msg
            m = re.search(r"ErrorID=([-]?\d+)", msg)
            if m:
                out["err_code"] = int(m.group(1))
            return out

    def _probe_pose_reachability_by_motion(self, pose_mm_deg: np.ndarray, stage_name: str) -> dict[str, Any]:
        """
        运动等效探测：
        - 默认 dry-run（不实际运动），避免危险动作
        - 若开启 --motion-probe-execute，执行一次保守探测并回传成功/失败
        """
        out: dict[str, Any] = {
            "ran": False,
            "ok": None,
            "mode": str(self.args.motion_probe_command),
            "err": "",
            "reason": "probe_not_executed",
        }
        if self.executor is None:
            out["reason"] = "no_executor"
            out["err"] = "no_executor"
            return out
        if not bool(self.args.motion_probe_execute):
            print(f"[ik-check][motion] dry-run only, skip actual probe for {stage_name}")
            return out
        try:
            if str(self.args.motion_probe_command) == "movl_pose":
                self.executor.movl_pose(np.asarray(pose_mm_deg, dtype=np.float64), f"probe_{stage_name}_movl")
            else:
                self.executor.movj_pose(np.asarray(pose_mm_deg, dtype=np.float64), f"probe_{stage_name}_movj")
            out["ran"] = True
            out["ok"] = True
            out["reason"] = "probe_success"
            return out
        except Exception as e:
            out["ran"] = True
            out["ok"] = False
            out["reason"] = "probe_failed"
            out["err"] = str(e)
            return out

    def _ik_check_pose(self, pose_mm_deg: np.ndarray, stage_name: str) -> dict[str, Any]:
        """
        统一可达性检查入口：将 API 逆解检查与 motion probe 分离并结构化输出。
        """
        mode = str(self.args.ik_check_mode)
        out: dict[str, Any] = {
            "mode": mode,
            "api_ok": False,
            "motion_probe_ok": None,
            "motion_probe_ran": False,
            "final_ok": False,
            "joint": None,
            "error_code": None,
            "raw": "",
            "pose_mm_deg": np.asarray(pose_mm_deg, dtype=np.float64).reshape(6).tolist(),
            "reason": "",
        }
        print(f"[pose][rounded] {self._format_pose_mm_deg(pose_mm_deg)}")
        print(f"[pose][full]    {self._format_pose_full_precision(pose_mm_deg)}")

        api_ret = None
        if mode in ("api_only", "api_then_motion_probe"):
            api_ret = self._ik_try_pose_with_reason(pose_mm_deg)
            out["api_ok"] = bool(api_ret["ok"])
            out["joint"] = api_ret["joint"]
            out["error_code"] = api_ret.get("err_code", None)
            out["raw"] = str(api_ret.get("raw", api_ret.get("err", "")))
            if out["api_ok"]:
                out["final_ok"] = True
                out["reason"] = "api_ok"
                print(f"[ik-check][api] OK stage={stage_name}")
                return out
            print(f"[ik-check][api] FAIL stage={stage_name} code={out['error_code']} raw={out['raw'][:120]}")
            if mode == "api_only":
                out["final_ok"] = False
                out["reason"] = "api_fail"
                return out

        probe_ret = self._probe_pose_reachability_by_motion(pose_mm_deg, stage_name)
        out["motion_probe_ran"] = bool(probe_ret.get("ran", False))
        out["motion_probe_ok"] = probe_ret.get("ok", None)
        if probe_ret.get("ran", False):
            print(
                f"[ik-check][motion] stage={stage_name} "
                f"ok={probe_ret.get('ok', None)} mode={probe_ret.get('mode', '')} err={str(probe_ret.get('err', ''))[:120]}"
            )
        if bool(out["motion_probe_ok"]) and bool(self.args.allow_motion_probe_success_as_reachable):
            out["final_ok"] = True
            out["reason"] = "api_failed_but_motion_ok"
            print("[check-mismatch] API check failed but motion probe succeeded")
            print("[check-mismatch] InverseSolution result may be unreliable for this pose")
            print("[check-mismatch] do not directly classify this candidate as unreachable")
            return out
        out["final_ok"] = False
        out["reason"] = "motion_probe_failed" if probe_ret.get("ran", False) else "probe_not_run"
        return out

    def _dedup_rpy_candidates(self, cands: list[dict[str, Any]], thr_deg: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in cands:
            rpy = np.asarray(c["rpy"], dtype=np.float64)
            keep = True
            for e in out:
                r2 = np.asarray(e["rpy"], dtype=np.float64)
                if float(np.linalg.norm(rpy - r2)) <= float(thr_deg):
                    keep = False
                    break
            if keep:
                out.append(c)
        return out

    def _expand_ik_rpy_candidates(
        self,
        seed_rpys: list[list[float]],
        visual_rpy: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        seeds: list[dict[str, Any]] = []
        for i, r in enumerate(seed_rpys):
            seeds.append({"source": "seed", "parent_idx": i, "delta": [0.0, 0.0, 0.0], "rpy": [float(r[0]), float(r[1]), float(r[2])]})
            print(f"[ik-cand][seed] #{i} rpy={r}")
        if not seeds and visual_rpy is not None:
            vr = np.asarray(visual_rpy, dtype=np.float64).reshape(3)
            seeds.append({"source": "vision_seed", "parent_idx": -1, "delta": [0.0, 0.0, 0.0], "rpy": vr.tolist()})
            print(f"[ik-cand][seed] vision rpy={vr.tolist()}")
        if not bool(self.args.ik_expand_enable):
            return seeds

        d_roll = _parse_csv_float_list(self.args.ik_expand_roll_deltas)
        d_pitch = _parse_csv_float_list(self.args.ik_expand_pitch_deltas)
        d_yaw = _parse_csv_float_list(self.args.ik_expand_yaw_deltas)
        expanded: list[dict[str, Any]] = []
        for s in seeds:
            base = np.asarray(s["rpy"], dtype=np.float64)
            for dr in d_roll:
                for dp in d_pitch:
                    for dy in d_yaw:
                        rpy = (base + np.asarray([dr, dp, dy], dtype=np.float64)).tolist()
                        expanded.append(
                            {
                                "source": "expand",
                                "parent_idx": int(s["parent_idx"]),
                                "delta": [float(dr), float(dp), float(dy)],
                                "rpy": rpy,
                            }
                        )
        all_cands = seeds + expanded
        dedup = self._dedup_rpy_candidates(all_cands, thr_deg=float(self.args.ik_expand_dedup_thr_deg))
        max_n = max(1, int(self.args.ik_expand_max_candidates))
        dedup = dedup[:max_n]
        print(
            f"[ik-cand][expand] enable={bool(self.args.ik_expand_enable)} "
            f"seed={len(seeds)} expanded={len(expanded)} dedup={len(dedup)} max={max_n}"
        )
        return dedup

    def _sort_ik_candidate_priority(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def _key(c: dict[str, Any]):
            source_rank = 0 if str(c.get("source", "")) in ("seed", "vision_seed") else 1
            delta = np.asarray(c.get("delta", [0.0, 0.0, 0.0]), dtype=np.float64)
            delta_norm = float(np.linalg.norm(delta))
            return (source_rank, delta_norm)

        return sorted(candidates, key=_key)

    def _run_fixed_rpy_reachability_debug(self, grasp_pos_base_mm: np.ndarray) -> dict[str, Any]:
        if not bool(self.args.ik_debug_fixed_rpy_enable):
            return {"enabled": False, "ran": False, "ok": False, "err": "disabled"}
        fixed_rpy = _parse_csv_floats(self.args.ik_debug_fixed_rpy, 3)
        grasp_pos_m = np.asarray(grasp_pos_base_mm, dtype=np.float64) / 1000.0
        pregrasp_xyz_m = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _euler_zyx_deg_to_rot(fixed_rpy[0], fixed_rpy[1], fixed_rpy[2])
        T[:3, 3] = grasp_pos_m
        T_pre = T @ _make_T_from_xyz_m(pregrasp_xyz_m)
        pose_pre = self._T_to_robot_pose_mm_deg(T_pre)
        print(f"[ik-debug][fixed-rpy] rpy={fixed_rpy.tolist()} pregrasp_pose={self._format_pose_mm_deg(pose_pre)}")
        print(f"[ik-debug][fixed-rpy] pregrasp_pose_full={self._format_pose_full_precision(pose_pre)}")
        ret = self._ik_check_pose(pose_pre, "fixed_rpy_pregrasp")
        ok = bool(ret["final_ok"])
        if ok:
            print("[ik-debug][result] SUCCESS: 固定 rpy 可达，问题更可能来自候选姿态定义过窄")
        else:
            print("[ik-debug][result] FAIL: 固定 rpy 仍不可达，当前位置或 user/tool/TCP 设置可能异常")
        return {"enabled": True, "ran": True, "ok": ok, "detail": ret, "fixed_rpy": fixed_rpy.tolist()}

    def _estimate_workspace_risk(self, grasp_pos_base_mm: np.ndarray) -> str:
        p = np.asarray(grasp_pos_base_mm, dtype=np.float64).reshape(3)
        xy = float(np.linalg.norm(p[:2]))
        z = float(p[2])
        if xy > 850.0 or z < -50.0 or z > 900.0:
            return "high"
        if xy > 750.0 or z < 50.0 or z > 800.0:
            return "medium"
        return "low"

    def _summarize_ik_attempts(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(attempts)
        ok = sum(1 for a in attempts if bool(a.get("ok", False)))
        seed_total = sum(1 for a in attempts if str(a.get("source", "")) in ("seed", "vision_seed"))
        seed_ok = sum(1 for a in attempts if str(a.get("source", "")) in ("seed", "vision_seed") and bool(a.get("ok", False)))
        expand_total = sum(1 for a in attempts if str(a.get("source", "")) == "expand")
        expand_ok = sum(1 for a in attempts if str(a.get("source", "")) == "expand" and bool(a.get("ok", False)))
        return {
            "total": total,
            "ok": ok,
            "seed_total": seed_total,
            "seed_ok": seed_ok,
            "expand_total": expand_total,
            "expand_ok": expand_ok,
        }

    def _summarize_reachability_check(self, attempts: list[dict[str, Any]]) -> dict[str, int]:
        out = {
            "both_ok": 0,
            "pregrasp_failed_only": 0,
            "grasp_failed_only": 0,
            "both_failed": 0,
            "api_failed_but_motion_ok": 0,
        }
        for a in attempts:
            pre_ok = bool(a.get("pre_final_ok", False))
            grasp_ok = bool(a.get("grasp_final_ok", False))
            if pre_ok and grasp_ok:
                out["both_ok"] += 1
            elif (not pre_ok) and grasp_ok:
                out["pregrasp_failed_only"] += 1
            elif pre_ok and (not grasp_ok):
                out["grasp_failed_only"] += 1
            else:
                out["both_failed"] += 1
            pre_reason = str(a.get("pre_reason", ""))
            grasp_reason = str(a.get("grasp_reason", ""))
            if ("api_failed_but_motion_ok" in pre_reason) or ("api_failed_but_motion_ok" in grasp_reason):
                out["api_failed_but_motion_ok"] += 1
        return out

    def _diagnose_api_vs_motion_mismatch(self, attempts: list[dict[str, Any]]) -> str:
        s = self._summarize_reachability_check(attempts)
        if int(s["api_failed_but_motion_ok"]) > 0:
            return "api_inverse_solution_false_negative"
        if int(s["pregrasp_failed_only"]) > 0 and int(s["grasp_failed_only"]) == 0:
            return "pregrasp_only_unreachable"
        if int(s["grasp_failed_only"]) > 0 and int(s["pregrasp_failed_only"]) == 0:
            return "grasp_only_unreachable"
        if int(s["both_failed"]) > 0:
            return "true_pose_unreachable"
        return "mixed_check_results"

    def _diagnose_ik_failure(
        self,
        attempts: list[dict[str, Any]],
        grasp_pos_base_mm: np.ndarray,
        pregrasp_pos_base_mm: np.ndarray,
        fixed_debug: dict[str, Any],
    ) -> None:
        stats = self._summarize_ik_attempts(attempts)
        reach_stats = self._summarize_reachability_check(attempts)
        workspace_risk = self._estimate_workspace_risk(grasp_pos_base_mm)
        g_xy = float(np.linalg.norm(np.asarray(grasp_pos_base_mm, dtype=np.float64)[:2]))
        p_xy = float(np.linalg.norm(np.asarray(pregrasp_pos_base_mm, dtype=np.float64)[:2]))
        farther = "pregrasp" if p_xy > g_xy else "grasp"
        likely_reason = self._diagnose_api_vs_motion_mismatch(attempts)
        suggestion = "增加候选姿态或检查 TCP/user/tool。"
        if fixed_debug.get("ran", False) and bool(fixed_debug.get("ok", False)):
            if int(stats["seed_ok"]) == 0 and int(stats["expand_ok"]) > 0:
                likely_reason = "candidate_rpy_too_narrow"
                suggestion = "原始候选库过窄，建议补充示教姿态并保留自动扩增。"
            else:
                likely_reason = "candidate_rpy_too_narrow"
                suggestion = "固定 rpy 可达，建议优化候选姿态库。"
        elif fixed_debug.get("ran", False) and (not bool(fixed_debug.get("ok", False))):
            likely_reason = "position_out_of_workspace" if workspace_risk != "low" else "tcp_or_user_tool_mismatch"
            suggestion = "检查抓取点位置是否越界，并核对 user/tool/TCP 标定。"
        if self._last_icp_pose_source == "refine":
            q = self._last_icp_quality or {}
            if float(q.get("refine_fitness", 1.0)) < float(q.get("coarse_fitness", 1.0)) * 0.6:
                likely_reason = "refine_pose_degraded"
                suggestion = "refine 相对 coarse 明显恶化，建议启用/收紧 refine 回退参数。"
        print(f"[ik-diagnose] grasp_xyz_mm={np.asarray(grasp_pos_base_mm, dtype=np.float64).tolist()}")
        print(f"[ik-diagnose] pregrasp_xyz_mm={np.asarray(pregrasp_pos_base_mm, dtype=np.float64).tolist()}")
        print(f"[ik-diagnose] workspace_risk={workspace_risk} farther_point={farther} pre_xy={p_xy:.1f} grasp_xy={g_xy:.1f}")
        print(
            f"[ik-diagnose] candidate_count={stats['total']} "
            f"seed_ok={stats['seed_ok']}/{stats['seed_total']} "
            f"expand_ok={stats['expand_ok']}/{stats['expand_total']}"
        )
        print(f"[ik-diagnose] fixed_rpy_debug={'SUCCESS' if bool(fixed_debug.get('ok', False)) else 'FAIL'}")
        print(
            f"[reachability] both_ok={reach_stats['both_ok']} "
            f"pregrasp_failed_only={reach_stats['pregrasp_failed_only']} "
            f"grasp_failed_only={reach_stats['grasp_failed_only']} "
            f"both_failed={reach_stats['both_failed']} "
            f"api_failed_but_motion_ok={reach_stats['api_failed_but_motion_ok']}"
        )
        print(f"[ik-diagnose] final_pose_source={self._last_icp_pose_source}")
        if self._last_icp_quality:
            print(f"[ik-diagnose] icp_quality={json.dumps(self._last_icp_quality, ensure_ascii=False)}")
        print(
            f"[ik-diagnose] config user={int(getattr(self.args, 'robot_user', 0))} "
            f"tool={int(getattr(self.args, 'robot_tool', 0))}"
        )
        print(f"[ik-diagnose] likely_reason={likely_reason}")
        print(f"[ik-diagnose] suggestion={suggestion}")

    def _try_all_candidate_ik(self, grasp_pos_base_mm: np.ndarray, visual_grasp_rpy_deg: Optional[np.ndarray] = None) -> list:
        """
        候选姿态试 IK（支持自动扩增与结构化结果记录）。
        """
        pregrasp_xyz_m = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz_m = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        grasp_pos_m = np.asarray(grasp_pos_base_mm, dtype=np.float64) / 1000.0
        pool = self._expand_ik_rpy_candidates(self._ik_candidates, visual_rpy=visual_grasp_rpy_deg)
        pool = self._sort_ik_candidate_priority(pool)
        results: list[dict[str, Any]] = []
        for i, c in enumerate(pool):
            rpy = np.asarray(c["rpy"], dtype=np.float64).reshape(3)
            T_cand = np.eye(4, dtype=np.float64)
            T_cand[:3, :3] = _euler_zyx_deg_to_rot(rpy[0], rpy[1], rpy[2])
            T_cand[:3, 3] = grasp_pos_m
            T_pre = T_cand @ _make_T_from_xyz_m(pregrasp_xyz_m)
            T_g = T_cand @ _make_T_from_xyz_m(grasp_xyz_m)
            pose_pre = self._T_to_robot_pose_mm_deg(T_pre)
            pose_grasp = self._T_to_robot_pose_mm_deg(T_g)
            pre_check = self._ik_check_pose(pose_pre, f"cand_{i}_pregrasp")
            grasp_check = self._ik_check_pose(pose_grasp, f"cand_{i}_grasp")
            pre_ok = bool(pre_check["final_ok"])
            grasp_ok = bool(grasp_check["final_ok"])
            ok = pre_ok and grasp_ok
            j_pre = pre_check.get("joint", None)
            j_grasp = grasp_check.get("joint", None)
            exec_mode = "joint_solution" if (j_pre is not None and j_grasp is not None) else "direct_pose_fallback"
            if pre_ok and grasp_ok:
                final_reason = "both_ok"
            elif (not pre_ok) and grasp_ok:
                final_reason = "pregrasp_failed_only"
            elif pre_ok and (not grasp_ok):
                final_reason = "grasp_failed_only"
            else:
                if ("api_failed_but_motion_ok" in str(pre_check.get("reason", ""))) or (
                    "api_failed_but_motion_ok" in str(grasp_check.get("reason", ""))
                ):
                    final_reason = "api_failed_but_motion_ok"
                else:
                    final_reason = "both_failed"
            r: dict[str, Any] = {
                "idx": i,
                "ok": ok,
                "source": str(c.get("source", "seed")),
                "parent_idx": int(c.get("parent_idx", -1)),
                "delta": list(c.get("delta", [0.0, 0.0, 0.0])),
                "rpy": rpy.tolist(),
                "pose_pre": pose_pre,
                "pose_grasp": pose_grasp,
                "j_pre": j_pre,
                "j_grasp": j_grasp,
                "pre_api_ok": bool(pre_check.get("api_ok", False)),
                "pre_motion_probe_ok": pre_check.get("motion_probe_ok", None),
                "pre_final_ok": pre_ok,
                "pre_reason": str(pre_check.get("reason", "")),
                "grasp_api_ok": bool(grasp_check.get("api_ok", False)),
                "grasp_motion_probe_ok": grasp_check.get("motion_probe_ok", None),
                "grasp_final_ok": grasp_ok,
                "grasp_reason": str(grasp_check.get("reason", "")),
                "err": "" if ok else (str(pre_check.get("raw", "")) or str(grasp_check.get("raw", ""))),
                "err_code": pre_check.get("error_code", grasp_check.get("error_code", None)),
                "exec_mode": exec_mode,
                "final_reason": final_reason,
            }
            status = "OK" if ok else f"FAIL({str(r['err'])[:80]})"
            print(
                f"[ik-cand][pre] #{i} api_ok={r['pre_api_ok']} motion_ok={r['pre_motion_probe_ok']} "
                f"final_ok={r['pre_final_ok']} reason={r['pre_reason']}"
            )
            print(
                f"[ik-cand][grasp] #{i} api_ok={r['grasp_api_ok']} motion_ok={r['grasp_motion_probe_ok']} "
                f"final_ok={r['grasp_final_ok']} reason={r['grasp_reason']}"
            )
            print(
                f"[ik-cand][summary] #{i} src={r['source']} parent={r['parent_idx']} delta={r['delta']} "
                f"rpy={r['rpy']} pose_pre={self._format_pose_mm_deg(pose_pre)} "
                f"pose_grasp={self._format_pose_mm_deg(pose_grasp)} final_reason={r['final_reason']} -> {status}"
            )
            results.append(r)
        return results

    def _update_stability(self, grasp_pos_base_mm: np.ndarray, candidate_idx: int, T_grasp_base: np.ndarray) -> bool:
        """
        稳定帧投票：记录最近 N 帧的抓取点位置和选中候选编号。
        当连续 N 帧位置偏差 < 阈值、姿态变化 < 阈值且候选一致时返回 True。
        """
        rpy = np.asarray(_rot_to_euler_zyx_deg(np.asarray(T_grasp_base[:3, :3], dtype=np.float64)), dtype=np.float64)
        self._stable_history.append({
            "pos_mm": np.asarray(grasp_pos_base_mm, dtype=np.float64).copy(),
            "cand_idx": int(candidate_idx),
            "rpy_deg": rpy,
        })
        n = self._stable_frames_required
        if len(self._stable_history) > n * 3:
            self._stable_history = self._stable_history[-n:]
        if len(self._stable_history) < n:
            return False
        recent = self._stable_history[-n:]
        positions = np.array([h["pos_mm"] for h in recent])
        ref = positions[-1]
        max_dev = float(np.max(np.linalg.norm(positions - ref, axis=1)))
        if max_dev > self._stable_pos_thr_mm:
            print(f"[stable] 位置偏差 {max_dev:.2f}mm > 阈值 {self._stable_pos_thr_mm:.2f}mm")
            return False
        rpys = np.array([h["rpy_deg"] for h in recent], dtype=np.float64)
        rpy_ref = rpys[-1]
        max_rot_dev = float(np.max(np.linalg.norm(rpys - rpy_ref, axis=1)))
        if max_rot_dev > self._stable_rot_thr_deg:
            print(f"[stable] 姿态偏差 {max_rot_dev:.2f}deg > 阈值 {self._stable_rot_thr_deg:.2f}deg")
            return False
        cands = set(h["cand_idx"] for h in recent)
        if len(cands) > 1:
            print(f"[stable] 候选编号不一致: {cands}")
            return False
        return True

    def _reset_stability(self):
        """清空稳定帧历史（ICP 失败/候选全不可达时调用）。"""
        self._stable_history.clear()

    def _save_handle_target_sample(self, pcd) -> bool:
        """
        调试能力：导出单帧局部抓取区域点云样本（preprocess 后）。
        """
        pts = int(len(pcd.points))
        min_pts = int(self.args.handle_target_sample_min_points)
        if pts < min_pts:
            print(f"[sample] 当前帧点云点数不足，跳过保存: points={pts} < min={min_pts}")
            return False
        out = Path(self.args.handle_target_sample_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(self.o3d.io.write_point_cloud(str(out), pcd, write_ascii=False, compressed=False, print_progress=False))
        if not ok:
            raise RuntimeError(f"写入点云失败: {out}")
        print(f"[sample] 已保存局部抓取区域点云样本: {out}")
        return True

    # ---------------------------------------------------------------- #
    #  FoundationPose bridge（双环境解耦）                                 #
    # ---------------------------------------------------------------- #

    def _init_fp_bridge(self, args) -> None:
        """初始化 FoundationPose 子进程桥接所需配置。"""
        self.T_region_in_obj: np.ndarray = _load_region_in_obj(args)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._fp_bridge_dir = str(getattr(args, "fp_bridge_dir", "") or "").strip()
        if not self._fp_bridge_dir:
            self._fp_bridge_dir = os.path.abspath(
                os.path.join(script_dir, "../third_party/runtime_bridge")
            )
        self._fp_script = str(getattr(args, "fp_script_path", "") or "").strip()
        if not self._fp_script:
            self._fp_script = os.path.abspath(os.path.join(
                script_dir,
                "../third_party/FoundationPose/mecheye_foundationpose_grasp_pipeline.py",
            ))
        self._fp_conda_env = str(getattr(args, "fp_conda_env", "foundationpose")).strip()
        self._fp_mesh_file = str(getattr(args, "fp_mesh_file", "") or "").strip()
        if not self._fp_mesh_file:
            raise ValueError("FoundationPose 模式需要 --fp-mesh-file 参数")
        self._fp_code_dir = str(getattr(args, "fp_code_dir", "") or "").strip()
        if not self._fp_code_dir:
            self._fp_code_dir = os.path.abspath(
                os.path.join(script_dir, "../FoundationPose")
            )
        self._fp_est_refine_iter = int(getattr(args, "fp_est_refine_iter", 5))
        self._fp_min_n_views = int(getattr(args, "fp_min_n_views", 20))
        self._fp_inplane_step = int(getattr(args, "fp_inplane_step", 120))
        self._fp_timeout = int(getattr(args, "fp_timeout", 120))
        self._fp_debug = int(getattr(args, "fp_debug", 0))
        self._fp_debug_dir = str(getattr(args, "fp_debug_dir", "/tmp/fp_debug"))

        os.makedirs(os.path.join(self._fp_bridge_dir, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(self._fp_bridge_dir, "outputs"), exist_ok=True)

        print(f"[fp-bridge] pose_backend=foundationpose")
        print(f"[fp-bridge] bridge_dir={self._fp_bridge_dir}")
        print(f"[fp-bridge] conda_env={self._fp_conda_env}")
        print(f"[fp-bridge] mesh={self._fp_mesh_file}")
        print(f"[fp-bridge] script={self._fp_script}")

    def _estimate_pose_via_fp_bridge(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        mask_pc: np.ndarray,
    ) -> np.ndarray:
        """
        写入 bridge/inputs → 调用 FoundationPose 子进程 → 读取 bridge/outputs。
        返回 T_obj_cam (4×4)。
        """
        inputs_dir = os.path.join(self._fp_bridge_dir, "inputs")
        outputs_dir = os.path.join(self._fp_bridge_dir, "outputs")
        request_id = f"frame_{self._frame_idx}"

        # ---- 写入输入 ----
        cv2.imwrite(os.path.join(inputs_dir, "color.png"), color_bgr)
        depth_u16_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(inputs_dir, "depth.png"), depth_u16_mm)
        cv2.imwrite(os.path.join(inputs_dir, "mask.png"), mask_pc)
        meta = {
            "K": self._K_3x3.tolist(),
            "depth_scale_to_m": 0.001,
            "image_height": color_bgr.shape[0],
            "image_width": color_bgr.shape[1],
            "mesh_file": self._fp_mesh_file,
            "est_refine_iter": self._fp_est_refine_iter,
            "fp_min_n_views": self._fp_min_n_views,
            "fp_inplane_step": self._fp_inplane_step,
            "fp_debug": self._fp_debug,
            "fp_debug_dir": self._fp_debug_dir,
            "request_id": request_id,
            "timestamp": time.time(),
        }
        with open(os.path.join(inputs_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # ---- 调用子进程 ----
        cmd = [
            "conda", "run", "-n", self._fp_conda_env, "--no-capture-output",
            "python", self._fp_script,
            "--bridge-dir", self._fp_bridge_dir,
            "--fp-code-dir", self._fp_code_dir,
        ]
        print(f"[fp-bridge] 调用子进程 (conda_env={self._fp_conda_env}) ...")
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._fp_timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"SKIP: FoundationPose 子进程超时 ({self._fp_timeout}s)"
            )
        dt_sub = (time.time() - t0) * 1000.0

        if proc.stdout.strip():
            print(f"[fp-bridge] stdout: {proc.stdout.strip()}")
        if proc.returncode != 0:
            err_tail = (proc.stderr or "").strip()[-500:]
            raise RuntimeError(
                f"SKIP: FoundationPose 子进程失败 "
                f"(rc={proc.returncode}): {err_tail}"
            )

        # ---- 读取输出 ----
        result_path = os.path.join(outputs_dir, "pose_result.json")
        if not os.path.exists(result_path):
            raise RuntimeError("SKIP: pose_result.json 不存在")
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") != "ok":
            raise RuntimeError(
                f"SKIP: FoundationPose 估计失败: {result.get('error', 'unknown')}"
            )

        T_obj_cam = np.array(
            result["T_obj_to_camera"], dtype=np.float64
        ).reshape(4, 4)
        score = float(result.get("score", -1))
        fp_ms = float(result.get("elapsed_ms", 0))
        print(
            f"[fp-bridge] 位姿接收成功  score={score:.4f}  "
            f"fp_elapsed={fp_ms:.0f}ms  subprocess={dt_sub:.0f}ms"
        )
        print(
            "[fp-bridge] T_obj_cam=\n"
            + np.array2string(T_obj_cam, precision=5, suppress_small=True)
        )
        return T_obj_cam

    def _compute_grasp_from_fp_pose(
        self,
        T_obj_cam: np.ndarray,
        T_base_cam: np.ndarray,
    ):
        """
        将 FoundationPose 的整物体位姿转换为抓取位姿（链路与 ICP 版后半段一致）。

        T_region_cam  = T_obj_cam  @ T_region_in_obj
        T_grasp_cam   = T_region_cam @ T_region_to_grasp
        T_grasp_base  = T_base_cam  @ T_grasp_cam
        """
        T_region_in_obj = np.asarray(self.T_region_in_obj, dtype=np.float64)
        T_region_cam = T_obj_cam @ T_region_in_obj
        T_region_to_grasp = np.asarray(self.T_region_to_grasp, dtype=np.float64)
        T_grasp_cam = T_region_cam @ T_region_to_grasp
        T_grasp_base = np.asarray(T_base_cam, dtype=np.float64) @ T_grasp_cam

        pregrasp_xyz = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        T_base_pregrasp = T_grasp_base @ _make_T_from_xyz_m(pregrasp_xyz)
        T_base_grasp = T_grasp_base @ _make_T_from_xyz_m(grasp_xyz)

        print("[grasp-ref] T_obj_to_camera=" + np.array2string(T_obj_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_region_in_obj=" + np.array2string(T_region_in_obj, precision=5, suppress_small=True))
        print("[grasp-ref] T_region_to_camera=" + np.array2string(T_region_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_region_to_grasp=" + np.array2string(T_region_to_grasp, precision=5, suppress_small=True))
        print("[grasp-ref] T_grasp_to_camera=" + np.array2string(T_grasp_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_grasp_to_base=" + np.array2string(T_grasp_base, precision=5, suppress_small=True))
        print("[grasp-ref] T_base_pregrasp=" + np.array2string(T_base_pregrasp, precision=5, suppress_small=True))
        print("[grasp-ref] T_base_grasp=" + np.array2string(T_base_grasp, precision=5, suppress_small=True))
        return T_base_pregrasp, T_base_grasp, T_region_cam, T_grasp_cam, T_grasp_base

    def _depth_raw_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        """将 Mech-Eye 原始深度转为米制 float32（全图，不裁剪）。"""
        depth_u16 = live_utils._depth_to_png_u16(depth_raw)
        return live_utils._depth_u16_to_m(depth_u16, unit=str(self.args.depth_unit))

    def loop(self):
        frame_all = live_utils.Frame2DAnd3D()
        self.move_robot_to_recognition_pose()
        print("在线闭环启动：q 退出（若 --auto-execute 开启则自动执行机械臂）")
        while True:
            self._frame_idx += 1
            st = self.camera.capture_2d_and_3d(frame_all)
            if not st.is_ok():
                live_utils.show_error(st)
                continue

            color_sdk = frame_all.frame_2d().get_color_image()
            depth = frame_all.frame_3d().get_depth_map()
            color_bgr = color_sdk.data()

            t0 = time.time()
            try:
                mask_pc = self.infer_mask_from_frame(color_bgr)

                if self._pose_backend == "foundationpose":
                    # ---- FoundationPose bridge 路径 ----
                    depth_m = self._depth_raw_to_meters(depth.data())
                    T_obj_cam = self._estimate_pose_via_fp_bridge(
                        color_bgr, depth_m, mask_pc
                    )
                    T_base_cam = self._compute_T_base_cam_dynamic()
                    T_base_pre, T_base_grasp, T_region_cam, T_grasp_cam, T_grasp_base = (
                        self._compute_grasp_from_fp_pose(T_obj_cam, T_base_cam)
                    )
                    self.execute_robot_plan(
                        T_base_pre,
                        T_base_grasp,
                        T_region_cam=T_region_cam,
                        T_grasp_cam=T_grasp_cam,
                        T_grasp_base=T_grasp_base,
                    )
                    if bool(self.args.print_pose_json):
                        payload = {
                            "T_base_to_camera": np.asarray(T_base_cam).tolist(),
                            "T_cam_to_flange": None if self.T_cam_to_flange is None else np.asarray(self.T_cam_to_flange).tolist(),
                            "T_obj_to_camera": np.asarray(T_obj_cam).tolist(),
                            "T_region_in_obj": np.asarray(self.T_region_in_obj).tolist(),
                            "T_region_to_camera": np.asarray(T_region_cam).tolist(),
                            "T_region_to_grasp": np.asarray(self.T_region_to_grasp).tolist(),
                            "T_grasp_to_camera": np.asarray(T_grasp_cam).tolist(),
                            "T_grasp_to_base": np.asarray(T_grasp_base).tolist(),
                            "T_base_pregrasp": np.asarray(T_base_pre).tolist(),
                            "T_base_grasp": np.asarray(T_base_grasp).tolist(),
                        }
                        print(json.dumps(payload, ensure_ascii=False))
                else:
                    # ---- ICP 路径（原有逻辑不变） ----
                    pcd = self.build_target_pointcloud(depth, mask_pc, color_bgr)
                    pcd = self.preprocess_target_pointcloud(pcd)
                    if len(pcd.points) < int(self.args.min_points):
                        print(f"[skip] 目标点云不足，points={len(pcd.points)}")
                    else:
                        if bool(self.args.handle_target_sample):
                            if self._save_handle_target_sample(pcd):
                                print("[sample] 导出完成，结束程序。")
                                break
                            continue

                        fused_source, fused_frames = self._update_and_fuse_source(pcd)
                        T_base_cam = self._compute_T_base_cam_dynamic()
                        T_base_pre, T_base_grasp, T_region_cam, T_grasp_cam, T_grasp_base = self.estimate_pose_and_grasp(
                            fused_source,
                            T_base_cam,
                            fused_frames=fused_frames,
                        )

                        best = None
                        runtime_done = False
                        runtime_mode = str(getattr(self.args, "ik_runtime_mode", "multi_rpy_search")).strip().lower()
                        ik_mode = bool(self._ik_candidates) or bool(self.args.ik_expand_enable)
                        if runtime_mode == "fixed_template_probe_execute":
                            grasp_pos_base_mm = np.asarray(T_grasp_base[:3, 3], dtype=np.float64) * 1000.0
                            if not self._update_stability(grasp_pos_base_mm, -2, T_grasp_base):
                                n_hist = len(self._stable_history)
                                print(f"[stable] 等待连续稳定帧 ({n_hist}/{self._stable_frames_required})")
                            else:
                                print(f"[stable] 连续 {self._stable_frames_required} 帧稳定，进入运行模式执行")
                                pose_pre_runtime, pose_grasp_runtime = self._build_fixed_template_pose(grasp_pos_base_mm)
                                if self._run_runtime_probe_and_execute(pose_pre_runtime, pose_grasp_runtime):
                                    self._reset_stability()
                                    runtime_done = True
                                else:
                                    print("[runtime][exec] probe或执行失败，重置稳定缓存并回到视觉循环")
                                    self._reset_stability()
                        elif ik_mode:
                            # ---- 新模式：ICP 稳定定位 + 候选法兰姿态试 IK ----
                            grasp_pos_base_mm = np.asarray(T_grasp_base[:3, 3], dtype=np.float64) * 1000.0
                            pregrasp_pos_base_mm = np.asarray(T_base_pre[:3, 3], dtype=np.float64) * 1000.0
                            print(
                                f"[vision] 抓取点位置(base,mm): "
                                f"x={grasp_pos_base_mm[0]:.2f} "
                                f"y={grasp_pos_base_mm[1]:.2f} "
                                f"z={grasp_pos_base_mm[2]:.2f}"
                            )
                            visual_grasp_rpy = np.asarray(
                                _rot_to_euler_zyx_deg(np.asarray(T_grasp_base[:3, :3], dtype=np.float64)),
                                dtype=np.float64,
                            )
                            ik_results = self._try_all_candidate_ik(grasp_pos_base_mm, visual_grasp_rpy_deg=visual_grasp_rpy)
                            ok_results = [r for r in ik_results if bool(r["ok"])]
                            if ok_results:
                                best = ok_results[0]
                                print(
                                    f"[ik-cand][best] idx={best['idx']} src={best['source']} "
                                    f"parent={best['parent_idx']} delta={best['delta']} rpy={best['rpy']}"
                                )
                            fixed_debug = {"enabled": bool(self.args.ik_debug_fixed_rpy_enable), "ran": False, "ok": False}
                            if bool(self.args.ik_debug_fixed_rpy_enable):
                                if (not bool(self.args.ik_debug_run_on_fail_only)) or (best is None):
                                    fixed_debug = self._run_fixed_rpy_reachability_debug(grasp_pos_base_mm)
                            if best is None:
                                print(
                                    "[ik-cand] 视觉定位成功，"
                                    "但当前抓取点在所有候选法兰姿态下均不可达"
                                )
                                self._diagnose_ik_failure(
                                    ik_results,
                                    grasp_pos_base_mm,
                                    pregrasp_pos_base_mm,
                                    fixed_debug,
                                )
                                self._reset_stability()
                            elif not self._update_stability(grasp_pos_base_mm, best["idx"], T_grasp_base):
                                n_hist = len(self._stable_history)
                                print(
                                    f"[stable] 等待连续稳定帧 "
                                    f"({n_hist}/{self._stable_frames_required})"
                                )
                            else:
                                print(
                                    f"[stable] 连续 {self._stable_frames_required} 帧稳定，"
                                    f"选中候选 #{best['idx']} "
                                    f"rpy={best['rpy']}"
                                )
                                print(f"[robot] pregrasp(mm,deg)={best['pose_pre'].tolist()}")
                                print(f"[robot] grasp(mm,deg)={best['pose_grasp'].tolist()}")
                                print(f"[pose][rounded] {self._format_pose_mm_deg(np.asarray(best['pose_pre'], dtype=np.float64))}")
                                print(f"[pose][full]    {self._format_pose_full_precision(np.asarray(best['pose_pre'], dtype=np.float64))}")
                                print(f"[pose][rounded] {self._format_pose_mm_deg(np.asarray(best['pose_grasp'], dtype=np.float64))}")
                                print(f"[pose][full]    {self._format_pose_full_precision(np.asarray(best['pose_grasp'], dtype=np.float64))}")
                                if best.get("j_pre", None) is not None and best.get("j_grasp", None) is not None:
                                    print(f"[robot] pregrasp_joint(deg)={best['j_pre'].tolist()}")
                                    print(f"[robot] grasp_joint(deg)={best['j_grasp'].tolist()}")
                                if self.executor is not None and bool(self.args.auto_execute):
                                    has_joint = (best.get("j_pre", None) is not None) and (best.get("j_grasp", None) is not None)
                                    if has_joint and bool(self.args.prefer_api_ik_joint_execution):
                                        print("[robot] exec_mode=joint_solution")
                                        self.executor.movj_joint(best["j_pre"], "pregrasp_movj_joint")
                                        self.executor.movj_joint(best["j_grasp"], "grasp_movj_joint")
                                    elif bool(self.args.direct_pose_fallback_enable):
                                        print("[robot] exec_mode=direct_pose_fallback")
                                        self.executor.movj_pose(np.asarray(best["pose_pre"], dtype=np.float64), "pregrasp_movj_pose_fallback")
                                        self.executor.movj_pose(np.asarray(best["pose_grasp"], dtype=np.float64), "grasp_movj_pose_fallback")
                                    else:
                                        print(
                                            "[robot] 候选可达但无关节解可执行（可能是 api_failed_but_motion_ok），"
                                            "请开启 --direct-pose-fallback-enable"
                                        )
                                    print("[robot] 候选姿态模式执行完成")
                                else:
                                    print("[robot] auto_execute=False，仅输出位姿")
                                self._reset_stability()
                        else:
                            # ---- 旧模式：ICP 结果直接输出完整位姿 ----
                            grasp_pos_base_mm = np.asarray(T_grasp_base[:3, 3], dtype=np.float64) * 1000.0
                            if not self._update_stability(grasp_pos_base_mm, -1, T_grasp_base):
                                n_hist = len(self._stable_history)
                                print(f"[stable] 等待连续稳定帧 ({n_hist}/{self._stable_frames_required})")
                            else:
                                self.execute_robot_plan(
                                    T_base_pre,
                                    T_base_grasp,
                                    T_region_cam=T_region_cam,
                                    T_grasp_cam=T_grasp_cam,
                                    T_grasp_base=T_grasp_base,
                                )
                                self._reset_stability()

                        if bool(self.args.print_pose_json):
                            payload = {
                                "T_base_to_camera": np.asarray(T_base_cam).tolist(),
                                "T_cam_to_flange": None if self.T_cam_to_flange is None else np.asarray(self.T_cam_to_flange).tolist(),
                                "T_region_to_camera": np.asarray(T_region_cam).tolist(),
                                "T_region_to_grasp": np.asarray(self.T_region_to_grasp).tolist(),
                                "T_grasp_to_camera": np.asarray(T_grasp_cam).tolist(),
                                "T_grasp_to_base": np.asarray(T_grasp_base).tolist(),
                                "T_base_pregrasp": np.asarray(T_base_pre).tolist(),
                                "T_base_grasp": np.asarray(T_base_grasp).tolist(),
                                "icp_final_pose_source": str(self._last_icp_pose_source),
                                "icp_quality": dict(self._last_icp_quality),
                            }
                            if runtime_mode == "fixed_template_probe_execute":
                                payload["ik_runtime_mode"] = "fixed_template_probe_execute"
                                payload["fixed_template_rpy"] = self._get_fixed_template_rpy().tolist()
                            elif ik_mode and best is not None:
                                payload["candidate_idx"] = best["idx"]
                                payload["candidate_rpy"] = best["rpy"]
                                payload["candidate_source"] = best.get("source", "seed")
                                payload["candidate_exec_mode"] = best.get("exec_mode", "")
                                payload["candidate_final_reason"] = best.get("final_reason", "")
                                payload["candidate_pose_pre"] = best["pose_pre"].tolist()
                                payload["candidate_pose_grasp"] = best["pose_grasp"].tolist()
                            print(json.dumps(payload, ensure_ascii=False))
                        if runtime_mode == "fixed_template_probe_execute" and runtime_done:
                            print("[runtime][done] 运行模式执行成功，结束当前任务流程")
                            break
            except Exception as e:
                msg = str(e)
                if msg.startswith("SKIP:"):
                    print(f"[skip] {msg[5:].strip()}")
                else:
                    print(f"[pipeline] 本帧失败: {type(e).__name__}: {e}")
                if self._pose_backend == "icp":
                    self._consecutive_icp_failures += 1
                    self._reset_stability()

            if not bool(self.args.no_gui):
                vis = live_utils._overlay_binary_mask(color_bgr, mask_pc if "mask_pc" in locals() else np.zeros(color_bgr.shape[:2], np.uint8))
                dt_ms = (time.time() - t0) * 1000.0
                cv2.putText(vis, f"loop {dt_ms:.1f} ms", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(str(self.args.win), vis)
                key = cv2.waitKey(int(self.args.wait)) & 0xFF
                if key == ord("q"):
                    break
            else:
                if int(self.args.max_loops) > 0:
                    self.args.max_loops -= 1
                    if int(self.args.max_loops) <= 0:
                        break

    def close(self):
        try:
            self.camera.disconnect()
        except Exception:
            pass
        if self.executor is not None:
            self.executor.close()
        if not bool(self.args.no_gui):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def build_argparser():
    p = argparse.ArgumentParser(description="Mech-Eye 在线抓取闭环：分割->目标点云->ICP->抓取位姿->机械臂执行")
    p.add_argument("--model-family", choices=["pointrend", "mask2former"], default="mask2former")
    p.add_argument("--config-file", default=live_utils._DEFAULT_POINTREND_CONFIG)
    p.add_argument("--config-file-prior", default=live_utils._DEFAULT_MASK2FORMER_QSP_CONFIG)
    p.add_argument("--mask2former-root", default=live_utils._DEFAULT_MASK2FORMER_ROOT)
    p.add_argument("--weights-prior", required=True, help="prior/QSP 权重")
    p.add_argument("--shape-prior-npy", default="")
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-classes", type=int, default=1)

    p.add_argument("--mask-mode", choices=["union", "maxscore"], default="union")
    p.add_argument("--pc-mask-mode", choices=["union", "maxscore", "iou"], default="iou")
    p.add_argument("--pc-iou-thresh", type=float, default=0.1)
    p.add_argument("--pc-join-dilate", type=int, default=25)
    p.add_argument("--mask-close", type=int, default=5)
    p.add_argument("--mask-dilate", type=int, default=0)
    p.add_argument("--mask-erode", type=int, default=0)
    p.add_argument("--invert-mask", action="store_true")
    p.add_argument("--auto-invert-mask", action="store_true")

    p.add_argument("--ip", default="", help="Mech-Eye 相机 IP")
    p.add_argument("--serial", default="", help="Mech-Eye 序列号")
    p.add_argument("--index", type=int, default=-1, help="discover index")
    p.add_argument("--exposure-seq", default="5,10")
    p.add_argument("--pc-smoothing", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-noise", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-outlier", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-edge", choices=["", "sharp", "normal", "smooth"], default="normal")
    p.add_argument("--save-userset", action="store_true")

    p.add_argument("--cleargrasp", action="store_true")
    p.add_argument("--cleargrasp-normals-weights", default="")
    p.add_argument("--cleargrasp-outlines-weights", default="")
    p.add_argument("--cleargrasp-depth2depth-exe", default="")
    p.add_argument("--cleargrasp-out-w", type=int, default=256)
    p.add_argument("--cleargrasp-out-h", type=int, default=144)
    p.add_argument("--cleargrasp-inertia", type=float, default=1000.0)
    p.add_argument("--cleargrasp-smoothness", type=float, default=0.0001)
    p.add_argument("--cleargrasp-tangent", type=float, default=1.0)
    p.add_argument("--cleargrasp-fill-thresh", type=float, default=0.0)
    p.add_argument("--cleargrasp-filter-d", type=int, default=0)
    p.add_argument("--cleargrasp-filter-sigma-color", type=float, default=5.0)
    p.add_argument("--cleargrasp-filter-sigma-space", type=float, default=10.0)
    p.add_argument("--depth-unit", choices=["mm", "m"], default="mm")

    p.add_argument("--pc-stride", type=int, default=1)
    p.add_argument("--min-points", type=int, default=500)
    p.add_argument("--pp-voxel", type=float, default=0.0, help="目标点云预处理体素大小（米）")
    p.add_argument("--pp-sor-nb", type=int, default=50)
    p.add_argument("--pp-sor-std", type=float, default=1.0)
    p.add_argument("--pp-ror-nb", type=int, default=0)
    p.add_argument("--pp-ror-radius", type=float, default=0.0)
    p.add_argument("--pp-dbscan-eps", type=float, default=0.006, help="米")
    p.add_argument("--pp-dbscan-min-points", type=int, default=30)
    p.add_argument("--pp-keep-top-k", type=int, default=1)
    p.add_argument(
        "--pp-cluster-select",
        choices=["largest", "closest_z", "farthest_z", "smallest_bbox", "largest_bbox"],
        default="largest",
    )

    p.add_argument("--cad-ply", default="", help="局部抓取区域 CAD 模板点云（ICP 模式必填）")
    p.add_argument(
        "--cad-is-grasp-region-template",
        action="store_true",
        help="兼容旧参数：当前版本始终将 --cad-ply 视为局部模板，此开关可保留但不再影响逻辑。",
    )
    p.add_argument(
        "--grasp-region-preserve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="局部模板模式下保留抓取区域点云结构（默认 true，避免 DBSCAN largest-cluster 破坏局部几何）",
    )
    p.add_argument(
        "--grasp-region-init-json",
        default="",
        help="局部模板配准初始位姿 json（T_source_to_target/T_init/T_source_to_grasp_region_cad）",
    )
    p.add_argument(
        "--grasp-region-init-T",
        nargs=16,
        type=float,
        default=None,
        help="局部模板配准初始位姿 T_source_to_target（16个数，行优先）",
    )
    p.add_argument(
        "--grasp-region-to-grasp-json",
        default="",
        help="固定抓取参考变换 json；按当前脚本记号右乘使用：T_grasp_cam = T_region_cam @ T_region_to_grasp",
    )
    p.add_argument(
        "--T-grasp-region-to-grasp",
        nargs=16,
        type=float,
        default=None,
        help="固定抓取参考变换 T_region_to_grasp（16个数，行优先）",
    )
    p.add_argument("--icp-voxel", type=float, default=0.003, help="米（融合后 source 与局部模板 target 的 ICP 体素）")
    p.add_argument("--ransac-mult", type=float, default=1.5)
    p.add_argument("--icp-mult", type=float, default=0.7)
    p.add_argument(
        "--online-init-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用在线粗初始化候选（center/PCA/flip）",
    )
    p.add_argument("--init-candidate-max", type=int, default=12, help="在线初始化候选最大数量")
    p.add_argument("--coarse-icp-dist-mult", type=float, default=1.8, help="coarse ICP 对应距离系数（相对 icp_voxel）")
    p.add_argument("--refine-icp-dist-mult", type=float, default=0.7, help="refine ICP 对应距离系数（相对 icp_voxel）")
    p.add_argument(
        "--refine-fallback-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="当 refine 相对 coarse 明显劣化时，是否回退使用 coarse 结果",
    )
    p.add_argument("--refine-min-fitness-ratio", type=float, default=0.55, help="触发回退阈值：refine_fitness/coarse_fitness 下限")
    p.add_argument("--refine-max-residual-trans-mm", type=float, default=30.0, help="触发回退阈值：refine 残差平移上限（mm）")
    p.add_argument("--refine-max-residual-rot-deg", type=float, default=35.0, help="触发回退阈值：refine 残差旋转上限（deg）")
    p.add_argument(
        "--keep-coarse-track-on-refine-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refine 失败时是否保留 coarse 结果作为下一帧 warm start",
    )
    p.add_argument("--max-ransac", type=int, default=100000)
    p.add_argument("--max-icp", type=int, default=80, help="兼容参数：等价 stage2 最大迭代次数")
    p.add_argument("--max-icp-stage1", type=int, default=40, help="两阶段 ICP：stage1(point-to-point) 最大迭代")
    p.add_argument("--max-icp-stage2", type=int, default=80, help="两阶段 ICP：stage2(point-to-plane) 最大迭代")
    p.add_argument("--icp-stage1-mult", type=float, default=1.8, help="stage1 对应距离系数（相对 icp_dist）")
    p.add_argument("--icp-stage1-fitness-thr", type=float, default=0.20, help="stage1(point-to-point) fitness 下限")
    p.add_argument("--icp-stage1-rmse-thr", type=float, default=0.004, help="stage1(point-to-point) rmse 上限（米）")
    p.add_argument("--icp-coarse-fitness-thr", type=float, default=0.05, help="coarse_fitness 失败阈值，低于则跳过本帧")
    p.add_argument("--icp-fine-fitness-thr", type=float, default=0.15, help="fine_fitness 失败阈值，低于则跳过本帧")
    p.add_argument("--icp-rmse-thr", type=float, default=0.003, help="ICP inlier_rmse 失败阈值（米），高于则跳过本帧")
    p.add_argument("--bbox-ratio-min", type=float, default=0.60, help="source/target bbox extent 比例最小阈值（收紧默认）")
    p.add_argument("--bbox-ratio-max", type=float, default=1.60, help="source/target bbox extent 比例最大阈值（收紧默认）")
    p.add_argument("--surface-thin-enable", action=argparse.BooleanOptionalAction, default=True, help="是否启用局部点云主曲面提纯")
    p.add_argument("--surface-thin-band-mm", type=float, default=12.0, help="主曲面厚度带宽（mm）")
    p.add_argument("--surface-thin-axis", choices=["auto", "x", "y", "z"], default="auto", help="主曲面厚度方向（默认 auto）")
    p.add_argument("--surface-thin-min-points", type=int, default=500, help="主曲面提纯后最少点数")
    p.add_argument("--fusion-frames", type=int, default=6, help="局部点云融合缓存帧数 N")
    p.add_argument("--fusion-min-valid-frames", type=int, default=3, help="进入 ICP 前最少有效融合帧数")
    p.add_argument("--fusion-voxel", type=float, default=0.002, help="融合点云体素（米）")
    p.add_argument("--fused-min-points", type=int, default=600, help="融合后最少点数，低于则跳过本帧")
    p.add_argument(
        "--fusion-after-coarse-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="开启后先 coarse 对齐再写入融合缓存（缓解相机位姿变化导致的融合糊化）",
    )
    p.add_argument("--vis-icp", action="store_true", help="可视化 ICP 配准叠加结果（Open3D 阻塞窗口）")
    p.add_argument("--vis-icp-every", type=int, default=1, help="每 N 帧显示一次 ICP 叠加（默认每帧）")
    p.add_argument(
        "--vis-icp-only-fail",
        action="store_true",
        help="仅当 fine_fitness 低于阈值时显示 ICP 叠加（配合 --vis-icp-fail-thr）",
    )
    p.add_argument("--vis-icp-fail-thr", type=float, default=0.2, help="ICP 失败可视化阈值：fine_fitness <= 该值")

    p.add_argument(
        "--eye-in-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否按 Eye-in-Hand 动态计算 T_base_cam(t)。默认 true。",
    )
    p.add_argument(
        "--cam-flange-json",
        default="",
        help="Eye-in-Hand: T_cam_to_flange 4x4 json（相机坐标系 -> 法兰中心坐标系）；不填则使用脚本内置本次标定矩阵",
    )
    p.add_argument(
        "--T-cam-flange",
        nargs=16,
        type=float,
        default=None,
        help="Eye-in-Hand: 直接输入 T_cam_to_flange 4x4（16个数，行优先）",
    )
    p.add_argument("--base-cam-json", default="", help="Eye-to-Hand: 固定 T_base_to_camera 4x4 json")
    p.add_argument("--T-base-cam", nargs=16, type=float, default=None, help="Eye-to-Hand: 直接输入 T_base_to_camera 4x4")
    p.add_argument("--pregrasp-offset-mm", default="0,0,80")
    p.add_argument("--grasp-offset-mm", default="0,0,20")
    p.add_argument("--print-pose-json", action="store_true")

    p.add_argument("--auto-execute", action="store_true", help="自动下发机械臂动作")
    p.add_argument("--robot-ip", default="192.168.5.2")
    p.add_argument("--robot-user", type=int, default=0)
    p.add_argument("--robot-tool", type=int, default=0)
    p.add_argument("--robot-a", type=int, default=20)
    p.add_argument("--robot-v", type=int, default=20)
    p.add_argument("--robot-cp", type=int, default=0)
    p.add_argument(
        "--gripper-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否接入夹爪控制。开启后在到达 p1 时初始化并张开夹爪（保持张开，不自动闭合）。",
    )
    p.add_argument("--gripper-port", default="/dev/ttyUSB0", help="夹爪串口")
    p.add_argument("--gripper-baudrate", type=int, default=115200, help="夹爪串口波特率")
    p.add_argument("--gripper-open-position", type=int, default=900, help="夹爪张开位置")
    p.add_argument("--gripper-init-timeout", type=float, default=5.0, help="夹爪初始化超时（秒）")
    p.add_argument("--debug-use-fixed-rpy", action="store_true", help="IK 诊断：保持 xyz 不变，临时替换为固定 rpy 再做逆解")
    p.add_argument("--debug-fixed-rpy", default="64.5293,79.9632,1.9803", help="固定姿态角 rx,ry,rz（deg，逗号分隔）")
    p.add_argument(
        "--ik-candidate-rpy-json",
        default="",
        help="候选法兰姿态 JSON 文件（包含 candidates/ik_candidates/rpy_list 键，每项 [rx,ry,rz] deg）",
    )
    p.add_argument(
        "--ik-candidate-rpy-list",
        default="",
        help="候选法兰姿态列表（分号分隔多组，逗号分隔 rx,ry,rz；例 '64.5,80.0,2.0;-118.0,-2.8,8.6'）",
    )
    p.add_argument(
        "--ik-candidate-source-type",
        choices=["euler_rpy", "joint_guess"],
        default="euler_rpy",
        help="候选姿态来源类型：euler_rpy=TCP姿态角(rx,ry,rz)；joint_guess(J4/J5/J6)将被拒绝。",
    )
    p.add_argument(
        "--ik-runtime-mode",
        choices=["multi_rpy_search", "fixed_template_probe_execute"],
        default="multi_rpy_search",
        help="IK运行模式：multi_rpy_search(调试) / fixed_template_probe_execute(运行)。",
    )
    p.add_argument(
        "--fixed-template-rpy",
        default="64.5293,79.9632,1.9803",
        help="固定模板TCP姿态角 RX/RY/RZ（deg，运行模式使用）。",
    )
    p.add_argument(
        "--fixed-template-source",
        choices=["cli", "current_robot_pose", "json"],
        default="cli",
        help="固定模板姿态来源：命令行/当前机器人姿态/json。",
    )
    p.add_argument(
        "--fixed-template-json",
        default="",
        help="fixed-template-source=json 时使用，键支持 fixed_template_rpy/template_rpy/rpy。",
    )
    p.add_argument(
        "--runtime-probe-grasp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="运行模式下是否对 grasp 也做 motion probe（默认仅 probe pregrasp）。",
    )
    p.add_argument(
        "--ik-check-mode",
        choices=["api_only", "api_then_motion_probe", "motion_probe_only"],
        default="api_only",
        help="IK可达性检查模式：仅API / API失败后运动探测 / 仅运动探测。",
    )
    p.add_argument(
        "--motion-probe-execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否真正执行运动探测（默认 false，为安全仅dry-run）。",
    )
    p.add_argument(
        "--motion-probe-command",
        choices=["movj_pose", "movl_pose"],
        default="movj_pose",
        help="运动探测命令类型。",
    )
    p.add_argument(
        "--allow-motion-probe-success-as-reachable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="API失败但运动探测成功时，是否允许判定为可达。",
    )
    p.add_argument(
        "--prefer-api-ik-joint-execution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="有IK关节解时优先使用 movj_joint 执行。",
    )
    p.add_argument(
        "--direct-pose-fallback-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="无关节解但运动探测成功时，允许回退到笛卡尔直达执行。",
    )
    p.add_argument(
        "--ik-expand-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在 seed 候选姿态周围自动扩增搜索",
    )
    p.add_argument("--ik-expand-roll-deltas", default="-15,0,15", help="IK 自动扩增：roll 扰动集合（deg，逗号分隔）")
    p.add_argument("--ik-expand-pitch-deltas", default="-15,0,15", help="IK 自动扩增：pitch 扰动集合（deg，逗号分隔）")
    p.add_argument("--ik-expand-yaw-deltas", default="-30,0,30", help="IK 自动扩增：yaw 扰动集合（deg，逗号分隔）")
    p.add_argument("--ik-expand-max-candidates", type=int, default=64, help="IK 自动扩增后最大候选数")
    p.add_argument("--ik-expand-dedup-thr-deg", type=float, default=2.0, help="IK 扩增候选去重阈值（deg）")
    p.add_argument(
        "--ik-debug-fixed-rpy-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用固定 rpy 可达性诊断",
    )
    p.add_argument("--ik-debug-fixed-rpy", default="64.5293,79.9632,1.9803", help="固定 rpy 诊断姿态（deg，rx,ry,rz）")
    p.add_argument(
        "--ik-debug-run-on-fail-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅当候选 IK 全失败时运行固定 rpy 诊断",
    )
    p.add_argument("--stable-frames", type=int, default=3, help="候选 IK 模式：连续多少帧稳定后才执行抓取")
    p.add_argument("--stable-pos-thr-mm", type=float, default=5.0, help="稳定帧位置偏差阈值（mm）")
    p.add_argument("--stable-rot-thr-deg", type=float, default=10.0, help="稳定帧姿态偏差阈值（deg）")
    p.add_argument("--icp-fallback-after", type=int, default=5, help="ICP 连续失败多少帧后回退到人工初值")
    p.add_argument("--plan-before-recog", action="store_true", help="在识别前先执行一次机械臂规划到固定识别位")
    p.add_argument("--p1", default="-19.1160,26.4384,-122.4939,71.5494,85.4363,4.4136", help="识别前固定关节位（6轴角度，逗号分隔）")
    p.add_argument("--handle-target-sample", action="store_true", help="调试模式：保存一帧局部抓取区域点云样本后退出（不进入ICP/机器人执行）")
    p.add_argument(
        "--handle-target-sample-path",
        default="/home/user/sjw/workspace/tmp/live_grasp_region_sample.ply",
        help="局部抓取区域样本点云输出路径（默认 /home/user/sjw/workspace/tmp/live_grasp_region_sample.ply）",
    )
    p.add_argument("--handle-target-sample-min-points", type=int, default=300, help="样本保存最少点数阈值")
    p.add_argument(
        "--source",
        default="",
        help="兼容参数：等价于 --handle-target-sample-path（可配合 --handle-target-sample 使用）",
    )

    p.add_argument("--win", default="Online Grasp Pipeline")
    p.add_argument("--wait", type=int, default=1)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--max-loops", type=int, default=0, help="仅 no-gui 时生效，>0 则循环指定次数后退出")

    # ---- 位姿估计后端选择（双环境解耦） ----
    p.add_argument(
        "--pose-backend",
        choices=["icp", "foundationpose"],
        default="icp",
        help="位姿估计后端：icp(默认，原有 ICP 链路) / foundationpose(子进程桥接)",
    )
    p.add_argument("--fp-conda-env", default="foundationpose", help="FoundationPose conda 环境名称")
    p.add_argument("--fp-bridge-dir", default="", help="runtime_bridge 目录（默认自动推断）")
    p.add_argument("--fp-script-path", default="", help="FP 桥接脚本路径（默认自动推断）")
    p.add_argument("--fp-code-dir", default="", help="FoundationPose 代码根目录（默认自动推断）")
    p.add_argument("--fp-mesh-file", default="", help="CAD 网格文件 (.obj/.stl)，FoundationPose 模式必填")
    p.add_argument("--fp-est-refine-iter", type=int, default=5, help="FoundationPose register 迭代次数")
    p.add_argument("--fp-min-n-views", type=int, default=20, help="FoundationPose 初始旋转网格视角数（减小可降显存）")
    p.add_argument("--fp-inplane-step", type=int, default=120, help="FoundationPose 初始旋转网格面内角步长（增大可降显存）")
    p.add_argument("--fp-timeout", type=int, default=120, help="子进程超时秒数")
    p.add_argument("--fp-debug", type=int, default=0, help="FoundationPose 内部 debug 级别")
    p.add_argument("--fp-debug-dir", default="/tmp/fp_debug", help="FP debug 输出目录")
    p.add_argument("--fp-region-in-obj-json", default="", help="T_region_in_obj JSON (整物体系→局部模板系)")
    p.add_argument("--T-fp-region-in-obj", nargs=16, type=float, default=None, help="T_region_in_obj 4×4（16 数，行优先）")

    return p


def main():
    args = build_argparser().parse_args()
    # 兼容旧参数 --max-icp：若用户未显式设置 stage2，则沿用 max-icp
    if int(args.max_icp_stage2) == 80 and int(args.max_icp) != 80:
        args.max_icp_stage2 = int(args.max_icp)
    if str(args.source).strip():
        args.handle_target_sample_path = str(args.source).strip()
        if not bool(args.handle_target_sample):
            args.handle_target_sample = True
    pipeline = OnlineGraspPipeline(args)
    try:
        pipeline.loop()
    finally:
        pipeline.close()
        print("pipeline stopped")


if __name__ == "__main__":
    main()

