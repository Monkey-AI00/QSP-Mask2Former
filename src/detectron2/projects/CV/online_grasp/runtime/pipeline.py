"""Runtime pipeline module (incremental migrated orchestration)."""

from __future__ import annotations

import json
import numpy as np
import cv2
import time
from pathlib import Path

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils
from online_grasp.grasp.ik_search import IKSearchRunner
from online_grasp.grasp.stability import StabilityChecker
from online_grasp.geometry.calibration import _load_cam_to_flange, _load_fixed_base_to_camera, _load_region_to_grasp, _load_T_from_json
from online_grasp.geometry.transforms import _as_T_4x4
from online_grasp.perception.depth_completion import DepthCompleter
from online_grasp.perception.pointcloud_builder import PointCloudBuilder
from online_grasp.perception.segmentor import Segmentor, _load_predictor
from online_grasp.geometry.transforms import _parse_csv_floats, _rot_to_euler_xyz_deg
from online_grasp.pose.foundationpose_backend import FoundationPoseBackend
from online_grasp.pose.icp_backend import ICPBackend
from online_grasp.robot.dobot_executor import DobotPoseExecutor
from online_grasp.robot.gripper import (
    close_gripper_after_grasp,
    format_gripper_status,
    handle_gripper_hotkey,
    maybe_print_gripper_stroke,
    wait_grip2_then_back_to_p1,
)


class OnlineGraspPipeline:
    """Online grasp runtime pipeline (fully split from legacy inheritance)."""

    def __init__(self, args):
        self.args = args
        try:
            import open3d as o3d  # type: ignore
        except Exception as e:
            raise RuntimeError(f"未找到 open3d，请先安装: pip install open3d; 原始错误: {e}")
        self.o3d = o3d
        self.predictor = _load_predictor(args)
        self.eye_in_hand = bool(args.eye_in_hand)
        self.T_cam_to_flange = None
        self.T_base_cam_fixed = None
        self.executor = None
        self.depthcomplete = None
        self._frame_idx = 0
        self.T_region_to_grasp = _load_region_to_grasp(args)
        self._pose_backend = str(getattr(args, "pose_backend", "icp")).strip().lower()
        self._ik_candidates = []
        self._last_T_source_to_target = None
        self._consecutive_icp_failures = 0
        self._icp_fallback_after = int(getattr(args, "icp_fallback_after", 5))
        self._stable_history = []
        self._stable_frames_required = int(getattr(args, "stable_frames", 3))
        self._stable_pos_thr_mm = float(getattr(args, "stable_pos_thr_mm", 5.0))
        self._stable_rot_thr_deg = float(getattr(args, "stable_rot_thr_deg", 10.0))
        self._last_icp_pose_source = "unknown"
        self._last_icp_quality = {}
        self._fixed_template_rpy_cache = None
        self._last_gripper_feedback_ts = 0.0
        self._last_gripper_state = None
        self._hotkey_help_printed = False
        self._last_final_pose_mm_deg = None
        self.reg_target_name = "grasp_region_cad(from_cad_ply)"
        self.T_source_to_target_manual_init = None
        self.use_local_region_template = True
        from collections import deque

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
        elif self._pose_backend == "foundationpose":
            self._init_fp_bridge(args)
        else:
            raise ValueError(f"未知 pose_backend: {self._pose_backend}")
        if self._pose_backend == "icp":
            self._ik_candidates = self._ik_runner._load_ik_candidates(args) if hasattr(self, "_ik_runner") else []
            if not self._ik_candidates:
                from online_grasp.grasp.ik_search import _load_ik_candidates
                self._ik_candidates = _load_ik_candidates(args)
            if self._ik_candidates:
                for ci, crpy in enumerate(self._ik_candidates):
                    print(f"  [ik-cand][#{ci}] rpy(deg)={crpy}")
            else:
                print("[ik-cand] 未提供候选法兰姿态，将使用 ICP 直接输出的完整位姿（旧行为）")
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
                gripper_close_position=int(args.gripper_close_position),
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
        self._depth_completer = DepthCompleter(self.args, self.intrinsics)
        self._depth_completer.depthcomplete = self.depthcomplete
        self._pc_builder = PointCloudBuilder(self.args, self.o3d, self.intrinsics, self._depth_completer)
        self._pc_builder.use_local_region_template = bool(self.use_local_region_template)
        self._segmentor = Segmentor(self.args, predictor=self.predictor)
        self._icp_backend = None
        self._fp_backend = None
        if self._pose_backend == "icp":
            self._icp_backend = ICPBackend(self.args, self.o3d)
            self._sync_state_to_icp_backend()
        elif self._pose_backend == "foundationpose":
            self._fp_backend = FoundationPoseBackend(self.args, self._K_3x3, skip_init=True)
            self._sync_state_to_fp_backend()
        self._stability_checker = StabilityChecker(
            self._stable_frames_required,
            self._stable_pos_thr_mm,
            self._stable_rot_thr_deg,
        )
        self._ik_runner = IKSearchRunner(self.args, self.executor, ctx=self)
        # 保持 legacy loop 中对 self._stable_history 的读取行为一致
        self._stable_history = self._stability_checker._stable_history
        if bool(args.cleargrasp):
            self._init_cleargrasp()

    def _sync_state_to_icp_backend(self) -> None:
        if self._icp_backend is None:
            return
        b = self._icp_backend
        b._frame_idx = int(self._frame_idx)
        b.source_buffer = self.source_buffer
        b.reg_target_pcd = self.reg_target_pcd
        b.reg_target_name = self.reg_target_name
        b.use_local_region_template = self.use_local_region_template
        b._last_T_source_to_target = self._last_T_source_to_target
        b._consecutive_icp_failures = int(self._consecutive_icp_failures)
        b._last_icp_pose_source = str(self._last_icp_pose_source)
        b._last_icp_quality = dict(self._last_icp_quality)
        b.T_source_to_target_manual_init = self.T_source_to_target_manual_init
        b.T_region_to_grasp = np.asarray(self.T_region_to_grasp, dtype=np.float64)

    def _sync_state_from_icp_backend(self) -> None:
        if self._icp_backend is None:
            return
        b = self._icp_backend
        self.source_buffer = b.source_buffer
        self._last_T_source_to_target = b._last_T_source_to_target
        self._consecutive_icp_failures = int(b._consecutive_icp_failures)
        self._last_icp_pose_source = str(b._last_icp_pose_source)
        self._last_icp_quality = dict(b._last_icp_quality)

    def _sync_state_to_fp_backend(self) -> None:
        if self._fp_backend is None:
            return
        b = self._fp_backend
        b._frame_idx = int(self._frame_idx)
        b.T_region_to_grasp = np.asarray(self.T_region_to_grasp, dtype=np.float64)
        for name in [
            "T_region_in_obj",
            "_fp_bridge_dir",
            "_fp_script",
            "_fp_conda_env",
            "_fp_mesh_file",
            "_fp_code_dir",
            "_fp_est_refine_iter",
            "_fp_min_n_views",
            "_fp_inplane_step",
            "_fp_timeout",
            "_fp_debug",
            "_fp_debug_dir",
        ]:
            if hasattr(self, name):
                setattr(b, name, getattr(self, name))

    def _init_fp_bridge(self, args) -> None:
        """
        覆盖 legacy 初始化入口：调用新 FoundationPoseBackend 的初始化逻辑，
        并将状态字段回填到当前 pipeline，保证后续 legacy loop 读取行为不变。
        """
        backend = FoundationPoseBackend(args, np.eye(3, dtype=np.float64), skip_init=True)
        backend._init_fp_bridge(args)
        self.T_region_in_obj = np.asarray(backend.T_region_in_obj, dtype=np.float64)
        for name in [
            "_fp_bridge_dir",
            "_fp_script",
            "_fp_conda_env",
            "_fp_mesh_file",
            "_fp_code_dir",
            "_fp_est_refine_iter",
            "_fp_min_n_views",
            "_fp_inplane_step",
            "_fp_timeout",
            "_fp_debug",
            "_fp_debug_dir",
        ]:
            setattr(self, name, getattr(backend, name))

    def _update_and_fuse_source(self, pcd):
        if self._pose_backend != "icp" or self._icp_backend is None:
            raise RuntimeError("SKIP: _update_and_fuse_source 仅支持 ICP 后端")
        self._sync_state_to_icp_backend()
        fused, n_valid = self._icp_backend.update_and_fuse_source(pcd)
        self._sync_state_from_icp_backend()
        return fused, n_valid

    def estimate_pose_and_grasp(self, fused_source_pcd, T_base_cam: np.ndarray, *, fused_frames: int):
        if self._pose_backend != "icp" or self._icp_backend is None:
            raise RuntimeError("SKIP: estimate_pose_and_grasp 仅支持 ICP 后端")
        self._sync_state_to_icp_backend()
        out = self._icp_backend.estimate_pose_and_grasp(
            fused_source_pcd,
            T_base_cam,
            fused_frames=fused_frames,
        )
        self._sync_state_from_icp_backend()
        return out

    def _update_stability(self, grasp_pos_base_mm: np.ndarray, candidate_idx: int, T_grasp_base: np.ndarray) -> bool:
        return bool(self._stability_checker.update(T_grasp_base, grasp_pos_base_mm, candidate_idx))

    def _reset_stability(self):
        self._stability_checker.reset()

    def _estimate_pose_via_fp_bridge(self, color_bgr, depth_m, mask_pc):
        if self._pose_backend != "foundationpose" or self._fp_backend is None:
            raise RuntimeError("SKIP: _estimate_pose_via_fp_bridge 仅支持 FoundationPose 后端")
        self._sync_state_to_fp_backend()
        return self._fp_backend._estimate_pose_via_fp_bridge(color_bgr, depth_m, mask_pc)

    def _compute_grasp_from_fp_pose(self, T_obj_cam: np.ndarray, T_base_cam: np.ndarray):
        if self._pose_backend != "foundationpose" or self._fp_backend is None:
            raise RuntimeError("SKIP: _compute_grasp_from_fp_pose 仅支持 FoundationPose 后端")
        self._sync_state_to_fp_backend()
        return self._fp_backend._compute_grasp_from_fp_pose(T_obj_cam, T_base_cam)

    def _T_to_robot_pose_mm_deg(self, T: np.ndarray) -> np.ndarray:
        xyz_m = np.asarray(T[:3, 3], dtype=np.float64)
        r, p, y = _rot_to_euler_xyz_deg(np.asarray(T[:3, :3], dtype=np.float64))
        xyz_mm = xyz_m * 1000.0
        return np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], r, p, y], dtype=np.float64)

    @staticmethod
    def _format_gripper_status(status):
        return format_gripper_status(status)

    def _depth_raw_to_meters(self, depth_raw):
        """将 Mech-Eye 原始深度转为米制 float32（全图，不裁剪）。"""
        depth_u16 = live_utils._depth_to_png_u16(depth_raw)
        return live_utils._depth_u16_to_m(depth_u16, unit=str(self.args.depth_unit))

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

    def execute_robot_plan(
        self,
        T_base_pregrasp: np.ndarray,
        T_base_grasp: np.ndarray,
        *,
        T_region_cam: np.ndarray = None,
        T_grasp_cam: np.ndarray = None,
        T_grasp_base: np.ndarray = None,
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
        self._close_gripper_after_grasp(context="execute_robot_plan")
        self._go_home_then_revisit_final_pose(pose_grasp, context="execute_robot_plan")
        print("[robot] 执行完成")

    def _go_home_then_revisit_final_pose(self, final_pose_mm_deg: np.ndarray, *, context: str) -> None:
        if self.executor is None:
            print(f"[robot][final-6d][skip] context={context} 无机器人连接")
            return
        final_pose = self._cache_final_pose_and_log(final_pose_mm_deg, context=context)
        preset_pose = final_pose.copy()
        offset_text = str(getattr(self.args, "final_pose_preset_offset_mm", "") or "").strip()
        if offset_text:
            offset_xyz_mm = _parse_csv_floats(offset_text, 3)
        else:
            retreat_mm = float(getattr(self.args, "final_pose_preset_retreat_mm", 10.0))
            offset_xyz_mm = np.array([0.0, 0.0, -retreat_mm], dtype=np.float64)
        preset_pose[0:3] = np.asarray(final_pose[0:3], dtype=np.float64) + np.asarray(offset_xyz_mm, dtype=np.float64)
        p1_joint = _parse_csv_floats(self.args.p1, 6)
        print(f"[robot][home] 先回 home(p1)={p1_joint.tolist()}")
        self.executor.movj_joint(p1_joint, f"{context}_back_to_home_p1")
        print(f"[robot][preset] offset_xyz_mm={np.asarray(offset_xyz_mm, dtype=np.float64).tolist()} before_final_6d={preset_pose.tolist()}")
        self.executor.movj_pose(np.asarray(preset_pose, dtype=np.float64), f"{context}_preset_offset")
        print("[robot][preset] 已到达 before_final_6d，等待 5.0 秒后回到 return_final_6d")
        time.sleep(5.0)
        print(f"[robot][revisit] return_final_6d={final_pose.tolist()}")
        self.executor.movj_pose(np.asarray(final_pose, dtype=np.float64), f"{context}_return_final_6d")
        print("[robot][revisit] 已完成 home -> 预设点(offset) -> 最终6D位姿")

    def _compute_T_base_cam_dynamic(self) -> np.ndarray:
        if self.eye_in_hand:
            if self.executor is None or self.T_cam_to_flange is None:
                raise RuntimeError("Eye-in-Hand 模式未初始化 robot executor 或 T_cam_to_flange")
            T_base_flange = self.executor.get_T_base_flange()
            T_flange_cam = np.linalg.inv(np.asarray(self.T_cam_to_flange, dtype=np.float64))
            return np.asarray(T_base_flange, dtype=np.float64) @ T_flange_cam
        if self.T_base_cam_fixed is None:
            raise RuntimeError("固定外参 T_base_cam 未设置")
        return np.asarray(self.T_base_cam_fixed, dtype=np.float64)

    def _close_gripper_after_grasp(self, *, context: str) -> None:
        close_gripper_after_grasp(self, context=context)

    def _wait_grip2_then_back_to_p1(self) -> None:
        wait_grip2_then_back_to_p1(self)

    def _handle_gripper_hotkey(self, key: int) -> None:
        handle_gripper_hotkey(self, key)

    def _maybe_print_gripper_stroke(self) -> None:
        maybe_print_gripper_stroke(self)

    def _init_cleargrasp(self):
        if not hasattr(self, "_depth_completer"):
            return
        self._depth_completer.init_cleargrasp()
        self.depthcomplete = self._depth_completer.depthcomplete

    def infer_mask_from_frame(self, color_bgr):
        return self._segmentor.infer_mask_from_frame(color_bgr)

    def _maybe_complete_depth(self, depth_np, color_bgr):
        out = self._depth_completer.maybe_complete_depth(depth_np, color_bgr)
        self.depthcomplete = self._depth_completer.depthcomplete
        return out

    def build_target_pointcloud(self, depth_obj, mask_pc, color_bgr):
        return self._pc_builder.build_target_pointcloud(depth_obj, mask_pc, color_bgr)

    def preprocess_target_pointcloud(self, pcd):
        self._pc_builder.use_local_region_template = bool(self.use_local_region_template)
        return self._pc_builder.preprocess_target_pointcloud(pcd)

    def _format_pose_mm_deg(self, pose_mm_deg: np.ndarray) -> str:
        return self._ik_runner._format_pose_mm_deg(pose_mm_deg)

    def _format_pose_full_precision(self, pose_mm_deg: np.ndarray) -> str:
        return self._ik_runner._format_pose_full_precision(pose_mm_deg)

    def _cache_final_pose_and_log(self, pose_mm_deg: np.ndarray, *, context: str) -> np.ndarray:
        return self._ik_runner._cache_final_pose_and_log(pose_mm_deg, context=context)

    def _get_fixed_template_rpy(self) -> np.ndarray:
        return self._ik_runner._get_fixed_template_rpy()

    def _build_fixed_template_pose(self, grasp_pos_base_mm: np.ndarray):
        return self._ik_runner._build_fixed_template_pose(grasp_pos_base_mm)

    def _run_runtime_probe_and_execute(self, pose_pre: np.ndarray, pose_grasp: np.ndarray) -> bool:
        return bool(self._ik_runner.run_runtime_probe_and_execute(pose_pre, pose_grasp))

    def _ik_try_pose_with_reason(self, pose_mm_deg: np.ndarray):
        return self._ik_runner._ik_try_pose_with_reason(pose_mm_deg)

    def _probe_pose_reachability_by_motion(self, pose_mm_deg: np.ndarray, stage_name: str):
        return self._ik_runner._probe_pose_reachability_by_motion(pose_mm_deg, stage_name)

    def _ik_check_pose(self, pose_mm_deg: np.ndarray, stage_name: str):
        return self._ik_runner._ik_check_pose(pose_mm_deg, stage_name)

    def _dedup_rpy_candidates(self, cands, thr_deg: float):
        return self._ik_runner._dedup_rpy_candidates(cands, thr_deg)

    def _expand_ik_rpy_candidates(self, seed_rpys, visual_rpy=None):
        return self._ik_runner._expand_ik_rpy_candidates(seed_rpys, visual_rpy=visual_rpy)

    def _sort_ik_candidate_priority(self, candidates):
        return self._ik_runner._sort_ik_candidate_priority(candidates)

    def _run_fixed_rpy_reachability_debug(self, grasp_pos_base_mm: np.ndarray):
        return self._ik_runner._run_fixed_rpy_reachability_debug(grasp_pos_base_mm)

    def _estimate_workspace_risk(self, grasp_pos_base_mm: np.ndarray) -> str:
        return self._ik_runner._estimate_workspace_risk(grasp_pos_base_mm)

    def _summarize_ik_attempts(self, attempts):
        return self._ik_runner._summarize_ik_attempts(attempts)

    def _summarize_reachability_check(self, attempts):
        return self._ik_runner._summarize_reachability_check(attempts)

    def _diagnose_api_vs_motion_mismatch(self, attempts) -> str:
        return self._ik_runner._diagnose_api_vs_motion_mismatch(attempts)

    def _diagnose_ik_failure(self, attempts, grasp_pos_base_mm: np.ndarray, pregrasp_pos_base_mm: np.ndarray, fixed_debug):
        return self._ik_runner._diagnose_ik_failure(
            attempts,
            grasp_pos_base_mm,
            pregrasp_pos_base_mm,
            fixed_debug,
        )

    def _try_all_candidate_ik(self, grasp_pos_base_mm: np.ndarray, visual_grasp_rpy_deg=None):
        return self._ik_runner._try_all_candidate_ik(grasp_pos_base_mm, visual_grasp_rpy_deg=visual_grasp_rpy_deg)

    def _save_handle_target_sample(self, pcd) -> bool:
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
                    depth_m = self._depth_raw_to_meters(depth.data())
                    T_obj_cam = self._estimate_pose_via_fp_bridge(color_bgr, depth_m, mask_pc)
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
                            grasp_pos_base_mm = np.asarray(T_grasp_base[:3, 3], dtype=np.float64) * 1000.0
                            pregrasp_pos_base_mm = np.asarray(T_base_pre[:3, 3], dtype=np.float64) * 1000.0
                            print(
                                f"[vision] 抓取点位置(base,mm): "
                                f"x={grasp_pos_base_mm[0]:.2f} "
                                f"y={grasp_pos_base_mm[1]:.2f} "
                                f"z={grasp_pos_base_mm[2]:.2f}"
                            )
                            visual_grasp_rpy = np.asarray(
                                _rot_to_euler_xyz_deg(np.asarray(T_grasp_base[:3, :3], dtype=np.float64)),
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
                                        self._close_gripper_after_grasp(context="candidate_joint_solution")
                                        self._go_home_then_revisit_final_pose(
                                            np.asarray(best["pose_grasp"], dtype=np.float64),
                                            context="candidate_joint_solution",
                                        )
                                    elif bool(self.args.direct_pose_fallback_enable):
                                        print("[robot] exec_mode=direct_pose_fallback")
                                        self.executor.movj_pose(np.asarray(best["pose_pre"], dtype=np.float64), "pregrasp_movj_pose_fallback")
                                        self.executor.movj_pose(np.asarray(best["pose_grasp"], dtype=np.float64), "grasp_movj_pose_fallback")
                                        self._close_gripper_after_grasp(context="candidate_direct_pose_fallback")
                                        self._go_home_then_revisit_final_pose(
                                            np.asarray(best["pose_grasp"], dtype=np.float64),
                                            context="candidate_direct_pose_fallback",
                                        )
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
                            self._wait_grip2_then_back_to_p1()
                            if self._last_final_pose_mm_deg is not None:
                                self._go_home_then_revisit_final_pose(
                                    np.asarray(self._last_final_pose_mm_deg, dtype=np.float64),
                                    context="runtime_fixed_template",
                                )
                            else:
                                print("[robot][final-6d][warn] runtime_done 后未找到缓存的最终6D位姿，跳过回访")
                            print("[runtime][done] 运行模式执行成功，已完成夹爪反馈与回 p1，结束视觉流程")
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
            self._maybe_print_gripper_stroke()
            if not bool(self.args.no_gui):
                if not bool(self._hotkey_help_printed):
                    print("[hotkey] q=退出, p=回到p1, o=打开夹爪, c=闭合夹爪")
                    self._hotkey_help_printed = True
                vis = live_utils._overlay_binary_mask(color_bgr, mask_pc if "mask_pc" in locals() else np.zeros(color_bgr.shape[:2], np.uint8))
                dt_ms = (time.time() - t0) * 1000.0
                cv2.putText(vis, f"loop {dt_ms:.1f} ms", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(str(self.args.win), vis)
                key = cv2.waitKey(int(self.args.wait)) & 0xFF
                self._handle_gripper_hotkey(key)
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


__all__ = ["OnlineGraspPipeline"]

