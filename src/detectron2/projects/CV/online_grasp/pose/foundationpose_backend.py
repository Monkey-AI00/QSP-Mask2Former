"""FoundationPose backend extracted from legacy pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import time

import cv2
import numpy as np

from online_grasp.geometry.calibration import _load_region_in_obj
from online_grasp.geometry.transforms import _make_T_from_xyz_m, _parse_csv_floats


class FoundationPoseBackend:
    def __init__(self, args, K_3x3, *, skip_init: bool = False):
        self.args = args
        self._K_3x3 = np.asarray(K_3x3, dtype=np.float64)
        self._frame_idx = 0
        self.T_region_to_grasp = np.eye(4, dtype=np.float64)
        if not bool(skip_init):
            self._init_fp_bridge(args)

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

    def _run_foundationpose_bridge(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        mask_pc: np.ndarray,
    ) -> np.ndarray:
        return self._estimate_pose_via_fp_bridge(color_bgr, depth_m, mask_pc)

    def _estimate_pose_via_fp_bridge(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        mask_pc: np.ndarray,
    ) -> np.ndarray:
        inputs_dir = os.path.join(self._fp_bridge_dir, "inputs")
        outputs_dir = os.path.join(self._fp_bridge_dir, "outputs")
        request_id = f"frame_{self._frame_idx}"
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

    def estimate_pose_and_grasp(self, color_bgr, depth_m, mask_pc, T_base_cam):
        T_obj_cam = self._run_foundationpose_bridge(color_bgr, depth_m, mask_pc)
        return self._compute_grasp_from_fp_pose(T_obj_cam, T_base_cam)


__all__ = ["FoundationPoseBackend"]

