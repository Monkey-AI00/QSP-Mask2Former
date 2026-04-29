"""Dobot executor migrated from legacy script."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from online_grasp.geometry.transforms import _pose_mm_deg_to_T


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
        gripper_close_position: int = 0,
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
        self._gripper_close_position = int(gripper_close_position)
        self._gripper_init_timeout = float(gripper_init_timeout)
        self._gripper = None
        self._gripper_opened_once = False

    def _ensure_gripper_ready(self) -> None:
        if self._gripper is None:
            self._gripper = self._GripperController(
                port=self._gripper_port,
                baudrate=self._gripper_baudrate,
                init_timeout_s=self._gripper_init_timeout,
            )
        if not self._gripper_opened_once:
            self._gripper.connect_and_initialize()
            self._gripper_opened_once = True

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
        if self._gripper_opened_once:
            return
        print("[gripper] 在识别初始位执行夹爪初始化并张开")
        self._ensure_gripper_ready()
        self._gripper.open_gripper(self._gripper_open_position)

    def open_gripper(self) -> None:
        if not self._gripper_enable:
            print("[gripper] 当前未启用（--gripper-enable=False），忽略 open 指令")
            return
        self._ensure_gripper_ready()
        self._gripper.open_gripper(self._gripper_open_position)

    def close_gripper(self) -> None:
        if not self._gripper_enable:
            print("[gripper] 当前未启用（--gripper-enable=False），忽略 close 指令")
            return
        self._ensure_gripper_ready()
        self._gripper.close_gripper(self._gripper_close_position)

    def get_gripper_status(self) -> Optional[dict[str, Any]]:
        if not self._gripper_enable:
            return None
        try:
            self._ensure_gripper_ready()
            return dict(self._gripper.get_status())
        except Exception:
            return None

    def get_gripper_stroke(self) -> Optional[int]:
        status = self.get_gripper_status()
        if status is None:
            return None
        pos = status.get("current_position", None)
        if pos is None:
            return None
        return int(pos)

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

__all__ = ["DobotPoseExecutor"]

