import numpy as np
import time
import sys
import os

# 确保导入当前目录的模块
sys.path.insert(0, os.path.dirname(__file__))

from wrist_force_rs485 import WristFTSensorRS485
from Gravity_compensation import (
    DobotRobotClient,
    GravityCompensation,
    process_force_data,
    calibrate_gravity_compensation,
)
# from force_frame_trans import force_moment_transform
from force_frame_trans_rot import force_moment_transform
from filter_module import RCLowPassFilter
from force_frame_trans_toB import wrench_tool_to_base

class ForceDataProcessor:
    def __init__(
        self,
        robot,
        serial_port="/dev/ttyUSB0",
        baudrate=115200,
        alpha_deg=-45.0,
        pose_is_radian=False,
    ):
        # ---------------- Modbus TCP 替换为 RS485 ----------------
        self.sensor = WristFTSensorRS485(
            serial_port=serial_port,
            baudrate=baudrate,
            timeout=0.1,
            auto_start=True,
            auto_stop_on_close=False,
        )
        if not self.sensor.open():
            raise ConnectionError(
                f"[sensor] open failed: serial_port={serial_port}, baudrate={baudrate}"
            )

        # 重力补偿初始化
        self.gravity_comp = GravityCompensation(alpha_deg)
        # 姿态单位需确认：
        # - False: 机器人返回角度（Dobot 常见）
        # - True : 机器人返回弧度（需 angle() 转换）
        self.pose_is_radian = bool(pose_is_radian)

        # 初始化低通滤波器 (RC低通滤波器)
        self.filter = RCLowPassFilter(alpha=0.5)

        # 进行重力补偿标定
        calibrate_gravity_compensation(self.gravity_comp, robot, self.sensor)

    def process_data(self, robot_pos, T_P_SORG, apply_rotation, R_TS):
        """
        处理传感器数据：应用重力补偿、坐标变换和滤波。
        """
        # (1)读取原始力数据
        raw_force = self.sensor.read_ft()
        if raw_force is None:
            raise RuntimeError("read_ft failed")

        # (2)应用重力补偿
        force, torque = process_force_data(
            self.gravity_comp,
            raw_force[:3],
            raw_force[3:],
            robot_pos,
            pose_is_radian=self.pose_is_radian,
        )

        # (3)S-T-B 坐标变换(工具坐标系在传感器坐标系中的位移T_P_SORG)
        transformed_force, transformed_torque = force_moment_transform(force, torque, T_P_SORG, apply_rotation, R_TS)
        T_force = np.hstack((transformed_force, transformed_torque))
        B_force = wrench_tool_to_base(robot_pos, T_force)

        # (4)应用低通滤波
        filtered_force = self.filter.update(B_force)

        return filtered_force, raw_force

    def close(self):
        """Close the sensor connection."""
        self.sensor.close()


def main():
    robot = None
    processor = None
    try:
        # ---------------- JAKA 替换为 Dobot ----------------
        robot_ip = "192.168.5.2"
        robot = DobotRobotClient(ip=robot_ip)
        robot.connect()
        robot.enable()

        T_P_SORG = [0, 0, 170 * 0.001]
        theta = 45 * np.pi / 180.0
        R_TS = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        # 创建力数据处理器实例
        processor = ForceDataProcessor(
            robot=robot,
            serial_port="/dev/ttyUSB0",
            baudrate=115200,
            alpha_deg=-45.0,
            pose_is_radian=False,
        )

        # 在循环中处理数据
        while True:
            try:
                robot_pos = robot.get_tcp_pose()
            except Exception as e:
                print(f"[robot][error] get pose failed: {type(e).__name__}: {e}")
                time.sleep(0.1)
                continue

            try:
                force, raw_force = processor.process_data(robot_pos, T_P_SORG, True, R_TS)
            except Exception as e:
                print(f"[sensor/process][error] {type(e).__name__}: {e}")
                time.sleep(0.1)
                continue

            # force,raw_force = processor.process_data(robot_pos, T_P_SORG, False, R_TS)
            # print(f"Raw Force: {raw_force}")
            print(f"Force: {force}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Data processing stopped.")
    finally:
        if processor is not None:
            processor.close()
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
