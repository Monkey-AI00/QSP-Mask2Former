from numpy import *
from numpy import cos, matrix, sin
import numpy as np
import math
import time
import sys
import re
import json
from pathlib import Path
from wrist_force_rs485 import WristFTSensorRS485

# ---------------- JAKA 替换为 Dobot：SDK 导入 ----------------
TCP_ROOT = Path(__file__).resolve().parents[1] / "TCP-IP-Python-V4-main" / "TCP-IP-Python-V4-main"
if str(TCP_ROOT) not in sys.path:
    sys.path.insert(0, str(TCP_ROOT))
from dobot_api import DobotApiDashboard, DobotApiFeedBack  # type: ignore


class DobotRobotClient:
    """
    JAKA 适配层替换为 Dobot 适配层：
    对外统一提供 connect()/enable()/get_tcp_pose()/close()。
    """

    DASHBOARD_PORT = 29999
    FEEDBACK_PORT = 30004

    def __init__(self, ip: str = "192.168.1.30"):
        self.ip = str(ip)
        self.dashboard = None
        self.feedback = None
        self._float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
        self._int_re = re.compile(r"-?\d+")

    def _parse_ints(self, resp: str):
        if resp is None:
            return []
        return [int(x) for x in self._int_re.findall(str(resp))]

    def connect(self):
        try:
            self.dashboard = DobotApiDashboard(self.ip, self.DASHBOARD_PORT)
            print(f"[connect] Dashboard connected: ip={self.ip}, port={self.DASHBOARD_PORT}")
        except Exception as e:
            raise RuntimeError(
                f"[connect] Dashboard connection failed: ip={self.ip}, port={self.DASHBOARD_PORT}, "
                f"error={type(e).__name__}: {e}"
            ) from e

        try:
            self.feedback = DobotApiFeedBack(self.ip, self.FEEDBACK_PORT)
            print(f"[connect] FeedBack connected: ip={self.ip}, port={self.FEEDBACK_PORT}")
        except Exception as e:
            raise RuntimeError(
                f"[connect] FeedBack connection failed: ip={self.ip}, port={self.FEEDBACK_PORT}, "
                f"error={type(e).__name__}: {e}"
            ) from e
        # 反馈通道可读性探测（失败只告警，不阻塞主流程）
        self.read_feedback_once(strict=False)

    def enable(self):
        if self.dashboard is None:
            raise RuntimeError("robot dashboard not connected")
        try:
            ret = self.dashboard.EnableRobot()
        except Exception as e:
            raise RuntimeError(
                f"[enable] EnableRobot call failed: ip={self.ip}, port={self.DASHBOARD_PORT}, "
                f"error={type(e).__name__}: {e}"
            ) from e
        nums = self._parse_ints(ret)
        if not nums or int(nums[0]) != 0:
            raise RuntimeError(
                f"[enable] EnableRobot rejected: ip={self.ip}, port={self.DASHBOARD_PORT}, resp={ret}"
            )
        print("[enable] EnableRobot OK")

    def read_feedback_once(self, strict: bool = False):
        if self.feedback is None:
            msg = (
                f"[feedback][error] feedback channel not connected: "
                f"ip={self.ip}, port={self.FEEDBACK_PORT}"
            )
            if strict:
                raise RuntimeError(msg)
            print(msg)
            return None
        try:
            return self.feedback.feedBackData()
        except Exception as e:
            msg = (
                f"[feedback][error] read feedback failed: ip={self.ip}, port={self.FEEDBACK_PORT}, "
                f"error={type(e).__name__}: {e}"
            )
            if strict:
                raise RuntimeError(msg) from e
            print(msg)
            return None

    def get_tcp_pose(self) -> np.ndarray:
        """
        返回末端位姿 [x, y, z, rx, ry, rz]。

        需确认 Dobot 姿态单位：
        - Dobot GetPose 通常返回角度（deg）
        - 若你的控制器固件返回弧度（rad），需在主流程里开启弧度转角度
        """
        if self.dashboard is None:
            raise RuntimeError("robot dashboard not connected")
        try:
            resp = self.dashboard.GetPose()
        except Exception as e:
            raise RuntimeError(
                f"[pose] GetPose call failed: ip={self.ip}, port={self.DASHBOARD_PORT}, "
                f"error={type(e).__name__}: {e}"
            ) from e
        vals = [float(v) for v in self._float_re.findall(str(resp))]
        if len(vals) < 7:
            raise RuntimeError(f"[pose] GetPose invalid response: {resp}")
        err = int(vals[0])
        if err != 0:
            raise RuntimeError(f"[pose] GetPose ErrorID={err}, raw={resp}")
        return np.asarray(vals[1:7], dtype=np.float64)

    def close(self):
        if self.feedback is not None:
            try:
                self.feedback.close()
            except Exception:
                pass
            self.feedback = None
        if self.dashboard is not None:
            try:
                self.dashboard.close()
            except Exception:
                pass
            self.dashboard = None

'''
重力补偿计算
统一 R_alpha 的使用，避免标定与实时补偿旋转链不一致。
'''
class GravityCompensation:
    def __init__(self, alpha_deg):
        # 标定中间量
        self.M = np.empty((0, 0))
        self.F = np.empty((0, 0))
        self.f = np.empty((0, 0))
        self.R = np.empty((0, 0))

        # A 参数
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0

        # B 参数
        self.U = 0.0
        self.V = 0.0
        self.g = 0.0

        self.F_x0 = 0.0
        self.F_y0 = 0.0
        self.F_z0 = 0.0

        self.M_x0 = 0.0
        self.M_y0 = 0.0
        self.M_z0 = 0.0

        # 传感器相对末端绕 z 轴固定安装角（单位：度）
        self.alpha_deg = alpha_deg

    def reset(self):
        """清空所有标定缓存。"""
        self.M = np.empty((0, 0))
        self.F = np.empty((0, 0))
        self.f = np.empty((0, 0))
        self.R = np.empty((0, 0))

    def Update_M(self, torque_data):
        M_x = torque_data[0]
        M_y = torque_data[1]
        M_z = torque_data[2]

        if any(self.M):
            M_1 = matrix([M_x, M_y, M_z]).transpose()
            self.M = vstack((self.M, M_1))
        else:
            self.M = matrix([M_x, M_y, M_z]).transpose()

    def Update_F(self, force_data):
        F_x = force_data[0]
        F_y = force_data[1]
        F_z = force_data[2]

        if any(self.F):
            F_1 = matrix([
                [0, F_z, -F_y, 1, 0, 0],
                [-F_z, 0, F_x, 0, 1, 0],
                [F_y, -F_x, 0, 0, 0, 1]
            ])
            self.F = vstack((self.F, F_1))
        else:
            self.F = matrix([
                [0, F_z, -F_y, 1, 0, 0],
                [-F_z, 0, F_x, 0, 1, 0],
                [F_y, -F_x, 0, 0, 0, 1]
            ])

    def Solve_A(self):
        A = dot(dot(linalg.inv(dot(self.F.transpose(), self.F)), self.F.transpose()), self.M)

        self.x = A[0, 0]
        self.y = A[1, 0]
        self.z = A[2, 0]
        self.k1 = A[3, 0]
        self.k2 = A[4, 0]
        self.k3 = A[5, 0]

        print("x= ", self.x)
        print("y= ", self.y)
        print("z= ", self.z)
        print("k1= ", self.k1)
        print("k2= ", self.k2)
        print("k3= ", self.k3)

    def Update_f(self, force_data):
        F_x = force_data[0]
        F_y = force_data[1]
        F_z = force_data[2]

        if any(self.f):
            f_1 = matrix([F_x, F_y, F_z]).transpose()
            self.f = vstack((self.f, f_1))
        else:
            self.f = matrix([F_x, F_y, F_z]).transpose()

    def get_sensor_rotation(self, euler_data):
        """
        统一生成补偿使用的旋转矩阵。
        关键点：标定(Update_R)和实时补偿(Solve_Force / Solve_Torque)必须完全一致。

        输入:
            euler_data: [rx, ry, rz]，单位：度

        返回:
            R_array: 当前补偿链使用的旋转矩阵
        """
        # 机械臂末端姿态矩阵（XYZ 欧拉序: R = Rz * Ry * Rx）
        R_tool = self.eulerAngles2rotationMat(euler_data)

        # 传感器相对末端的固定 z 轴安装角
        alpha = self.alpha_deg * math.pi / 180.0
        R_alpha = np.array([
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0,                0,               1]
        ])

        # Update_R 中的乘法顺序
        R_array = np.dot(R_alpha, R_tool.transpose())
        return R_array

    def Update_R(self, euler_data):
        R_array = self.get_sensor_rotation(euler_data)

        if any(self.R):
            R_1 = hstack((R_array, np.eye(3)))
            self.R = vstack((self.R, R_1))
        else:
            self.R = hstack((R_array, np.eye(3)))

    def Solve_B(self):
        B = dot(dot(linalg.inv(dot(self.R.transpose(), self.R)), self.R.transpose()), self.f)

        b0 = float(B[0, 0])
        b1 = float(B[1, 0])
        b2 = float(B[2, 0])
        self.g = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
        self.U = math.asin(-b1 / self.g)
        self.V = math.atan2(-b0, b2)    # 使用 atan2 避免除零错误

        self.F_x0 = B[3, 0]
        self.F_y0 = B[4, 0]
        self.F_z0 = B[5, 0]

        print("g= ", self.g / 9.81)
        print("U= ", self.U * 180 / math.pi)
        print("V= ", self.V * 180 / math.pi)
        print("F_x0= ", self.F_x0)
        print("F_y0= ", self.F_y0)
        print("F_z0= ", self.F_z0)

    def Solve_Force(self, force_data, euler_data):
        Force_input = matrix([force_data[0], force_data[1], force_data[2]]).transpose()

        my_f = matrix([
            cos(self.U) * sin(self.V) * self.g,
            -sin(self.U) * self.g,
            -cos(self.U) * cos(self.V) * self.g,
            self.F_x0,
            self.F_y0,
            self.F_z0
        ]).transpose()

        # 与 Update_R 完全一致的旋转定义
        R_array = self.get_sensor_rotation(euler_data)
        R_1 = hstack((R_array, np.eye(3)))

        Force_ex = Force_input - dot(R_1, my_f)
        return Force_ex.T

    def Solve_Torque(self, torque_data, euler_data):
        Torque_input = matrix([torque_data[0], torque_data[1], torque_data[2]]).transpose()

        M_x0 = self.k1 - self.F_y0 * self.z + self.F_z0 * self.y
        M_y0 = self.k2 - self.F_z0 * self.x + self.F_x0 * self.z
        M_z0 = self.k3 - self.F_x0 * self.y + self.F_y0 * self.x

        Torque_zero = matrix([M_x0, M_y0, M_z0]).transpose()

        Gravity_param = matrix([
            [0, -self.z, self.y],
            [self.z, 0, -self.x],
            [-self.y, self.x, 0]
        ])

        Gravity_input = matrix([
            cos(self.U) * sin(self.V) * self.g,
            -sin(self.U) * self.g,
            -cos(self.U) * cos(self.V) * self.g
        ]).transpose()

        # 与 Update_R 完全一致的旋转定义
        R_array = self.get_sensor_rotation(euler_data)

        Torque_ex = Torque_input - Torque_zero - dot(dot(Gravity_param, R_array), Gravity_input)
        return Torque_ex.T

    def eulerAngles2rotationMat(self, theta):
        theta = [i * math.pi / 180.0 for i in theta]  # 角度转弧度

        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]),  math.cos(theta[0])]
        ])

        R_y = np.array([
            [ math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])]
        ])

        R_z = np.array([
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]),  math.cos(theta[2]), 0],
            [0, 0, 1]
        ])

        # XYZ 欧拉序: R = Rz * Ry * Rx
        R = np.dot(R_z, np.dot(R_y, R_x))
        # R = np.dot(R_x, np.dot(R_y, R_z))
        return R


# 封装数据处理过程
def process_force_data(comp, raw_force, raw_torque, robot_pos, pose_is_radian=False):
    """
    参数:
    comp: 重力补偿器对象
    raw_force: 原始力数据 (3维)
    raw_torque: 原始力矩数据 (3维)
    robot_pos: 机器人位置 (6维,包含姿态)

    返回:
    force: 补偿后的力 (3维)
    torque: 补偿后的力矩 (3维)
    """
    # 需确认 Dobot 姿态单位：
    # - 若 GetPose 返回角度（常见），不要再做 angle() 二次转换
    # - 若实际返回弧度，把 pose_is_radian 设为 True
    if pose_is_radian:
        realtime_euler = [angle(robot_pos[3]), angle(robot_pos[4]), angle(robot_pos[5])]
    else:
        realtime_euler = [robot_pos[3], robot_pos[4], robot_pos[5]]

    tem1 = comp.Solve_Force(raw_force, realtime_euler)
    tem2 = comp.Solve_Torque(raw_torque, realtime_euler)
    force = np.array(tem1)
    torque = np.array(tem2)

    return force, torque


def get_robot_pose(robot):
    """
    统一读取机器人位姿，返回 [x, y, z, rx, ry, rz]。
    """
    if hasattr(robot, "get_tcp_pose"):
        return np.asarray(robot.get_tcp_pose(), dtype=np.float64)

    if hasattr(robot, "get_tcp_position"):
        _, robot_pos = robot.get_tcp_position()
        return np.asarray(robot_pos, dtype=np.float64)

    raise AttributeError("robot 对象不支持 get_tcp_pose() 或 get_tcp_position()")


def collect_calibration_sample(
    robot,
    Get_force,
    pose_index,
    settle_time=1.5,
    sample_count=200,
    min_valid_frames=50,
    sample_interval=0.01,
):
    """
    交互式采集单个标定姿态样本。
    返回:
        dict | None
    """
    while True:
        user_input = input(
            f"\n请将机械臂调整到第 {pose_index} 个标定姿态后按回车开始采集，输入 q 退出标定: "
        ).strip().lower()
        if user_input == "q":
            return None

        print(f"[标定] 第 {pose_index} 个姿态等待停稳 {settle_time:.1f} 秒...")
        time.sleep(settle_time)

        force_samples = []
        torque_samples = []
        euler_samples = []

        print(f"[标定] 第 {pose_index} 个姿态开始采集，共尝试 {sample_count} 帧...")
        for _ in range(sample_count):
            F = Get_force.read_ft()
            if F is None:
                time.sleep(sample_interval)
                continue

            try:
                pos = get_robot_pose(robot)
            except Exception as e:
                print(f"[标定][warn] 第 {pose_index} 个姿态读取位姿失败: {type(e).__name__}: {e}")
                time.sleep(sample_interval)
                continue

            force_samples.append(np.asarray(F[:3], dtype=np.float64))
            torque_samples.append(np.asarray(F[3:], dtype=np.float64))
            euler_samples.append(np.asarray(pos[3:6], dtype=np.float64))
            time.sleep(sample_interval)

        valid_count = len(force_samples)
        print(f"[标定] 第 {pose_index} 个姿态有效帧数: {valid_count}/{sample_count}")

        if valid_count < min_valid_frames:
            print(f"[标定][warn] 第 {pose_index} 个姿态有效帧不足 {min_valid_frames}，请重新采集。")
            continue

        force_mean = np.mean(np.asarray(force_samples), axis=0)
        torque_mean = np.mean(np.asarray(torque_samples), axis=0)
        euler_mean = np.mean(np.asarray(euler_samples), axis=0)

        print(f"[标定] 第 {pose_index} 个姿态 force 均值: {force_mean}")
        print(f"[标定] 第 {pose_index} 个姿态 torque 均值: {torque_mean}")
        print(f"[标定] 第 {pose_index} 个姿态 euler 均值(度): {euler_mean}")

        return {
            'force': force_mean.tolist(),
            'torque': torque_mean.tolist(),
            'euler': euler_mean.tolist(),
        }


def save_calibration_samples_to_json(
    calibration_data,
    num_poses,
    settle_time,
    sample_count,
    min_valid_frames,
    sample_interval,
    json_path=None,
):
    """
    将实时标定得到的均值样本保存为 JSON，便于复现。
    """
    if json_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = Path(__file__).resolve().parent / f"calibration_samples_{timestamp}.json"
    else:
        json_path = Path(json_path)

    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(calibration_data),
        "sampling_config": {
            "num_poses": int(num_poses),
            "settle_time": float(settle_time),
            "sample_count": int(sample_count),
            "min_valid_frames": int(min_valid_frames),
            "sample_interval": float(sample_interval),
        },
        "samples": calibration_data,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[标定] 样本已保存到 JSON: {json_path}")
    return str(json_path)


def load_calibration_samples_from_json(json_path=None):
    """
    从 JSON 加载标定样本。
    - json_path 为 None 时，自动加载当前目录最新的 calibration_samples_*.json
    """
    if json_path is None:
        candidates = sorted(
            Path(__file__).resolve().parent.glob("calibration_samples_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("当前目录未找到 calibration_samples_*.json")
        target = candidates[0]
    else:
        target = Path(json_path)
        if not target.is_absolute():
            target = (Path(__file__).resolve().parent / target).resolve()

    if not target.exists():
        raise FileNotFoundError(f"标定 JSON 不存在: {target}")

    with open(target, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", payload)
    if not isinstance(samples, list):
        raise ValueError(f"标定 JSON 格式错误，samples 不是列表: {target}")

    calibration_data = []
    for idx, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"第 {idx} 组样本格式错误（应为 dict）: {sample}")
        force = sample.get("force")
        torque = sample.get("torque")
        euler = sample.get("euler")
        if force is None or torque is None or euler is None:
            raise ValueError(f"第 {idx} 组样本缺少 force/torque/euler 字段")
        if len(force) != 3 or len(torque) != 3 or len(euler) != 3:
            raise ValueError(f"第 {idx} 组样本长度错误，需均为 3 维")

        calibration_data.append({
            "force": [float(v) for v in force],
            "torque": [float(v) for v in torque],
            "euler": [float(v) for v in euler],
        })

    if len(calibration_data) < 4:
        raise ValueError(f"标定样本不足 4 组: {len(calibration_data)}")

    print(f"[标定] 已从 JSON 加载样本: {target}")
    print(f"[标定] JSON 样本数量: {len(calibration_data)}")
    return calibration_data, str(target)


# 封装标定过程
def calibrate_gravity_compensation(
    comp,
    robot=None,
    Get_force=None,
    num_poses=12,
    settle_time=1.5,
    sample_count=200,
    min_valid_frames=50,
    sample_interval=0.01,
    save_samples_json=True,
    json_path=None,
    load_samples_json=True,
    load_json_path=None,
):
    """
    现场实时采集标定数据并执行重力补偿器标定。
    """
    print("开始重力补偿器标定...")
    comp.reset()

    calibration_data = None

    if load_samples_json:
        try:
            calibration_data, loaded_path = load_calibration_samples_from_json("calibration_samples_20260420_093954.json")
            print(f"[标定] 使用历史 JSON 样本进行标定: {loaded_path}")
        except Exception as e:
            print(f"[标定][warn] 加载历史 JSON 失败，回退到实时采集: {type(e).__name__}: {e}")

    if calibration_data is None:
        if robot is None or Get_force is None:
            raise ValueError("实时标定需要同时提供 robot 和 Get_force。")

        calibration_data = []
        for pose_index in range(1, num_poses + 1):
            sample = collect_calibration_sample(
                robot=robot,
                Get_force=Get_force,
                pose_index=pose_index,
                settle_time=settle_time,
                sample_count=sample_count,
                min_valid_frames=min_valid_frames,
                sample_interval=sample_interval,
            )
            if sample is None:
                break
            calibration_data.append(sample)

        if len(calibration_data) < 4:
            raise RuntimeError(
                f"标定成功样本数不足，至少需要 4 组，当前仅采集到 {len(calibration_data)} 组。"
            )

        print(f"[标定] 总共采集成功的样本数: {len(calibration_data)}")

        if save_samples_json:
            save_calibration_samples_to_json(
                calibration_data=calibration_data,
                num_poses=num_poses,
                settle_time=settle_time,
                sample_count=sample_count,
                min_valid_frames=min_valid_frames,
                sample_interval=sample_interval,
                json_path=json_path,
            )

    for data in calibration_data:
        comp.Update_F(data['force'])
        comp.Update_M(data['torque'])
        comp.Update_f(data['force'])
        comp.Update_R(data['euler'])

    comp.Solve_A()
    comp.Solve_B()
    print("重力补偿器标定完成！")


def angle(v):
    return v * 180.0 / math.pi


def main():
    robot = None
    Get_force = None
    try:
        # ---------------- JAKA 替换为 Dobot：连接/使能 ----------------
        robot = DobotRobotClient(ip="192.168.1.30")
        robot.connect()
        robot.enable()

        # ---------------- Modbus TCP 替换为 RS485 ----------------
        Get_force = WristFTSensorRS485(
            serial_port="/dev/ttyUSB1",
            baudrate=115200,
            timeout=0.1,
            auto_start=True,
            auto_stop_on_close=False,
        )
        if not Get_force.open():
            return

        # 初始化重力补偿器
        comp = GravityCompensation(alpha_deg=-45)
        print("重力补偿器初始化完成")

        # 执行标定
        save_samples_json = True
        load_samples_json = True
        calibrate_gravity_compensation(
            comp,
            robot,
            Get_force,
            num_poses=12,
            save_samples_json=save_samples_json,
            json_path=None,
            load_samples_json=load_samples_json,
            load_json_path=None,
        )

        while True:
            # 读取传感器数据
            F = Get_force.read_ft()
            if F is None:
                print("读取力传感器失败")
                time.sleep(0.1)
                continue

            # Dobot 读取位姿 [x, y, z, rx, ry, rz]
            try:
                pos = robot.get_tcp_pose()
            except Exception as e:
                print(f"[robot][error] get_tcp_pose failed: {type(e).__name__}: {e}")
                time.sleep(0.1)
                continue
            # 需确认 Dobot 姿态单位：默认按角度处理；若返回弧度请改为 True
            dobot_pose_is_radian = False

            # 使用封装的函数处理数据
            Fe, Me = process_force_data(
                comp,
                raw_force=F[:3],
                raw_torque=F[3:],
                robot_pos=pos,
                pose_is_radian=dobot_pose_is_radian,
            )
            current_force = np.concatenate((np.asarray(Fe).ravel(), np.asarray(Me).ravel()))
            print("current_force", current_force)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序结束")
    finally:
        if Get_force is not None:
            Get_force.close()
        if robot is not None:
            # ---------------- JAKA 替换为 Dobot：资源释放 ----------------
            robot.close()


if __name__ == '__main__':
    main()