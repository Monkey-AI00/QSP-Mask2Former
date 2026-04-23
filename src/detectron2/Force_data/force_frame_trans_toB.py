import math
import numpy as np


def euler_xyz_deg_to_rot_matrix(euler_deg):
    """
    纯 NumPy 欧拉角转旋转矩阵（去 JAKA 依赖）。
    按 XYZ 欧拉序计算：R = Rx * Ry * Rz。

    参数:
      euler_deg: [rx, ry, rz]，单位角度

    返回:
      R_B_T (3x3)
    """
    rx, ry, rz = [float(v) * math.pi / 180.0 for v in euler_deg]
    r_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ],
        dtype=np.float64,
    )
    r_y = np.array(
        [
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)],
        ],
        dtype=np.float64,
    )
    r_z = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return r_x @ r_y @ r_z

def wrench_tool_to_base(pose_w, wrench_t):
    """
    将工具坐标系下的 [Fx,Fy,Fz,Mx,My,Mz]_T 转换到基坐标系
    :param pose_w: [x, y, z, rx, ry, rz]  末端在基坐标系下的位姿
    :param wrench_t: [Fx, Fy, Fz, Mx, My, Mz] 工具坐标系 T 下的力和力矩
    :return: [Fx, Fy, Fz, Mx, My, Mz] 基坐标系 B 下的力和力矩
    """
    # 姿态单位/定义需确认：
    # 1) 当前按角度处理 [rx, ry, rz]
    # 2) 当前按 XYZ 欧拉序（Rx*Ry*Rz）
    # 若控制器返回弧度或不同欧拉序，需要在此同步调整。
    euler_deg = np.asarray(pose_w[3:], dtype=np.float64)
    r_b_t = euler_xyz_deg_to_rot_matrix(euler_deg)  # B->T
    r_t_b = r_b_t.T  # T->B

    f_t = np.asarray(wrench_t[:3], dtype=np.float64)
    m_t = np.asarray(wrench_t[3:], dtype=np.float64)

    f_b = r_t_b @ f_t
    m_b = r_t_b @ m_t

    return np.concatenate([f_b, m_b])

# 示例
if __name__ == "__main__":
    pose_w = [0, 0, 0, -113.5, -2.0, 9.0]  # [x,y,z,rx,ry,rz]，姿态按角度示例
    wrench_t = [10, 0, 0, 0.1, 0, 0]
    wrench_b = wrench_tool_to_base(pose_w, wrench_t)
    print("基坐标系下的力和力矩:", wrench_b)
