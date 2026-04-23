import numpy as np


def skew_symmetric(p):
    """构造向量 p 的反对称矩阵，使得 skew(p) @ a = p × a"""
    px, py, pz = p
    return np.array([
        [0,   -pz,  py],
        [pz,   0,  -px],
        [-py, px,   0]
    ], dtype=float)


def rotate_wrench(S_F, S_M, R_TS):
    """
    先做旋转：将力/力矩从传感器坐标系 S 旋转到工具坐标系 T

    参数:
        S_F: S系中的力向量 [fx, fy, fz]
        S_M: S系中的力矩向量 [mx, my, mz]
        R_TS: 从 S 到 T 的旋转矩阵，满足 v_T = R_TS @ v_S

    返回:
        T_F_rot: 旋转后的力（T系）
        T_M_rot: 旋转后的力矩（T系，但参考点仍是传感器原点）
    """
    S_F = np.asarray(S_F, dtype=float).reshape(3, 1)
    S_M = np.asarray(S_M, dtype=float).reshape(3, 1)
    R_TS = np.asarray(R_TS, dtype=float).reshape(3, 3)

    T_F_rot = R_TS @ S_F
    T_M_rot = R_TS @ S_M

    return T_F_rot.flatten(), T_M_rot.flatten()


def translate_wrench_in_tool(T_F, T_M_at_sensor_origin, T_P_SORG):
    """
    在工具坐标系 T 下做参考点平移修正

    已知:
        - T_F: 力，表达在 T 系
        - T_M_at_sensor_origin: 力矩，表达在 T 系，且参考点是传感器原点 SORG
        - T_P_SORG: 传感器原点 SORG 在工具坐标系 T 下的位置向量

    计算:
        工具原点处的力矩 = 传感器原点处的力矩 + p × F

    参数:
        T_F: T系中的力向量
        T_M_at_sensor_origin: T系中的力矩向量（参考点为传感器原点）
        T_P_SORG: 传感器原点在工具坐标系 T 下的位置向量 [dx, dy, dz]

    返回:
        T_F_out: 工具坐标系下的力
        T_M_out: 工具坐标系下、工具原点处的力矩
    """
    T_F = np.asarray(T_F, dtype=float).reshape(3, 1)
    T_M_at_sensor_origin = np.asarray(T_M_at_sensor_origin, dtype=float).reshape(3, 1)
    T_P_SORG = np.asarray(T_P_SORG, dtype=float).reshape(3)

    P_skew = skew_symmetric(T_P_SORG)

    T_F_out = T_F
    T_M_out = T_M_at_sensor_origin + P_skew @ T_F

    return T_F_out.flatten(), T_M_out.flatten()


def force_moment_transform(
    S_F,
    S_M,
    T_P_SORG=None,
    apply_rotation=False,
    R_TS=None
):
    """
    将补偿后的力/力矩从传感器坐标系 S 转换到工具坐标系 T

    正确物理顺序：
        1) 先旋转：S -> T
        2) 再平移：从传感器原点换算到工具原点

    参数:
        S_F: S系中的力
        S_M: S系中的力矩（参考点为传感器原点）
        T_P_SORG: 传感器原点在工具坐标系 T 中的位置向量
        apply_rotation: 是否执行旋转
        R_TS: 从 S 到 T 的旋转矩阵，满足 v_T = R_TS @ v_S

    返回:
        F_out, M_out
        - 若 apply_rotation=False:
            返回仍在 S 系下的数据（此时不建议再使用 T_P_SORG）
        - 若 apply_rotation=True:
            返回 T 系下的数据，其中力矩参考点为工具原点
    """
    F_out = np.asarray(S_F, dtype=float).flatten()
    M_out = np.asarray(S_M, dtype=float).flatten()

    if not apply_rotation:
        # 不旋转时，理论上 T_P_SORG 也不应直接参与，因为它是 T 系下定义的。
        # 为避免误用，这里直接返回 S 系结果。
        return F_out, M_out

    if R_TS is None:
        raise ValueError("apply_rotation=True 时必须提供 R_TS")

    # 第一步：先从 S 旋转到 T
    T_F, T_M_at_sensor_origin = rotate_wrench(F_out, M_out, R_TS)

    # 第二步：如果提供了 T_P_SORG，则在 T 系下做参考点平移
    if T_P_SORG is not None:
        T_F, T_M = translate_wrench_in_tool(T_F, T_M_at_sensor_origin, T_P_SORG)
    else:
        T_M = T_M_at_sensor_origin

    return T_F, T_M


# ------------------------------
# 测试
# ------------------------------
if __name__ == "__main__":
    S_F = [10, 20, 30]
    S_M = [50, 20, 0]

    # 传感器原点在工具系 T 中的位置
    T_P_SORG = [1, 0, 3]

    theta = np.deg2rad(90)
    R_TS = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    F_out, M_out = force_moment_transform(
        S_F, S_M,
        T_P_SORG=T_P_SORG,
        apply_rotation=True,
        R_TS=R_TS
    )

    print("T系下的力:", F_out)
    print("T系下的力矩:", M_out)