import numpy as np

def skew_symmetric(p):
    """构造平移向量p的反对称矩阵"""
    px, py, pz = p
    return np.array([
        [ 0,  -pz,  py],
        [pz,   0,  -px],
        [-py, px,   0]
    ])

def force_moment_transform(S_F, S_M, T_P_SORG):
    """
    将力和力矩从传感器坐标系(S)转换到工具坐标系(T)
    输入:
        S_F: 传感器坐标系中的力向量 [f_x, f_y, f_z]
        S_M: 传感器坐标系中的力矩向量 [m_x, m_y, m_z]
        T_P_SORG: 工具坐标系中传感器原点的平移向量 [dx, dy, dz]
    输出:
        T_F: 工具坐标系中的力向量
        T_M: 工具坐标系中的力矩向量
    """
    # 转换为列向量
    S_F = np.array(S_F).reshape(3, 1)
    S_M = np.array(S_M).reshape(3, 1)
    T_P_SORG = np.array(T_P_SORG)
    
    # 构造反对称矩阵（叉乘项）
    P_skew = skew_symmetric(T_P_SORG)
    
    # 力分量直接相等（无旋转）
    T_F = S_F
    
    # 力矩分量需要叠加平移产生的附加力矩
    T_M = S_M + P_skew @ S_F
    
    return T_F.flatten(), T_M.flatten()

# ------------------------------
# 测试
# ------------------------------
if __name__ == "__main__":
    # 传感器坐标系中的力和力矩
    S_F = [10, 20, 30]  # [f_x, f_y, f_z]
    S_M = [50, 20, 0]     # [m_x, m_y, m_z]
    # 工具坐标系中传感器原点的位置
    T_P_SORG = [1, 0, 3]  # [dx, dy, dz]
    
    # 转换到工具坐标系
    T_F, T_M = force_moment_transform(S_F, S_M, T_P_SORG)
    
    print("工具坐标系中的力:", T_F)
    print("工具坐标系中的力矩:", T_M)