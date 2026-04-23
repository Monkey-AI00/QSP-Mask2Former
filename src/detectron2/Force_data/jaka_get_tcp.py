import struct
import time
from typing import List, Dict, Optional
from pyModbusTCP.client import ModbusClient
import numpy as np

# ------------------------------------------------------------------
# Modbus读取 TCP 末端速度、位置：X/Y/Z (m/s、m), RX/RY/RZ (rad/s、rad)
# ------------------------------------------------------------------
def regs_to_float32_be(regs: List[int]) -> float:
    """两个 16-bit big-endian 寄存器 -> float32"""
    if len(regs) != 2:
        raise ValueError(f"需要 2 个寄存器来组成 float32，当前为 {len(regs)} 个寄存器")
    b = regs[0].to_bytes(2, 'big') + regs[1].to_bytes(2, 'big')
    return struct.unpack('>f', b)[0]

def get_tcp_speed(client: ModbusClient,
                  start_addr: int) -> Optional[Dict[str, float]]:
    """
    读取 TCP 末端速度：X/Y/Z (m/s), RX/RY/RZ (rad/s)
    start_addr: PDF 中 TCP速度X 的输入寄存器地址
    """
    # 一次读 12 个寄存器 = 6 个 float32
    regs = client.read_input_registers(start_addr, 12)

    if (regs is None) or (len(regs) != 12):
        print(f"[get_tcp_speed] 读取失败，返回寄存器数量异常: {regs}")
        return None

    try:
        vx  = regs_to_float32_be(regs[0:2]) / 1000
        vy  = regs_to_float32_be(regs[2:4]) / 1000
        vz  = regs_to_float32_be(regs[4:6]) / 1000
        vrx = regs_to_float32_be(regs[6:8]) /180 * np.pi
        vry = regs_to_float32_be(regs[8:10]) /180 * np.pi
        vrz = regs_to_float32_be(regs[10:12]) /180 * np.pi
    except Exception as e:
        print(f"[get_tcp_speed] 解析 float32 失败: {e}")
        return None

     # 返回numpy数组格式: [vx, vy, vz, vrx, vry, vrz]
    return np.array([vx, vy, vz, vrx, vry, vrz])

def get_tcp_pos(client: ModbusClient,
                  start_addr: int) -> Optional[Dict[str, float]]:
    """
    读取 TCP 末端位置：X/Y/Z (m), RX/RY/RZ (rad)
    start_addr: PDF 中 TCP位置X 的输入寄存器地址
    """
    # 一次读 12 个寄存器 = 6 个 float32
    regs = client.read_input_registers(start_addr, 12)

    if (regs is None) or (len(regs) != 12):
        print(f"[get_tcp_speed] 读取失败，返回寄存器数量异常: {regs}")
        return None

    try:
        x  = regs_to_float32_be(regs[0:2]) / 1000         
        y  = regs_to_float32_be(regs[2:4]) / 1000
        z  = regs_to_float32_be(regs[4:6]) / 1000
        rx = regs_to_float32_be(regs[6:8]) /180 * np.pi
        ry = regs_to_float32_be(regs[8:10]) /180 * np.pi
        rz = regs_to_float32_be(regs[10:12]) /180 * np.pi
    except Exception as e:
        print(f"[get_tcp_pos] 解析 float32 失败: {e}")
        return None

     # 返回numpy数组格式: [x, y, z, rx, ry, rz]
    return np.array([x, y, z, rx, ry, rz])

def main():
    ROBOT_IP = "192.168.1.131"
    ROBOT_PORT = 6502
    UNIT_ID = 1

    # TCP速度X的真实地址
    TCP_SPEED_X_ADDR = 418  # TODO: 对照1.7文档改

    # TCP位置X的真实地址
    TCP_POS_X_ADDR = 406  # TODO: 对照1.7文档改

    client = ModbusClient(
        host=ROBOT_IP,
        port=ROBOT_PORT,
        unit_id=UNIT_ID,      
        auto_open=True,
        auto_close=False
    )

    if not client.open():
        print("连接 Modbus 失败")
        return

    print("连接成功，开始循环读取 TCP 速度...")

    try:
        while True:
            # start_time = time.time()
            speed = get_tcp_speed(client, TCP_SPEED_X_ADDR)
            if speed is not None:
                print(
                    f"TCP 速度: "
                    f"vx={speed[0]:.3f} m/s, vy={speed[1]:.3f} m/s, vz={speed[2]:.3f} m/s, "
                    f"vrx={speed[3]:.3f} rad/s, vry={speed[4]:.3f} rad/s, vrz={speed[5]:.3f} rad/s"
                )
            else:
                print("读取 TCP 速度失败")
            # print(f"读取时间: {time.time() - start_time:.6f} 秒") # 极限频率大概是0.00015s，6666.666666666667 Hz

            # 读取 TCP 位置
            pos = get_tcp_pos(client, TCP_POS_X_ADDR)
            if pos is not None:
                print(
                    f"TCP 位置: "
                    f"x={pos[0]:.6f} m, y={pos[1]:.6f} m, z={pos[2]:.6f} m, "
                    f"rx={pos[3]:.6f} rad, ry={pos[4]:.6f} rad, rz={pos[5]:.6f} rad"
                )
            else:
                print("读取 TCP 位置失败")
            
            time.sleep(1)  # 50 Hz，可根据你的控制周期调整

    finally:
        client.close()
        print("已关闭 Modbus 连接")


if __name__ == "__main__":
    main()
