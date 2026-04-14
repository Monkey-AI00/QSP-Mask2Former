import sys
import os
sys.path.append(os.path.dirname(__file__))
from time import sleep
import dh_modbus_gripper

# 单个夹爪 纯净版
class Gripper:
    def __init__(self):
        self.initstate = 0
        self.speed = 0.2

        # ------------------- 关键：只留一个夹爪 -------------------
        port = '/dev/ttyUSB0'  # 你现在唯一的USB串口
        # ---------------------------------------------------------

        baudrate = 115200
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()

        # 打开串口
        self.m_gripper.open(port, baudrate)
        print("串口打开成功")

        # 使能 + 初始化（这一步会让红灯变绿）
        self.m_gripper.Initialization()
        print("夹爪初始化中...")

        # 等待初始化完成
        while self.initstate != 1:
            self.initstate = self.m_gripper.GetInitState()
            sleep(0.2)

        print("✅ 夹爪初始化成功！可以控制了！")

    def open(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.SetTargetPosition(900)
        print("夹爪打开")

    def close(self):
        self.m_gripper.SetTargetPosition(0)
        print("夹爪闭合")

if __name__ == '__main__':
    # 1. 先给串口权限（必须运行一次）
    # sudo chmod 666 /dev/ttyUSB0

    # 2. 创建夹爪对象（自动初始化 + 使能）
    gripper = Gripper()
    sleep(1)
    gripper.open()
    sleep(2)
    gripper.close()
