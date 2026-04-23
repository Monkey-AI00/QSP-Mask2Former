import sys
sys.path.append('.')
from time import sleep
import dh_modbus_gripper
import time



class Gripper:
    def __init__(self):
        self.initstate = 0
        self.g_state = 0
        self.force = 20
        self.speed = 0.2

        # 初始化（开闭一次）j

        port = '/dev/ttyUSB0'
        baudrate = 115200
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()
        self.m_gripper.open(port, baudrate)
        self.m_gripper.Initialization()
        print('Send grip init')
        while (self.initstate != 1):
            self.initstate = self.m_gripper.GetInitState()
            sleep(0.2)

    def open(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.SetTargetPosition(900)
        time.sleep(3)
        print("gripper打开")

    def open1(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.SetTargetPosition(450)
        time.sleep(3)
        print("gripper打开")

# 0-1000
    def close(self):
        self.m_gripper.SetTargetPosition(220)
        time.sleep(3)
        print("gripper关闭")

    def close1(self):
        self.m_gripper.SetTargetPosition(0)
        time.sleep(3)
        print("gripper关闭")

    def gripper_close(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.close()

    def close3(self):
        self.m_gripper.SetTargetPosition(168)
        time.sleep(3)
        print("机械爪理线　mid")

if __name__ == '__main__':
    gripper = Gripper()
    # sleep(2)
    # gripper.open()

    # input()
    # sleep(5)
    # gripper.close3()
    # sleep(5)
    # gripper.close1()  #完全关闭

    # input()
    # gripper.gripper_close()
    # gripper.open()
    # gripper.open()
