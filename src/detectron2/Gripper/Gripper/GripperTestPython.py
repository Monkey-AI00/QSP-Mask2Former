import sys
import os
sys.path.append(os.path.dirname(__file__))  # 确保当前目录在 Python 路径中
sys.path.append('.')
from time import sleep
import dh_modbus_gripper
import time
import mecheye          


class left_Gripper:
    def __init__(self):
        self.initstate = 0
        self.g_state = 0
        self.force = 140
        self.speed = 0.2

        # 初始化
        port_left = '/dev/ttyUSB0'

        baudrate = 115200
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()

        self.m_gripper.open(port_left, baudrate)

        #self.m_gripper.Initialization()
        print('Left  grip init')
        while (self.initstate != 1):
            self.initstate = self.m_gripper.GetInitState()
            # sleep(0.2)

    def open(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.SetTargetPosition(900)
        # time.sleep(0.5)
        print("gripper打开")

    def close(self):
        self.m_gripper.SetTargetPosition(0)
        # time.sleep(0.5)
        print("gripper关闭")

    def gripper_close(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.close()

    def close1(self):
        self.m_gripper.SetTargetPosition(200)
        time.sleep(0.5)
        print("gripper关闭")

    def close3(self):
        self.m_gripper.SetTargetPosition(450)
        self.m_gripper.SetTargetForce(100)
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")
    # def parameter_grip(self, parameter):
    #     self.m_gripper.SetTargetPosition(parameter)
    #     time.sleep(1)

    # def force_min(self):
    #     self.m_gripper.SetTargetForce(20)

    # def force_max(self):
    #     self.m_gripper.SetTargetForce(100)

    # def force_parameter(self,parameter):
    #     self.m_gripper.SetTargetForce(parameter)

    # def read_gripper_force(self):
    #     self.m_gripper.GetCurrentTargetForce()

class right_Gripper:
    def __init__(self):
        self.initstate = 0
        self.g_state = 0
        self.force = 140
        self.speed = 0.01
        # 初始化（开闭一次）
        port_right = '/dev/ttyUSB2'

        baudrate = 115200
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()

        self.m_gripper.open(port_right, baudrate)

        self.m_gripper.Initialization()#每次开机初始化之后再注释掉
        print('Right  grip init')
        while (self.initstate != 1):
            self.initstate = self.m_gripper.GetInitState()
            sleep(0.2)

    def open(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.SetTargetPosition(900)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        # time.sleep(0.5)
        print("gripper打开")

    def close(self):#夹管子
        self.m_gripper.SetTargetPosition(30)
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")

    def close1(self):#夹管子
        self.m_gripper.SetTargetPosition(305)#30
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")

    def close2(self):#夹紧
        self.m_gripper.SetTargetPosition(200)
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")

    def close3(self):
        self.m_gripper.SetTargetPosition(450)
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")

    def close4(self):
        self.m_gripper.SetTargetPosition(0)
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")
    def close5(self):
        self.m_gripper.SetTargetPosition(500)
        # time.sleep(0.5)                                                                                       
        print("gripper关闭")
    def gripper_close(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.close()

if __name__ == '__main__':
    left_gripper = left_Gripper()
    # # # time.sleep(1)
    left_gripper.open()
    # # # time.sleep(2)
    left_gripper.close()

    # right_gripper = right_Gripper()
    # right_gripper.open()
    # right_gripper.close4()