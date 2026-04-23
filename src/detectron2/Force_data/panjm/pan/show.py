import json
import os
import random
import socket

import cv2
import math
import time

import numpy as np
import pandas as pd
import rtde_control
import rtde_receive

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
        self.m_gripper.SetTargetPosition(1000)
        time.sleep(3)
        print("gripper打开")

# 0-1000
    def close(self):
        self.m_gripper.SetTargetPosition(200)
        time.sleep(3)
        print("gripper关闭")

    def close1(self):
        self.m_gripper.SetTargetPosition(0)
        time.sleep(3)
        print("gripper关闭")

    def gripper_close(self):
        self.m_gripper.SetTargetSpeed = self.speed
        self.m_gripper.close()


# 连接机械臂
rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.2")

force_torque = {"time": [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [], 'z_torque': [], 'x_pos': [], 'y_pos': [], 'z_pos': [], 'x_rot': [], 'y_rot': [], 'z_rot': []}

class PID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.0
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.output = 0.0

    def update(self, feedback_value):
        error = self.SetPoint - feedback_value
        # print(error)
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        # print(delta_time, self.sample_time)
        delta_error = error - self.last_error
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error  # 比例
            self.ITerm += error * delta_time  # 积分
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time  # 微分
            self.last_time = self.current_time
            self.last_error = error
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            # print(1)

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time


def force_control(setpoint, force_con, action,pid):
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500

    start = time.time()

    p_now = rtde_r.getActualTCPPose()
    f_now = rtde_r.getActualTCPForce()

    for i in range(6):
        pid[i].SetPoint = setpoint[i]
    pos = []

    # pos_keep = []
    # print("keep:", keep)

    if force_con == 1:
        for i in range(6):
            # x:侧面,y:正面
            pid[i].update(f_now[i])
            p = pid[i].output
            if i == 4 or i == 1:
                p = -p + action[i]
            elif i == 0:
                p = p - action[i]
            else:
                p = p - action[i]
            pos.append(p)
    pos_keep = rtde_c.poseTrans(p_now, pos)
    # print("pos_keep:", pos_keep)

    for y in range(50):
        rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)

    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)
    return f_now,p_now

def servo_move(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 1.0, 1, 0.5, 0.2, 500
    rtde_c.servoL(pos,velocity,acceleration,dt,lookahead_time,gain)
    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)

def assembly_over(force_con, f, setpoint, start_pos):
    print("回到初始位置")
    pos = start_pos
    pos[2] = rtde_r.getActualTCPPose()[2]
    servo_move(pos)
    # f = [0, 0, 0, 0, 0, 0]
    # z = [0, 0, -0.005, 0, 0, 0]
    # while True:
    #     force_control(force_con, f, setpoint, z)
    #     p = rtde_r.getActualTCPPose()
    #     if p[2] > 0.145:
    #         break
    # print("111111111111111111111111111111111")
    # return f

def save_history( path, history, name):
    if not os.path.exists(path):
        os.makedirs(path)
    name = os.path.join(path, name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False, encoding='utf-8')

def force_control_1(setpoint1, force_torque, limit, pid):
    force_con, assembly_times = 1, 0
    # setpoint2 = [F_zero[0], F_zero[1], F_zero[2] , F_zero[3], F_zero[4], F_zero[5]]
    # action = [0,0,-0.002,0,0,0]
    action = [0, 0, -0.002, 0, 0, 0]
    i = 0
    while force_con==1:
        f, pos = force_control(setpoint1, force_con, action, pid)
        # print(setpoint1)
        # print(f)
        # print('步数',i)
        force_torque["time"].append(time.time() - starttime)
        force_torque["x_force"].append(f[0])
        force_torque["y_force"].append(f[1])
        force_torque["z_force"].append(f[2])
        force_torque["x_torque"].append(f[3])
        force_torque["y_torque"].append(f[4])
        force_torque["z_torque"].append(f[5])

        # force_torque["time"].append(time.time() - starttime)

        force_torque["x_pos"].append(pos[0])
        force_torque["y_pos"].append(pos[1])
        force_torque["z_pos"].append(pos[2])
        force_torque["x_rot"].append(pos[3])
        force_torque["y_rot"].append(pos[4])
        force_torque["z_rot"].append(pos[5])
        i += 1
        # print("11111111111111111111111")
        if f[2] > F_zero[2] + limit:
            sleep(1)
            force_con=0
            print("break")
            break
            # sleep(2)
        # a = time.time()-starttime
        # if a > 3:
        #     break



if __name__ == '__main__':
    starttime = time.time()
    pid = []
    dt = 1.0 / 300

    m = 0.00008
    o = m / 100
    P = [0, 0, m, 0, 0, 0]
    I = [0, 0, o / 10, 0, 0, 0]
    # m = 0.00008
    # o = m / 100
    # n =  0.00008
    # # # P = [0, 0, m, 0, 0, 0]
    # # # I = [0, 0, o / 10, 0, 0, 0]
    # P = [m, m, m, m * 200, m * 200, m * 200]
    # I = [o / 10, o / 10, o / 10, o * 400, o * 400, o * 400]
    D = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        p = PID(P[i], I[i], D[i])
        p.setSampleTime(dt)
        pid.append(p)
    # usb
    # startpos = [-0.6208340688217945, -0.01799439049253482, 0.3288208554012073,0, math.pi, 0]
    # rtde_c.moveL(startpos)

    # usblan
    # startpos = [-0.5466338645551002, 0.12309744671836524, 0.32679097131888485,0, math.pi, 0]
    # rtde_c.moveL(startpos)

    # usbhei
    # startpos = [-0.5207727322669193, -0.2852935382944224,  0.23004783341917945, 0, math.pi, 0]
    # # 轴孔倾斜
    # startpos1 = [-0.5308802766926926, -0.2841791986840468, 0.23005302621701806, -0.0074098798123915345, 3.085885446781733, 0.02582846166225255]
    # 初始位置，10.10确定
    # startpos2 = [-0.7087141336571776, 0.027219858594480797, 0.28762611818876105, 2.2212916169815555, 2.221449605195147, -3.453311062760046e-05]
    # startpos2 = [-0.7080141336571776, 0.027219858594480797, 0.28762611818876105, 2.2212916169815555, 2.221449605195147,-3.453311062760046e-05]
    # carpos ，10.11
    # startpos2 = [-0.7067979817771683, 0.021868895639689058, 0.3380254587131028, -2.234900550971707, -2.194083304870828, 0.008448228732142685]

    # start = [-0.7080565613778299, 0.021099214273542683, 0.3962058585687662, -2.2310903943814284, -2.1923785440750523, 0.0006015838817492948]
    start = [-0.7080565613778299, 0.021099214273542683, 0.3982058585687662,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    rtde_c.moveL(start)
    sleep(2)
    gripper = Gripper()
    sleep(2)
    startpose_wang = [-0.759830465201794, 0.037658880147330646, 0.26871837126993375,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    rtde_c.moveL(startpose_wang)
    sleep(3)
    gripper.close1()
    startpose_wang1 = [-0.759830465201794, 0.037658880147330646, 0.36671837126993375,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    rtde_c.moveL(startpose_wang1)
    sleep(2)
    '''力控前偏置位置设置1'''
    # startpose_wang2 = [-0.7601350312821558, 0.007918180791062308, 0.28248145580719985, -2.231107322777417, -2.1926159744580653, 0.0003942823008356865]
    startpose_wang2 = [-0.7604350312821558, 0.007418180791062308, 0.28448145580719985,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    rtde_c.moveL(startpose_wang2)
    sleep(3)
    # print(rtde_r.getActualTCPPose())


    force_con, assembly_times = 1, 0
    F_zero = rtde_r.getActualTCPForce()
    print('力',F_zero)
    setpoint1 = [F_zero[0], F_zero[1], F_zero[2]+10 , F_zero[3], F_zero[4], F_zero[5]]
    limit = 11

    force_control_1(setpoint1, force_torque, limit, pid)
    # setpoint2 = [F_zero[0], F_zero[1], F_zero[2] , F_zero[3], F_zero[4], F_zero[5]]
    # action = [0,0,-0.002,0,0,0]
    # action = [0, 0,-0.002,0, 0, 0]
    # i = 0
    # while True:
    #     f, pos = force_control(setpoint1, force_con, action, pid)
    #     # print(setpoint1)
    #     # print(f)
    #     # print('步数',i)
    #     force_torque["time"].append(time.time()-starttime)
    #     force_torque["x_force"].append(f[0])
    #     force_torque["y_force"].append(f[1])
    #     force_torque["z_force"].append(f[2])
    #     force_torque["x_torque"].append(f[3])
    #     force_torque["y_torque"].append(f[4])
    #     force_torque["z_torque"].append(f[5])
    #
    #     # force_torque["time"].append(time.time() - starttime)
    #
    #     force_torque["x_pos"].append(pos[0])
    #     force_torque["y_pos"].append(pos[1])
    #     force_torque["z_pos"].append(pos[2])
    #     force_torque["x_rot"].append(pos[3])
    #     force_torque["y_rot"].append(pos[4])
    #     force_torque["z_rot"].append(pos[5])
    #     i += 1
    #
    #     if f[2]>F_zero[2] + 15:
    #         break
    #         gripper.open()
    #         sleep(2)
    #     # a = time.time()-starttime
    #     # if a > 3:
    #     #     break


    # path = './shuju'
    # save_history( path, force_torque, 'pih2-xie2-2.csv')
    print('网线装配结束')
    gripper.open()
    time.sleep(1)


    startpose_wang3 = [-0.7601350312821558, 0.007918180791062308, 0.38448145580719985,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose_wang3)
    sleep(3)
    print('ok1')


    startpose2 = [-0.7629484314365551, -0.024775176535773425, 0.28614695561079817, -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose2)
    sleep(3)
    gripper.close1()
    sleep(2)
    startpose2_1 = [-0.7629484314365551, -0.024775176535773425, 0.38414695561079817,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose2_1)
    sleep(3)
    '''力控前偏置位置设置2'''
    # startpose2_2 = [-0.7634069232449348, -0.05436830404194509, 0.2953524693930417, -2.2309874044503006, -2.19262841471277, 0.0003764139921783979]
    startpose2_2 = [-0.7637069232449348, -0.05486830404194509, 0.2973524693930417,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose2_2)
    sleep(4)

    F_zero = rtde_r.getActualTCPForce()
    print('力', F_zero)
    setpoint1 = [F_zero[0], F_zero[1], F_zero[2]+15 , F_zero[3], F_zero[4], F_zero[5]]
    limit = 20
    force_control_1(setpoint1, force_torque, limit, pid)

    print('2装配结束')
    gripper.open()
    time.sleep(2)
    # for i in range(300):
    startpose2_3 = [-0.7634069232449348, -0.05436830404194509, 0.3973524693930417,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    servo_move(startpose2_3)
    sleep(2)
    print('ok2')

    startpose_usb = [-0.7664212158739068, -0.0934059415346996, 0.27565079717608665,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose_usb)
    sleep(3)
    gripper.close1()
    sleep(1)
    startpose_usb1 = [-0.7664212158739068, -0.0934059415346996, 0.37365079717608665, -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose_usb1)
    sleep(3)
    '''力控前偏置位置设置3'''
    # startpose_usb2 = [-0.7672419051791606, -0.12308895368642439, 0.28782421245593776, -2.2312220675596737, -2.192494235471103, 0.0005160420137148046]
    startpose_usb2 = [-0.7675419051791606, -0.12308895368642439, 0.28982421245593776,  -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose_usb2)
    sleep(3)

    F_zero = rtde_r.getActualTCPForce()
    print('力', F_zero)
    setpoint1 = [F_zero[0], F_zero[1], F_zero[2] + 15, F_zero[3], F_zero[4], F_zero[5]]

    limit = 22
    force_control_1(setpoint1, force_torque, limit, pid)

    print('usb装配结束')
    gripper.open()
    time.sleep(1)
    startpose_usb3 = [-0.7672419051791606, -0.12308895368642439, 0.38982421245593776, -2.221331775940252, 2.221468531666118, -3.860149961047668e-05]
    # for i in range(300):
    servo_move(startpose_usb3)
    # servo_move(start)
    sleep(2)
    print('ok3')