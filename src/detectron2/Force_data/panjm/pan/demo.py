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
    # print("keep:",pos_keep)

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
    print("pos_keep:", pos_keep)

    for y in range(50):
        rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)

    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)
    return f_now,p_now

def servo_move(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500
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



if __name__ == '__main__':
    starttime = time.time()
    pid = []
    dt = 1.0 / 300

    # m = 0.00008
    # o = m / 100
    # P = [0, 0, m, 0, 0, 0]
    # I = [0, 0, o / 10, 0, 0, 0]
    m = 0.000008
    o = m / 100
    P = [m, m, 0.000008 * 5, m * 200, m * 200, m * 200]
    I = [o / 10, o / 10, 0.000008 / 2000, o * 400, o * 400, o * 400]
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
    # 10.25
    # startpos2 = [-0.5716569195689681, -0.1349873890313325, 0.4356084409834532, -2.213731452268282, -2.220085875959799, -0.0220181522906126]
    # startpos2 = [-0.4671028074005579, -0.04743910813709854, 0.3250749899212768, -2.21847984492004, -2.2185724772433555, -0.0074050923337874485]
    startpos2 = [-0.6109597138742326, 0.17391764063464732, 0.20634294000316677, 4.369815159031885e-05, -3.141566669628136, 1.7497104552545863e-05]
    #startpos2 = [-0.5387962204608504, -0.062284644712982756, 0.17473851717393266, 2.4780600040804553e-05,3.1415649673401322, 6.859270044526057e-05]
    rtde_c.moveL(startpos2)
    print(rtde_r.getActualTCPPose())


    force_con, assembly_times = 1, 0
    F_zero = rtde_r.getActualTCPForce()
    print('力',F_zero)
    setpoint1 = [F_zero[0], F_zero[1], F_zero[2]+10 , F_zero[3], F_zero[4], F_zero[5]]
    # setpoint2 = [F_zero[0], F_zero[1], F_zero[2] , F_zero[3], F_zero[4], F_zero[5]]
    # action = [0,0,-0.002,0,0,0]
    action = [0, 0,-0.002,0, 0, 0]

    i = 0
    while True:
        f, pos = force_control(setpoint1, force_con, action, pid)
        # print(setpoint1)
        # print(f)
        # print('步数',i)
        force_torque["time"].append(time.time()-starttime)
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

        #if f[2]>F_zero[2] + 15:
             #break
        if pos[2]<0.15254469767114052:
            break
        # a = time.time()-starttime
        # if a > 3:
        #     break


    path = './shuju'
    save_history( path, force_torque, 'pih2-0802-2.csv')
    print('装配结束')
    time.sleep(3)
    servo_move(startpos2)
    print('ok')
