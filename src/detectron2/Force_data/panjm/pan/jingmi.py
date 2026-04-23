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
#定义数据结构，存储机械臂的力矩数据和位姿数据
force_torque = {"time": [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [], 'z_torque': [], 'x_pos': [], 'y_pos': [], 'z_pos': [], 'x_rot': [], 'y_rot': [], 'z_rot': []}
#一、PID控制器类，用于调整机械臂的运动，使其在接触物体时保持恒定的接触力
class PID:
    def __init__(self, P, I, D):
        self.Kp = P #比例系数
        self.Ki = I#积分系数
        self.Kd = D
        self.sample_time = 0.0#采样时间
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        self.SetPoint = 0.0#目标值
        self.PTerm = 0.0#积分项
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.output = 0.0#PID输出

    def update(self, feedback_value):
        error = self.SetPoint - feedback_value#计算误差
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error  # 比例项
            self.ITerm += error * delta_time  # 积分项
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time  # 微分
            self.last_time = self.current_time
            self.last_error = error
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)


    def setSampleTime(self, sample_time):
        self.sample_time = sample_time
#二、力控制核心函数
def force_control(setpoint,mode, action,pid):
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 500, 0.1, 0.05, 0.2, 800
    start = time.time()

    p_now = rtde_r.getActualTCPPose()
    f_now = rtde_r.getActualTCPForce()
    pos = []

    for i in range(6):
            pid[i].SetPoint = setpoint[i]#设置PID目标值（期望的力或者力矩）


        #if force_con == 1:
        # for i in range(6):
        #     pid[i].update(f_now[i])#更新PID计算
        #p = pid[i].output
    if mode=="insert":
            for i in range(6):
                    pid[i].update(f_now[i])  # 更新PID计算
                    p = pid[i].output
                    if i == 4 or i == 1:
                        p = -p + action[i]#调整y轴和RY轴的方向
                    elif i == 0:
                        p = p - action[i]#调整x轴的方向
                    else:
                        p = p - action[i]#其他方向
                    pos.append(p)
    elif mode == "extract":
            for i in range(6):
                    pid[i].update(f_now[i])  # 更新PID计算
                    p = pid[i].output
                    if i == 4 or i == 1:
                        p = -p + action[i]#调整y轴和RY轴的方向
                    elif i == 0:
                        p = p - action[i]#调整x轴的方向
                    else:
                        p = p - action[i]#其他方向
                    pos.append(p)
    pos_keep = rtde_c.poseTrans(p_now, pos)#计算新的目标位姿
    print("pos_keep:", pos_keep)

    for y in range(50):
        rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)#伺服控制，用于柔顺运动

    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)#确保控制频率稳定
    return f_now,p_now

def servo_move(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 500, 0.1, 0.05, 0.2, 800
    rtde_c.servoL(pos,velocity,acceleration,dt,lookahead_time,gain)
    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)

# def assembly_over(force_con, f, setpoint, start_pos):
#     print("回到初始位置")
#     pos = start_pos
#     pos[2] = rtde_r.getActualTCPPose()[2]
#     servo_move(pos)

def save_history( path, history, name):
    if not os.path.exists(path):
        os.makedirs(path)
    name = os.path.join(path, name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False, encoding='utf-8')


#三、主程序逻辑
if __name__ == '__main__':
    starttime = time.time()
    pid = []
    dt = 1.0 / 500#控制频率500HZ

    # P = [0, 0, m, 0, 0, 0]
    # I = [0, 0, o / 10, 0, 0, 0]
    m = 0.000008
    o = m / 100
    P = [m*0.3, m*0.3, 0.000008 * 5, m / 800, m / 800, m / 800]#PID比例系数
    I = [o / 10, o / 10, 0.000008 / 8000, o * 400, o * 400, o * 400]
    D = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        p = PID(P[i], I[i], D[i])
        p.setSampleTime(dt)
        pid.append(p)

    #初始位置（接近装配点）
    #startpos2 = [-0.5398239136863351, -0.06236168774875655, 0.1913874221127928, 2.524648847472695e-05, 3.1415505957307315, -2.689401435416394e-05]
    startpos2 = [-0.566057217953994, 0.07387238006164573, 0.20973752315104616, -4.856623721607159e-05, 3.141532893001445, 2.579752147526794e-05]
    #移动到初始位置
    rtde_c.moveL(startpos2)
    print('初始位置',rtde_r.getActualTCPPose())


    force_con, assembly_times = 1, 0
    F_zero = rtde_r.getActualTCPForce()#读取初始受力（零点校准）
    print('初始受力',F_zero)
    setpoint_insert = [F_zero[0], F_zero[1], F_zero[2]+2 , F_zero[3], F_zero[4], F_zero[5]]#目标：Z轴增加10N的力
    action_insert = [0, 0,-0.0018,0, 0, 0]#Z轴持续下压微调，步长0.002m，注意单位是米

    setpoint_extract = [F_zero[0], F_zero[1], F_zero[2]-2, F_zero[3], F_zero[4], F_zero[5]]
    action_extract = [0, 0, 0.002, 0, 0, 0]

    i = 0
    print("开始插入阶段")
    insert_success = False
    while True:
        f, pos = force_control(setpoint_insert, "insert",action_insert, pid)
    #四、存储数据
        force_torque["time"].append(time.time()-starttime)
        force_torque["x_force"].append(f[0])
        force_torque["y_force"].append(f[1])
        force_torque["z_force"].append(f[2])
        force_torque["x_torque"].append(f[3])
        force_torque["y_torque"].append(f[4])
        force_torque["z_torque"].append(f[5])
        force_torque["x_pos"].append(pos[0])
        force_torque["y_pos"].append(pos[1])
        force_torque["z_pos"].append(pos[2])
        force_torque["x_rot"].append(pos[3])
        force_torque["y_rot"].append(pos[4])
        force_torque["z_rot"].append(pos[5])
#五、终止条件，Z轴位置太低或者受力太大
        #if f[2]>F_zero[2] + 15:
            #insert_success=True
            #print("插入完成，当前Z轴位置",pos[2])
            #break
        if pos[2]<0.12:
            insert_success = True
            print("插入完成，当前Z轴位置", pos[2])
            break

    if insert_success:
        print("开始拔出阶段")
        while True:
            f, pos = force_control(setpoint_extract, "extract", action_extract, pid)

            force_torque["time"].append(time.time() - starttime)
            force_torque["x_force"].append(f[0])
            force_torque["y_force"].append(f[1])
            force_torque["z_force"].append(f[2])
            force_torque["x_torque"].append(f[3])
            force_torque["y_torque"].append(f[4])
            force_torque["z_torque"].append(f[5])
            force_torque["x_pos"].append(pos[0])
            force_torque["y_pos"].append(pos[1])
            force_torque["z_pos"].append(pos[2])
            force_torque["x_rot"].append(pos[3])
            force_torque["y_rot"].append(pos[4])
            force_torque["z_rot"].append(pos[5])
        # 五、终止条件，Z轴位置太低或者受力太大
        # if f[2]>F_zero[2] + 15:
        #     break
            if pos[2] > 0.186:
                print("拔出完成")
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