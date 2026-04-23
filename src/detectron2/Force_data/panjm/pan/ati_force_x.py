# 导入库
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

import ati_test as ati
from GripperTestPython import Gripper

# 连接机械臂
rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.2")

force_torque = {"time": [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [], 'z_torque': [], 'x_pos': [], 'y_pos': [], 'z_pos': [], 'x_rot': [], 'y_rot': [], 'z_rot': []}
# 定义了一个名为 force_torque 的字典，用于存储机械臂在实验过程中收集到的 力、力矩、位置 以及 旋转角度 的数据。这些数据可以用于分析机械臂的运动状态或在控制系统中反馈用于调整。

#这段代码定义了一个 PID控制器类，用于实现比例-积分-微分（PID）控制算法。PID控制器用于自动调整一个系统的输出，使其达到期望的设定值（SetPoint），通过反馈循环来最小化误差。
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
        error = self.SetPoint - feedback_value                      #误差计算，表示当前设定值与反馈值的差异。
        # print(error)
        self.current_time = time.time()                             #获取当前的时间戳
        delta_time = self.current_time - self.last_time             #两次更新之间的时间差，用于计算积分和微分项。
        # print(delta_time, self.sample_time)
        delta_error = error - self.last_error                       #表示当前误差与上次误差之间的差异。该值用于微分项的计算，反映了误差的变化速度（或趋势）。
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error  # 比例
            self.ITerm += error * delta_time  # 积分
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time  # 微分
            self.last_time = self.current_time   #更新时间
            self.last_error = error              #更新误差
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            # print(1)

    def setSampleTime(self, sample_time):                           #设置控制器的采样时间，控制PID控制器的更新频率。只有当经过了设定的采样时间，PID控制器才会更新输出。
        self.sample_time = sample_time

#setpoint：设定的目标值（6维向量）。force_con：一个控制标志，决定是否启用力控制模式。action：一个动作修正量，用于调整控制输出（6维向量）。pid：一个包含6个 PID 控制器的列表，每个控制器控制一个维度的力。
def force_control(setpoint, force_con, action, pid, udp_socket):    

    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500

    start = time.time()

    p_now = rtde_r.getActualTCPPose()       #p_now：获取当前机械臂末端工具的位置信息（TCP位置，包括位置和姿态的6个值）。
    # f_now = rtde_r.getActualTCPForce()    #f_now：获取当前机械臂末端的受力情况（6个力和力矩值，x、y、z方向的力和扭矩）。
    f_now = ati.read_data(udp_socket)       #api六维力传感器
    print(f_now)

    for i in range(6):
        pid[i].SetPoint = setpoint[i]       #将每个维度的目标值（设定值）传递给对应的PID控制器。
    pos = []

    # pos_keep = []
    # print("keep:", keep)

    if force_con == 1:
        for i in range(6):
            # x:侧面,y:正面
            pid[i].update(f_now[i])              #对每个维度（X、Y、Z位置和旋转）进行PID控制，根据当前反馈的力或力矩值 f_now[i] 更新 PID 控制器的输出。
            p = pid[i].output                    #获取 PID 控制器的输出 p，表示计算后的修正量。
            if i == 4 or i == 1 or i == 2:
                p = -p + action[i]
            elif i == 0:
                p = p - action[i]
            else:
                p = p - action[i]
            pos.append(p)                       #将修正后的输出 p 添加到位置列表 pos 中。
    pos_keep = rtde_c.poseTrans(p_now, pos)     #poseTrans 方法用于将一个相对变换应用到给定的位姿上，生成一个新的位姿。
    # print("pos_keep:", pos_keep)
    for y in range(50):
        rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)  #通过 servoL 命令，将机械臂通过直线运动（Linear motion）逐步移动到新的目标位置 pos_keep。这个过程重复 50 次以保持平稳过渡。

    end = time.time()               #计算整个控制周期所用的时间，并根据 dt 进行延时补偿，确保每次控制更新以 dt 为步长。
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)
    return f_now,p_now              #返回当前的力反馈 f_now 和位置反馈 p_now，这些值可以用于后续的分析或控制反馈。

#3秒1次
def servo_move_slow(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 3.0 , 3, 1.5, 0.2, 500
    rtde_c.servoL(pos,velocity,acceleration,dt,lookahead_time,gain)
    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)
    #关闭伺服，防止占用线程
    rtde_c.servoStop()

def save_history( path, history, name):
    if not os.path.exists(path):
        os.makedirs(path)
    name = os.path.join(path, name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False, encoding='utf-8')

if __name__ == '__main__':

    # 夹子初始化
    gripper = Gripper()
    time.sleep(1)

    starttime = time.time()
    pid = []
    dt = 1.0 / 300

    m = 0.00008
    o = m / 100
    P = [m, 0, 0, 0, 0, 0]
    I = [o / 10, 0, 0, 0, 0, 0]
    D = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        p = PID(P[i], I[i], D[i])
        p.setSampleTime(dt)
        pid.append(p)
   
    # 1. 创建套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 2. 绑定一个本地信息
    localaddr = ("192.168.1.100", 8899)
    udp_socket.bind(localaddr)  # 必须绑定自己电脑的ip以及port，其他的不行
    udp_socket.sendto(b"\x12\x34\x00\x42\x00\x00\x00\xFF", ("192.168.1.1", 49152))

    # 初始位置
    startpos = [-0.619373412041113, 0.10508543862005018, 0.28772181893224714, -3.135287857806526, -5.8234522048500313e-05, -0.0005165615376406536]
    servo_move_slow(startpos)
    time.sleep(2)

    # 初始位置
    startpos = [-0.619373412041113, 0.10508543862005018, 0.23772181893224714, -3.135287857806526, -5.8234522048500313e-05, -0.0005165615376406536]
    servo_move_slow(startpos)
    time.sleep(2)

    gripper.close1()

    # safepos
    safepos = [-0.619373412041113, 0.10508543862005018, 0.28772181893224714, -3.135287857806526, -5.8234522048500313e-05, -0.0005165615376406536]
    servo_move_slow(safepos)
    time.sleep(1)
    # insertpos
    insertpos = [-0.7184309955909052, -0.10764831759172329, 0.2803751600291965, -3.135319280429827, -0.00014496615100887926, -0.0005276249232834103]
    servo_move_slow(insertpos)

    force_con, assembly_times = 1, 0
    # F_zero = rtde_r.getActualTCPForce()
    F_zero = ati.read_data(udp_socket)
    print(F_zero)
    setpoint1 = [F_zero[0] + 9, F_zero[1], F_zero[2], F_zero[3], F_zero[4], F_zero[5]]
    # action = [0,0,-0.002,0,0,0] 
    action = [0, 0, 0, 0, 0, 0]
    while True:
        f, pos = force_control(setpoint1, force_con, action, pid, udp_socket)
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

        if abs(f[2] - F_zero[2]) >= 9:

            gripper.close1()
            time.sleep(1)

            outpos2 = [-0.7181543061002071, -0.14395913522860007, 0.2686252697122423, 3.1364096186503945, -0.01723700236756983, 0.009730743533185687]
            servo_move_slow(outpos2)
            time.sleep(2)
            gripper.open()
            break
        # a = time.time()-starttime
        # if a > 18:
        #     break

# 关闭套接字
udp_socket.close()

path = './shuju'
save_history( path, force_torque, 'shu3_28_wen8.csv')


rtde_c.moveJ()