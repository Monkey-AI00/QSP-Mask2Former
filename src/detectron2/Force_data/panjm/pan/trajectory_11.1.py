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
import dh_modbus_gripper
import sys
sys.path.append('.')
from time import sleep

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

#setpoint：设定的目标值（6维向量）。force_con：一个控制标志，决定是否启用力控制模式。action：一个动作修正量，用于调整控制输出（6维向量）。pid：一个包含6个 PID 控制器的列表，每个控制器控制一个维度的力。
def force_control(setpoint, force_con, action, pid, desired_y, desired_z):
    
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500
    # dt：时间步长，机械臂每次更新之间的间隔时间，这里是 1/300 秒。即1/300秒运行一次
    # velocity：机械臂的运动速度。
    # acceleration：机械臂的加速度。
    # lookahead_time 和 gain：这些参数用于控制机械臂的路径规划和伺服控制，定义运动的响应速度和平滑性。

    start = time.time()

    p_now = rtde_r.getActualTCPPose()       #p_now：获取当前机械臂末端工具的位置信息（TCP位置，包括位置和姿态的6个值）。
    f_now = rtde_r.getActualTCPForce()      #f_now：获取当前机械臂末端的受力情况（6个力和力矩值，x、y、z方向的力和扭矩）。

    for i in range(6):
        pid[i].SetPoint = setpoint[i]       #将每个维度的目标值（设定值）传递给对应的PID控制器。
    pos = []

    if force_con == 1:
        for i in range(6):
            # x:侧面,y:正面
            pid[i].update(f_now[i])              #对每个维度（X、Y、Z位置和旋转）进行PID控制，根据当前反馈的力或力矩值 f_now[i] 更新 PID 控制器的输出。
            p = pid[i].output                    #获取 PID 控制器的输出 p，表示计算后的修正量。
            if i == 4 or i == 1 or i == 0:
                p = -p + action[i]
            # elif i == 0:
            #     p = p - action[i]
            else:
                p = p - action[i]
            pos.append(p)                       #将修正后的输出 p 添加到位置列表 pos 中。

    # print('x方向当前位置', p_now[0])
    # print('x方向增量', pos[0])
    pos_keep = rtde_c.poseTrans(p_now, pos)     #poseTrans 方法用于将一个相对变换应用到给定的位姿上，生成一个新的位姿。
    # print('x方向目标位置', pos_keep[0])
    pos_keep[0] = p_now[0] + pos[0]
    pos_keep[1] = desired_y
    pos_keep[2] = desired_z
    # print('新位置x', pos_keep[0])
    # print('新位置y', pos_keep[1])

    for y in range(100):
        rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)  #通过 servoL 命令，将机械臂通过直线运动（Linear motion）逐步移动到新的目标位置 pos_keep。这个过程重复 50 次以保持平稳过渡。

    end = time.time()               #计算整个控制周期所用的时间，并根据 dt 进行延时补偿，确保每次控制更新以 dt 为步长。
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)
    return f_now, p_now              #返回当前的力反馈 f_now 和位置反馈 p_now，这些值可以用于后续的分析或控制反馈。

def servo_move(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 1.0 / 3, 3, 1.5, 0.2, 500
    rtde_c.servoL(pos,velocity,acceleration,dt,lookahead_time,gain)
    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)

def save_history(path, history, name):
    if not os.path.exists(path):
        os.makedirs(path)
    name = os.path.join(path, name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False, encoding='utf-8')

def servo_move_slow(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 1.0 , 3, 1.5, 0.2, 500
    rtde_c.servoL(pos,velocity,acceleration,dt,lookahead_time,gain)
    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)


if __name__ == '__main__':
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

    # 师姐
    # startpos = [-0.5716569195689681, -0.1349873890313325, 0.4356084409834532, -2.213731452268282, -2.220085875959799, -0.0220181522906126]
    # 插入前紧贴位置,10.31日，LYH
    #[-0.7246677139217135, -0.10592810075748238, 0.2781135437313704, -3.1353200143776094, -5.078079473915725e-05, -0.00045776113673870976]
    # 抓取插头位置，10.31日，LYH
    # [-0.6232025295357336, -0.10592179912709998, 0.22742261853087054, -3.135332457486525, 0.00011016229219391172, -0.0003855980137974397]

    gripper = Gripper()
    sleep(2)
    # 抓取插头位置，10.31日，LYH
    startpos = [-0.6232025295357336, -0.10592179912709998, 0.22642261853087054, -3.135332457486525, 0.00011016229219391172, -0.0003855980137974397]
    servo_move_slow(startpos)
    time.sleep(3)
    gripper.close1()

    force_con, assembly_times = 1, 0
    F_zero = rtde_r.getActualTCPForce()
    # print(F_zero)
    setpoint1 = [F_zero[0] + 15, F_zero[1], F_zero[2], F_zero[3], F_zero[4], F_zero[5]]
    # action = [0,0,-0.002,0,0,0] 
    action = [0, 0, 0, 0, 0, 0]

    # 参数定义
    c = np.linspace(0, 18, 6000)  # z 生成范围
    r = 0.0001  # 半径缩放因子
    # 定义位置突变阈值
    POSITION_JUMP_THRESHOLD = 0.003  # 3毫米

    # 移动到紧贴位置（带误差，y方向3mm，z方向3mm）
    # startpos2 = [-0.7246677139217135, -0.10292810075748238, 0.2811135437313704, -3.135320014377609 sleep(2)4, -5.078079473915725e-05, -0.00045776113673870976]
    # 师姐相机目标位置，先上后前进
    startpos2 = [-0.6232025295357336, -0.10592179912709998, 0.27912186707054654, -3.1353784169112777,3.955488120540533e-05, -2.1846927787759983e-05]
    servo_move(startpos2)
    time.sleep(3)

    # startpos2 = [-0.7448544289462319, -0.10853027785696967, 0.28012186707054654, -3.1353784169112777, 3.955488120540533e-05, -2.1846927787759983e-05]

    startpos2 = [ -0.7448605473200092, -0.10755249415082235, 0.2781307197147147, -3.135339474830095, 0.0001551260927985827, -0.00030621688245537017]
    servo_move(startpos2)
    time.sleep(3)

     # 初始化位置跟踪变量
    found_hole = False
    previous_x = startpos2[0]
    fixed_y = startpos2[1]      
    fixed_z = startpos2[2]

    for i in range(len(c)):
        current_time = time.time() - starttime

        # 获取当前机械臂的位置
        p_now = rtde_r.getActualTCPPose()
        current_x = p_now[0]

        # 计算x方向的位置变化
        delta_x = abs(current_x - previous_x)

        # 检测是否找到孔
        # print(found_hole)
        if not found_hole and delta_x > POSITION_JUMP_THRESHOLD:
            found_hole = True
            print(f"孔已找到 at time {current_time:.2f}s, x变化: {delta_x:.6f}m")
            setpoint1 = [F_zero[0] + 25, F_zero[1], F_zero[2], F_zero[3], F_zero[4], F_zero[5]]
            # 固定当前的x,y和z位置
            fixed_x = p_now[0]
            fixed_y = p_now[1]
            fixed_z = p_now[2]

        # previous_x = current_x

        if found_hole:
            # 一旦找到孔，停止y和z的移动，保持固定
            desired_y = fixed_y
            desired_z = fixed_z
        else:
            # 继续计算YZ方向的螺旋轨迹
            y = c[i] * r * np.cos(np.pi * 5 * c[i])
            z = c[i] * r * np.sin(np.pi * 5 * c[i]) * 1.5

            desired_y = startpos2[1] + y
            desired_z = startpos2[2] + z
            # print('轨迹值',c[i])
            # print('y期望',desired_y)
            # print('z期望',desired_z)

        if found_hole and (abs(p_now[0] - fixed_x) >= 0.006):  # 找到孔之后，插入6mm退出，闭合夹爪
            print('已插入6mm，退出，闭合夹爪')
            sleep(0.5)
            gripper.open1()
            # 卡扣之前，先退出4cm
            startpos3 = [-0.7038544289462319, -0.10613027785696967, 0.27912186707054654, -3.1353784169112777,3.955488120540533e-05, -2.1846927787759983e-05]
            servo_move_slow(startpos3)
            sleep(2)
            gripper.close1()
            sleep(2)
            #最后卡扣位置
            startpos4 = [-0.7523508683355917, -0.10713027785696967, 0.28012186707054654, -3.1353784169112777, 3.955488120540533e-05, -2.1846927787759983e-05]
            servo_move_slow(startpos4)
            sleep(2)
            #卡扣结束，退出的位置
            startpos5 = [-0.7038544289462319, -0.10613027785696967, 0.27912186707054654, -3.1353784169112777,
                         3.955488120540533e-05, -2.1846927787759983e-05]
            servo_move_slow(startpos5)
            sleep(2)
            break


        f, pos = force_control(setpoint1, force_con, action, pid, desired_y, desired_z)
        # print('力控')
        # 记录数据
        force_torque["time"].append(current_time)
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


        # print('判断条件')
        # if  f[0] > F_zero[0] + 50:
        #     break
        # #
        # print('走完一次')
        # if found_hole and (abs(p_now[0]-fixed_x) >= 0.008):  #找到孔之后，插入5mm结束
        #     print('已插入8mm，结束')
        #     break
        #
        # 可选：设置运行时间限制
        # if current_time > 30:  # 30秒后停止
        #     break
