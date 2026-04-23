import math
import os
import random
import time
import socket
import json
import numpy as np
import pandas as pd
import rtde_control
import rtde_receive
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

writer = SummaryWriter("runs/" + "force")


class PID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
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


class Force_control(object):
    def __init__(self):
        # 连接学习算法
        HOST = ''
        PORT = 10888
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.bind((HOST, PORT))
        self.soc.listen()
        self.conn, self.addr = self.soc.accept()
        print('...connected from:', self.addr)

        # 连接机械臂
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.2")
        self.dt = 1.0 / 300
        self.velocity = 3
        self.acceleration = 1.5
        self.lookahead_time = 0.2
        self.gain = 500
        self.pid = []

        # 初始化力控
        m = 0.000008
        o = m / 100
        P = [m, m, 0.000008 * 5, m * 200, m * 200, m * 200]
        I = [o / 10, o / 10, 0.000008 / 2000, o * 400, o * 400, o * 400]
        D = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            p = PID(P[i], I[i], D[i])
            p.setSampleTime(self.dt)
            self.pid.append(p)

    def receive(self):
        data = 0
        try:
            data = self.conn.recv(2048, 0x40)
        except BlockingIOError as err:
            pass
        if data:
            data = data.decode('utf-8')
            data = json.loads(data)
            data = [float(i) for i in data]
        else:
            data = [0, 0, 0, 0, 0, 0, 0]
        return data

    def send(self, data):
        data = json.dumps(list([str(i) for i in data]))
        data = data.encode('UTF-8')
        self.conn.sendall(data)

    def Getdate(self, f):
        pos = self.rtde_r.getActualTCPPose()
        data = np.append(pos, f)
        return data

    def Getdatej(self, f):
        pos = self.rtde_r.getActualQ()  # 获取关节角度为弧度
        data = np.append(pos, f)
        return data

    def reset(self, start_pos, setpoint):
        # print(start_pos)
        for i in range(300):
            self.servo_move(start_pos)

        random_x = random.uniform(0, 4)
        # 生成-10至10之间的随机数
        random_x = (random_x - 2) / 2000
        # np.random.seed(self.seed)
        random_y = random.uniform(0, 4)
        # 生成-10至10之间的随机数
        random_y = (random_y - 2) / 2000
        # 生成-0.05至0.05之间的随机数
        # np.random.seed(self.seed)
        random_alaph = random.uniform(0, 6)
        # random_alaph = (random_alaph - 3) * 2.5 / 1.5
        random_alaph = (random_alaph - 3) / 1.5

        # 生成-0.05至0.05之间的随机数
        # np.random.seed(self.seed)
        random_beta = random.uniform(0, 6)
        # random_beta = (random_beta - 3) * 2.5 / 1.5
        random_beta = (random_beta - 3) / 1.5
        # random_pos = [start_pos[0] + random_x, start_pos[1] + random_y,
        #               start_pos[2], start_pos[3] + random_alaph * math.pi / 180,
        #               start_pos[4] + random_beta * math.pi / 180, start_pos[5]]
        random_pos = [start_pos[0], start_pos[1],
                      start_pos[2], start_pos[3] + random_alaph * math.pi / 180,
                      start_pos[4] + random_beta * math.pi / 180, start_pos[5]]
        # self.rtde_c.moveL(random_pos)
        for i in range(100):
            self.servo_move(random_pos)

        f1 = self.rtde_r.getActualTCPForce()
        for i in range(6):
            f[i] = abs(f1[i] - setpoint[i])
        random_state = self.Getdatej(f)

        return random_state, f

    def next_step(self, f, data, setpoint):
        # self.rtde_c.moveL(data)
        data1 = [i for i in data]
        for i in range(1):
            f, f_now = self.force_control(1, f, setpoint, data1)
        # p_now = self.rtde_r.getActualTCPPose()
        # pos1 = self.rtde_c.poseTrans(p_now, data)
        # for i in range(100):
        #     self.servo_move(pos1)
        # f = self.force_control(force_con, f, setpoint, data)
        state_pos = self.Getdate(f)
        state_joint = self.Getdatej(f)
        return state_pos, state_joint, f, f_now

    def assembly_over(self, force_con, f, setpoint, start_pos):
        print("回到初始位置")
        pos = start_pos
        pos[2] = self.rtde_r.getActualTCPPose()[2]
        self.servo_move(pos)
        f = [0, 0, 0, 0, 0, 0]
        z = [0, 0, -0.005, 0, 0, 0]
        while True:
            self.force_control(force_con, f, setpoint, z)
            p = self.rtde_r.getActualTCPPose()
            if p[2] > 0.145:
                break
        print("111111111111111111111111111111111")
        return f

    def servo_move(self, pos):
        start = time.time()
        self.rtde_c.servoL(pos,
                           self.velocity,
                           self.acceleration,
                           self.dt,
                           self.lookahead_time,
                           self.gain)
        end = time.time()
        duration = end - start
        if duration < self.dt:
            time.sleep(self.dt - duration)

    def force_control(self, force_con, f, setpoint, action):
        start = time.time()

        p_now = self.rtde_r.getActualTCPPose()
        f_now = self.rtde_r.getActualTCPForce()

        for i in range(6):
            writer.add_scalar("force" + str(i), f_now[i])
            self.pid[i].SetPoint = setpoint[i]
            # print(f_now)
            # print(setpoint)
            # print(f)
            if abs(f_now[i]) > f[i]:
                f[i] = abs(f_now[i])
                writer.add_scalar("force_max" + str(i), f[i])
            # print(f)
        pos = []

        if force_con == 1:
            for i in range(6):
                self.pid[i].update(f_now[i])
                p = self.pid[i].output
                if i == 4 or i == 1:
                    p = -p - action[i]
                else:
                    if i == 3:
                        p = p + action[5]
                    elif i == 5:
                        p = p + action[3]
                    else:
                        p = p + action[i]
                pos.append(p)
        pos1 = self.rtde_c.poseTrans(p_now, pos)
        if action[2] != 0 and action[0] != 0:
            print("*************************************************************")
            for y in range(50):
                self.rtde_c.servoL(pos1,
                                   self.velocity,
                                   self.acceleration,
                                   self.dt,
                                   self.lookahead_time,
                                   self.gain)
        else:
            self.rtde_c.servoL(pos1,
                               self.velocity,
                               self.acceleration,
                               self.dt,
                               self.lookahead_time,
                               self.gain)

        end = time.time()
        duration = end - start
        if duration < self.dt:
            time.sleep(self.dt - duration)
        return f, f_now

    def save_history(self, history, name):
        if not os.path.exists('./history/test/real_test'):
            os.makedirs('./history/test/real_test')
        name = os.path.join('history/test/real_test', name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')

if __name__ == '__main__':
    print("##################开启力控#####################")
    force_torque = {'time': [], 'step': [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [],
                    'z_torque': []}
    force_control = Force_control()
    force_con, u = 1, 0
    f = [0, 0, 0, 0, 0, 0]
    start_pos = [-0.5182844909030594, -0.2834964395061327, 0.22098068953923522, 0, math.pi, 0]
    F_zero = force_control.rtde_r.getActualTCPForce()
    setpoint = [F_zero[0], F_zero[1], F_zero[2], F_zero[3], F_zero[4], F_zero[5]]
    setpoint1 = [F_zero[0], F_zero[1], F_zero[2] + 40, F_zero[3], F_zero[4], F_zero[5]]
    t0 = time.time()
    while True:
        sig = force_control.receive()
        # 无数据传输
        if sig[6] == 0:
            action = [0, 0, 0, 0, 0, 0]
            f, f_now = force_control.force_control(force_con, f, setpoint1, action)
            # t1 = time.time()-t0
            # force_torque['time'].append(t1)
            # force_torque['step'].append(u)
            # force_torque['x_force'].append(f_now[0] - setpoint[0])
            # force_torque['y_force'].append(f_now[1] - setpoint[1])
            # force_torque['z_force'].append(f_now[2] - setpoint[2])
            # force_torque['x_torque'].append(f_now[3] - setpoint[3])
            # force_torque['y_torque'].append(f_now[4] - setpoint[4])
            # force_torque['z_torque'].append(f_now[5] - setpoint[5])
            # force_control.save_history(force_torque, 'force_torque.csv')
        # 回到初始位置
        elif sig[6] == 1:
            print("_______________reset_________________________")
            # start_pos = [0.3455031180364852, 0.15030846974605286, 0.14409661811206914, 0, math.pi, 0]
            start_pos = [-0.5182844909030594, -0.2834964395061327, 0.22098068953923522, 0, math.pi, 0]
            data, f = force_control.reset(start_pos, setpoint)
            F_zero = force_control.rtde_r.getActualTCPForce()
            setpoint = [F_zero[0], F_zero[1], F_zero[2], F_zero[3], F_zero[4], F_zero[5]]
            setpoint1 = [F_zero[0], F_zero[1], F_zero[2] + 40, F_zero[3], F_zero[4], F_zero[5]]
            force_control.send(data)
        # 处理网络输出
        elif sig[6] == 2:
            print("_______________next step_________________________")
            action_step = sig[:6]
            print(f)
            pos_s, joint_s, f, f_now = force_control.next_step(f, action_step, setpoint1)
            t1 = time.time() - t0
            # force_torque['time'].append(t1)
            # force_torque['step'].append(u)
            # force_torque['x_force'].append(f_now[0] - setpoint[0])
            # force_torque['y_force'].append(f_now[1] - setpoint[1])
            # force_torque['z_force'].append(f_now[2] - setpoint[2])
            # force_torque['x_torque'].append(f_now[3] - setpoint[3])
            # force_torque['y_torque'].append(f_now[4] - setpoint[4])
            # force_torque['z_torque'].append(f_now[5])
            # force_control.save_history(force_torque, 'force_torque.csv')
            state = np.append(pos_s, joint_s)
            force_control.send(state)
            f = [0, 0, 0, 0, 0, 0]
        # 装配结束
        elif sig[6] == 3:
            u += 1
            print("_______________assembly over_________________________")
            f = force_control.assembly_over(force_con, f, setpoint, start_pos)
