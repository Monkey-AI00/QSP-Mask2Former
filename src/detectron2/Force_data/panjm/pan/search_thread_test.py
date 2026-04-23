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

# 连接学习算法
HOST = ''
PORT = 10888
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.bind((HOST, PORT))
soc.listen()
conn, addr = soc.accept()
print('...connected from:', addr)

force_torque = {"time": [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [], 'z_torque': []}
tar = {'X': [], 'Y': [], 'Z': []}
starttime = time.time()

# 连接机械臂
rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.2")

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

class Search(object):
    def __init__(self):
        # 初始化参数
        self.reset_x = 0
        self.reset_y = 0

        self.dt = 1.0 / 300
        self.velocity = 3
        self.acceleration = 1.5
        self.lookahead_time = 0.2
        self.gain = 500
        self.pid = []

        # m = 0.000008
        # o = m / 100
        # P = [m, m, 0.000008 * 5, m * 200, m * 200, m * 200]
        # I = [o / 10, o / 10, 0.000008 / 2000, o * 400, o * 400, o * 400]
        # D = [0, 0, 0, 0, 0, 0]

        m = 0.00008
        o = m / 100
        P = [0, 0, m, 0, 0, 0]
        I = [0, 0, o / 10, 0, 0, 0]
        D = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            p = PID(P[i], I[i], D[i])
            p.setSampleTime(self.dt)
            self.pid.append(p)

    def receive(self):
        data = 0
        try:
            data = conn.recv(2048, 0x40)
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
        conn.sendall(data)

    def servo_move(self, pos):
        start = time.time()
        rtde_c.servoL(pos,
                           self.velocity,
                           self.acceleration,
                           self.dt,
                           self.lookahead_time,
                           self.gain)
        end = time.time()
        duration = end - start
        if duration < self.dt:
            time.sleep(self.dt - duration)

    def force_control(self, setpoint, force_con, action):
        dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500
        startpos = [-0.5524743627854106, -0.2819489527065604, 0.23034620802736447, 0, 2.9671, 0]
        # startpos = [-0.515286983672204, -0.2821514697386087, 0.2202129229116446, 0, math.pi, 0]
        start = time.time()

        p_now = rtde_r.getActualTCPPose()
        f_now = rtde_r.getActualTCPForce()

        for i in range(6):
            self.pid[i].SetPoint = setpoint[i]
        pos = []

        # pos_keep = []
        # print("keep:", keep)

        if force_con == 1:
            for i in range(6):
                # x:侧面,y:正面
                self.pid[i].update(f_now[i])
                p = self.pid[i].output
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
        return f_now

    def force_control_step(self, setpoint, force_con, action, step_pos, keep, pos_keep):
        dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500
        startpos = [-0.55234743627854106, -0.2819489527065604, 0.23034620802736447, 0, 2.9671, 0]
        # startpos = [-0.515286983672204, -0.2821514697386087, 0.2202129229116446, 0, math.pi, 0]
        start = time.time()

        p_now = rtde_r.getActualTCPPose()
        f_now = rtde_r.getActualTCPForce()

        for i in range(6):
            self.pid[i].SetPoint = setpoint[i]
        pos = []

        # pos_keep = []
        # print("keep:", keep)

        if force_con == 1:
            if keep == 0:
                for i in range(6):
                    # x:侧面,y:正面
                    self.pid[i].update(f_now[i])
                    p = self.pid[i].output
                    if i == 4 or i == 1:
                        p = -p - action[i]
                    elif i == 0:
                        p = p + action[i]
                    else:
                        p = p - action[i]
                    pos.append(p)
                pos_keep = rtde_c.poseTrans(pos_keep, pos)
            else:
                for i in range(6):
                    # x:侧面,y:正面
                    self.pid[i].update(f_now[i])
                    p = self.pid[i].output
                    if i == 4 or i == 1:
                        p = -p
                    elif i == 0:
                        p = p
                    else:
                        p = p - action[i]
                    pos.append(p)
                pos_keep = rtde_c.poseTrans(pos_keep, pos)
        sig = 1
        print("pos_keep:", pos_keep)
        for y in range(50):
            rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)
            pos2 = rtde_r.getActualTCPPose()
            print(abs(abs(step_pos[0] - pos2[0]) - abs(action[0])), abs(abs(step_pos[1] - pos2[1]) - abs(action[0])))
            print("pos2", pos2)
            if math.sqrt(abs(pos2[0] - startpos[0]) * abs(pos2[0] - startpos[0]) + abs(pos2[1] - startpos[1])
                         * abs(pos2[1] - startpos[1])) < 0.0035:
                # print("success:", f[2], pos[2] - 0.23034620802736447, pos[0] + 0.5567663774747263)
                # print("success:", math.sqrt(abs(pos[0] - startpos[0]) * abs(pos[0] - startpos[0]) + abs(pos[0] - startpos[0])
                #          * abs(pos[0] - startpos[0])))
                sig = 0
                break
            if abs(abs(step_pos[0] - pos2[0]) - abs(action[0])) < 0.0005 and abs(
                    abs(step_pos[1] - pos2[1]) - abs(action[1])) < 0.0005:
                sig = 2
                break

        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)
        return f_now, sig, pos_keep

    def Getdatepos(self, force):
        pos = rtde_r.getActualTCPPose()
        state = np.append(pos, force)
        return state

    def reset(self, start_pos, setpoint):
        print("start_pos", start_pos)

        for i in range(600):
            self.servo_move(start_pos)

        random_x = random.uniform(0, 4)
        # 生成-10至10之间的随机数
        random_x = (random_x - 2) / 200
        # np.random.seed(self.seed)
        random_y = random.uniform(0, 4)
        # 生成-10至10之间的随机数
        random_y = (random_y - 2) / 200

        random_pos = [random_x, random_y, 0, 0, 0, 0]
        # random_pos = [0, 0, 0, 0, 0, 0]
        pos_target = rtde_c.poseTrans(start_pos, random_pos)
        # self.rtde_c.moveL(pos_target)

        # random_pos = [start_pos[0] + random_x, start_pos[1] + random_y,
        #               start_pos[2], start_pos[3], start_pos[4], start_pos[5]]

        for i in range(300):
            self.servo_move(pos_target)

        pos = rtde_r.getActualTCPPose()
        self.reset_x = pos[0]
        self.reset_y = pos[1]
        tar['X'].append(pos[0])
        tar['Y'].append(pos[1])
        tar['Z'].append(pos[2])
        # print("初始位置:", self.reset_x, self.reset_y)
        if math.sqrt(abs(pos[0] - startpos[0]) * abs(pos[0] - startpos[0]) + abs(pos[1] - startpos[1])
                     * abs(pos[1] - startpos[1])) < 0.0035:
            sig = 0
        else:
            sig = 1

        print("sig::::::::", sig)

        f1 = rtde_r.getActualTCPForce()
        print(setpoint[2], f1[2])

        while sig == 1:
            f = self.force_control(setpoint, 1, [0, 0, 0, 0, 0, 0])
            pos1 = rtde_r.getActualTCPPose()
            if f[2] > (f1[2] - 5) or abs(pos1[2] - pos[2]) > 0.002:
                break

        force = rtde_r.getActualTCPForce()
        random_state = self.Getdatepos(force)
        random_state = np.append(np.append(random_state, pos[:2]), sig)
        return random_state

    def assembly_over(self, F_zero):
        print("回到初始位置")
        pos = rtde_r.getActualTCPPose()
        pos1 = rtde_r.getActualTCPPose()
        # setpoint = [F_zero[0], F_zero[1], F_zero[2] - 10, F_zero[3], F_zero[4], F_zero[5]]
        while True:
            action = [0, 0, -0.005, 0, 0, 0]
            pos2 = rtde_c.poseTrans(pos1, action)
            self.servo_move(pos2)
            # f = self.force_control(setpoint, 1, [0, 0, 0.005, 0, 0, 0])
            f = rtde_r.getActualTCPForce()
            pos1 = rtde_r.getActualTCPPose()
            # print(pos1[2] - pos[2])
            # if (pos1[2] - pos[2]) > 0.06:
            if (pos1[2] - pos[2]) > 0.01:
                self.save_history(path, tar, "tar.csv")
                break
        print("111111111111111111111111111111111")

    def insert(self, F_zero):
        startpos = [-0.5539743627854106, -0.28194895270656043, 0.23534620802736447, 0, 2.952, 0]
        # startpos = [-0.515286983672204, -0.2821514697386087, 0.2202129229116446, 0, math.pi, 0]
        pos = rtde_r.getActualTCPPose()
        # startpos[2] =
        for i in range(200):
            self.servo_move(startpos)
        pos1 = rtde_r.getActualTCPPose()
        while math.sqrt(abs(pos1[0] - startpos[0]) * abs(pos1[0] - startpos[0]) + abs(pos1[1] - startpos[1])
                         * abs(pos1[1] - startpos[1])) > 0.0001:
            self.servo_move(startpos)
        setpoint = [F_zero[0], F_zero[1], F_zero[2] + 20, F_zero[3], F_zero[4], F_zero[5]]
        inserttime = 0
        while True:
            inserttime += 1
            # action = [0, 0, -0.005, 0, 0, 0]
            # pos2 = rtde_c.poseTrans(pos1, action)
            # self.servo_move(pos2)
            f = self.force_control(setpoint, 1, [0, 0, -0.001, 0, 0, 0])
            # f = self.force_control(setpoint, 1, [0, 0, 0, 0, 0, 0])
            # print(f, setpoint)
            pos1 = rtde_r.getActualTCPPose()
            if inserttime % 5 == 0:
                tar['X'].append(pos1[0])
                tar['Y'].append(pos1[1])
                tar['Z'].append(pos1[2])
            if abs(pos1[2] - pos[2]) > 0.045:
                break

        print("insert over")

    def next_step(self, data, setpoint):
        dt, velocity, acceleration, lookahead_time, gain = 1.0 / 300, 3, 1.5, 0.2, 500
        startpos = [-0.5524743627854106, -0.28194895270656043, 0.23034620802736447, 0, 2.9671, 0]
        # startpos = [-0.515286983672204, -0.2821514697386087, 0.2202129229116446, 0, math.pi, 0]
        action = [i for i in data]
        # print(data1)
        sig = 1
        t1 = time.time()
        force = rtde_r.getActualTCPForce()
        step_pos = rtde_r.getActualTCPPose()

        for i in range(6):
            self.pid[i].SetPoint = setpoint[i]

        for y in range(50):
            pos = []
            p_now = rtde_r.getActualTCPPose()
            f_now = rtde_r.getActualTCPForce()
            # print(f_now)
            force = [force[i] if abs(force[i]) > abs(f[i]) else f[i] for i in range(6)]
            # print(force)
            if force_con == 1:
                for i in range(6):
                    # x:侧面,y:正面
                    # print("*******************************************")
                    self.pid[i].update(f_now[i])
                    p = self.pid[i].output
                    if i == 4 or i == 1:
                        p = -p - action[i]
                    elif i == 0:
                        p = p + action[i]
                    else:
                        p = p - action[i]
                    pos.append(p)
            pos_keep = rtde_c.poseTrans(p_now, pos)
            # print(pos_keep, "**************")
            rtde_c.servoL(pos_keep, velocity, acceleration, dt, lookahead_time, gain)
            pos2 = rtde_r.getActualTCPPose()
            tar['X'].append(pos2[0])
            tar['Y'].append(pos2[1])
            tar['Z'].append(pos2[2])
            # print(abs(abs(step_pos[0] - pos2[0]) - abs(action[0])), abs(abs(step_pos[1] - pos2[1]) - abs(action[0])))
            if math.sqrt(abs(pos2[0] - startpos[0]) * abs(pos2[0] - startpos[0]) + abs(pos2[1] - startpos[1])
                         * abs(pos2[1] - startpos[1])) < 0.0035:
                sig = 0
                break
            if abs(abs(step_pos[0] - pos2[0]) - abs(data[0])) < 0.0001 and abs(
                    abs(step_pos[1] - pos2[1]) - abs(data[1])) < 0.0001:
                sig = 2
                break
            # print(sig, "****************")
        # print(sig, "_________________")
        state_pos = rtde_r.getActualTCPPose()
        state = self.Getdatepos(force)
        state = np.append(state, [0, 0, sig])
        t2 = time.time()
        print("search time:", t2 - t1)
        return state_pos, state

    def save_history(self, path, history, name):
        if not os.path.exists(path):
            os.makedirs(path)
        name = os.path.join(path, name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')

if __name__ == '__main__':
    print("####################搜索开始##################")
    path = "./history"
    env = Search()
    startpos = [-0.5524743627854106, -0.2819489527065604, 0.23034620802736447, 0, 2.9671, 0]
    # startpos = [-0.515286983672204, -0.2821514697386087, 0.2202129229116446, 0, math.pi, 0]
    rtde_c.moveL(startpos)

    force_con, assembly_times = 1, 0
    F_zero = rtde_r.getActualTCPForce()
    setpoint1 = [F_zero[0], F_zero[1], F_zero[2] + 10, F_zero[3], F_zero[4], F_zero[5]]
    # startpos = [-0.5529743627854106, -0.2819489527065604, 0.23034620802736447, 0, 2.952, 0]
    #
    while True:
        sig = env.receive()
        # print(sig)
        # 无数据传输
        if sig[6] == 0:
            setpoint = F_zero
            action = [0, 0, 0, 0, 0, 0]
            pos = rtde_r.getActualTCPPose()
            f = env.force_control(setpoint, force_con, action)
        # 回到初始位置
        elif sig[6] == 1:
            print("_______________reset_________________________")
            setpoint = setpoint1
            data = env.reset(startpos, setpoint)
            print(data)
            env.send(data)
        # 处理网络输出
        elif sig[6] == 2:
            print("_______________next step_________________________")
            action_step = sig[:6]
            setpoint = F_zero
            # print("action:", action_step)
            pos_sta = rtde_r.getActualTCPPose()
            pos_s, feature_s = env.next_step(action_step, setpoint)
            pos = [pos_sta[i] - pos_s[i] for i in range(6)]
            # print("move:", pos)
            env.send(feature_s)
            # f = [0, 0, 0, 0, 0, 0]
        # 装配结束
        elif sig[6] == 3:
            assembly_times += 1
            print("_______________search over_________________________")
            # env.insert(F_zero)
            env.assembly_over(F_zero)