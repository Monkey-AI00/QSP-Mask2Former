from force_data import ForceDataProcessor

import time
import os
import pandas as pd
import math
import numpy as np

import ctypes
path = ("/home/adminrt/code/libjakaAPI.so")
ctypes.CDLL(path)
import jkrc

def save_history( path, history, name):
    if not os.path.exists(path):
        os.makedirs(path)
    name = os.path.join(path, name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False, encoding='utf-8')

def main():

    try:
        robot = jkrc.RC("192.168.1.129")
        robot.login()
        robot.power_on()
        robot.enable_robot()

        processor = ForceDataProcessor(force_sensor_ip="192.168.1.20", force_sensor_port=502)

        ABS = 0
        T_P_SORG = [0, 0, 125.5 * 0.001]
 
        #数据记录
        data_log = {
            'raw_x_force': [], 'raw_y_force': [], 'raw_z_force': [],
            'raw_x_torque': [], 'raw_y_torque': [], 'raw_z_torque': [],
            'x_force': [], 'y_force': [], 'z_force': [],
            'x_torque': [], 'y_torque': [], 'z_torque': []}
        keys1 = ['raw_x_force', 'raw_y_force', 'raw_z_force', 'raw_x_torque', 'raw_y_torque', 'raw_z_torque']
        keys2 = ['x_force', 'y_force', 'z_force', 'x_torque', 'y_torque', 'z_torque']

        # pos1 = [-114.710999, 483.334015, 440.908203, 3.141592653589793, -0.0, -1.5707963267948966]
        # robot.linear_move(pos1, ABS, True, 500)
        # time.sleep(5)
        # # 主循环
        # for i in range(10):
        #     # 读取传感器数据
        #     F = Get_force.read_sensor_data()
        #     ret = robot.get_tcp_position()
        #     pos = ret[1]
        #     print('实际受力：',F)
        #     print('实际姿态角：', [angle(pos[3]), angle(pos[4]), angle(pos[5])])
        #     realtime_force = F[:3]
        #     realtime_torque = F[3:]
        #     realtime_euler = [angle(pos[3]), angle(pos[4]), angle(pos[5])]
        #
        #     # 执行重力补偿
        #     tem1 = comp.Solve_Force(realtime_force, realtime_euler)
        #     tem2 = comp.Solve_Torque(realtime_torque, realtime_euler)
        #     force = np.array(tem1)
        #     torque = np.array(tem2)
        #
        #     # 执行坐标系变换
        #     T_F, T_M = trans.force_moment_transform(force[0], torque[0], T_P_SORG)
        #
        #
        #     for g, key in enumerate(keys[:3]):
        #         data_log[key].append(T_F[g])
        #     for g, key in enumerate(keys[3:]):
        #         data_log[key].append(T_M[g])
        #
        #     i = i+1
        #
        # pos2 = [-472.827082, 226.43587, 607.04603, 2.067999650357244, 0.9077361795579499, -0.8590817058132739]
        # robot.linear_move(pos2, ABS, True, 500)
        # time.sleep(5)
        # # 主循环
        # for i in range(10):
        #     # 读取传感器数据
        #     F = Get_force.read_sensor_data()
        #     ret = robot.get_tcp_position()
        #     pos = ret[1]
        #     print('实际受力：', F)
        #     print('实际姿态角：', [angle(pos[3]), angle(pos[4]), angle(pos[5])])
        #     realtime_force = F[:3]
        #     realtime_torque = F[3:]
        #     realtime_euler = [angle(pos[3]), angle(pos[4]), angle(pos[5])]
        #
        #     # 执行重力补偿
        #     tem1 = comp.Solve_Force(realtime_force, realtime_euler)
        #     tem2 = comp.Solve_Torque(realtime_torque, realtime_euler)
        #     force = np.array(tem1)
        #     torque = np.array(tem2)
        #
        #     # 执行坐标系变换
        #     T_F, T_M = trans.force_moment_transform(force[0], torque[0], T_P_SORG)
        #
        #
        #     for g, key in enumerate(keys[:3]):
        #         data_log[key].append(T_F[g])
        #     for g, key in enumerate(keys[3:]):
        #         data_log[key].append(T_M[g])
        #
        #     i = i + 1

        # 主循环
        while True:
            # 读取传感器数据
            _, pos = robot.get_tcp_position()
            force,raw_force = processor.process_data(pos, T_P_SORG)

            print('实际受力：', raw_force)
            print('处理后受力：', force)

            for g, key in enumerate(keys1):
                data_log[key].append(raw_force[g])
            for g, key in enumerate(keys2):
                data_log[key].append(force[g])
            
            time.sleep(0.2)

    finally:
        robot.logout()
        path = '/home/adminrt/LYH_Graduation/Chapter_2/2.1Force_data/experiment/force_processing'
        save_history(path, data_log, 'force_data.csv')

if __name__ == '__main__':
    main()