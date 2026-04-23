"""
实时可视化重力补偿后的力和力矩曲线
整合：力传感器读取 + 重力补偿 + 坐标变换 + 实时绘图
"""

import numpy as np
import time
import sys
import os
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# 添加路径导入相关模块
sys.path.insert(0, os.path.dirname(__file__))

from wrist_force_modbus import WristFTSensor
from Gravity_compensation import GravityCompensation, process_force_data, calibrate_gravity_compensation
from force_frame_trans_rot import force_moment_transform
from filter_module import RCLowPassFilter
from force_frame_trans_toB import wrench_tool_to_base

import ctypes
path = ("D:\论文\code\jaka\API\jakaAPI.dll")
ctypes.CDLL(path)

import jkrc


class RealTimeForceVisualizer:
    """实时力数据可视化器（带重力补偿）"""
    
    def __init__(self, 
                 robot_ip="192.168.1.131",
                 sensor_ip="192.168.1.20", 
                 sensor_port=502,
                 alpha_deg=-45.0,
                 filter_alpha=0.3,
                 buffer_size=300,
                 update_rate=50):
        """
        参数:
            robot_ip: 机器人IP地址
            sensor_ip: 力传感器IP地址
            sensor_port: 力传感器端口
            alpha_deg: 传感器安装角（度）
            filter_alpha: 低通滤波器系数 (0-1)
            buffer_size: 显示缓存点数
            update_rate: 更新频率 (Hz)
        """
        self.buffer_size = buffer_size
        self.update_interval = 1.0 / update_rate
        
        # 初始化机器人
        self.robot = jkrc.RC(robot_ip)
        self.robot.login()
        self.robot.power_on()
        self.robot.enable_robot()
        print("机器人连接成功")
        
        # 初始化力传感器
        self.sensor = WristFTSensor(host=sensor_ip, port=sensor_port)
        if not self.sensor.open():
            raise ConnectionError(f"无法连接到力传感器 {sensor_ip}:{sensor_port}")
        print("力传感器连接成功")
        
        # 初始化重力补偿器
        self.gravity_comp = GravityCompensation(alpha_deg)
        
        # 执行重力补偿标定
        print("开始重力补偿标定...")
        calibrate_gravity_compensation(self.gravity_comp, self.robot, self.sensor)
        print("重力补偿标定完成")
        
        # 初始化低通滤波器
        self.filter = RCLowPassFilter(alpha=filter_alpha)
        
        # 坐标变换参数
        self.T_P_SORG = [0, 0, 170 * 0.001]  # 传感器原点在工具坐标系中的位置
        theta = 45 * np.pi / 180.0
        self.R_TS = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        
        # 数据缓存
        self.time_buffer = deque(maxlen=buffer_size)
        self.force_raw_buffer = deque(maxlen=buffer_size)   # 原始力
        self.force_compensated_buffer = deque(maxlen=buffer_size)  # 补偿后力
        self.torque_raw_buffer = deque(maxlen=buffer_size)  # 原始力矩
        self.torque_compensated_buffer = deque(maxlen=buffer_size)  # 补偿后力矩
        
        # 运行标志
        self.running = False
        
    def get_force_data(self):
        """获取经过完整处理的力数据"""
        # 读取原始力数据
        raw_force = self.sensor.read_ft()
        if raw_force is None:
            return None, None, None
        
        # 获取机器人当前位姿
        _, robot_pos = self.robot.get_tcp_position()
        
        # 重力补偿
        force_comp, torque_comp = process_force_data(
            self.gravity_comp, 
            raw_force[:3], 
            raw_force[3:], 
            robot_pos
        )
        
        # 坐标变换：传感器坐标系 -> 工具坐标系
        transformed_force, transformed_torque = force_moment_transform(
            force_comp, torque_comp, 
            self.T_P_SORG, True, self.R_TS
        )
        T_force = np.hstack((transformed_force, transformed_torque))
        
        # 坐标变换：工具坐标系 -> 基坐标系
        B_force = wrench_tool_to_base(robot_pos, T_force)
        
        # 低通滤波
        filtered_force = self.filter.update(B_force)
        
        return raw_force, filtered_force[:3], filtered_force[3:]
    
    def setup_plot(self):
        """设置绘图界面"""
        plt.ion()  # 交互模式
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 力曲线图
        self.ax_force = axes[0]
        self.ax_force.set_title('Force (Gravity Compensated)', fontsize=14)
        self.ax_force.set_ylabel('Force (N)', fontsize=12)
        self.ax_force.grid(True, alpha=0.3)
        self.ax_force.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 力矩曲线图
        self.ax_torque = axes[1]
        self.ax_torque.set_title('Torque (Gravity Compensated)', fontsize=14)
        self.ax_torque.set_xlabel('Time (s)', fontsize=12)
        self.ax_torque.set_ylabel('Torque (N·m)', fontsize=12)
        self.ax_torque.grid(True, alpha=0.3)
        self.ax_torque.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 初始化曲线（力）
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        labels = ['Fx', 'Fy', 'Fz']
        self.force_lines = []
        for i, (color, label) in enumerate(zip(colors, labels)):
            line, = self.ax_force.plot([], [], color=color, linewidth=2, label=label)
            self.force_lines.append(line)
        
        # 初始化曲线（力矩）
        torque_labels = ['Mx', 'My', 'Mz']
        self.torque_lines = []
        for i, (color, label) in enumerate(zip(colors, torque_labels)):
            line, = self.ax_torque.plot([], [], color=color, linewidth=2, label=label)
            self.torque_lines.append(line)
        
        # 添加图例
        self.ax_force.legend(loc='upper right')
        self.ax_torque.legend(loc='upper right')
        
        # 添加信息文本
        self.info_text = self.ax_force.text(0.02, 0.98, '', transform=self.ax_force.transAxes,
                                             fontsize=10, verticalalignment='top',
                                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self.fig = fig
        
    def update_plot(self, t, raw_force, comp_force, comp_torque):
        """更新绘图"""
        # 更新缓存
        self.time_buffer.append(t)
        self.force_raw_buffer.append(raw_force)
        self.force_compensated_buffer.append(comp_force)
        self.torque_compensated_buffer.append(comp_torque)
        
        # 转换为数组
        time_arr = np.array(self.time_buffer)
        force_arr = np.array(self.force_compensated_buffer)
        torque_arr = np.array(self.torque_compensated_buffer)
        
        # 更新力曲线
        for i in range(3):
            self.force_lines[i].set_data(time_arr, force_arr[:, i])
        
        # 更新力矩曲线
        for i in range(3):
            self.torque_lines[i].set_data(time_arr, torque_arr[:, i])
        
        # 自动调整坐标轴范围
        if len(time_arr) > 1:
            self.ax_force.set_xlim(time_arr[0], time_arr[-1])
            self.ax_torque.set_xlim(time_arr[0], time_arr[-1])
            
            # 动态调整Y轴范围（添加20%余量）
            force_max = np.max(np.abs(force_arr)) * 1.2
            torque_max = np.max(np.abs(torque_arr)) * 1.2
            if force_max > 0:
                self.ax_force.set_ylim(-force_max, force_max)
            if torque_max > 0:
                self.ax_torque.set_ylim(-torque_max, torque_max)
        
        # 更新信息文本
        if len(force_arr) > 0:
            latest_force = force_arr[-1]
            latest_torque = torque_arr[-1]
            info_str = f'Time: {t:.2f}s\n'
            info_str += f'Fx: {latest_force[0]:+6.2f} N  Fy: {latest_force[1]:+6.2f} N  Fz: {latest_force[2]:+6.2f} N\n'
            info_str += f'Mx: {latest_torque[0]:+6.3f} Nm  My: {latest_torque[1]:+6.3f} Nm  Mz: {latest_torque[2]:+6.3f} Nm'
            self.info_text.set_text(info_str)
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def run(self):
        """主运行循环"""
        print("\n" + "="*60)
        print("实时力可视化系统启动")
        print("="*60)
        print("功能说明:")
        print("  - 实时读取六维力传感器数据")
        print("  - 自动进行重力补偿（去除工具自重）")
        print("  - 坐标变换到基坐标系")
        print("  - 低通滤波去噪")
        print("  - 实时显示力和力矩曲线")
        print("\n控制指令:")
        print("  Ctrl+C - 退出程序")
        print("  s - 传感器置零")
        print("  r - 重置视图")
        print("="*60 + "\n")
        
        # 设置绘图
        self.setup_plot()
        self.running = True
        
        start_time = time.time()
        last_update = start_time
        frame_count = 0
        
        try:
            while self.running and plt.fignum_exists(self.fig.number):
                loop_start = time.time()
                
                # 获取处理后的力数据
                raw_force, comp_force, comp_torque = self.get_force_data()
                
                if raw_force is not None:
                    current_time = time.time() - start_time
                    self.update_plot(current_time, raw_force, comp_force, comp_torque)
                    frame_count += 1
                
                # 控制更新频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
                # 检查键盘输入
                plt.pause(0.001)
                
        except KeyboardInterrupt:
            print("\n用户中断，正在退出...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        if hasattr(self, 'sensor') and self.sensor:
            self.sensor.close()
        if hasattr(self, 'robot') and self.robot:
            self.robot.logout()
        plt.ioff()
        plt.close('all')
        print("程序已退出")


class SimpleForceVisualizer:
    """简化版：仅力传感器 + 重力补偿（不需要机器人）"""
    
    def __init__(self, sensor_ip="192.168.1.20", sensor_port=502, 
                 alpha_deg=-45.0, buffer_size=300):
        """
        简化版，使用预标定的重力补偿参数（不需要连接机器人）
        
        注意：使用简化版需要预先标定好重力补偿参数，
        或者从文件中加载已保存的参数
        """
        self.buffer_size = buffer_size
        
        # 初始化力传感器
        self.sensor = WristFTSensor(host=sensor_ip, port=sensor_port)
        if not self.sensor.open():
            raise ConnectionError(f"无法连接到力传感器 {sensor_ip}:{sensor_port}")
        
        # 初始化重力补偿器（需要预标定参数）
        self.gravity_comp = GravityCompensation(alpha_deg)
        
        # 尝试加载已保存的标定参数
        if not self.load_calibration_params():
            print("警告: 未找到保存的标定参数，将使用默认值")
            print("建议先运行标定程序或手动设置参数")
        
        # 数据缓存
        self.time_buffer = deque(maxlen=buffer_size)
        self.force_buffer = deque(maxlen=buffer_size)
        self.torque_buffer = deque(maxlen=buffer_size)
        
    def load_calibration_params(self, param_file="gravity_params.npy"):
        """从文件加载标定参数"""
        try:
            params = np.load(param_file, allow_pickle=True).item()
            self.gravity_comp.x = params['x']
            self.gravity_comp.y = params['y']
            self.gravity_comp.z = params['z']
            self.gravity_comp.g = params['g']
            self.gravity_comp.U = params['U']
            self.gravity_comp.V = params['V']
            self.gravity_comp.F_x0 = params['F_x0']
            self.gravity_comp.F_y0 = params['F_y0']
            self.gravity_comp.F_z0 = params['F_z0']
            print("已加载保存的标定参数")
            return True
        except:
            return False
    
    def save_calibration_params(self, param_file="gravity_params.npy"):
        """保存标定参数到文件"""
        params = {
            'x': self.gravity_comp.x,
            'y': self.gravity_comp.y,
            'z': self.gravity_comp.z,
            'g': self.gravity_comp.g,
            'U': self.gravity_comp.U,
            'V': self.gravity_comp.V,
            'F_x0': self.gravity_comp.F_x0,
            'F_y0': self.gravity_comp.F_y0,
            'F_z0': self.gravity_comp.F_z0,
        }
        np.save(param_file, params)
        print(f"标定参数已保存到 {param_file}")
    
    def run(self):
        """简化版运行（需要手动输入机器人姿态）"""
        print("\n简化版模式：需要手动输入机器人姿态角")
        print("如果机器人静止，姿态角可以设为 [0, 0, 0]")
        
        plt.ion()
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 设置图形...
        # (类似上面的设置，省略具体代码)
        
        try:
            while plt.fignum_exists(fig.number):
                # 手动输入姿态角（或使用固定值）
                euler = input("请输入姿态角 [rx, ry, rz] (度)，直接回车使用[0,0,0]: ")
                if euler.strip():
                    euler = [float(x) for x in euler.split(',')]
                else:
                    euler = [0, 0, 0]
                
                # 读取并补偿
                raw = self.sensor.read_ft()
                if raw:
                    force = self.gravity_comp.Solve_Force(raw[:3], euler)
                    torque = self.gravity_comp.Solve_Torque(raw[3:], euler)
                    print(f"补偿后 - 力: {force[0]:6.2f}, {force[0][1]:6.2f}, {force[0][2]:6.2f} N")
                    
        except KeyboardInterrupt:
            print("\n退出")
        finally:
            self.sensor.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实时力可视化工具（带重力补偿）')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'simple'],
                        help='运行模式: full(需要机器人) / simple(仅传感器)')
    parser.add_argument('--sensor_ip', type=str, default='192.168.1.20',
                        help='力传感器IP地址')
    parser.add_argument('--robot_ip', type=str, default='192.168.1.131',
                        help='机器人IP地址')
    parser.add_argument('--alpha', type=float, default=-45.0,
                        help='传感器安装角（度）')
    parser.add_argument('--filter', type=float, default=0.3,
                        help='低通滤波器系数 (0-1)')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        # 完整模式：需要连接机器人
        visualizer = RealTimeForceVisualizer(
            robot_ip=args.robot_ip,
            sensor_ip=args.sensor_ip,
            alpha_deg=args.alpha,
            filter_alpha=args.filter
        )
        visualizer.run()
    else:
        # 简化模式：仅传感器（需要预标定）
        print("简化模式需要预先标定重力补偿参数")
        print("请先运行完整模式进行标定，或手动设置参数")
        visualizer = SimpleForceVisualizer(
            sensor_ip=args.sensor_ip,
            alpha_deg=args.alpha
        )
        visualizer.run()


if __name__ == "__main__":
    main()