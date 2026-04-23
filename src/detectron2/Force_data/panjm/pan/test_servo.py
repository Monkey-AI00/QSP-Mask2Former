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

def servo_move(pos):
    start = time.time()
    dt, velocity, acceleration, lookahead_time, gain = 2.0 , 3, 1.5, 0.2, 500
    rtde_c.servoL(pos,velocity,acceleration,dt,lookahead_time,gain)
    end = time.time()
    duration = end - start
    if duration < dt:
        time.sleep(dt - duration)


if __name__ == '__main__':
    starttime = time.time()
    startpos = [-0.5716569195689681, -0.1349873890313325, 0.4356084409834532, -2.213731452268282, -2.220085875959799, -0.0220181522906126]
    pos = [-0.5716569195689681, -0.1349873890313325, 0.4276084409834532, -2.213731452268282, -2.220085875959799, -0.0220181522906126]
    rtde_c.moveL(startpos)
    time.sleep(1)
    servo_move(pos)
