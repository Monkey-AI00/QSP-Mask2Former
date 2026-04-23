import time
import rtde_control
import rtde_receive
import math

# 连接机械臂
rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.2")

def joint_move(target_q):
    velocity = 1.0
    acceleration = 1.0
    rtde_c.moveJ(target_q, velocity, acceleration)

if __name__ == '__main__':
    # [0, -90, 90, -90, -90, 90]
    # start_joint = [-2.462068666631012e-05, -1.5708161793150843, 1.5708039442645472, -1.5708023510374964, -1.5708258787738245, 1.5707777738571167]
    # joint_move(start_joint)
    # # [0, -90, 90, -90, -90, 0]
    # test_joint = [-3.225008119756012e-05, -1.5707880815319477, 1.5707467238055628, -1.5707864533299762, -1.5707815329181116, -3.6541615621388246e-05]
    # joint_move(test_joint)

    # startpos = [-0.5466338645551002, 0.12309744671836524, 0.32679097131888485, 0, math.pi, 0]
    startpos = [-0.4745579401540578, -0.28237865725707884, 0.272486164474702, 0, math.pi, 0]
    rtde_c.moveL(startpos)