import socket
import time
import matplotlib.pyplot as plt
import numpy as np


class ATI80:
    def __init__(self):
        # 1. 创建套接字
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 2. 绑定一个本地信息
        self.localaddr = ("192.168.1.111", 8899)
        self.udp_socket.bind(self.localaddr)  # 

    def read_data(self):
        self.udp_socket.sendto(b"\x12\x34\x00\x02\x00\x00\x00\x01", ("192.168.1.1", 49152))
        recv_data = self.udp_socket.recvfrom(1024)
        recv_msg = recv_data[0]  # 存储接收的数据，
        # send_addr = recv_data[1]  # 存储发送方的地址信息
        recv_msg_1 = recv_msg.hex()  # 字节串转16进制
        f = []  # 每32bit，4字节分割
        i = len(recv_msg_1) // 8
        for j in range(0, i):
            k = 0 + 8 * j
            f.append(recv_msg_1[k: k + 8])

        width = 32  # 16进制数所占位数
        F_o = [0, 0, 0, 0, 0, 0]
        for m in range(3, i):
            data = int(f[m], 16)
            if data > 2 ** (width - 1) - 1:
                data = 2 ** width - data
                data = 0 - data

            F_o[m-3] = data / 1000000  

        return np.array(F_o)

def read_data(udp_socket):
    udp_socket.sendto(b"\x12\x34\x00\x02\x00\x00\x00\x01", ("192.168.1.２", 49152))
    recv_data = udp_socket.recvfrom(1024)
    recv_msg = recv_data[0]  # 存储接收的数据，
    # send_addr = recv_data[1]  # 存储发送方的地址信息
    recv_msg_1 = recv_msg.hex()  # 字节串转16进制
    f = []  # 每32bit，4字节分割
    i = len(recv_msg_1) // 8
    for j in range(0, i):
        k = 0 + 8 * j
        f.append(recv_msg_1[k: k + 8])

    width = 32  # 16进制数所占位数
    F_o = [0, 0, 0, 0, 0, 0]
    for m in range(3, i):
        data = int(f[m], 16)
        if data > 2 ** (width - 1) - 1:
            data = 2 ** width - data
            data = 0 - data

        F_o[m-3] = data / 1000000  

    return np.array(F_o)

def main():
    # 1. 创建套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 2. 绑定一个本地信息
    # localaddr = ("192.168.1.1", 8899)
    localaddr = ("192.168.1.100", 8999)
    udp_socket.bind(localaddr)  # 必须绑定自己电脑的ip以及port，其他的不行
    
    # set the zero force
    udp_socket.sendto(b"\x12\x34\x00\x42\x00\x00\x00\xFF", ("192.168.1.111", 49152))
    #recv_data = udp_socket.recvfrom(1024)

    F_o_0 = read_data(udp_socket)
    _, ax = plt.subplots()
    
    i = 0
    x = []
    # Tx = []
    # Ty = []
    # Tz = []
    Test = []
    while True:
        F_o_i = read_data(udp_socket)
        print(F_o_i)
        # x.append(i)
        #Tx.append(F_o_i[3])
        #Ty.append(F_o_i[4])
        # Tz.append(F_o_i[2])
        Test.append(F_o_i[2])
        ax.cla() # clear plot
        # ax.plot(x, Tx, 'r', lw=1) # draw line chart
        # ax.plot(x, Ty, 'g', lw=1) # draw line chart
        # ax.plot(x, Tz, 'b', lw=1) # draw line chart
        ax.plot(x, Test, 'y', lw=1)  # draw line chart

        # do_something()
        plt.ylim(-100, 100)
        plt.pause(0.1)
        i+=1
        # print(recv_data)

    F_o_i = read_data(udp_socket)
    print(F_o_i)
    # 6. 关闭套接字
    udp_socket.close()


if __name__ == "__main__":
    # while True:
     main()

