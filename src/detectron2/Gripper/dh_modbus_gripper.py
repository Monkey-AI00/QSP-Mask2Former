import dh_device

#全局变量改为类变量
# m_device = dh_device.dh_device()


class dh_modbus_gripper(object):
    gripper_ID = 0x01
    # PGC 夹爪通过 RS485(Modbus RTU) 控制：
    # - 写单寄存器(功能码 0x06)：下发初始化/目标位置/力/速度
    # - 读保持寄存器(功能码 0x03)：读取初始化状态/夹持状态/当前位置等
    REG_INIT = 0x0100
    REG_TARGET_FORCE = 0x0101
    REG_TARGET_POSITION = 0x0103
    REG_TARGET_SPEED = 0x0104
    REG_INIT_STATE = 0x0200
    REG_GRIP_STATE = 0x0201
    REG_CURRENT_POSITION = 0x0202
    INIT_STATE_TEXT = {
        0: "未初始化",
        1: "初始化成功",
    }
    GRIP_STATE_TEXT = {
        0: "运动中",
        1: "停止且未检测到物体",
        2: "停止且检测到物体",
        3: "检测到物体后掉落",
    }

    def __init__(self):  # 添加构造函数
        self.m_device = dh_device.dh_device() 

    def CRC16(self,nData, wLength) :
        if nData==0x00:
            return 0x0000
        wCRCWord=0xFFFF
        poly=0xA001
        for num in range(wLength):
            date = nData[num]
            wCRCWord = (date & 0xFF)^ wCRCWord
            for bit in range(8) : 
                if(wCRCWord&0x01)!=0:
                    wCRCWord>>=1
                    wCRCWord^= poly
                else:
                    wCRCWord>>=1
        return wCRCWord

    def open(self,PortName,BaudRate) :
        ret = 0
        ret = self.m_device.connect_device(PortName, BaudRate)
        if(ret < 0) :
            print('open failed')
            return ret
        else :
            print('open successful')
            return ret

    def close(self,) :
        self.m_device.disconnect_device()
        print('gripper close')

    def WriteRegisterFunc(self,index, value) :
        send_buf = [0,0,0,0,0,0,0,0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x06
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = (value >> 8) & 0xFF
        send_buf[5] = value & 0xFF

        crc = self.CRC16(send_buf,len(send_buf)-2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        send_temp = send_buf
        ret = False
        retrycount = 3

        while ( ret == False ):
            ret = False

            if(retrycount < 0) :
                break
            retrycount = retrycount - 1

            wdlen = self.m_device.device_wrire(send_temp)
            if(len(send_temp) != wdlen) :
                print('write error ! write : ', send_temp)
                continue

            rev_buf = self.m_device.device_read(8)
            if(len(rev_buf) == wdlen) :
                ret = True
        return ret

    def ReadRegisterFunc(self,index) :
        send_buf = [0,0,0,0,0,0,0,0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x03
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = 0x00
        send_buf[5] = 0x01

        crc = self.CRC16(send_buf,len(send_buf)-2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        send_temp = send_buf
        ret = False
        retrycount = 3

        while ( ret == False ):
            ret = False

            if(retrycount < 0) :
                break
            retrycount = retrycount - 1

            wdlen = self.m_device.device_wrire(send_temp)
            if(len(send_temp) != wdlen) :
                print('write error ! write : ', send_temp)
                continue

            rev_buf = self.m_device.device_read(7)
            if(len(rev_buf) == 7) :
                value = ((rev_buf[4]&0xFF)|(rev_buf[3] << 8))
                ret = True
            #('read value : ', value)
        return value

    def Initialization(self) :
        self.WriteRegisterFunc(self.REG_INIT,0xA5)
        
    def SetTargetPosition(self,refpos) :
        self.WriteRegisterFunc(self.REG_TARGET_POSITION,refpos)

    def SetTargetForce(self,force) :
        self.WriteRegisterFunc(self.REG_TARGET_FORCE,force)
        
    def SetTargetSpeed(self,speed) :
        self.WriteRegisterFunc(self.REG_TARGET_SPEED,speed)

    def GetCurrentPosition(self) :
        return self.ReadRegisterFunc(self.REG_CURRENT_POSITION)

    def GetTargetPosition(self) :
        return self.ReadRegisterFunc(self.REG_TARGET_POSITION)

    def GetCurrentTargetForce(self) :
        return self.ReadRegisterFunc(self.REG_TARGET_FORCE)

    def GetCurrentTargetSpeed(self) :
        return self.ReadRegisterFunc(self.REG_TARGET_SPEED)

    def GetInitState(self) :
        return self.ReadRegisterFunc(self.REG_INIT_STATE)

    def GetGripState(self) :
        return self.ReadRegisterFunc(self.REG_GRIP_STATE)

    def DecodeInitState(self, v):
        return self.INIT_STATE_TEXT.get(int(v), "未知")

    def DecodeGripState(self, v):
        return self.GRIP_STATE_TEXT.get(int(v), "未知")

    def GetStatusSnapshot(self):
        # 输出原始寄存器值，避免在未知固件版本下误判状态语义。
        init_state = int(self.GetInitState())
        grip_state = int(self.GetGripState())
        return {
            "init_state": init_state,
            "init_state_desc": self.DecodeInitState(init_state),
            "grip_state": grip_state,
            "grip_state_desc": self.DecodeGripState(grip_state),
            "current_position": int(self.GetCurrentPosition()),
            "target_position": int(self.GetTargetPosition()),
            "target_force": int(self.GetCurrentTargetForce()),
            "target_speed": int(self.GetCurrentTargetSpeed()),
        }

    """description of class"""


