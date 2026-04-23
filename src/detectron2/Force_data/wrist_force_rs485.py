import argparse
import struct
import time
from typing import Optional, Tuple

import serial  # type: ignore[reportMissingImports]


class WristFTSensorRS485:
    """
    WRIST 六维力传感器 RS485 驱动（量产 485 通讯协议 A 正式版）。

    注意：
    - 当前实现严格按“量产 485 通讯协议 A”。
    - 不是 3110 测试脚本协议，也不是 Modbus TCP。
    - 三者在启动命令、数据帧结构、数据类型与换算公式上都不同，不能混用。
    """

    # 协议 A：100Hz/500Hz 上传命令
    START_CMD_100HZ = bytes([0x09, 0x10, 0x01, 0x9A, 0x00, 0x01, 0x02, 0x02, 0x00, 0xCD, 0xCA])
    START_CMD_500HZ = bytes([0x09, 0x10, 0x01, 0x9A, 0x00, 0x01, 0x02, 0x02, 0x02, 0x4C, 0x0B])
    # 协议 A：停止上传命令（50+ 个 0xFF）
    STOP_CMD = bytes([0xFF] * 64)
    # 协议 A：清零命令
    ZERO_CMD = bytes([0x09, 0x10, 0x01, 0x9A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6C, 0x96])

    # 协议 A 数据帧：20 4E + 12 字节数据 + 2 字节 CRC = 16 字节
    FRAME_LEN = 16
    HEADER_0 = 0x20
    HEADER_1 = 0x4E

    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB1",
        baudrate: int = 115200,
        timeout: float = 0.1,
        auto_start: bool = True,
        auto_stop_on_close: bool = False,
        stream_rate_hz: int = 500,
        debug_hex: bool = False,
    ) -> None:
        self.serial_port = str(serial_port)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.auto_start = bool(auto_start)
        self.auto_stop_on_close = bool(auto_stop_on_close)
        self.stream_rate_hz = self._normalize_stream_rate(stream_rate_hz)
        self.debug_hex = bool(debug_hex)
        self.ser: Optional[serial.Serial] = None
        self._sync_fail_count = 0

    @staticmethod
    def _normalize_stream_rate(rate_hz: int) -> int:
        v = int(rate_hz)
        if v not in (100, 500):
            raise ValueError(f"stream_rate_hz must be 100 or 500, got {rate_hz}")
        return v

    @staticmethod
    def _to_hex(data: bytes) -> str:
        return " ".join(f"{b:02X}" for b in data)

    @staticmethod
    def _crc16_modbus(payload: bytes) -> int:
        """
        协议 A 要求“除 CRC 外的所有数据参与计算”。
        这里沿用 CRC16(Modbus) 算法，和协议 A 样例帧可对上。
        """
        crc = 0xFFFF
        for b in payload:
            crc ^= int(b)
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    def _read_exact(self, n: int) -> Optional[bytes]:
        if not self.is_open():
            return None
        buf = bytearray()
        t0 = time.time()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
                continue
            if (time.time() - t0) > max(0.5, self.timeout * 10.0):
                return None
        return bytes(buf)

    def _get_start_cmd(self) -> bytes:
        if self.stream_rate_hz == 100:
            return self.START_CMD_100HZ
        return self.START_CMD_500HZ

    def open(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
            )
        except Exception as e:
            print(
                f"[sensor][error] open serial failed: serial_port={self.serial_port}, "
                f"baudrate={self.baudrate}, error={type(e).__name__}: {e}"
            )
            self.ser = None
            return False

        print(
            f"[sensor][connect] RS485 opened: serial_port={self.serial_port}, baudrate={self.baudrate}, "
            f"timeout={self.timeout}, proto=production_485_A, stream_rate_hz={self.stream_rate_hz}"
        )
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass
        if self.auto_start:
            self.send_start()
        return True

    def close(self) -> None:
        if self.ser is None:
            return
        try:
            if self.auto_stop_on_close:
                self.send_stop()
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass
        self.ser = None

    def is_open(self) -> bool:
        return self.ser is not None and bool(self.ser.is_open)

    def send_start(self) -> bool:
        if not self.is_open():
            print(
                f"[sensor][error] send_start failed: serial not open, serial_port={self.serial_port}, "
                f"baudrate={self.baudrate}"
            )
            return False
        try:
            cmd = self._get_start_cmd()
            self.ser.write(cmd)
            self.ser.flush()
            print(f"[sensor] start command sent ({self.stream_rate_hz}Hz): {self._to_hex(cmd)}")
            return True
        except Exception as e:
            print(
                f"[sensor][error] send_start failed: serial_port={self.serial_port}, baudrate={self.baudrate}, "
                f"error={type(e).__name__}: {e}"
            )
            return False

    def send_stop(self) -> bool:
        if not self.is_open():
            print(
                f"[sensor][error] send_stop failed: serial not open, serial_port={self.serial_port}, "
                f"baudrate={self.baudrate}"
            )
            return False
        try:
            self.ser.write(self.STOP_CMD)
            self.ser.flush()
            print(f"[sensor] stop command sent: {len(self.STOP_CMD)} bytes of FF")
            return True
        except Exception as e:
            print(
                f"[sensor][error] send_stop failed: serial_port={self.serial_port}, baudrate={self.baudrate}, "
                f"error={type(e).__name__}: {e}"
            )
            return False

    def set_zero(self) -> bool:
        if not self.is_open():
            print(
                f"[sensor][error] set_zero failed: serial not open, serial_port={self.serial_port}, "
                f"baudrate={self.baudrate}"
            )
            return False
        try:
            self.ser.write(self.ZERO_CMD)
            self.ser.flush()
            print(f"[sensor] zero command sent: {self._to_hex(self.ZERO_CMD)}")
            return True
        except Exception as e:
            print(
                f"[sensor][error] set_zero failed: serial_port={self.serial_port}, baudrate={self.baudrate}, "
                f"error={type(e).__name__}: {e}"
            )
            return False

    def _read_one_frame(self) -> Optional[bytes]:
        """
        协议 A 稳定同步：
        - 持续寻找帧头 20 4E
        - 读满 16 字节
        - 校验头、长度、CRC 成功后返回
        """
        if not self.is_open():
            print(
                f"[sensor][error] read frame failed: serial not open, serial_port={self.serial_port}, "
                f"baudrate={self.baudrate}"
            )
            return None

        for _ in range(128):
            b0 = self._read_exact(1)
            if not b0:
                self._sync_fail_count += 1
                print(
                    f"[sensor][error] frame sync timeout: serial_port={self.serial_port}, "
                    f"baudrate={self.baudrate}"
                )
                if self._sync_fail_count % 5 == 0:
                    print("[sensor][recover] consecutive sync timeout, resend start command")
                    self.send_start()
                return None
            if b0[0] != self.HEADER_0:
                continue

            b1 = self._read_exact(1)
            if not b1:
                return None
            if b1[0] != self.HEADER_1:
                continue

            remain = self._read_exact(self.FRAME_LEN - 2)
            if not remain:
                return None

            frame = bytes([self.HEADER_0, self.HEADER_1]) + remain
            if len(frame) != self.FRAME_LEN:
                continue

            recv_crc = struct.unpack("<H", frame[14:16])[0]
            calc_crc = self._crc16_modbus(frame[:14])
            crc_ok = calc_crc == recv_crc
            if self.debug_hex:
                print(
                    f"[sensor][debug] frame16 hex={self._to_hex(frame)}, "
                    f"crc_ok={crc_ok}, recv_crc=0x{recv_crc:04X}, calc_crc=0x{calc_crc:04X}"
                )
            if not crc_ok:
                print(
                    f"[sensor][error] crc mismatch, drop frame: recv=0x{recv_crc:04X}, "
                    f"calc=0x{calc_crc:04X}"
                )
                continue

            self._sync_fail_count = 0
            return frame

        print(
            f"[sensor][error] frame header invalid repeatedly: serial_port={self.serial_port}, "
            f"baudrate={self.baudrate}"
        )
        return None

    def read_ft(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        frame = self._read_one_frame()
        if frame is None:
            return None

        # 协议 A：6 个 little-endian 有符号 int16
        raw_fx, raw_fy, raw_fz, raw_mx, raw_my, raw_mz = struct.unpack("<6h", frame[2:14])

        # 协议 A 换算规则：力=VAL/100，力矩=VAL/1000
        fx = raw_fx / 100.0
        fy = raw_fy / 100.0
        fz = raw_fz / 100.0
        mx = raw_mx / 1000.0
        my = raw_my / 1000.0
        mz = raw_mz / 1000.0

        if self.debug_hex:
            print(
                f"[sensor][debug] raw_counts=({raw_fx}, {raw_fy}, {raw_fz}, {raw_mx}, {raw_my}, {raw_mz}), "
                f"scaled_ft=({fx:.6f}, {fy:.6f}, {fz:.6f}, {mx:.6f}, {my:.6f}, {mz:.6f})"
            )
        return fx, fy, fz, mx, my, mz


def main() -> None:
    parser = argparse.ArgumentParser(description="WRIST RS485 量产协议A 读取测试")
    parser.add_argument("--serial_port", type=str, default="/dev/ttyUSB1", help="串口设备")
    parser.add_argument("--baudrate", type=int, default=115200, help="波特率(协议A固定115200)")
    parser.add_argument("--timeout", type=float, default=0.1, help="串口读超时(秒)")
    parser.add_argument("--stream_rate_hz", type=int, default=500, choices=[100, 500], help="上传频率")
    parser.add_argument("--period", type=float, default=0.05, help="打印周期(秒)")
    parser.add_argument("--count", type=int, default=0, help="读取次数，0 表示无限循环")
    parser.add_argument("--debug_hex", action="store_true", help="打印原始16字节hex、CRC、计数与工程值")
    args = parser.parse_args()

    sensor = WristFTSensorRS485(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        auto_start=True,
        auto_stop_on_close=False,
        stream_rate_hz=args.stream_rate_hz,
        debug_hex=args.debug_hex,
    )

    if not sensor.open():
        return

    print("[demo] 开始输出协议A工程值（Fx/Fy/Fz: N, Mx/My/Mz: N·m），按 Ctrl+C 退出")
    idx = 0
    try:
        while args.count <= 0 or idx < args.count:
            ft = sensor.read_ft()
            if ft is None:
                print("[demo] 读取失败")
            else:
                fx, fy, fz, mx, my, mz = ft
                print(
                    f"[ft] Fx={fx:.6f} N, Fy={fy:.6f} N, Fz={fz:.6f} N, "
                    f"Mx={mx:.6f} N·m, My={my:.6f} N·m, Mz={mz:.6f} N·m"
                )
            idx += 1
            time.sleep(max(0.0, args.period))
    except KeyboardInterrupt:
        print("\n[demo] 用户中断，结束读取")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()

