"""
兼容模块：原 `wrist_force_modbus.py` 迁移为 RS485 实现入口。

说明：
- 原项目上层大量 `from wrist_force_modbus import WristFTSensor` 依赖仍保留；
- 底层通信已从 Modbus TCP 替换为 WRIST RS485 串口通信；
- 建议新代码直接使用 `wrist_force_rs485.WristFTSensorRS485`。
"""

import time
from typing import Optional

from filter_module import MoveMeanFilter
from wrist_force_rs485 import WristFTSensorRS485


class WristFTSensor(WristFTSensorRS485):
    """
    向后兼容类名：
    - 旧接口: WristFTSensor(host="192.168.5.20", port=502)
    - 新接口: WristFTSensor(serial_port="/dev/ttyUSB0", baudrate=115200)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 0.1,
        auto_open: bool = False,
        serial_port: Optional[str] = None,
        baudrate: int = 115200,
        auto_start: bool = True,
        auto_stop_on_close: bool = False,
    ) -> None:
        # Modbus TCP -> RS485 迁移提示
        if serial_port is None:
            serial_port = host if host else "/dev/ttyUSB0"
        if host is not None or port is not None:
            print(
                "[sensor][warn] Modbus 参数 host/port 已废弃，当前使用 RS485："
                f" serial_port={serial_port}, baudrate={baudrate}"
            )
        super().__init__(
            serial_port=serial_port,
            baudrate=baudrate,
            timeout=timeout,
            auto_start=auto_start,
            auto_stop_on_close=auto_stop_on_close,
        )
        if auto_open:
            self.open()


def demo_MoveMeanFilter():
    sensor = WristFTSensor(serial_port="/dev/ttyUSB0", baudrate=115200)
    filt = MoveMeanFilter(window_size=10)
    if not sensor.open():
        return
    try:
        while True:
            ft = sensor.read_ft()
            if ft is None:
                continue
            fx, fy, fz, mx, my, mz = filt.update(ft)
            print(
                f"Fx={fx:.5f} N, Fy={fy:.5f} N, Fz={fz:.5f} N, "
                f"Mx={mx:.5f} N·m, My={my:.5f} N·m, Mz={mz:.5f} N·m"
            )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()


if __name__ == "__main__":
    sensor = WristFTSensor(serial_port="/dev/ttyUSB0", baudrate=115200)
    if not sensor.open():
        raise SystemExit(1)
    start = time.time()
    try:
        while True:
            ft = sensor.read_ft()
            if ft is None:
                continue
            print(
                f"t={time.time()-start:7.4f}s | 原始力N={ft[:3]} 原始力矩N·m={ft[3:]}"
            )
            time.sleep(1)
    finally:
        sensor.close()