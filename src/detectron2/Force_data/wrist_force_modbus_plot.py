import time
from typing import List, Optional, Tuple
from collections import deque

import matplotlib.pyplot as plt
from pyModbusTCP.client import ModbusClient


class MoveMeanFilter:
    """
    简单滑动均值滤波器
    输入可以是长度固定的 tuple/list，例如六维力 (Fx, Fy, Fz, Mx, My, Mz)
    """
    def __init__(self, window_size: int = 10):
        if window_size <= 0:
            raise ValueError("window_size 必须大于 0")
        self.window_size = window_size
        self.buffers = None

    def update(self, data):
        data = list(data)
        if self.buffers is None:
            self.buffers = [deque(maxlen=self.window_size) for _ in range(len(data))]

        for i, v in enumerate(data):
            self.buffers[i].append(float(v))

        mean_vals = [sum(buf) / len(buf) for buf in self.buffers]
        return tuple(mean_vals)


class WristFTSensor:
    """
    WRIST ST 系列六维力传感器 Modbus TCP

    主要接口：
      - open() / close()
      - read_ft() -> (Fx, Fy, Fz, Mx, My, Mz)
      - set_zero()
      - read_system_info()
    """

    def __init__(
        self,
        host: str = "192.168.1.20",
        port: int = 502,
        timeout: int = 5,
        auto_open: bool = False,
    ) -> None:
        self.client = ModbusClient(
            host=host,
            port=port,
            timeout=timeout,
            auto_open=auto_open,
        )

    # ------------ 连接管理 ------------

    def open(self) -> bool:
        """主动建立 TCP 连接。"""
        ok = self.client.open()
        if not ok:
            print(f"连接失败 {self.client.host}:{self.client.port}，错误信息：{self.client.last_error}")
        return ok

    def close(self) -> None:
        """关闭 TCP 连接。"""
        self.client.close()

    def is_open(self) -> bool:
        """当前 TCP 连接是否打开。"""
        return self.client.is_open()

    # ------------ 低层寄存器读写 ------------

    def _read_raw_registers(self) -> Optional[List[int]]:
        """
        读取原始 12 个寄存器（0x0000 开始）。
        """
        regs = self.client.read_holding_registers(0x0000, 12)
        if not regs:
            print("读取传感器数据失败，错误信息：", self.client.last_error)
            return None
        return regs

    # ------------ 对外主要接口 ------------

    def read_ft(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        读取当前六维力数据，单位：
          - 力：N
          - 力矩：N·m
        """
        regs = self._read_raw_registers()
        if regs is None:
            return None

        # 有效数据位于 0,2,4,6,8,10 这 6 个寄存器
        values: List[float] = []
        for idx, reg_index in enumerate(range(0, 12, 2)):
            raw = regs[reg_index]
            if idx < 3:
                # 力：F(N) = (value - 32768) / 32.768
                converted = (raw - 32768) / 32.768
            else:
                # 力矩：M(N·m) = (value - 32768) / 327.68
                converted = (raw - 32768) / 327.68
            values.append(converted)

        fx, fy, fz, mx, my, mz = values
        return fx, fy, fz, mx, my, mz

    def set_zero(self) -> bool:
        """
        软件置零：往 0x1000 写入 1
        """
        ok = self.client.write_single_register(0x1000, 1)
        if not ok:
            print("设置零点失败，错误信息：", self.client.last_error)
        return ok

    def read_system_info(self) -> Optional[List[int]]:
        """
        读取传感器系统信息(0x1088, 长度 8 寄存器）
        """
        regs = self.client.read_holding_registers(0x1088, 8)
        if not regs:
            print("读取系统信息失败，错误信息：", self.client.last_error)
            return None
        return regs

    def write_system_info(self, regs: List[int]) -> bool:
        """
        一次性写入 8 个系统信息寄存器(0x1078~0x107F)
        """
        if len(regs) != 8:
            raise ValueError("系统信息寄存器长度必须为 8")
        ok = self.client.write_multiple_registers(0x1078, regs)
        if not ok:
            print("写入系统信息失败，错误信息：", self.client.last_error)
        return ok


def demo_realtime_plot(
    host: str = "192.168.1.20",
    port: int = 502,
    use_filter: bool = True,
    filter_window: int = 10,
    sample_interval: float = 0.02,
    max_points: int = 300,
    print_data: bool = True,
    auto_zero: bool = False,
):
    """
    六维力实时绘图

    参数：
        host: 传感器 IP
        port: 传感器端口
        use_filter: 是否使用滑动均值滤波
        filter_window: 滤波窗口长度
        sample_interval: 采样周期，0.02 表示约 50Hz
        max_points: 图中最多保留多少个点
        print_data: 是否同时在终端打印数据
        auto_zero: 启动后是否自动置零
    """
    sensor = WristFTSensor(host=host, port=port)
    filter_obj = MoveMeanFilter(window_size=filter_window)

    if not sensor.open():
        return

    if auto_zero:
        print("正在执行传感器置零...")
        sensor.set_zero()
        time.sleep(0.5)

    # 时间与六维数据缓存
    t_buf = deque(maxlen=max_points)
    fx_buf = deque(maxlen=max_points)
    fy_buf = deque(maxlen=max_points)
    fz_buf = deque(maxlen=max_points)
    mx_buf = deque(maxlen=max_points)
    my_buf = deque(maxlen=max_points)
    mz_buf = deque(maxlen=max_points)

    # 打开交互模式
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上图：力
    line_fx, = ax1.plot([], [], label="Fx (N)")
    line_fy, = ax1.plot([], [], label="Fy (N)")
    line_fz, = ax1.plot([], [], label="Fz (N)")
    ax1.set_title("Real-time Force")
    ax1.set_ylabel("Force (N)")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # 下图：力矩
    line_mx, = ax2.plot([], [], label="Mx (N·m)")
    line_my, = ax2.plot([], [], label="My (N·m)")
    line_mz, = ax2.plot([], [], label="Mz (N·m)")
    ax2.set_title("Real-time Torque")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (N·m)")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    fig.tight_layout()

    start = time.time()

    try:
        while plt.fignum_exists(fig.number):
            ft = sensor.read_ft()

            if ft is None:
                plt.pause(0.001)
                time.sleep(sample_interval)
                continue

            if use_filter:
                ft = filter_obj.update(ft)

            fx, fy, fz, mx, my, mz = ft
            t_now = time.time() - start

            if print_data:
                print(
                    f"t={t_now:8.3f}s | "
                    f"Fx={fx:9.4f} N, Fy={fy:9.4f} N, Fz={fz:9.4f} N, "
                    f"Mx={mx:9.4f} N·m, My={my:9.4f} N·m, Mz={mz:9.4f} N·m"
                )

            # 写入缓存
            t_buf.append(t_now)
            fx_buf.append(fx)
            fy_buf.append(fy)
            fz_buf.append(fz)
            mx_buf.append(mx)
            my_buf.append(my)
            mz_buf.append(mz)

            # 更新曲线数据
            line_fx.set_data(t_buf, fx_buf)
            line_fy.set_data(t_buf, fy_buf)
            line_fz.set_data(t_buf, fz_buf)

            line_mx.set_data(t_buf, mx_buf)
            line_my.set_data(t_buf, my_buf)
            line_mz.set_data(t_buf, mz_buf)

            # 自动重算坐标范围
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            # x 轴跟随滚动
            if len(t_buf) > 1:
                ax1.set_xlim(t_buf[0], t_buf[-1])
                ax2.set_xlim(t_buf[0], t_buf[-1])

            # 刷新图像
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\n用户中断，程序退出。")
    finally:
        sensor.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    demo_realtime_plot(
        host="192.168.1.20",
        port=502,
        use_filter=True,       # 是否滤波
        filter_window=10,      # 滑动均值窗口
        sample_interval=0.02,  # 约 50Hz
        max_points=300,        # 图中显示最近300个点
        print_data=True,       # 是否终端打印
        auto_zero=False        # 是否启动后自动置零
    )