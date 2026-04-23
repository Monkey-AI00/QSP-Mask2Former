import argparse
import re
import select
import shlex
import subprocess
import threading
import sys
import termios
import traceback
import tty
from pathlib import Path
from time import sleep
from typing import Optional

from dobot_api import DobotApiDashboard, DobotApiFeedBack



def parse_ints(resp: str):
    if resp is None:
        return []
    if "Not Tcp" in resp:
        return [1]
    return [int(x) for x in re.findall(r"-?\d+", resp)]


def parse_point(text: str):
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", str(text))]
    if len(nums) != 6:
        raise ValueError(f"点位必须恰好包含 6 个关节值，当前得到 {len(nums)} 个: {text}")
    return nums


def scale_exposure_seq(text: str, scale: float = 2.0) -> str:
    raw = str(text).strip()
    if not raw:
        return raw
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if not vals:
            return raw
        scaled = [max(0.0, v * float(scale)) for v in vals]
        return ",".join(
            str(int(v)) if abs(v - int(v)) < 1e-9 else f"{v:.4f}".rstrip("0").rstrip(".")
            for v in scaled
        )
    except Exception:
        # 解析失败时保持原值，避免影响已有调用
        return raw


def plan_from_four_points(p1, p2, p3, p4, samples: int = 20):
    """
    用 4 个关节点做三次 Bezier 插值，自动生成一条平滑关节轨迹。
    p1: 起点，p2/p3: 中间控制点，p4: 终点
    """
    n = max(2, int(samples))
    path = []
    for i in range(n):
        t = float(i) / float(n - 1)
        omt = 1.0 - t
        point = [
            (omt ** 3) * a + 3.0 * (omt ** 2) * t * b + 3.0 * omt * (t ** 2) * c + (t ** 3) * d
            for a, b, c, d in zip(p1, p2, p3, p4)
        ]
        path.append(point)
    return path


def build_argparser():
    p = argparse.ArgumentParser(description="传入 4 个点，自动规划 Dobot 关节轨迹并执行。")
    p.add_argument("--ip", default="192.168.5.2", help="机械臂 IP")
    p.add_argument(
        "--p1",
        default="-3.6910,38.9994,-143.6169,93.9034,55.9985,11.5497",
        help="起点 6 关节值，逗号分隔",
    )
    p.add_argument(
        "--p2",
        default="-3.6325,19.4963,-124.7558,16.3779,91.2283,1.5363",
        help="中间控制点 6 关节值，逗号分隔",
    )
    p.add_argument(
        "--p3",
        default="1.3565,-40.4180,-109.1531,63.3551,91.2968,1.5364",
        help="中间控制点2（P3）6 关节值，逗号分隔",
    )
    p.add_argument(
        "--p4",
        default="1.3565,-40.4180,-109.1531,63.3551,91.2968,1.5364",
        help="终止点（P4）6 关节值，逗号分隔",
    )
    p.add_argument("--samples", type=int, default=20, help="由 4 个点生成多少个轨迹点，至少 2")
    p.add_argument("--user", type=int, default=0)
    p.add_argument("--tool", type=int, default=0)
    p.add_argument("--a", type=int, default=20, help="加速度 1~100")
    p.add_argument("--v", type=int, default=20, help="速度 1~100")
    p.add_argument("--cp", type=int, default=50, help="平滑指数 0~100")
    p.add_argument(
        "--disable-gripper-close-at-p4",
        "--disable-gripper-close-at-p3",
        dest="disable_gripper_close_at_p4",
        action="store_true",
        help="禁用到达 P4 后自动闭合夹爪（兼容旧参数 --disable-gripper-close-at-p3）。",
    )
    p.add_argument("--gripper-close-delay", type=float, default=2.0, help="到达 P4 后闭合夹爪前的延时（秒）")
    p.add_argument("--gripper-port", default="/dev/ttyUSB0", help="夹爪串口，例如 /dev/ttyUSB0")
    p.add_argument("--gripper-baudrate", type=int, default=115200, help="夹爪串口波特率")
    p.add_argument("--gripper-close-position", type=int, default=0, help="夹爪闭合位置（默认 0）")
    p.add_argument("--gripper-open-position", type=int, default=900, help="夹爪打开位置（默认 900）")
    p.add_argument("--gripper-init-timeout", type=float, default=5.0, help="夹爪初始化超时（秒）")
    p.add_argument(
        "--disable-camera-at-p1",
        action="store_true",
        help="禁用到达 P1 后触发相机程序。",
    )
    p.add_argument(
        "--camera-script",
        default=str(Path(__file__).resolve().parents[2] / "projects" / "PointRend" / "mecheye_live_pointrend_pointcloud_shape_prior.py"),
        help="相机脚本路径（默认指向 mecheye_live_pointrend_pointcloud_shape_prior.py）",
    )
    p.add_argument("--camera-python", default=sys.executable, help="调用相机脚本的 Python 解释器")
    p.add_argument("--camera-no-gui", action="store_true", help="到达 P1 调相机时不显示可视化窗口")
    p.add_argument(
        "--camera-show-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="到达 P1 调相机时显示/不显示 depth 可视化窗口",
    )
    p.add_argument(
        "--camera-capture-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="到达 P1 时是否只做采集不加载分割模型。默认 true。",
    )
    p.add_argument("--camera-ip", default="", help="相机 IP（可选，不传则按脚本默认 discover/connect）")
    p.add_argument("--camera-serial", default="", help="相机序列号（可选）")
    p.add_argument("--camera-index", type=int, default=-1, help="discover 列表索引（可选）")
    p.add_argument(
        "--camera-exposure-seq",
        default="5,10",
        help="相机曝光序列，例如 '5,10'。默认两次曝光。",
    )
    p.add_argument("--camera-max-captures", type=int, default=2, help="到达 P1 后相机自动采集多少帧后退出")
    p.add_argument("--camera-timeout", type=float, default=120.0, help="相机程序超时（秒）")
    p.add_argument(
        "--camera-required",
        action="store_true",
        help="相机采集失败时中止流程（默认失败仅告警并继续执行 P2/P3/P4）。",
    )
    p.add_argument("--camera-extra-args", default="", help="透传给相机脚本的额外参数（字符串）")
    p.add_argument(
        "--disable-hotkey-home",
        action="store_true",
        help="禁用快捷键回 home；执行完成后直接退出。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印规划结果，不真正下发机械臂运动命令",
    )
    return p


class DobotMove:
    def __init__(self, ip: str):
        self.ip = ip
        self.dashboard_port = 29999
        self.feedback_port = 30004

        try:
            self.dashboard = DobotApiDashboard(ip, self.dashboard_port)
            print(f"[connect] Dashboard connected: {ip}:{self.dashboard_port}")
        except Exception as e:
            raise RuntimeError(
                f"[connect] Dashboard connection failed: ip={ip}, port={self.dashboard_port}, error={type(e).__name__}: {e}"
            ) from e

        try:
            self.feed = DobotApiFeedBack(ip, self.feedback_port)
            print(f"[connect] FeedBack connected: {ip}:{self.feedback_port}")
        except Exception as e:
            raise RuntimeError(
                f"[connect] FeedBack connection failed: ip={ip}, port={self.feedback_port}, error={type(e).__name__}: {e}"
            ) from e

        self._lock = threading.Lock()
        self.robotMode = -1
        self.currentCmdId = -1
        self._stop = False
        self._feed_error = None
        self._paused_by_hotkey = False

    def _feed_loop(self):
        while not self._stop:
            try:
                info = self.feed.feedBackData()
            except Exception as e:
                self._feed_error = e
                print(
                    f"[feed] FeedBack read failed: ip={self.ip}, port={self.feedback_port}, "
                    f"error={type(e).__name__}: {e}"
                )
                break
            if info is None:
                continue

            try:
                ok = (hex(info["TestValue"][0]).lower() == "0x123456789abcdef")
            except Exception:
                ok = False
            if not ok:
                continue

            with self._lock:
                self.robotMode = int(info["RobotMode"][0])
                self.currentCmdId = int(info["CurrentCommandId"][0])

            sleep(0.005)

    def start(self):
        try:
            ret = self.dashboard.EnableRobot()
        except Exception as e:
            raise RuntimeError(
                f"[start] EnableRobot failed on Dashboard: ip={self.ip}, port={self.dashboard_port}, "
                f"error={type(e).__name__}: {e}"
            ) from e
        nums = parse_ints(ret)
        if not nums or nums[0] != 0:
            raise RuntimeError(
                f"[start] EnableRobot rejected by robot: ip={self.ip}, port={self.dashboard_port}, resp={ret}"
            )
        print("EnableRobot OK")

        t = threading.Thread(target=self._feed_loop, daemon=True)
        t.start()

    def pause_motion(self):
        try:
            ret = self.dashboard.Pause()
            print(f"[hotkey] Pause() sent. resp={ret}")
            self._paused_by_hotkey = True
        except Exception as e:
            print(f"[hotkey][warn] Pause() failed: {type(e).__name__}: {e}")

    def continue_motion(self):
        try:
            ret = self.dashboard.Continue()
            print(f"[hotkey] Continue() sent. resp={ret}")
            self._paused_by_hotkey = False
        except Exception as e:
            print(f"[hotkey][warn] Continue() failed: {type(e).__name__}: {e}")

    def wait_done(self, target_cmd_id: int, timeout_s: float = 120.0):
        t = 0.0
        step = 0.02
        tty_attrs = None
        tty_fd = None
        if sys.stdin.isatty():
            try:
                tty_fd = sys.stdin.fileno()
                tty_attrs = termios.tcgetattr(tty_fd)
                tty.setcbreak(tty_fd)
            except Exception:
                tty_fd = None
                tty_attrs = None
        try:
            while t < timeout_s:
                with self._lock:
                    mode = self.robotMode
                    cid = self.currentCmdId

                if self._feed_error is not None:
                    raise RuntimeError(
                        f"[wait_done] FeedBack thread aborted: ip={self.ip}, port={self.feedback_port}, "
                        f"error={type(self._feed_error).__name__}: {self._feed_error}"
                    )

                if mode == 5 and cid == target_cmd_id:
                    return True

                # 任意时刻热键：S 暂停，C 继续
                if tty_fd is not None:
                    try:
                        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if readable:
                            ch = sys.stdin.read(1)
                            key = str(ch).lower()
                            if key == "s":
                                self.pause_motion()
                            elif key == "c":
                                self.continue_motion()
                    except Exception:
                        pass

                sleep(step)
                # 暂停状态不计入超时，避免用户暂停调试时误触发 timeout
                if not self._paused_by_hotkey:
                    t += step

            return False
        finally:
            if tty_fd is not None and tty_attrs is not None:
                try:
                    termios.tcsetattr(tty_fd, termios.TCSADRAIN, tty_attrs)
                except Exception:
                    pass

    def run(self, points, user=0, tool=0, a=20, v=50, cp=0):
        total = len(points)
        for i, p in enumerate(points, 1):
            j1, j2, j3, j4, j5, j6 = p

            try:
                ret = self.dashboard.MovJ(
                    j1, j2, j3, j4, j5, j6,
                    1,              # coordinateMode=1：joint  0：位姿
                    user=user,
                    tool=tool,
                    a=a,            # 加速度1~100
                    v=v,            # 速度1~100
                    cp=cp           # 平滑指数0~100
                )
            except Exception as e:
                point_str = f"[{j1:.4f}, {j2:.4f}, {j3:.4f}, {j4:.4f}, {j5:.4f}, {j6:.4f}]"
                raise RuntimeError(
                    f"[run] MovJ send failed at step {i}/{total} on Dashboard: "
                    f"ip={self.ip}, port={self.dashboard_port}, point={point_str}, "
                    f"error={type(e).__name__}: {e}"
                ) from e

            nums = parse_ints(ret)
            if len(nums) < 2 or nums[0] != 0:
                raise RuntimeError(f"[{i}/{total}] MovJ failed: {ret}")

            cmd_id = nums[1]
            print(f"[{i}/{total}] MovJ sent, ResultID={cmd_id}")

            ok = self.wait_done(cmd_id, timeout_s=180.0)
            if not ok:
                raise TimeoutError(
                    f"[{i}/{total}] wait_done timeout. robotMode={self.robotMode}, currentCmdId={self.currentCmdId}"
                )

            print(f"[{i}/{total}] reached ✅")

        print("结束")

    def move_home(self, home_point, user=0, tool=0, a=20, v=50, cp=0):
        print(f"[home] 手动恢复到 home(P1): {home_point}")
        self.run([home_point], user=user, tool=tool, a=a, v=v, cp=cp)

    def close(self):
        self._stop = True


class GripperController:
    def __init__(self, *, port: str, baudrate: int, init_timeout_s: float):
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.init_timeout_s = float(init_timeout_s)
        self._opened = False
        self._gripper = None
        self._import_gripper_module()

    def _import_gripper_module(self):
        # 兼容两种目录布局：
        # 1) src/detectron2/Gripper
        # 2) src/detectron2/Gripper/Gripper
        cand_dirs = [
            Path(__file__).resolve().parents[2] / "Gripper",
            Path(__file__).resolve().parents[2] / "Gripper" / "Gripper",
        ]
        for d in cand_dirs:
            if d.exists() and str(d) not in sys.path:
                sys.path.append(str(d))
        import dh_modbus_gripper  # noqa: PLC0415  # type: ignore[reportMissingImports]
        import time  # noqa: PLC0415
        self._dh_modbus_gripper = dh_modbus_gripper
        self._time = time

    def connect_and_initialize(self) -> None:
        if self._opened and self._gripper is not None:
            return
        g = self._dh_modbus_gripper.dh_modbus_gripper()
        ret = g.open(self.port, self.baudrate)
        if int(ret) < 0:
            raise RuntimeError(f"open failed: port={self.port}, baudrate={self.baudrate}, ret={ret}")

        g.Initialization()
        t0 = self._time.time()
        initstate = 0
        while int(initstate) != 1 and (self._time.time() - t0) < self.init_timeout_s:
            initstate = g.GetInitState()
            sleep(0.1)
        if int(initstate) != 1:
            try:
                g.close()
            except Exception:
                pass
            raise TimeoutError(
                f"gripper init timeout after {self.init_timeout_s}s on {self.port}, last_initstate={initstate}"
            )

        self._gripper = g
        self._opened = True
        print(
            f"[gripper] initialized once: port={self.port}, baudrate={self.baudrate}, "
            "control=RS485(Modbus RTU)"
        )

    def set_position(self, position: int, *, action_name: str) -> None:
        if not self._opened or self._gripper is None:
            raise RuntimeError("gripper not initialized")
        self._gripper.SetTargetPosition(int(position))
        print(f"[gripper] {action_name} command sent: port={self.port}, pos={int(position)}")

    def open_gripper(self, position: int) -> None:
        self.set_position(int(position), action_name="open")

    def close_gripper(self, position: int) -> None:
        self.set_position(int(position), action_name="close")

    def get_status(self) -> dict:
        if not self._opened or self._gripper is None:
            raise RuntimeError("gripper not initialized")
        return dict(self._gripper.GetStatusSnapshot())

    def release(self) -> None:
        if self._opened and self._gripper is not None:
            try:
                self._gripper.close()
            except Exception:
                pass
        self._opened = False
        self._gripper = None


def wait_hotkey_home_then_run(
    bot: DobotMove,
    home_point,
    planned_points,
    *,
    user=0,
    tool=0,
    a=20,
    v=20,
    cp=50,
    gripper_port="/dev/ttyUSB0",
    gripper_baudrate=115200,
    gripper_close_position=0,
    gripper_open_position=900,
    gripper_init_timeout=5.0,
    gripper_controller: Optional[GripperController] = None,
    gripper_close_delay=2.0,
    disable_gripper_close_at_p4=False,
    disable_camera_at_p1=False,
    camera_script="",
    camera_python=sys.executable,
    camera_no_gui=False,
    camera_show_depth=True,
    camera_capture_only=True,
    camera_ip="",
    camera_serial="",
    camera_index=-1,
    camera_exposure_seq="5,10",
    camera_max_captures=2,
    camera_timeout=120.0,
    camera_required=False,
    camera_extra_args="",
):
    """
    轨迹执行后等待键盘输入：
    - H: 回到 home(P1)
    - K: 重复执行 P1->P4 轨迹
    - O: 打开夹爪
    - Q: 直接退出
    """
    if not sys.stdin.isatty():
        print("[hotkey] 非交互终端，无法监听按键。跳过快捷键监听。")
        return

    print("\n轨迹执行完成。按 'H' 回到 home(P1)，按 'K' 重复轨迹(P1->P4)，按 'O' 打开夹爪，按 'S' 暂停，按 'C' 继续，按 'Q' 退出。")
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                continue
            key = ch.lower()
            if key == "h":
                print("[hotkey] 检测到 H，开始回 home(P1)...")
                bot.move_home(home_point, user=user, tool=tool, a=a, v=v, cp=cp)
                print("[hotkey] 已回到 home(P1)。")
                continue
            if key == "k":
                print("[hotkey] 检测到 K，开始重复执行 P1->P4 轨迹...")
                execute_p1_to_p4_cycle(
                    bot,
                    planned_points=planned_points,
                    user=user,
                    tool=tool,
                    a=a,
                    v=v,
                    cp=cp,
                    disable_camera_at_p1=disable_camera_at_p1,
                    camera_script=camera_script,
                    camera_python=camera_python,
                    camera_no_gui=camera_no_gui,
                    camera_show_depth=camera_show_depth,
                    camera_capture_only=camera_capture_only,
                    camera_ip=camera_ip,
                    camera_serial=camera_serial,
                    camera_index=camera_index,
                    camera_exposure_seq=camera_exposure_seq,
                    camera_max_captures=camera_max_captures,
                    camera_timeout=camera_timeout,
                    camera_required=camera_required,
                    camera_extra_args=camera_extra_args,
                    disable_gripper_close_at_p4=disable_gripper_close_at_p4,
                    gripper_close_delay=gripper_close_delay,
                    gripper_port=gripper_port,
                    gripper_baudrate=gripper_baudrate,
                    gripper_close_position=gripper_close_position,
                    gripper_init_timeout=gripper_init_timeout,
                    gripper_controller=gripper_controller,
                )
                print("[hotkey] 重复轨迹执行完成。")
                continue
            if key == "o":
                print("[hotkey] 检测到 O，开始打开夹爪...")
                if gripper_controller is not None:
                    gripper_controller.open_gripper(int(gripper_open_position))
                else:
                    close_gripper_once(
                        port=str(gripper_port),
                        baudrate=int(gripper_baudrate),
                        close_position=int(gripper_open_position),
                        init_timeout_s=float(gripper_init_timeout),
                        action_name="open",
                    )
                print("[hotkey] 夹爪打开完成。")
                continue
            if key == "q":
                print("[hotkey] 检测到 Q，退出，不回 home。")
                return
            if key == "s":
                bot.pause_motion()
                continue
            if key == "c":
                bot.continue_motion()
                continue
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def close_gripper_once(
    *,
    port: str,
    baudrate: int,
    close_position: int,
    init_timeout_s: float,
    action_name: str = "close",
) -> None:
    """
    参考 GripperTestPython.py 的逻辑：
    - 打开串口
    - 初始化夹爪
    - 设置闭合位置（默认 0）
    - 关闭串口
    """
    gripper_dir = Path(__file__).resolve().parents[2] / "Gripper" / "Gripper"
    if str(gripper_dir) not in sys.path:
        sys.path.append(str(gripper_dir))

    import dh_modbus_gripper  # noqa: PLC0415  # type: ignore[reportMissingImports]
    import time  # noqa: PLC0415

    g = dh_modbus_gripper.dh_modbus_gripper()
    opened = False
    try:
        ret = g.open(port, int(baudrate))
        if int(ret) < 0:
            raise RuntimeError(f"open failed: port={port}, baudrate={baudrate}, ret={ret}")
        opened = True

        g.Initialization()
        t0 = time.time()
        initstate = 0
        while int(initstate) != 1 and (time.time() - t0) < float(init_timeout_s):
            initstate = g.GetInitState()
            sleep(0.1)
        if int(initstate) != 1:
            raise TimeoutError(
                f"gripper init timeout after {init_timeout_s}s on {port}, last_initstate={initstate}"
            )

        g.SetTargetPosition(int(close_position))
        print(f"[gripper] {action_name} command sent: port={port}, pos={close_position}")
    finally:
        if opened:
            try:
                g.close()
            except Exception:
                pass


def run_camera_program_at_p1(
    *,
    camera_python: str,
    camera_script: str,
    camera_no_gui: bool,
    camera_show_depth: bool,
    camera_capture_only: bool,
    camera_ip: str,
    camera_serial: str,
    camera_index: int,
    camera_exposure_seq: str,
    camera_max_captures: int,
    camera_timeout: float,
    camera_extra_args: str,
) -> bool:
    script = str(camera_script).strip()
    if not script:
        raise ValueError("camera_script 为空，无法调用相机程序。")
    script_path = Path(script).expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"camera_script 不存在: {script_path}")

    cmd = [str(camera_python).strip() or sys.executable, str(script_path)]
    if bool(camera_no_gui):
        cmd.append("--no-gui")
    if bool(camera_show_depth):
        cmd.append("--show-depth")
    if bool(camera_capture_only):
        cmd.append("--capture-only")
    cmd += ["--max-captures", str(max(1, int(camera_max_captures)))]
    if str(camera_exposure_seq).strip():
        doubled_exposure_seq = scale_exposure_seq(str(camera_exposure_seq).strip(), scale=2.0)
        cmd += ["--exposure-seq", doubled_exposure_seq]
    if str(camera_ip).strip():
        cmd += ["--ip", str(camera_ip).strip()]
    if str(camera_serial).strip():
        cmd += ["--serial", str(camera_serial).strip()]
    if int(camera_index) >= 0:
        cmd += ["--index", str(int(camera_index))]
    if str(camera_extra_args).strip():
        cmd += shlex.split(str(camera_extra_args).strip())

    print("[camera] 调用相机脚本：")
    print("  " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=float(camera_timeout))
        print("[camera] 曝光采集完成并自动退出。")
        return True
    except subprocess.TimeoutExpired as e:
        print(f"[camera][warn] 相机程序超时（>{camera_timeout}s）：{e}")
        return False
    except subprocess.CalledProcessError as e:
        if int(e.returncode) < 0:
            print(
                f"[camera][warn] 相机程序异常终止（signal={-int(e.returncode)}），"
                "可能是 SDK/网络异常。将按策略决定是否继续。"
            )
        else:
            print(f"[camera][warn] 相机程序退出码非 0：returncode={e.returncode}")
        return False
    except Exception as e:
        print(f"[camera][warn] 相机程序调用失败：{type(e).__name__}: {e}")
        return False


def execute_p1_to_p4_cycle(
    bot: DobotMove,
    *,
    planned_points,
    user=0,
    tool=0,
    a=20,
    v=20,
    cp=50,
    disable_camera_at_p1=False,
    camera_script="",
    camera_python=sys.executable,
    camera_no_gui=False,
    camera_show_depth=True,
    camera_capture_only=True,
    camera_ip="",
    camera_serial="",
    camera_index=-1,
    camera_exposure_seq="5,10",
    camera_max_captures=2,
    camera_timeout=120.0,
    camera_required=False,
    camera_extra_args="",
    disable_gripper_close_at_p4=False,
    gripper_close_delay=2.0,
    gripper_port="/dev/ttyUSB0",
    gripper_baudrate=115200,
    gripper_close_position=0,
    gripper_init_timeout=5.0,
    gripper_controller: Optional[GripperController] = None,
):
    if not planned_points:
        raise ValueError("planned_points 为空，无法执行轨迹。")

    # 先到 P1
    bot.run([planned_points[0]], user=user, tool=tool, a=a, v=v, cp=cp)

    # 在 P1 触发相机：两次曝光后自动退出
    if not disable_camera_at_p1:
        ok = run_camera_program_at_p1(
            camera_python=camera_python,
            camera_script=camera_script,
            camera_no_gui=camera_no_gui,
            camera_show_depth=camera_show_depth,
            camera_capture_only=camera_capture_only,
            camera_ip=camera_ip,
            camera_serial=camera_serial,
            camera_index=camera_index,
            camera_exposure_seq=camera_exposure_seq,
            camera_max_captures=camera_max_captures,
            camera_timeout=camera_timeout,
            camera_extra_args=camera_extra_args,
        )
        if not ok and bool(camera_required):
            raise RuntimeError("[camera] 采集失败且 --camera-required 已启用，流程中止。")
        if not ok and not bool(camera_required):
            print("[camera][warn] 采集失败，继续执行 P2/P3/P4。")

    # 再走 P2->P4（插值后的剩余点）
    if len(planned_points) > 1:
        bot.run(planned_points[1:], user=user, tool=tool, a=a, v=v, cp=cp)

    # 到达 P4 后夹爪闭合
    if not disable_gripper_close_at_p4:
        print(f"[gripper] 到达 P4，等待 {gripper_close_delay:.2f}s 后执行夹爪闭合...")
        sleep(max(0.0, float(gripper_close_delay)))
        if gripper_controller is not None:
            gripper_controller.close_gripper(int(gripper_close_position))
        else:
            close_gripper_once(
                port=str(gripper_port),
                baudrate=int(gripper_baudrate),
                close_position=int(gripper_close_position),
                init_timeout_s=float(gripper_init_timeout),
                action_name="close",
            )


if __name__ == "__main__":
    try:
        args = build_argparser().parse_args()
        p1 = parse_point(args.p1)
        p2 = parse_point(args.p2)
        p3 = parse_point(args.p3)
        p4 = parse_point(args.p4)
        home_point = p1[:]  # 约定 home 点为 P1
        planned_points = plan_from_four_points(p1, p2, p3, p4, samples=args.samples)

        print("规划输入四点:")
        print("  p1 =", p1)
        print("  p2 =", p2)
        print("  p3 =", p3)
        print("  p4 =", p4)
        print("  home(P1) =", home_point)
        print(f"自动生成轨迹点数: {len(planned_points)}")
        for idx, point in enumerate(planned_points, 1):
            print(f"  [{idx:02d}] {point}")
        print("\n提示：默认不会自动回 home。")
        print("轨迹完成后可在终端按 H 回 home(P1)，按 K 重复轨迹(P1->P4)，按 O 打开夹爪，按 Q 退出。")

        if args.dry_run:
            print("dry-run 模式：未下发运动命令。")
        else:
            bot = DobotMove(args.ip)
            gripper = None
            try:
                print("[gripper] 启动阶段先执行一次夹爪张开...")
                gripper = GripperController(
                    port=str(args.gripper_port),
                    baudrate=int(args.gripper_baudrate),
                    init_timeout_s=float(args.gripper_init_timeout),
                )
                gripper.connect_and_initialize()
                gripper.open_gripper(int(args.gripper_open_position))
                bot.start()
                execute_p1_to_p4_cycle(
                    bot,
                    planned_points=planned_points,
                    user=args.user,
                    tool=args.tool,
                    a=args.a,
                    v=args.v,
                    cp=args.cp,
                    disable_camera_at_p1=args.disable_camera_at_p1,
                    camera_script=args.camera_script,
                    camera_python=args.camera_python,
                    camera_no_gui=args.camera_no_gui,
                    camera_show_depth=args.camera_show_depth,
                    camera_capture_only=args.camera_capture_only,
                    camera_ip=args.camera_ip,
                    camera_serial=args.camera_serial,
                    camera_index=args.camera_index,
                    camera_exposure_seq=args.camera_exposure_seq,
                    camera_max_captures=args.camera_max_captures,
                    camera_timeout=args.camera_timeout,
                    camera_required=args.camera_required,
                    camera_extra_args=args.camera_extra_args,
                    disable_gripper_close_at_p4=args.disable_gripper_close_at_p4,
                    gripper_close_delay=args.gripper_close_delay,
                    gripper_port=args.gripper_port,
                    gripper_baudrate=args.gripper_baudrate,
                    gripper_close_position=args.gripper_close_position,
                    gripper_init_timeout=args.gripper_init_timeout,
                    gripper_controller=gripper,
                )
                if not args.disable_hotkey_home:
                    wait_hotkey_home_then_run(
                        bot,
                        home_point,
                        planned_points,
                        user=args.user,
                        tool=args.tool,
                        a=args.a,
                        v=args.v,
                        cp=args.cp,
                        gripper_port=args.gripper_port,
                        gripper_baudrate=args.gripper_baudrate,
                        gripper_close_position=args.gripper_close_position,
                        gripper_open_position=args.gripper_open_position,
                        gripper_init_timeout=args.gripper_init_timeout,
                        gripper_controller=gripper,
                        gripper_close_delay=args.gripper_close_delay,
                        disable_gripper_close_at_p4=args.disable_gripper_close_at_p4,
                        disable_camera_at_p1=args.disable_camera_at_p1,
                        camera_script=args.camera_script,
                        camera_python=args.camera_python,
                        camera_no_gui=args.camera_no_gui,
                        camera_show_depth=args.camera_show_depth,
                        camera_capture_only=args.camera_capture_only,
                        camera_ip=args.camera_ip,
                        camera_serial=args.camera_serial,
                        camera_index=args.camera_index,
                        camera_exposure_seq=args.camera_exposure_seq,
                        camera_max_captures=args.camera_max_captures,
                        camera_timeout=args.camera_timeout,
                        camera_required=args.camera_required,
                        camera_extra_args=args.camera_extra_args,
                    )
            finally:
                if gripper is not None:
                    gripper.release()
                bot.close()
    except Exception as e:
        print("\n[move.py] 执行失败")
        print(f"  error_type: {type(e).__name__}")
        print(f"  error: {e}")
        print("  traceback:")
        print(traceback.format_exc())
        raise

