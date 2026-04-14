import argparse
import select
import sys
import termios
import tty
import traceback
from time import sleep
from typing import Optional

from move import (
    DobotMove,
    GripperController,
    close_gripper_once,
    parse_point,
    plan_from_four_points,
)


def build_argparser():
    p = argparse.ArgumentParser(
        description="转运热键脚本：H 执行 P1->P4；K 先回 P1 再执行到 P5。"
    )
    p.add_argument("--ip", default="192.168.5.2", help="机械臂 IP")
    p.add_argument(
        "--p1",
        default="-19.1160,26.4384,-122.4939,71.5494,85.4363,4.4136",
        help="起点 P1（6 关节值，逗号分隔）",
    )
    p.add_argument(
        "--p2",
        default="-20.8615,4.9481,-133.9164,125.4493,85.0759,4.4135",
        help="中间控制点 P2（6 关节值，逗号分隔）",
    )
    p.add_argument(
        "--p3",
        default="-20.5522,-17.4762,-109.6515,124.6196,82.4423,4.5724",
        help="中间控制点 P3（6 关节值，逗号分隔）",
    )
    p.add_argument(
        "--p4",
        default="-20.5917,-21.7659,-106.6855,127.1042,82.0431,4.5954",
        help="终点 P4（H 触发的轨迹终点）",
    )
    p.add_argument(
        "--p5",
        default="-58.3833,-27.0070,-102.9371,130.5150,83.3058,4.4146",
        help="终点 P5（K 触发的轨迹终点）",
    )
    p.add_argument("--samples", type=int, default=20, help="每条轨迹采样点数，至少 2")
    p.add_argument("--user", type=int, default=0)
    p.add_argument("--tool", type=int, default=0)
    p.add_argument("--a", type=int, default=20, help="加速度 1~100")
    p.add_argument("--v", type=int, default=20, help="速度 1~100")
    p.add_argument("--cp", type=int, default=50, help="平滑指数 0~100")
    p.add_argument(
        "--disable-gripper-close-at-p4",
        action="store_true",
        help="禁用到达 P4 后自动闭合夹爪。",
    )
    p.add_argument("--gripper-close-delay", type=float, default=2.0, help="到达 P4 后闭合夹爪前的延时（秒）")
    p.add_argument("--gripper-port", default="/dev/ttyUSB0", help="夹爪串口，例如 /dev/ttyUSB0")
    p.add_argument("--gripper-baudrate", type=int, default=115200, help="夹爪串口波特率")
    p.add_argument("--gripper-close-position", type=int, default=0, help="夹爪闭合位置（默认 0）")
    p.add_argument("--gripper-open-position", type=int, default=900, help="夹爪打开位置（默认 900）")
    p.add_argument("--gripper-init-timeout", type=float, default=5.0, help="夹爪初始化超时（秒）")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印规划结果，不连接机械臂。",
    )
    return p


def run_route(bot: DobotMove, points, *, user: int, tool: int, a: int, v: int, cp: int, tag: str):
    print(f"[route] 开始执行 {tag}，共 {len(points)} 个点")
    bot.run(points, user=user, tool=tool, a=a, v=v, cp=cp)
    print(f"[route] {tag} 执行完成")


def close_gripper_at_endpoint(
    *,
    endpoint_name: str,
    gripper_close_delay: float,
    gripper_port: str,
    gripper_baudrate: int,
    gripper_close_position: int,
    gripper_init_timeout: float,
    gripper_controller: Optional[GripperController],
):
    print(f"[gripper] 到达 {endpoint_name}，等待 {gripper_close_delay:.2f}s 后执行夹爪闭合...")
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


def hotkey_loop(
    bot: DobotMove,
    *,
    p1,
    path_p1_p4,
    path_p1_p5,
    user: int,
    tool: int,
    a: int,
    v: int,
    cp: int,
    disable_gripper_close_at_p4: bool,
    gripper_close_delay: float,
    gripper_port: str,
    gripper_baudrate: int,
    gripper_close_position: int,
    gripper_open_position: int,
    gripper_init_timeout: float,
    gripper_controller: Optional[GripperController],
):
    if not sys.stdin.isatty():
        raise RuntimeError("当前不是交互终端，无法监听热键。")

    print("\n热键说明：")
    print("  H: 按 P1->P4 轨迹执行转运")
    print("  K: 先回退到 P1，再按 P1->P5 轨迹执行转运")
    print("  O: 打开夹爪")
    print("  P: 闭合夹爪")
    print("  S: 暂停运动")
    print("  C: 继续运动")
    print("  Q: 退出程序")

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
                run_route(
                    bot,
                    path_p1_p4,
                    user=user,
                    tool=tool,
                    a=a,
                    v=v,
                    cp=cp,
                    tag="P1->P4",
                )
                if not disable_gripper_close_at_p4:
                    close_gripper_at_endpoint(
                        endpoint_name="P4",
                        gripper_close_delay=gripper_close_delay,
                        gripper_port=gripper_port,
                        gripper_baudrate=gripper_baudrate,
                        gripper_close_position=gripper_close_position,
                        gripper_init_timeout=gripper_init_timeout,
                        gripper_controller=gripper_controller,
                    )
            elif key == "k":
                print("[route] K 触发：先回退到 P1")
                bot.move_home(p1, user=user, tool=tool, a=a, v=v, cp=cp)
                run_route(
                    bot,
                    path_p1_p5,
                    user=user,
                    tool=tool,
                    a=a,
                    v=v,
                    cp=cp,
                    tag="P1->P5",
                )
            elif key == "o":
                print("[gripper] O 触发：打开夹爪...")
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
                print("[gripper] 打开夹爪完成。")
            elif key == "p":
                print("[gripper] P 触发：闭合夹爪...")
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
                print("[gripper] 闭合夹爪完成。")
            elif key == "s":
                bot.pause_motion()
            elif key == "c":
                bot.continue_motion()
            elif key == "q":
                print("[route] 收到 Q，退出。")
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


if __name__ == "__main__":
    try:
        args = build_argparser().parse_args()
        p1 = parse_point(args.p1)
        p2 = parse_point(args.p2)
        p3 = parse_point(args.p3)
        p4 = parse_point(args.p4)
        p5 = parse_point(args.p5)

        path_p1_p4 = plan_from_four_points(p1, p2, p3, p4, samples=args.samples)
        path_p1_p5 = plan_from_four_points(p1, p2, p3, p5, samples=args.samples)

        print("已规划两条轨迹：")
        print(f"  H: P1->P4，共 {len(path_p1_p4)} 点")
        print(f"  K: 回 P1 后 P1->P5，共 {len(path_p1_p5)} 点")

        if args.dry_run:
            print("dry-run 模式：不连接机械臂，程序结束。")
            sys.exit(0)

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
            hotkey_loop(
                bot,
                p1=p1,
                path_p1_p4=path_p1_p4,
                path_p1_p5=path_p1_p5,
                user=args.user,
                tool=args.tool,
                a=args.a,
                v=args.v,
                cp=args.cp,
                disable_gripper_close_at_p4=args.disable_gripper_close_at_p4,
                gripper_close_delay=args.gripper_close_delay,
                gripper_port=args.gripper_port,
                gripper_baudrate=args.gripper_baudrate,
                gripper_close_position=args.gripper_close_position,
                gripper_open_position=args.gripper_open_position,
                gripper_init_timeout=args.gripper_init_timeout,
                gripper_controller=gripper,
            )
        finally:
            if gripper is not None:
                gripper.release()
            bot.close()
    except Exception as e:
        print("\n[move_transfer_hotkey.py] 执行失败")
        print(f"  error_type: {type(e).__name__}")
        print(f"  error: {e}")
        print("  traceback:")
        print(traceback.format_exc())
        raise
