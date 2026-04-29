"""Gripper helpers migrated from legacy pipeline."""

from __future__ import annotations

import time

from online_grasp.geometry.transforms import _parse_csv_floats


def format_gripper_status(status) -> str:
    init_state = status.get("init_state", "?")
    init_desc = status.get("init_state_desc", "")
    grip_state = status.get("grip_state", "?")
    grip_desc = status.get("grip_state_desc", "")
    cur = status.get("current_position", "?")
    tgt = status.get("target_position", "?")
    force = status.get("target_force", "?")
    speed = status.get("target_speed", "?")
    init_text = f"{init_state}({init_desc})" if init_desc else f"{init_state}"
    grip_text = f"{grip_state}({grip_desc})" if grip_desc else f"{grip_state}"
    return (
        f"init={init_text} grip={grip_text} "
        f"pos={cur} target={tgt} force={force} speed={speed}"
    )


def close_gripper_after_grasp(ctx, *, context: str):
    if ctx.executor is None:
        return
    if not bool(ctx.args.gripper_enable):
        return
    try:
        print(f"[gripper] 到达抓取点后闭合夹爪并停留等待（{context}）")
        ctx.executor.close_gripper()
        timeout_s = max(0.0, float(getattr(ctx.args, "gripper_close_feedback_timeout", 2.0)))
        poll_s = max(0.05, float(getattr(ctx.args, "gripper_feedback_interval", 0.5)))
        if timeout_s > 0.0:
            t0 = time.time()
            last_status_line = ""
            got_object = False
            while (time.time() - t0) < timeout_s:
                status = ctx.executor.get_gripper_status()
                if status is not None:
                    grip_state = int(status.get("grip_state", -1))
                    ctx._last_gripper_state = grip_state
                    line = format_gripper_status(status)
                    if line != last_status_line:
                        print(f"[gripper][feedback][close] {line}")
                        last_status_line = line
                    if grip_state == 2:
                        got_object = True
                        print("[gripper][feedback][close] 已检测到物体（grip=2）")
                        break
                time.sleep(poll_s)
            if last_status_line:
                print(f"[gripper][feedback][close] 闭合完成，最终状态: {last_status_line}")
            else:
                print("[gripper][feedback][close] 未读取到夹爪状态反馈")
            if not got_object:
                print("[gripper][feedback][close] 未在等待窗口内检测到 grip=2")
    except Exception as e:
        print(f"[gripper][warn] 自动闭合失败（{context}）: {type(e).__name__}: {e}")


def wait_grip2_then_back_to_p1(ctx):
    if ctx.executor is None:
        return
    if not bool(ctx.args.gripper_enable):
        return
    timeout_s = max(0.0, float(getattr(ctx.args, "gripper_close_feedback_timeout", 2.0)))
    poll_s = max(0.05, float(getattr(ctx.args, "gripper_feedback_interval", 0.5)))
    t0 = time.time()
    grip2 = (ctx._last_gripper_state == 2)
    while (not grip2) and ((time.time() - t0) < timeout_s):
        status = ctx.executor.get_gripper_status()
        if status is not None:
            ctx._last_gripper_state = int(status.get("grip_state", -1))
            print(f"[gripper][feedback][wait] {format_gripper_status(status)}")
            if ctx._last_gripper_state == 2:
                grip2 = True
                break
        time.sleep(poll_s)
    if not grip2:
        print("[gripper][wait] 未检测到 grip=2，跳过回 p1")
        return
    p1_joint = _parse_csv_floats(ctx.args.p1, 6)
    print(f"[robot] grip=2，执行回到 p1={p1_joint.tolist()}")
    ctx.executor.movj_joint(p1_joint, "post_grasp_back_to_p1")


def handle_gripper_hotkey(ctx, key: int):
    if key not in (ord("o"), ord("c"), ord("p")):
        return
    if ctx.executor is None:
        print("[gripper][hotkey] 当前无机器人连接，忽略按键")
        return
    if key == ord("p"):
        try:
            p1_joint = _parse_csv_floats(ctx.args.p1, 6)
            print(f"[robot][hotkey] p -> 回到 p1={p1_joint.tolist()}")
            ctx.executor.movj_joint(p1_joint, "hotkey_back_to_p1")
        except Exception as e:
            print(f"[robot][hotkey][warn] 回到 p1 失败: {type(e).__name__}: {e}")
    elif key == ord("o"):
        try:
            ctx.executor.open_gripper()
            print("[gripper][hotkey] o -> 打开夹爪")
        except Exception as e:
            print(f"[gripper][hotkey][warn] 打开夹爪失败: {type(e).__name__}: {e}")
    elif key == ord("c"):
        try:
            ctx.executor.close_gripper()
            print("[gripper][hotkey] c -> 闭合夹爪")
        except Exception as e:
            print(f"[gripper][hotkey][warn] 闭合夹爪失败: {type(e).__name__}: {e}")


def maybe_print_gripper_stroke(ctx):
    if ctx.executor is None or (not bool(ctx.args.gripper_enable)):
        return
    interval_s = float(getattr(ctx.args, "gripper_feedback_interval", 0.5))
    if interval_s <= 0.0:
        return
    now = time.time()
    if (now - float(ctx._last_gripper_feedback_ts)) < interval_s:
        return
    ctx._last_gripper_feedback_ts = now
    status = ctx.executor.get_gripper_status()
    if status is not None:
        print(f"[gripper][feedback] {format_gripper_status(status)}")


__all__ = [
    "format_gripper_status",
    "close_gripper_after_grasp",
    "wait_grip2_then_back_to_p1",
    "handle_gripper_hotkey",
    "maybe_print_gripper_stroke",
]

