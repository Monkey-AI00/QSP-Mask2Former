#!/usr/bin/env python3
"""
从 icp_registration.py 的 json 输出中，计算插头(CAD)在相机坐标系下的 6D 位姿，
并把“抓取点（相对 CAD 原点的偏移）”变换到相机坐标系。

关键约定（与 icp_registration.py 一致）：
  - json 内的 T_source_to_target == T_final :  source(相机点云) -> target(CAD) 的变换
  - 我们需要的插头位姿：T_CAD_to_Camera = inv(T_source_to_target)

用法示例：
  python pose_from_icp.py --icp-json output_live/icp_result_refined.json --offset 28 0 -80 --offset-unit mm
  # 若需要 Base 坐标系（最常见）：再提供 T_base_to_camera（Base->Camera）
  python pose_from_icp.py --icp-json output_live/icp_result_refined.json --base-cam-json T_base_cam.json --pretty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _as_T(mat) -> np.ndarray:
    T = np.asarray(mat, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"期望 4x4 矩阵，得到 {T.shape}")
    return T


def _rot_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    旋转矩阵 -> 四元数 (w, x, y, z)，右手系。
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("R 必须是 3x3")

    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # 找到对角线最大项，数值更稳定
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q = q / n
    # 统一符号：让 w >= 0（同一旋转对应 q 与 -q）
    if q[0] < 0:
        q = -q
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _rot_to_euler_zyx_rad(R: np.ndarray) -> Tuple[float, float, float]:
    """
    旋转矩阵 -> 欧拉角（ZYX：yaw-pitch-roll），返回 (roll, pitch, yaw) 弧度。
    约定：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("R 必须是 3x3")

    sy = -R[2, 0]
    sy_clamped = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy_clamped))

    # 接近万向节锁时，yaw/roll 不唯一；这里给一个稳定解
    if abs(abs(sy_clamped) - 1.0) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def _load_icp_json(path: Path) -> Dict:
    data = json.loads(path.read_text())
    if "T_source_to_target" not in data:
        raise KeyError("json 中找不到键：T_source_to_target")
    return data


def _load_base_cam_T(path: Path) -> np.ndarray:
    """
    读取 Base->Camera 外参矩阵（4x4）。
    允许的 key（任一存在即可）：
      - T_base_to_camera
      - T_base_to_cam
      - T_base_cam
      - T_base2cam
      - T_base_to_sensor / T_base_to_rgb / T_base_to_depth（有些项目会这样命名）
    """
    data = json.loads(path.read_text())
    keys = [
        "T_base_to_camera",
        "T_base_to_cam",
        "T_base_cam",
        "T_base2cam",
        "T_base_to_sensor",
        "T_base_to_rgb",
        "T_base_to_depth",
    ]
    for k in keys:
        if k in data:
            return _as_T(data[k])
    raise KeyError(f"base-cam json 中未找到外参矩阵。支持的 key: {keys}")


def main() -> int:
    ap = argparse.ArgumentParser(description="从 ICP 结果计算插头 6D 位姿与抓取点（相机系）")
    ap.add_argument("--icp-json", required=True, help="icp_registration.py 保存的 json 路径（包含 T_source_to_target）")
    ap.add_argument(
        "--offset",
        nargs=3,
        type=float,
        default=[28.0, 0.0, -80.0],
        help="抓取点相对 CAD 原点的偏移 (x y z)，默认 28 0 -80",
    )
    ap.add_argument(
        "--offset-unit",
        choices=["mm", "m", "same"],
        default="mm",
        help="offset 的单位：mm(默认，自动乘 0.001)、m(不缩放)、same(不缩放)",
    )
    ap.add_argument(
        "--base-cam-json",
        default="",
        help="Base->Camera 外参 json 路径（4x4）。提供后将输出 Base 坐标系下的 XYZ+RPY 以及抓取点 Base 坐标。",
    )
    ap.add_argument(
        "--T-base-cam",
        nargs=16,
        type=float,
        default=None,
        help="直接在命令行给 Base->Camera 4x4（16 个数，按行展开 row-major）。优先级高于 --base-cam-json。",
    )
    ap.add_argument("--pretty", action="store_true", help="以更易读的格式打印")
    args = ap.parse_args()

    json_path = Path(args.icp_json).expanduser().resolve()
    data = _load_icp_json(json_path)
    T_s2t = _as_T(data["T_source_to_target"])
    T_t2s = np.linalg.inv(T_s2t)  # CAD -> Camera

    R_cam = T_t2s[:3, :3]
    t_cam = T_t2s[:3, 3]

    roll, pitch, yaw = _rot_to_euler_zyx_rad(R_cam)
    euler_deg = np.rad2deg([roll, pitch, yaw]).astype(np.float64)
    qw, qx, qy, qz = _rot_to_quat_wxyz(R_cam)

    offset = np.asarray(args.offset, dtype=np.float64).reshape(3)
    if args.offset_unit == "mm":
        offset = offset * 0.001
    # m/same 都不缩放
    p_grasp_cad = np.array([offset[0], offset[1], offset[2], 1.0], dtype=np.float64)
    p_grasp_cam = (T_t2s @ p_grasp_cad)[:3]

    out = {
        "icp_json": str(json_path),
        "T_CAD_to_Camera": T_t2s.tolist(),
        "translation_camera": {"x": float(t_cam[0]), "y": float(t_cam[1]), "z": float(t_cam[2])},
        "euler_zyx_deg": {"roll_x": float(euler_deg[0]), "pitch_y": float(euler_deg[1]), "yaw_z": float(euler_deg[2])},
        "quaternion_wxyz": {"w": float(qw), "x": float(qx), "y": float(qy), "z": float(qz)},
        "grasp_point": {
            "offset_cad": {"x": float(offset[0]), "y": float(offset[1]), "z": float(offset[2])},
            "point_camera": {"x": float(p_grasp_cam[0]), "y": float(p_grasp_cam[1]), "z": float(p_grasp_cam[2])},
        },
        "note": "RPY 为 ZYX(yaw-pitch-roll) 分解后按 (roll_x,pitch_y,yaw_z) 输出。若要 Base 坐标系：T_CAD_to_Base = T_base_to_camera @ T_CAD_to_Camera。",
    }

    # === Base 坐标系（若提供 T_base_to_camera）===
    T_b2c = None
    if args.T_base_cam is not None:
        T_b2c = np.asarray(args.T_base_cam, dtype=np.float64).reshape(4, 4)
    elif args.base_cam_json:
        T_b2c = _load_base_cam_T(Path(args.base_cam_json).expanduser().resolve())

    if T_b2c is not None:
        T_cad2base = np.asarray(T_b2c, dtype=np.float64) @ np.asarray(T_t2s, dtype=np.float64)
        R_base = T_cad2base[:3, :3]
        t_base = T_cad2base[:3, 3]
        r2, p2, y2 = _rot_to_euler_zyx_rad(R_base)
        euler_base_deg = np.rad2deg([r2, p2, y2]).astype(np.float64)
        p_grasp_base = (np.asarray(T_b2c, dtype=np.float64) @ np.array([p_grasp_cam[0], p_grasp_cam[1], p_grasp_cam[2], 1.0]))[:3]

        out["T_base_to_camera"] = np.asarray(T_b2c, dtype=np.float64).tolist()
        out["T_CAD_to_Base"] = T_cad2base.tolist()
        out["translation_base"] = {"x": float(t_base[0]), "y": float(t_base[1]), "z": float(t_base[2])}
        out["rpy_zyx_deg_base"] = {
            "roll_x": float(euler_base_deg[0]),
            "pitch_y": float(euler_base_deg[1]),
            "yaw_z": float(euler_base_deg[2]),
        }
        out["grasp_point"]["point_base"] = {"x": float(p_grasp_base[0]), "y": float(p_grasp_base[1]), "z": float(p_grasp_base[2])}

    if args.pretty:
        print("=== 插头(CAD)在相机系的 6D 位姿 ===")
        print("T_CAD_to_Camera = inv(T_source_to_target):")
        print(np.asarray(T_t2s))
        print(f"translation(camera): x={t_cam[0]:.6f}, y={t_cam[1]:.6f}, z={t_cam[2]:.6f}")
        print(
            "euler(ZYX) deg: roll_x={:.3f}, pitch_y={:.3f}, yaw_z={:.3f}".format(
                euler_deg[0], euler_deg[1], euler_deg[2]
            )
        )
        print("quat(wxyz): w={:.6f}, x={:.6f}, y={:.6f}, z={:.6f}".format(qw, qx, qy, qz))
        print("=== 抓取点（相机系）===")
        print(
            "offset(CAD) = ({:.6f}, {:.6f}, {:.6f}) [{}]".format(
                offset[0], offset[1], offset[2], args.offset_unit
            )
        )
        print("grasp_point(camera) = ({:.6f}, {:.6f}, {:.6f})".format(p_grasp_cam[0], p_grasp_cam[1], p_grasp_cam[2]))

        if T_b2c is not None:
            print("=== 插头(CAD)在 Base 坐标系的 6D 位姿（XYZ + RPY）===")
            print("T_CAD_to_Base = T_base_to_camera @ T_CAD_to_Camera:")
            print(np.asarray(out["T_CAD_to_Base"], dtype=np.float64))
            tb = out["translation_base"]
            rb = out["rpy_zyx_deg_base"]
            print(f"translation(base): x={tb['x']:.6f}, y={tb['y']:.6f}, z={tb['z']:.6f}")
            print("RPY(ZYX) deg: roll_x={:.3f}, pitch_y={:.3f}, yaw_z={:.3f}".format(rb["roll_x"], rb["pitch_y"], rb["yaw_z"]))
            gb = out["grasp_point"].get("point_base")
            if gb:
                print("grasp_point(base) = ({:.6f}, {:.6f}, {:.6f})".format(gb["x"], gb["y"], gb["z"]))

        print("=== 机器可用 JSON（可直接复制）===")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


