#!/usr/bin/env python3
"""
离线录制数据集（Mech-Eye NANO / Area Scan 3D Camera）

输出目录结构（默认 ./dataset）：
  dataset/
    color/ 000000.png ...
    depth/ 000000.png ...   # 16-bit PNG，单位=毫米（0 表示无效/背景）
    meta/
      intrinsics.json       # Open3D PinholeCameraIntrinsic 需要的参数

设计原则：
- 不做实时重建，只做“稳定录制”，方便后续离线 PointRend 分割 + Open3D 重建
- 深度统一转换为 uint16(mm)，替换 NaN/负值为 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mecheye.shared import *  # type: ignore # noqa: F401,F403
from mecheye.area_scan_3d_camera import *  # type: ignore # noqa: F401,F403
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d  # type: ignore


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _depth_to_u16_mm(depth_any: np.ndarray) -> np.ndarray:
    """
    Mech-Eye SDK 的 depth_map.data() 常见为 float(mm) 或 uint16(mm)。
    这里做稳健转换：NaN/inf/负值 -> 0，其他四舍五入并 clip 到 uint16。
    """
    depth = np.asarray(depth_any)
    if depth.dtype == np.uint16:
        return depth
    depth_f = depth.astype(np.float32, copy=False)
    depth_f[~np.isfinite(depth_f)] = 0.0
    depth_f[depth_f < 0.0] = 0.0
    depth_u16 = np.clip(np.rint(depth_f), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_u16


def _save_intrinsics_json(meta_dir: Path, intrinsics: "CameraIntrinsics", width: int, height: int) -> Path:
    fx = float(intrinsics.depth.camera_matrix.fx)
    fy = float(intrinsics.depth.camera_matrix.fy)
    cx = float(intrinsics.depth.camera_matrix.cx)
    cy = float(intrinsics.depth.camera_matrix.cy)

    payload = {
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "depth_unit": "mm",
        "depth_scale_for_open3d": 1000.0,  # Open3D: depth_scale=1000 表示输入深度单位=毫米
    }
    out = meta_dir / "intrinsics.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Mech-Eye NANO 离线录制 color/depth 数据集（供 Open3D 重建）")
    parser.add_argument("--out", default=str(Path.cwd() / "dataset"), help="输出数据集根目录")
    parser.add_argument("--num-frames", type=int, default=80, help="录制帧数（建议 50-100）")
    parser.add_argument("--interval-ms", type=int, default=120, help="两帧间隔（毫秒）。手持转动建议 >=80ms")
    parser.add_argument("--start-idx", type=int, default=0, help="起始帧编号")
    parser.add_argument("--preview", action="store_true", help="显示预览窗口（按 q 退出）")
    parser.add_argument("--save-raw-tiff", action="store_true", help="额外保存原始深度 tiff（调试用）")
    args = parser.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    color_dir = out_root / "color"
    depth_dir = out_root / "depth"
    meta_dir = out_root / "meta"
    _ensure_dir(color_dir)
    _ensure_dir(depth_dir)
    _ensure_dir(meta_dir)

    cam = Camera()
    if not find_and_connect(cam):
        print("✗ 未能连接到 Mech-Eye 相机")
        return 2

    try:
        if not confirm_capture_3d():
            print("已取消 3D 采集。")
            return 0

        # 读取内参（写到 meta/intrinsics.json）
        intr = CameraIntrinsics()
        show_error(cam.get_camera_intrinsics(intr))

        # 首帧用于确定分辨率
        frame = Frame2DAnd3D()
        st = cam.capture_2d_and_3d(frame)
        show_error(st)
        if not st.is_ok():
            return 3

        color0 = frame.frame_2d().get_color_image().data()
        depth0_any = frame.frame_3d().get_depth_map().data()
        depth0 = _depth_to_u16_mm(depth0_any)

        h, w = int(depth0.shape[0]), int(depth0.shape[1])
        intr_path = _save_intrinsics_json(meta_dir, intr, width=w, height=h)
        print(f"✓ 已保存内参: {intr_path}")
        print(f"深度分辨率: {w}x{h}（单位=mm, uint16 PNG）")
        if color0 is not None:
            print(f"彩色分辨率: {int(color0.shape[1])}x{int(color0.shape[0])}")

        # 保存首帧
        idx0 = int(args.start_idx)
        cv2.imwrite(str(color_dir / f"{idx0:06d}.png"), color0)
        cv2.imwrite(str(depth_dir / f"{idx0:06d}.png"), depth0)
        if args.save_raw_tiff:
            cv2.imwrite(str(depth_dir / f"{idx0:06d}_raw.tiff"), np.asarray(depth0_any))

        if args.preview:
            cv2.imshow("color", color0)
            d8 = cv2.convertScaleAbs(np.clip(depth0, 0, 2000), alpha=255.0 / 2000.0)
            cv2.imshow("depth(mm<=2000)", cv2.applyColorMap(d8, cv2.COLORMAP_JET))
            cv2.waitKey(1)

        # 连续录制剩余帧
        captured = 1
        t0 = time.time()
        for k in range(1, int(args.num_frames)):
            before = time.time()
            if args.preview:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("用户退出，提前结束录制。")
                    break

            frame_k = Frame2DAnd3D()
            st = cam.capture_2d_and_3d(frame_k)
            show_error(st)
            if not st.is_ok():
                print("✗ 采集失败，停止。")
                break

            color = frame_k.frame_2d().get_color_image().data()
            depth_any = frame_k.frame_3d().get_depth_map().data()
            depth = _depth_to_u16_mm(depth_any)

            idx = idx0 + k
            cv2.imwrite(str(color_dir / f"{idx:06d}.png"), color)
            cv2.imwrite(str(depth_dir / f"{idx:06d}.png"), depth)
            if args.save_raw_tiff:
                cv2.imwrite(str(depth_dir / f"{idx:06d}_raw.tiff"), np.asarray(depth_any))

            captured += 1
            if args.preview:
                cv2.imshow("color", color)
                d8 = cv2.convertScaleAbs(np.clip(depth, 0, 2000), alpha=255.0 / 2000.0)
                cv2.imshow("depth(mm<=2000)", cv2.applyColorMap(d8, cv2.COLORMAP_JET))
                cv2.waitKey(1)

            # 控制帧间隔
            used = (time.time() - before) * 1000.0
            remain = float(args.interval_ms) - used
            if remain > 0:
                time.sleep(remain / 1000.0)

        dt = time.time() - t0
        fps = captured / dt if dt > 0 else 0.0
        print(f"✓ 录制完成：{captured} 帧，耗时 {dt:.2f}s，平均 {fps:.2f} FPS")
        print(f"输出目录：{out_root}")
        return 0
    finally:
        try:
            cam.disconnect()
        except Exception:
            pass
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())


