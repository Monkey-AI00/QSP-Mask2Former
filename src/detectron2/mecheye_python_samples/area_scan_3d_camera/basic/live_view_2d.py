"""
实时 2D 预览（Mech-Eye NANO / Area Scan 3D Camera）

按官方建议：
- discover_cameras() 扫描相机
- connect() 连接（支持按 IP/序列号/索引）
- 循环 capture_2d() 获取 Frame2D，并用 OpenCV cv2.imshow 实时显示

退出：
- 按 'q' 退出

无 GUI 环境：
- OpenCV imshow 需要图形界面支持；若在纯命令行服务器上，请配置 X11 转发或在本机运行。
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:  # pragma: no cover
    from mecheye.area_scan_3d_camera import Camera, Frame2D  # type: ignore
    from mecheye.shared import show_error  # type: ignore
else:
    from mecheye.shared import *  # type: ignore # noqa: F401,F403
    from mecheye.area_scan_3d_camera import *  # type: ignore # noqa: F401,F403


try:  # pragma: no cover
    from mecheye.area_scan_3d_camera_utils import print_camera_info as _print_camera_info  # type: ignore
except Exception:  # pragma: no cover
    _print_camera_info = None


def _print_camera_info_fallback(ci: Any) -> None:
    ip = getattr(ci, "ip_address", None)
    sn = getattr(ci, "serial_number", None)
    model = getattr(ci, "model", None)
    fw = getattr(ci, "firmware_version", None)
    print(f"  ip_address={ip} serial_number={sn} model={model} firmware={fw}")


def _save_frame(img2d: Any, save_dir: str, save_idx: int) -> str:
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = os.path.join(save_dir, f"save{int(save_idx)}_{ts}.png")
    cv2.imwrite(out, img2d)
    return out


def discover_and_print_cameras() -> list:
    print("Discovering all available cameras...")
    camera_infos = Camera.discover_cameras()
    if len(camera_infos) == 0:
        print("No cameras found.")
        return []
    for i in range(len(camera_infos)):
        print(f"Camera index: {i}")
        ci = camera_infos[i]
        if _print_camera_info is not None:
            _print_camera_info(ci)
        else:
            _print_camera_info_fallback(ci)
    return camera_infos


def connect_camera(camera: "Camera", *, ip: str = "", serial: str = "", index: int = -1) -> bool:
    if ip:
        print(f"Connecting by IP: {ip} ...")
        st = camera.connect(ip)
        if not st.is_ok():
            show_error(st)
            return False
        print("Connected to the camera successfully.")
        return True

    camera_infos = discover_and_print_cameras()
    if not camera_infos:
        return False

    if serial:
        serial = str(serial).strip()
        for i, ci in enumerate(camera_infos):
            if str(getattr(ci, "serial_number", "")).strip() == serial:
                print(f"Connecting by serial_number={serial} (index={i}) ...")
                st = camera.connect(ci)
                if not st.is_ok():
                    show_error(st)
                    return False
                print("Connected to the camera successfully.")
                return True
        print(f"未在 discover_cameras() 列表中找到 serial_number={serial}")
        return False

    if index < 0:
        index = 0
    if not (0 <= int(index) < len(camera_infos)):
        print(f"--index 越界：{index}，有效范围 [0, {len(camera_infos)-1}]")
        return False

    print(f"Connecting by index: {index} ...")
    st = camera.connect(camera_infos[int(index)])
    if not st.is_ok():
        show_error(st)
        return False
    print("Connected to the camera successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Mech-Eye：实时显示 2D 图（cv2.imshow）")
    parser.add_argument("--discover", action="store_true", help="仅扫描并打印相机列表，然后退出")
    parser.add_argument("--ip", default="", help="按相机 IP 直连，例如 169.254.5.157")
    parser.add_argument("--serial", default="", help="按相机序列号连接（会先 discover 再匹配）")
    parser.add_argument("--index", type=int, default=-1, help="按 discover_cameras 的索引连接（默认 0）")
    parser.add_argument("--win", default="Mech-Eye 2D Live", help="窗口标题")
    parser.add_argument("--wait", type=int, default=1, help="cv2.waitKey 等待毫秒数（越小越流畅）")
    parser.add_argument("--save", default="", help="可选：保存目录。按 s 保存当前帧；不再自动逐帧保存。")
    args = parser.parse_args()

    camera = Camera()
    if args.discover:
        discover_and_print_cameras()
        return

    if not connect_camera(camera, ip=str(args.ip).strip(), serial=str(args.serial).strip(), index=int(args.index)):
        return

    try:
        save_dir = str(args.save).strip()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        frame_2d = Frame2D()
        print("开始实时预览：按 'q' 退出，按 's' 保存当前帧。")
        idx = 0
        save_count = 0
        while True:
            idx += 1
            st = camera.capture_2d(frame_2d)
            if not st.is_ok():
                show_error(st)
                continue

            # 单色/彩色相机兼容
            if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
                img2d = frame_2d.get_gray_scale_image().data()
            else:
                img2d = frame_2d.get_color_image().data()  # BGR

            cv2.imshow(str(args.win), img2d)

            k = cv2.waitKey(int(args.wait)) & 0xFF
            if k == ord("s"):
                target_dir = save_dir or os.path.abspath("./captures")
                save_count += 1
                out = _save_frame(img2d, target_dir, save_count)
                print(f"已保存第{save_count}张: {out}")
                continue
            if k == ord("q"):
                break
    finally:
        camera.disconnect()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    main()


