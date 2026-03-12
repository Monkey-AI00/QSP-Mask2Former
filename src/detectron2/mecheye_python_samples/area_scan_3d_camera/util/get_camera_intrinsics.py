"""
获取并打印 Mech-Eye（如 NANO / Area Scan 3D Camera）相机内参。

用法示例：
  - 按 IP 直连（推荐）：
      python3 get_camera_intrinsics.py --ip 169.254.5.157
  - 扫描并打印相机列表：
      python3 get_camera_intrinsics.py --discover
  - 按序列号连接：
      python3 get_camera_intrinsics.py --serial XXXXX
  - 按 discover_cameras 的索引连接：
      python3 get_camera_intrinsics.py --index 0
"""

from __future__ import annotations

import argparse
from typing import Any

from mecheye.shared import show_error  # type: ignore
from mecheye.area_scan_3d_camera import Camera, CameraIntrinsics  # type: ignore

try:  # pragma: no cover
    # 该工具模块通常随 Mech-Eye SDK 一起提供；若环境中没有它，我们用 fallback 逻辑。
    from mecheye.area_scan_3d_camera_utils import print_camera_intrinsics as _print_camera_intrinsics  # type: ignore
    from mecheye.area_scan_3d_camera_utils import print_camera_info as _print_camera_info  # type: ignore
except Exception:  # pragma: no cover
    _print_camera_intrinsics = None
    _print_camera_info = None


def _print_camera_info_fallback(ci: Any) -> None:
    ip = getattr(ci, "ip_address", None)
    sn = getattr(ci, "serial_number", None)
    model = getattr(ci, "model", None)
    fw = getattr(ci, "firmware_version", None)
    print(f"  ip_address={ip} serial_number={sn} model={model} firmware={fw}")


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


def connect_camera(camera: Camera, *, ip: str = "", serial: str = "", index: int = -1) -> bool:
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


def _print_intrinsics_fallback(intr: Any) -> None:
    # SDK 版本差异可能导致字段/方法名不同，这里尽量“有啥打印啥”，避免崩溃。
    print("CameraIntrinsics (raw):", intr)
    for name in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"):
        try:
            attr = getattr(intr, name, None)
            if callable(attr):
                print(f"{name}:", attr())
            elif attr is not None:
                print(f"{name}:", attr)
        except Exception:
            pass

    for name in ("camera_matrix", "dist_coeffs", "distortion_coefficients"):
        try:
            attr = getattr(intr, name, None)
            if callable(attr):
                print(f"{name}:", attr())
            elif attr is not None:
                print(f"{name}:", attr)
        except Exception:
            pass


def _extract_fx_fy_cx_cy(intr: Any) -> tuple[float, float, float, float]:
    """
    尽量从 SDK 的 CameraIntrinsics 中提取 fx, fy, cx, cy。
    兼容不同 SDK 版本：优先直接取 fx/fy/cx/cy，其次从（Texture/Depth）相机矩阵 K 里读。
    """
    def _read_from_camera_matrix(cm: Any) -> tuple[float, float, float, float] | None:
        """
        兼容你这版 SDK：CameraMatrix 提供 fx/fy/cx/cy 方法（不可下标）。
        """
        if cm is None:
            return None
        try:
            fx = getattr(cm, "fx", None)
            fy = getattr(cm, "fy", None)
            cx = getattr(cm, "cx", None)
            cy = getattr(cm, "cy", None)
            if callable(fx) and callable(fy) and callable(cx) and callable(cy):
                return float(fx()), float(fy()), float(cx()), float(cy())
            # 某些版本可能是属性
            if fx is not None and fy is not None and cx is not None and cy is not None:
                return float(fx), float(fy), float(cx), float(cy)
        except Exception:
            return None
        return None

    # 1) 你当前 SDK 的标准结构：intr.texture.camera_matrix / intr.depth.camera_matrix
    for path in (("texture", "camera_matrix"), ("depth", "camera_matrix")):
        try:
            obj = intr
            for name in path:
                obj = getattr(obj, name, None)
            got = _read_from_camera_matrix(obj)
            if got is not None:
                return got
        except Exception:
            pass

    # 2) 兼容其它 SDK：直接 fx/fy/cx/cy 在 intr 上
    direct: dict[str, float] = {}
    for k in ("fx", "fy", "cx", "cy"):
        try:
            v = getattr(intr, k, None)
            if callable(v):
                v = v()
            if v is not None:
                direct[k] = float(v)
        except Exception:
            pass
    if len(direct) == 4:
        return direct["fx"], direct["fy"], direct["cx"], direct["cy"]

    raise RuntimeError("无法从 CameraIntrinsics 提取 fx/fy/cx/cy（请确认 SDK 版本与 intrinsics 结构）。")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mech-Eye：获取并打印相机内参")
    parser.add_argument("--discover", action="store_true", help="仅扫描并打印相机列表，然后退出")
    parser.add_argument("--ip", default="", help="按相机 IP 直连，例如 169.254.5.157")
    parser.add_argument("--serial", default="", help="按相机序列号连接（会先 discover 再匹配）")
    parser.add_argument("--index", type=int, default=-1, help="按 discover_cameras 的索引连接（默认 0）")
    args = parser.parse_args()

    camera = Camera()
    if args.discover:
        discover_and_print_cameras()
        return 0

    if not connect_camera(camera, ip=str(args.ip).strip(), serial=str(args.serial).strip(), index=int(args.index)):
        return 2

    try:
        intrinsics = CameraIntrinsics()
        show_error(camera.get_camera_intrinsics(intrinsics))
        fx, fy, cx, cy = _extract_fx_fy_cx_cy(intrinsics)
        print(f"fx: {fx}")
        print(f"fy: {fy}")
        print(f"cx: {cx}")
        print(f"cy: {cy}")
        return 0
    finally:
        camera.disconnect()
        print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    raise SystemExit(main())
