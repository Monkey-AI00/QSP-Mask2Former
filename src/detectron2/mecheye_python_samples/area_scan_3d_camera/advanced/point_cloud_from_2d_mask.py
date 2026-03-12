"""
从 2D Mask 生成 3D 点云（Mech-Eye SDK: get_point_cloud_after_mapping）

用途：
- 输入：PointRend/Mask R-CNN 导出的 2D 实例掩码（png/jpg/npy，单通道）
- 相机：Mech-Eye 工业 3D 相机（通过 MechEyeAPI 采集 2D + 深度）
- 输出：掩码区域对应的点云（Untextured / Textured），保存为 .ply

核心逻辑（与官方 mapping_2d_image_to_depth_map.py 一致）：
1) capture_2d_and_3d -> color/depth
2) get_camera_intrinsics
3) 将外部 mask 转为 SDK 需要的 GrayScale2DImage
4) get_point_cloud_after_mapping(depth, mask, intrinsics, point_cloud)

注意（非常重要）：
- 官方示例里 mask 的“灰度=255”用于屏蔽区域，灰度=0 表示保留区域。
  因此：若你的分割 mask 是“前景=255(白)”，通常需要把它映射为“保留=0(黑)”，背景映射为 255。
- 本脚本通过 --input_foreground 与 --invert 来适配你的 mask 语义。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

import numpy as np

# 说明：
# - 运行时需要 MechEyeAPI（会提供 mecheye.* 模块）
# - 但很多编辑器/静态检查环境没有装该 SDK，会报“无法解析导入/未定义”
#   这里用 TYPE_CHECKING 分支让静态检查通过，不影响运行时行为。
if TYPE_CHECKING:  # pragma: no cover
    from mecheye.area_scan_3d_camera import (  # type: ignore
        Camera,
        CameraIntrinsics,
        FileFormat_PLY,
        Frame2DAnd3D,
        Frame3D,
        PointCloudEdgePreservation,
        PointCloudNoiseRemoval,
        PointCloudOutlierRemoval,
        PointCloudSurfaceSmoothing,
        Scanning3DExposureSequence,
        TexturedPointCloud,
        UntexturedPointCloud,
        get_point_cloud_after_mapping,
    )
    from mecheye.shared import GrayScale2DImage, show_error  # type: ignore
else:
    from mecheye.shared import *  # type: ignore # noqa: F401,F403 - Mech-Eye SDK 风格
    from mecheye.area_scan_3d_camera import *  # type: ignore # noqa: F401,F403

# 兼容：有些环境里官方 utils 模块可用（包含 print_camera_info/confirm_capture_3d 等）
try:  # pragma: no cover
    from mecheye.area_scan_3d_camera_utils import confirm_capture_3d as _confirm_capture_3d  # type: ignore
    from mecheye.area_scan_3d_camera_utils import print_camera_info as _print_camera_info  # type: ignore
except Exception:  # pragma: no cover
    _confirm_capture_3d = None
    _print_camera_info = None


def _confirm_capture_3d_fallback() -> bool:
    """
    无官方 confirm_capture_3d 时的降级实现。
    """
    try:
        ans = input("即将触发 3D 采集（激光/投影），请确认现场安全。继续？[y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except Exception:
        return False


def _print_camera_info_fallback(ci: Any) -> None:
    """
    无官方 print_camera_info 时，尽量打印常用字段（ip/serial/model）。
    """
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


def connect_camera(camera: "Camera", *, ip: str = "", serial: str = "", index: int = -1) -> bool:
    """
    连接方式：
    - 优先 --ip：camera.connect(ip)
    - 其次 --serial/--index：discover_cameras 后按条件选择，camera.connect(camera_info)
    """
    if ip:
        print(f"Connecting by IP: {ip} ...")
        st = camera.connect(ip)
        if st.is_ok():
            print("Connected to the camera successfully.")
            return True
        # 某些环境下 SDK 无法判断“哪块网卡连着相机”，导致按 IP 直连失败。
        # 回退到 discover_cameras() 并用 camera_info.connect() 尝试一次。
        show_error(st)
        print("按 IP 直连失败，尝试通过 discover_cameras() 回退连接（按相机信息匹配 IP）...")
        camera_infos = discover_and_print_cameras()
        for ci in camera_infos:
            if str(getattr(ci, "ip_address", "")).strip() == str(ip).strip():
                st2 = camera.connect(ci)
                if st2.is_ok():
                    print("Connected to the camera successfully (fallback via camera_info).")
                    return True
                show_error(st2)
                break
        return False

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


def _load_grayscale_mask(mask_path: str) -> np.ndarray:
    p = Path(mask_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到 mask 文件: {mask_path}")

    # 1) .npy：直接读二维数组
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"mask .npy 必须是 2D 或 HxWx1，实际 shape={arr.shape}")
        return arr.astype(np.uint8, copy=False)

    # 2) 图片：优先 cv2，其次 PIL
    try:
        import cv2  # type: ignore

        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"无法读取 mask 图片: {mask_path}")
        return m.astype(np.uint8, copy=False)
    except Exception:
        from PIL import Image  # type: ignore

        im = Image.open(str(p)).convert("L")
        return np.array(im, dtype=np.uint8)


def _resize_mask_nearest(mask: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    w, h = int(size_wh[0]), int(size_wh[1])
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    try:
        import cv2  # type: ignore

        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8, copy=False)
    except Exception:
        from PIL import Image  # type: ignore

        im = Image.fromarray(mask)
        im = im.resize((w, h), resample=Image.NEAREST)
        return np.array(im, dtype=np.uint8)


def _mask_to_sdk_grayscale2d(
    input_mask: np.ndarray,
    width: int,
    height: int,
    *,
    input_foreground: Literal["white", "black"] = "white",
    threshold: int = 127,
    invert: bool = False,
    sdk_keep_value: int = 0,
    sdk_exclude_value: int = 255,
) -> "GrayScale2DImage":
    """
    将外部二值/灰度 mask 转为 Mech-Eye SDK 需要的 GrayScale2DImage。

    - input_foreground="white"：mask > threshold 视为前景/保留
    - input_foreground="black"：mask <= threshold 视为前景/保留
    - invert=True：交换保留/剔除

    SDK 语义（来自官方示例习惯）：
    - sdk_keep_value=0：保留区域（生成点云）
    - sdk_exclude_value=255：剔除区域（不生成点云）
    """
    if input_mask.ndim != 2:
        raise ValueError(f"input_mask 必须是 2D，实际 shape={input_mask.shape}")

    input_mask = _resize_mask_nearest(input_mask, (width, height))

    if input_foreground == "white":
        keep = input_mask > int(threshold)
    elif input_foreground == "black":
        keep = input_mask <= int(threshold)
    else:
        raise ValueError(f"不支持的 input_foreground={input_foreground}")

    if invert:
        keep = ~keep

    sdk_mask = GrayScale2DImage()
    sdk_mask.resize(int(width), int(height))

    # 初始化：默认全剔除（255），再把保留区域写成 0
    # （官方示例的写法等价：只对某些像素写 255，其他默认 0）
    for i in range(int(height)):
        for j in range(int(width)):
            sdk_mask.at(i, j).gray = int(sdk_exclude_value)
    for i in range(int(height)):
        row = keep[i]
        for j in range(int(width)):
            if bool(row[j]):
                sdk_mask.at(i, j).gray = int(sdk_keep_value)
    return sdk_mask


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return []
    return [float(p) for p in parts]


def _apply_mecheye_params(
    camera: "Camera",
    *,
    exposure_sequence: list[float] | None = None,
    pc_surface_smoothing: str = "",
    pc_noise_removal: str = "",
    pc_outlier_removal: str = "",
    pc_edge_preservation: str = "",
    save_to_device: bool = False,
) -> None:
    current_user_set = camera.current_user_set()
    if exposure_sequence:
        show_error(current_user_set.set_float_array_value(Scanning3DExposureSequence.name, exposure_sequence))

    def _set_level(param_cls, level: str):
        if not level:
            return
        key = f"Value_{level.capitalize()}"
        v = getattr(param_cls, key, None)
        if v is None:
            raise ValueError(f"不支持 {param_cls.name} 的 level={level}，期望 off/weak/normal/strong")
        show_error(current_user_set.set_enum_value(param_cls.name, v))

    def _set_edge(level: str):
        if not level:
            return
        key = f"Value_{level.capitalize()}"
        v = getattr(PointCloudEdgePreservation, key, None)
        if v is None:
            raise ValueError("不支持 PointCloudEdgePreservation 的 level，期望 sharp/normal/smooth")
        show_error(current_user_set.set_enum_value(PointCloudEdgePreservation.name, v))

    _set_level(PointCloudSurfaceSmoothing, pc_surface_smoothing)
    _set_level(PointCloudNoiseRemoval, pc_noise_removal)
    _set_level(PointCloudOutlierRemoval, pc_outlier_removal)
    _set_edge(pc_edge_preservation)

    if save_to_device:
        show_error(current_user_set.save_all_parameters_to_device())


def main():
    parser = argparse.ArgumentParser(description="Mech-Eye：输入 2D Mask 输出 3D 点云（PLY）")
    parser.add_argument("--mask", required=True, help="2D mask 路径（png/jpg/npy），单通道，来自 PointRend/MaskRCNN")
    parser.add_argument("--output", default="", help="输出 ply 路径；不填则自动命名到当前目录")
    parser.add_argument("--textured", action="store_true", help="输出带纹理点云（需要同时传 color 到 API）")

    # 连接方式（按 IP/序列号/索引）
    parser.add_argument("--discover", action="store_true", help="仅扫描并打印相机列表，然后退出")
    parser.add_argument("--ip", default="", help="按相机 IP 直连（GigE/TCP-IP），例如 192.168.1.110")
    parser.add_argument("--serial", default="", help="按相机序列号连接（会先 discover_cameras 再匹配）")
    parser.add_argument("--index", type=int, default=-1, help="按 discover_cameras 的索引连接（默认 0）")
    parser.add_argument("--no_confirm", action="store_true", help="跳过 3D 采集安全确认（无人值守时使用）")

    parser.add_argument("--input_foreground", choices=["white", "black"], default="white", help="输入 mask 中，前景/保留区域的颜色语义")
    parser.add_argument("--threshold", type=int, default=127, help="二值化阈值（仅用于灰度 mask）")
    parser.add_argument("--invert", action="store_true", help="反转保留/剔除（语义不确定时用）")

    # 采集/点云质量相关（HDR + Post-processing）
    parser.add_argument("--exposure-seq", default="", help="HDR 多曝光序列（3D），例如 '5,10' 或 '3,6,12'；为空则不修改")
    parser.add_argument("--pc-smoothing", choices=["", "off", "weak", "normal", "strong"], default="", help="点云表面平滑")
    parser.add_argument("--pc-noise", choices=["", "off", "weak", "normal", "strong"], default="", help="点云噪声去除")
    parser.add_argument("--pc-outlier", choices=["", "off", "weak", "normal", "strong"], default="", help="点云离群点去除")
    parser.add_argument("--pc-edge", choices=["", "sharp", "normal", "smooth"], default="", help="边缘保持（Sharp/Normal/Smooth）")
    parser.add_argument("--save-userset", action="store_true", help="把上述参数保存到相机当前 user set（下次也生效）")

    args = parser.parse_args()

    camera = Camera()
    if args.discover:
        discover_and_print_cameras()
        return

    if not connect_camera(camera, ip=str(args.ip).strip(), serial=str(args.serial).strip(), index=int(args.index)):
        return
    try:
        if not args.no_confirm:
            if _confirm_capture_3d is not None:
                if not _confirm_capture_3d():
                    return
            else:
                if not _confirm_capture_3d_fallback():
                    return

        # 采集前设置 HDR + 点云后处理（从源头减少空洞/碎片）
        exp_seq = _parse_float_list(args.exposure_seq)
        _apply_mecheye_params(
            camera,
            exposure_sequence=exp_seq if exp_seq else None,
            pc_surface_smoothing=str(args.pc_smoothing),
            pc_noise_removal=str(args.pc_noise),
            pc_outlier_removal=str(args.pc_outlier),
            pc_edge_preservation=str(args.pc_edge),
            save_to_device=bool(args.save_userset),
        )

        # 1) 采集 2D+3D
        frame_all_2d_3d = Frame2DAnd3D()
        show_error(camera.capture_2d_and_3d(frame_all_2d_3d))
        color = frame_all_2d_3d.frame_2d().get_color_image()
        depth = frame_all_2d_3d.frame_3d().get_depth_map()

        # 2) 内参
        intrinsics = CameraIntrinsics()
        show_error(camera.get_camera_intrinsics(intrinsics))

        # 3) 读取并转换 mask -> GrayScale2DImage
        input_mask = _load_grayscale_mask(args.mask)
        sdk_mask = _mask_to_sdk_grayscale2d(
            input_mask,
            width=int(color.width()),
            height=int(color.height()),
            input_foreground=args.input_foreground,
            threshold=int(args.threshold),
            invert=bool(args.invert),
        )

        # 4) 映射生成点云并保存
        out_path = args.output.strip()
        if not out_path:
            out_path = "MaskTexturedPointCloud.ply" if args.textured else "MaskUntexturedPointCloud.ply"

        if args.textured:
            points_xyz_bgr = TexturedPointCloud()
            show_error(get_point_cloud_after_mapping(depth, sdk_mask, color, intrinsics, points_xyz_bgr))
            show_error(Frame2DAnd3D.save_point_cloud(points_xyz_bgr, FileFormat_PLY, out_path), f"保存点云到: {out_path}")
        else:
            points_xyz = UntexturedPointCloud()
            show_error(get_point_cloud_after_mapping(depth, sdk_mask, intrinsics, points_xyz))
            show_error(Frame3D.save_point_cloud(points_xyz, FileFormat_PLY, out_path), f"保存点云到: {out_path}")

        print(f"完成：{out_path}")
        print("如果点云为空/不对：优先尝试加 --invert，或切换 --input_foreground black/white。")
    finally:
        camera.disconnect()
        print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    main()


