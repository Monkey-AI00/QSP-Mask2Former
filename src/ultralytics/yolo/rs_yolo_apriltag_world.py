#!/usr/bin/env python3
"""
实时整合：RealSense + AprilTag + YOLO -> 输出目标在 Tag(世界)坐标系下的 (x,y,z)

坐标系约定：
- 相机坐标系（RealSense/OpenCV常用）：x 右，y 下，z 前（朝向镜头外）
- Tag 坐标系：由 OpenCV/aruco 的 marker 坐标定义（在 marker 平面上，z 垂直于平面）
- 世界坐标系：直接把 Tag 坐标系当世界坐标系（World Origin = Tag）

输出：
每个 YOLO 检测框中心 (u,v) -> depth z -> 相机 3D 点 X_cam -> 变换到 X_tag(=X_world)
"""

import argparse
from typing import Optional, Union

import numpy as np

# 允许从任意工作目录运行该脚本：把脚本所在目录加入 sys.path，方便本地导入同目录模块
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from apriltag_utils import detect_apriltag_poses, draw_tag_poses
from realsense_utils import (
    deproject_uv_depth_to_cam_xyz,
    depth_at_uv,
    frames_to_aligned_color_depth,
    get_color_intrinsics,
    start_pipeline,
)
from yolo_utils import infer_detections, load_yolo


def main():
    parser = argparse.ArgumentParser(description="RealSense + AprilTag + YOLO：输出 Tag(世界)坐标")
    parser.add_argument("--weights", required=True, help="YOLO 权重路径")
    parser.add_argument("--device", default=0, help="YOLO 推理设备（0/1/... 或 'cpu'）")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="（保留参数）")

    parser.add_argument("--width", type=int, default=1280, help="RealSense 分辨率宽")
    parser.add_argument("--height", type=int, default=720, help="RealSense 分辨率高")
    parser.add_argument("--fps", type=int, default=30, help="RealSense FPS")

    parser.add_argument("--tag_size", type=float, required=True, help="AprilTag 边长（米），例如 0.05")
    parser.add_argument("--tag_dict", default="apriltag_36h11", help="AprilTag 字典：apriltag_36h11/36h10/25h9/16h5")
    parser.add_argument("--tag_id", type=int, default=-1, help="只使用指定 tag id；-1 表示用检测到的第一个")

    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    parser.add_argument("--max_det", type=int, default=5, help="每帧最多输出多少个 YOLO 检测")
    parser.add_argument("--print_every", type=int, default=30, help="每隔多少帧在终端输出一次状态/xyz（默认约1秒）")
    args = parser.parse_args()

    model = load_yolo(args.weights)

    pipeline, profile, align = start_pipeline(width=args.width, height=args.height, fps=args.fps, align_to_color=True)
    intr = get_color_intrinsics(profile)
    K = intr.camera_matrix()
    dist = intr.dist_coeffs

    # 延迟导入 cv2，避免无显示环境报错
    cv2 = None
    if args.show:
        import cv2 as _cv2

        cv2 = _cv2

    try:
        frame_idx = 0
        while True:
            frame_idx += 1
            frames = pipeline.wait_for_frames()
            color_bgr, depth_frame = frames_to_aligned_color_depth(frames, align)
            if color_bgr is None or depth_frame is None:
                continue

            # 1) AprilTag 位姿：得到 tag->cam 的 R,t
            tag_poses = detect_apriltag_poses(
                image_bgr=color_bgr,
                camera_matrix=K,
                dist_coeffs=dist,
                tag_size_m=float(args.tag_size),
                dict_name=str(args.tag_dict),
            )
            if len(tag_poses) == 0:
                if frame_idx % int(args.print_every) == 0:
                    print("[status] 未检测到 AprilTag", flush=True)
                if args.show and cv2 is not None:
                    cv2.imshow("rs_yolo_apriltag_world", color_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # 选用的 tag
            tag_pose = tag_poses[0]
            if int(args.tag_id) >= 0:
                for p in tag_poses:
                    if p.tag_id == int(args.tag_id):
                        tag_pose = p
                        break

            # 2) YOLO：取检测框中心点
            dets = infer_detections(model, color_bgr, conf=float(args.conf), device=args.device)[: int(args.max_det)]

            # 3) 深度 + 反投影 + 坐标变换到 tag/world
            lines = []
            valid_cnt = 0
            for i, d in enumerate(dets):
                u, v = d.uv
                z = depth_at_uv(depth_frame, u, v)
                if z <= 0:
                    continue
                X_cam = deproject_uv_depth_to_cam_xyz(intr, u, v, z)
                X_tag = tag_pose.cam_to_tag(X_cam)
                valid_cnt += 1

                lines.append(
                    f"[{i}] cls={d.cls} conf={d.conf:.3f} uv=({u:.1f},{v:.1f})  "
                    f"cam_xyz=({X_cam[0]:.3f},{X_cam[1]:.3f},{X_cam[2]:.3f})m  "
                    f"tag_xyz=({X_tag[0]:.3f},{X_tag[1]:.3f},{X_tag[2]:.3f})m  tag_id={tag_pose.tag_id}"
                )

            # 终端输出：每隔 N 帧输出一次，避免刷屏；并且对“没输出”的原因给状态提示
            if frame_idx % int(args.print_every) == 0:
            if lines:
                    print("\n".join(lines), flush=True)
                else:
                    if len(dets) == 0:
                        print(f"[status] tag_id={tag_pose.tag_id} 已检测到，但 YOLO 未检出目标（dets=0）", flush=True)
                    else:
                        print(
                            f"[status] tag_id={tag_pose.tag_id} dets={len(dets)}，但中心点深度无效（valid_depth=0）"
                            "（常见原因：深度空洞/超出测距范围/中心点落在无深度区域）",
                            flush=True,
                        )

            # 4) 可视化
            if args.show and cv2 is not None:
                vis = draw_tag_poses(color_bgr, tag_poses, K, dist, axis_len_m=float(args.tag_size) * 0.5)
                # 叠加状态/xyz，确保不看终端也能看到结果
                status = f"tag_id={tag_pose.tag_id} dets={len(dets)} valid_depth={valid_cnt}"
                cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                if lines:
                    # 只叠加第一个有效目标的 tag_xyz（避免遮挡太多）
                    first = lines[0]
                    # 从字符串里截取 tag_xyz=... 部分
                    idx = first.find("tag_xyz=")
                    show_line = first[idx:] if idx >= 0 else first
                    cv2.putText(vis, show_line, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                for d in dets:
                    x1, y1, x2, y2 = d.xyxy.astype(int).tolist()
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    u, v = d.uv
                    cv2.circle(vis, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
                cv2.imshow("rs_yolo_apriltag_world", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        if args.show:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()


