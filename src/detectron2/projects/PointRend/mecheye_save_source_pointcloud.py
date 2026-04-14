#!/usr/bin/env python3
"""
仅用于导出一帧 source 点云：
同帧采集 -> 实例分割 -> 点云构建 -> 轻量预处理 -> 保存并退出
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 当前脚本位于 detectron2/projects/PointRend
_D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _D2_ROOT not in sys.path:
    sys.path.insert(0, _D2_ROOT)

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils  # noqa: E402


def _import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:
        raise RuntimeError(f"未找到 open3d，请先安装: pip install open3d; 原始错误: {e}")


def _load_predictor(args):
    model_family = str(args.model_family).strip().lower()
    score_thr = float(args.score_thr)
    prior_path_override = str(args.shape_prior_npy).strip()
    if prior_path_override:
        prior_path_override = live_utils._require_existing_file(prior_path_override, "prior npy")

    if model_family == "mask2former":
        config_prior = live_utils._require_existing_file(
            str(args.config_file_prior).strip() or live_utils._DEFAULT_MASK2FORMER_QSP_CONFIG,
            "Mask2Former prior config",
        )
        weights_prior = live_utils._require_existing_file(str(args.weights_prior).strip(), "prior weights")
        predictor = live_utils.build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root),
            config_file=config_prior,
            weights=weights_prior,
            score_thresh=score_thr,
            device=str(args.device),
            num_classes=int(args.num_classes),
            prior_path_override=prior_path_override,
        )
    else:
        config = live_utils._require_existing_file(
            str(args.config_file).strip() or live_utils._DEFAULT_POINTREND_CONFIG,
            "PointRend config",
        )
        weights_prior = live_utils._require_existing_file(str(args.weights_prior).strip(), "prior weights")
        if prior_path_override:
            os.environ["SHAPE_PRIOR_PATH"] = str(prior_path_override)
        predictor = live_utils.build_pointrend_predictor(
            config_file=config,
            weights=weights_prior,
            mask_head_name="ShapeAwareCoarseMaskHead",
            score_thresh=score_thr,
            device=str(args.device),
            num_classes=int(args.num_classes),
        )
    return predictor


def _build_source_pcd(o3d, depth_obj, color_bgr: np.ndarray, mask_pc: np.ndarray, intrinsics, args):
    depth_np = depth_obj.data()
    depth_m = live_utils._depth_u16_to_m(
        live_utils._depth_to_png_u16(depth_np),
        unit=str(args.depth_unit),
    )
    fx, fy, cx, cy = live_utils._get_depth_k_from_mecheye_intrinsics(intrinsics)
    xyz_m, rgb = live_utils._backproject_masked_xyzrgb(
        depth_m,
        color_bgr,
        mask_pc,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        stride=max(1, int(args.pc_stride)),
    )
    pcd = o3d.geometry.PointCloud()
    if xyz_m.shape[0] == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(xyz_m.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float64) / 255.0))
    return pcd


def _preprocess_source_pcd(pcd, args):
    if len(pcd.points) == 0:
        return pcd
    p = pcd
    if float(args.pp_voxel) > 0:
        p = p.voxel_down_sample(float(args.pp_voxel))
    if len(p.points) == 0:
        return p
    if int(args.pp_sor_nb) > 0:
        _, ind = p.remove_statistical_outlier(
            nb_neighbors=int(args.pp_sor_nb),
            std_ratio=float(args.pp_sor_std),
        )
        p = p.select_by_index(ind)
    if int(args.pp_ror_nb) > 0 and float(args.pp_ror_radius) > 0:
        _, ind = p.remove_radius_outlier(
            nb_points=int(args.pp_ror_nb),
            radius=float(args.pp_ror_radius),
        )
        p = p.select_by_index(ind)
    return p


def build_argparser():
    p = argparse.ArgumentParser(description="仅导出一帧 source 点云（分割->点云->预处理->保存）")
    p.add_argument("--model-family", choices=["pointrend", "mask2former"], default="mask2former")
    p.add_argument("--config-file", default=live_utils._DEFAULT_POINTREND_CONFIG)
    p.add_argument("--config-file-prior", default=live_utils._DEFAULT_MASK2FORMER_QSP_CONFIG)
    p.add_argument("--mask2former-root", default=live_utils._DEFAULT_MASK2FORMER_ROOT)
    p.add_argument("--weights-prior", required=True, help="prior/QSP 权重")
    p.add_argument("--shape-prior-npy", default="")
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-classes", type=int, default=1)

    p.add_argument("--mask-mode", choices=["union", "maxscore"], default="union")
    p.add_argument("--pc-mask-mode", choices=["union", "maxscore", "iou"], default="maxscore")
    p.add_argument("--pc-iou-thresh", type=float, default=0.1)
    p.add_argument("--pc-join-dilate", type=int, default=0)
    p.add_argument("--mask-close", type=int, default=35)
    p.add_argument("--mask-dilate", type=int, default=30)
    p.add_argument("--mask-erode", type=int, default=1)
    p.add_argument("--invert-mask", action="store_true")
    p.add_argument("--auto-invert-mask", action="store_true")

    p.add_argument("--ip", default="", help="Mech-Eye 相机 IP")
    p.add_argument("--serial", default="", help="Mech-Eye 序列号")
    p.add_argument("--index", type=int, default=-1, help="discover index")
    p.add_argument("--exposure-seq", default="5,10,15")
    p.add_argument("--pc-smoothing", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-noise", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-outlier", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-edge", choices=["", "sharp", "normal", "smooth"], default="normal")
    p.add_argument("--save-userset", action="store_true")
    p.add_argument("--depth-unit", choices=["mm", "m"], default="mm")
    p.add_argument("--pc-stride", type=int, default=1)

    p.add_argument("--pp-voxel", type=float, default=0.0)
    p.add_argument("--pp-sor-nb", type=int, default=50)
    p.add_argument("--pp-sor-std", type=float, default=1.0)
    p.add_argument("--pp-ror-nb", type=int, default=0)
    p.add_argument("--pp-ror-radius", type=float, default=0.0)

    p.add_argument("--output", default="/home/user/sjw/workspace/tmp/live_grasp_region_sample.ply", help="source 点云输出路径")
    p.add_argument("--min-points", type=int, default=300, help="保存最小点数")
    p.add_argument("--max-loops", type=int, default=200, help="最多尝试采集帧数")
    p.add_argument("--key-save", action="store_true", help="按键保存模式：按 save-key 保存，按 quit-key 退出")
    p.add_argument("--save-key", default="s", help="保存按键（默认 s）")
    p.add_argument("--quit-key", default="q", help="退出按键（默认 q）")
    p.add_argument("--wait", type=int, default=1, help="GUI waitKey 延时（ms）")
    return p


def main():
    args = build_argparser().parse_args()
    o3d = _import_open3d()
    predictor = _load_predictor(args)

    camera = live_utils.Camera()
    if not live_utils.connect_camera(
        camera,
        ip=str(args.ip).strip(),
        serial=str(args.serial).strip(),
        index=int(args.index),
    ):
        raise RuntimeError("连接相机失败")

    exp_seq = live_utils._parse_float_list(args.exposure_seq)
    live_utils._apply_mecheye_params(
        camera,
        exposure_sequence=exp_seq if exp_seq else None,
        pc_surface_smoothing=str(args.pc_smoothing),
        pc_noise_removal=str(args.pc_noise),
        pc_outlier_removal=str(args.pc_outlier),
        pc_edge_preservation=str(args.pc_edge),
        save_to_device=bool(args.save_userset),
    )
    intrinsics = live_utils.CameraIntrinsics()
    live_utils.show_error(camera.get_camera_intrinsics(intrinsics))

    frame_all = live_utils.Frame2DAnd3D()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_key = ord(str(args.save_key).strip().lower()[:1] or "s")
    quit_key = ord(str(args.quit_key).strip().lower()[:1] or "q")
    if bool(args.key_save):
        print(f"[key-save] 已启用：按 {chr(save_key)} 保存，按 {chr(quit_key)} 退出")

    saved = False
    try:
        for i in range(1, int(args.max_loops) + 1):
            st = camera.capture_2d_and_3d(frame_all)
            if not st.is_ok():
                live_utils.show_error(st)
                continue
            color_sdk = frame_all.frame_2d().get_color_image()
            depth = frame_all.frame_3d().get_depth_map()
            color_bgr = color_sdk.data()

            out_pred = predictor(color_bgr)
            inst = out_pred["instances"].to("cpu")
            _, mask_pc, invert_applied = live_utils._build_output_masks(
                inst,
                mask_mode=str(args.mask_mode),
                pc_mask_mode=str(args.pc_mask_mode),
                pc_iou_thresh=float(args.pc_iou_thresh),
                pc_join_dilate=int(args.pc_join_dilate),
                mask_close=int(args.mask_close),
                mask_dilate=int(args.mask_dilate),
                mask_erode=int(args.mask_erode),
                invert_mask=bool(args.invert_mask),
                auto_invert_mask=bool(args.auto_invert_mask),
            )
            print(f"[frame {i}] " + live_utils._mask_stats(mask_pc) + f" invert={invert_applied}")

            pcd = _build_source_pcd(o3d, depth, color_bgr, mask_pc, intrinsics, args)
            pcd = _preprocess_source_pcd(pcd, args)
            n = int(len(pcd.points))
            print(f"[frame {i}] source_points={n}")

            if bool(args.key_save):
                vis = live_utils._overlay_binary_mask(color_bgr, mask_pc)
                tip = (
                    f"frame={i} points={n} min={int(args.min_points)} | "
                    f"press '{chr(save_key)}' to save, '{chr(quit_key)}' to quit"
                )
                cv2.putText(vis, tip, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Save Source PointCloud", vis)
                key = cv2.waitKey(int(args.wait)) & 0xFF
                if key == quit_key:
                    print("[key-save] 用户退出，未保存")
                    break
                if key != save_key:
                    continue
                if n < int(args.min_points):
                    print(f"[key-save] 点数不足，拒绝保存: {n} < {int(args.min_points)}")
                    continue
            else:
                if n < int(args.min_points):
                    continue

            ok = bool(o3d.io.write_point_cloud(str(out), pcd, write_ascii=False, compressed=False, print_progress=False))
            if not ok:
                raise RuntimeError(f"写入点云失败: {out}")
            print(f"[saved] source 点云已保存: {out}")
            saved = True
            break
    finally:
        try:
            camera.disconnect()
        except Exception:
            pass
        if bool(args.key_save):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if not saved:
        raise RuntimeError(f"在 max_loops={int(args.max_loops)} 内未采到满足 min_points={int(args.min_points)} 的 source 点云")


if __name__ == "__main__":
    main()

