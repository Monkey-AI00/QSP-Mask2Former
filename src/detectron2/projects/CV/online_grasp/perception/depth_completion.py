"""Depth completion wrapper scaffold."""

from __future__ import annotations

import cv2
import numpy as np

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils


class DepthCompleter:
    """
    ClearGrasp 封装骨架。
    当前阶段由 legacy pipeline 保持既有 cleargrasp 行为和日志。
    """

    def __init__(self, args, intrinsics):
        self.args = args
        self.intrinsics = intrinsics
        self.depthcomplete = None

    def init_cleargrasp(self):
        if (
            not str(self.args.cleargrasp_normals_weights).strip()
            or not str(self.args.cleargrasp_outlines_weights).strip()
            or not str(self.args.cleargrasp_depth2depth_exe).strip()
        ):
            print("[cleargrasp] 缺少权重/可执行文件，跳过")
            return
        live_utils._add_cleargrasp_to_syspath()
        from api import depth_completion_api  # type: ignore

        fx, fy, cx, cy = live_utils._get_depth_k_from_mecheye_intrinsics(self.intrinsics)
        self.depthcomplete = {
            "api": depth_completion_api,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "out_w": int(self.args.cleargrasp_out_w),
            "out_h": int(self.args.cleargrasp_out_h),
        }
        print("[cleargrasp] 已启用")

    def maybe_complete_depth(self, depth_np, color_bgr):
        if not bool(self.args.cleargrasp) or self.depthcomplete is None:
            return live_utils._depth_u16_to_m(
                live_utils._depth_to_png_u16(depth_np),
                unit=str(self.args.depth_unit),
            )

        if "model" not in self.depthcomplete:
            H, W = depth_np.shape[:2]
            fx0, fy0, cx0, cy0 = (
                float(self.depthcomplete["fx"]),
                float(self.depthcomplete["fy"]),
                float(self.depthcomplete["cx"]),
                float(self.depthcomplete["cy"]),
            )
            out_w, out_h = int(self.depthcomplete["out_w"]), int(self.depthcomplete["out_h"])
            fx2, fy2, cx2, cy2 = live_utils._scale_k_for_resize(fx0, fy0, cx0, cy0, in_w=W, in_h=H, out_w=out_w, out_h=out_h)
            api = self.depthcomplete["api"]
            self.depthcomplete["model"] = api.DepthToDepthCompletion(
                normalsWeightsFile=str(self.args.cleargrasp_normals_weights),
                outlinesWeightsFile=str(self.args.cleargrasp_outlines_weights),
                masksWeightsFile="",
                normalsModel="drn",
                outlinesModel="drn",
                depth2depthExecutable=str(self.args.cleargrasp_depth2depth_exe),
                outputImgHeight=out_h,
                outputImgWidth=out_w,
                fx=fx2,
                fy=fy2,
                cx=cx2,
                cy=cy2,
                filter_d=int(self.args.cleargrasp_filter_d),
                filter_sigmaColor=float(self.args.cleargrasp_filter_sigma_color),
                filter_sigmaSpace=float(self.args.cleargrasp_filter_sigma_space),
                normalsInferenceHeight=out_h,
                normalsInferenceWidth=out_w,
                outlinesInferenceHeight=out_h,
                outlinesInferenceWidth=out_w,
                min_depth=0.0,
                max_depth=3.0,
            )
            print("[cleargrasp] model loaded")

        depth_m = live_utils._depth_u16_to_m(
            live_utils._depth_to_png_u16(depth_np),
            unit=str(self.args.depth_unit),
        )
        rgb = color_bgr[:, :, ::-1]
        out_small, _ = self.depthcomplete["model"].depth_completion(
            rgb,
            depth_m,
            inertia_weight=float(self.args.cleargrasp_inertia),
            smoothness_weight=float(self.args.cleargrasp_smoothness),
            tangent_weight=float(self.args.cleargrasp_tangent),
            mode_modify_input_depth="",
        )
        out_up = cv2.resize(out_small, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
        completed = depth_m.copy()
        thr = float(self.args.cleargrasp_fill_thresh)
        holes = (~np.isfinite(completed)) | (completed <= max(0.0, thr))
        completed[holes] = out_up[holes]
        return completed


__all__ = ["DepthCompleter"]

