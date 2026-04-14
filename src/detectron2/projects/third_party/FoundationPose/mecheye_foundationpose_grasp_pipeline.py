#!/usr/bin/env python3
"""
FoundationPose 单次位姿估计 — 文件桥接模式
============================================
在 **foundationpose** conda 环境中由子进程调用，与主管线完全依赖隔离。

桥接 I/O 协议
--------------
inputs/color.png      uint8 BGR 图像 (cv2 原生顺序)
inputs/depth.png      uint16 深度 (毫米)
inputs/mask.png       uint8 mask (255=目标, 0=背景)
inputs/meta.json      内参 K、mesh 路径、参数

outputs/pose_result.json
    status            "ok" | "error"
    T_obj_to_camera   4×4 行优先 list (物体坐标系→相机坐标系)
    score             FoundationPose 最佳得分
    elapsed_ms        总耗时 (含初始化)
    timings           各阶段耗时明细
    error             错误文本 (仅 status="error")

调用方式
--------
conda run -n foundationpose --no-capture-output python \\
    .../third_party/FoundationPose/mecheye_foundationpose_grasp_pipeline.py \\
    --bridge-dir .../third_party/runtime_bridge

退出码：0 = 成功，1 = 失败（pose_result.json 仍会写入，含 error 字段）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 实际 FoundationPose 代码：../../FoundationPose  (相对于 third_party/FoundationPose/)
_FP_CODE_DIR_DEFAULT = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "../../FoundationPose")
)


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "outofmemoryerror" in msg or "cuda out of memory" in msg


# ------------------------------------------------------------------ #
#  主入口                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="FoundationPose 单次位姿估计（文件桥接模式）"
    )
    parser.add_argument(
        "--bridge-dir", required=True,
        help="runtime_bridge 目录 (含 inputs/ 和 outputs/)",
    )
    parser.add_argument(
        "--fp-code-dir", default="",
        help="FoundationPose 代码根目录（自动推断为 ../../FoundationPose）",
    )
    args = parser.parse_args()

    bridge_dir = os.path.abspath(args.bridge_dir)
    inputs_dir = os.path.join(bridge_dir, "inputs")
    outputs_dir = os.path.join(bridge_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    result: dict = {
        "status": "error",
        "T_obj_to_camera": None,
        "score": -1.0,
        "request_id": "",
        "elapsed_ms": 0.0,
        "timings": {},
        "error": None,
        "debug": {},
    }

    t_all = time.time()

    try:
        # -------- 1. 读取输入 --------
        t0 = time.time()

        meta_path = os.path.join(inputs_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        result["request_id"] = meta.get("request_id", "")

        color_bgr = cv2.imread(os.path.join(inputs_dir, "color.png"))
        if color_bgr is None:
            raise FileNotFoundError("color.png 读取失败")
        rgb = color_bgr[:, :, ::-1].copy()

        depth_u16 = cv2.imread(
            os.path.join(inputs_dir, "depth.png"), cv2.IMREAD_UNCHANGED
        )
        if depth_u16 is None:
            raise FileNotFoundError("depth.png 读取失败")
        depth_scale = float(meta.get("depth_scale_to_m", 0.001))
        depth_m = depth_u16.astype(np.float32) * depth_scale

        mask = cv2.imread(
            os.path.join(inputs_dir, "mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise FileNotFoundError("mask.png 读取失败")

        K = np.array(meta["K"], dtype=np.float64)
        mesh_file = str(meta["mesh_file"])

        mask_valid = int((mask > 0).sum())
        depth_valid = int(((depth_m >= 0.001) & (mask > 0)).sum())
        if mask_valid < 4:
            raise RuntimeError(f"mask 有效像素过少 ({mask_valid})")
        if depth_valid < 4:
            raise RuntimeError(f"mask 内有效深度点过少 ({depth_valid})")

        t_read = time.time()

        # -------- 2. 导入 & 初始化 FoundationPose --------
        fp_code_dir = args.fp_code_dir or _FP_CODE_DIR_DEFAULT
        fp_code_dir = os.path.abspath(fp_code_dir)
        if fp_code_dir not in sys.path:
            sys.path.insert(0, fp_code_dir)

        # 减少显存碎片导致的分配失败
        # 兼容老版本 PyTorch：仅使用稳定可识别的 allocator 选项。
        # `expandable_segments` 在部分版本会报
        # "Unrecognized CachingAllocator option"。
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:64",
        )

        import trimesh  # noqa: E402
        import torch  # noqa: E402
        from estimater import FoundationPose  # noqa: E402
        from learning.training.predict_score import ScorePredictor  # noqa: E402
        from learning.training.predict_pose_refine import (  # noqa: E402
            PoseRefinePredictor,
        )
        import nvdiffrast.torch as dr  # noqa: E402

        mesh = trimesh.load(mesh_file, force="mesh")
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        est = FoundationPose(
            model_pts=mesh.vertices.copy(),
            model_normals=mesh.vertex_normals.copy(),
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=int(meta.get("fp_debug", 0)),
            debug_dir=str(meta.get("fp_debug_dir", "/tmp/fp_debug")),
        )

        t_init = time.time()

        # -------- 3. 位姿估计 --------
        est_iter = int(meta.get("est_refine_iter", 5))
        base_min_n_views = int(meta.get("fp_min_n_views", 20))
        base_inplane_step = int(meta.get("fp_inplane_step", 120))
        fallback_schedule = [
            (base_min_n_views, base_inplane_step),
            (16, 120),
            (12, 180),
            (8, 180),
            (6, 180),
        ]
        retry_schedule = []
        for mv, st in fallback_schedule:
            key = (int(mv), int(st))
            if key not in retry_schedule:
                retry_schedule.append(key)

        last_oom_exc: Exception | None = None
        used_grid = None
        T_obj_cam = None
        for idx, (min_views, inplane_step) in enumerate(retry_schedule, start=1):
            try:
                print(
                    f"[fp] register attempt {idx}/{len(retry_schedule)} "
                    f"min_n_views={min_views} inplane_step={inplane_step}"
                )
                est.make_rotation_grid(
                    min_n_views=int(min_views),
                    inplane_step=int(inplane_step),
                )
                T_obj_cam = est.register(
                    K=K,
                    rgb=rgb,
                    depth=depth_m,
                    ob_mask=mask,
                    iteration=est_iter,
                )
                used_grid = {
                    "min_n_views": int(min_views),
                    "inplane_step": int(inplane_step),
                    "attempt_index": idx,
                }
                break
            except Exception as exc:
                if _is_oom_error(exc):
                    last_oom_exc = exc
                    print(f"[fp][oom] attempt {idx} 失败: {exc}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                raise

        if T_obj_cam is None:
            if last_oom_exc is not None:
                raise RuntimeError(
                    f"FoundationPose OOM after retries: {last_oom_exc}"
                )
            raise RuntimeError("FoundationPose register 失败（未知原因）")

        T_obj_cam = np.asarray(T_obj_cam, dtype=np.float64).reshape(4, 4)

        best_score = -1.0
        if (
            hasattr(est, "scores")
            and est.scores is not None
            and len(est.scores) > 0
        ):
            best_score = float(est.scores[0])

        t_est = time.time()

        # -------- 4. 组装结果 --------
        result["status"] = "ok"
        result["T_obj_to_camera"] = T_obj_cam.tolist()
        result["score"] = best_score
        result["timings"] = {
            "read_inputs_ms": (t_read - t0) * 1000,
            "init_model_ms": (t_init - t_read) * 1000,
            "estimate_ms": (t_est - t_init) * 1000,
        }
        result["debug"] = {
            "mask_valid_px": mask_valid,
            "depth_valid_in_mask_px": depth_valid,
            "mesh_file": mesh_file,
            "mesh_vertices": len(mesh.vertices),
            "mesh_faces": len(mesh.faces),
            "est_refine_iter": est_iter,
            "retry_schedule": retry_schedule,
            "used_grid": used_grid,
        }

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["debug"]["traceback"] = traceback.format_exc()

    result["elapsed_ms"] = (time.time() - t_all) * 1000.0

    # -------- 写入输出 --------
    out_path = os.path.join(outputs_dir, "pose_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary = {
        "status": result["status"],
        "score": result["score"],
        "elapsed_ms": round(result["elapsed_ms"], 1),
    }
    if result["status"] != "ok":
        summary["error"] = result["error"]
    if "timings" in result and result["timings"]:
        summary["timings"] = {
            k: round(v, 1) for k, v in result["timings"].items()
        }
    print(json.dumps(summary, ensure_ascii=False))

    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
