#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _parse_csv_floats(text: str, n: int) -> np.ndarray:
    arr = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(arr) != n:
        raise ValueError(f"期望 {n} 个数值，实际 {len(arr)}: {text}")
    return np.asarray(arr, dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return x / n


def _pca_basis(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 10:
        return np.eye(3, dtype=np.float64)
    c = np.mean(pts, axis=0, keepdims=True)
    x = pts - c
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    v = v[:, order]
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1.0
    return v


def _make_T_from_xyz_m(xyz_m: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return T


class ICPRegistrationCore:
    def __init__(
        self,
        args,
        o3d,
        reg_target_pcd,
        *,
        reg_target_name: str = "grasp_region_cad(from_cad_ply)",
        use_local_region_template: bool = True,
        T_source_to_target_manual_init: Optional[np.ndarray] = None,
        last_T_source_to_target: Optional[np.ndarray] = None,
        consecutive_icp_failures: int = 0,
    ):
        self.args = args
        self.o3d = o3d
        self.reg_target_pcd = reg_target_pcd
        self.reg_target_name = reg_target_name
        self.use_local_region_template = bool(use_local_region_template)
        self.T_source_to_target_manual_init = (
            None if T_source_to_target_manual_init is None else np.asarray(T_source_to_target_manual_init, dtype=np.float64)
        )
        self._last_T_source_to_target = (
            None if last_T_source_to_target is None else np.asarray(last_T_source_to_target, dtype=np.float64)
        )
        self._consecutive_icp_failures = int(consecutive_icp_failures)
        self._last_icp_pose_source = "unknown"
        self._last_icp_quality: dict[str, Any] = {}

    @property
    def last_T_source_to_target(self) -> Optional[np.ndarray]:
        return None if self._last_T_source_to_target is None else np.asarray(self._last_T_source_to_target, dtype=np.float64)

    @property
    def consecutive_icp_failures(self) -> int:
        return int(self._consecutive_icp_failures)

    @property
    def last_icp_pose_source(self) -> str:
        return str(self._last_icp_pose_source)

    @property
    def last_icp_quality(self) -> dict[str, Any]:
        return dict(self._last_icp_quality)

    def set_track_state(self, last_T_source_to_target: Optional[np.ndarray], consecutive_icp_failures: int) -> None:
        self._last_T_source_to_target = (
            None if last_T_source_to_target is None else np.asarray(last_T_source_to_target, dtype=np.float64)
        )
        self._consecutive_icp_failures = int(consecutive_icp_failures)

    def _clone_pcd(self, pcd):
        if hasattr(pcd, "clone"):
            return pcd.clone()
        import copy

        return copy.deepcopy(pcd)

    def _pcd_stats_str(self, pcd, name: str) -> str:
        n = int(len(pcd.points))
        if n <= 0:
            return f"{name}: points=0"
        pts = np.asarray(pcd.points, dtype=np.float64)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        ext = mx - mn
        return (
            f"{name}: points={n} "
            f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
            f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}) "
            f"extent=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f})"
        )

    def _pcd_extent(self, pcd) -> np.ndarray:
        if len(pcd.points) <= 0:
            return np.zeros(3, dtype=np.float64)
        pts = np.asarray(pcd.points, dtype=np.float64)
        return pts.max(axis=0) - pts.min(axis=0)

    def _check_extent_consistency(self, source_pcd, target_pcd) -> None:
        es = self._pcd_extent(source_pcd)
        et = self._pcd_extent(target_pcd)
        if np.any(es <= 1e-9) or np.any(et <= 1e-9):
            raise RuntimeError(f"SKIP: bbox extent 异常: source={es.tolist()} target={et.tolist()}")
        ratio = es / et
        rmin = float(np.min(ratio))
        rmax = float(np.max(ratio))
        print(f"[bbox] source_extent={es.tolist()} target_extent={et.tolist()}")
        print(f"[bbox] source/target extent ratio={ratio.tolist()} min={rmin:.3f} max={rmax:.3f}")
        if rmin < float(self.args.bbox_ratio_min) or rmax > float(self.args.bbox_ratio_max):
            raise RuntimeError(
                f"SKIP: bbox 尺寸不一致，ratio_min={rmin:.3f}, ratio_max={rmax:.3f}, "
                f"阈值=[{float(self.args.bbox_ratio_min):.3f},{float(self.args.bbox_ratio_max):.3f}]"
            )

    def _make_center_align_init(self, source_pcd, target_pcd) -> np.ndarray:
        pts_s = np.asarray(source_pcd.points, dtype=np.float64)
        pts_t = np.asarray(target_pcd.points, dtype=np.float64)
        cs = pts_s.mean(axis=0)
        ct = pts_t.mean(axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = ct - cs
        return T

    def _make_pca_align_init(self, source_pcd, target_pcd) -> np.ndarray:
        pts_s = np.asarray(source_pcd.points, dtype=np.float64)
        pts_t = np.asarray(target_pcd.points, dtype=np.float64)
        Rs = _pca_basis(pts_s)
        Rt = _pca_basis(pts_t)
        R = Rt @ Rs.T
        cs = pts_s.mean(axis=0)
        ct = pts_t.mean(axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = ct - (R @ cs)
        return T

    def _expand_rotation_flip_candidates(self, T_base: np.ndarray) -> list[dict[str, Any]]:
        def _rot_axis_180(axis: np.ndarray) -> np.ndarray:
            a = _unit(np.asarray(axis, dtype=np.float64))
            return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(a, a)

        R0 = np.asarray(T_base[:3, :3], dtype=np.float64)
        t0 = np.asarray(T_base[:3, 3], dtype=np.float64)
        thickness_axis = _unit(R0[:, 2])
        main_axis = _unit(R0[:, 0])
        cands = [{"name": "pca", "R_delta": np.eye(3, dtype=np.float64)}]
        cands.append({"name": "pca_flip_thickness", "R_delta": _rot_axis_180(thickness_axis)})
        cands.append({"name": "pca_flip_main", "R_delta": _rot_axis_180(main_axis)})
        cands.append({"name": "pca_flip_combo", "R_delta": _rot_axis_180(main_axis) @ _rot_axis_180(thickness_axis)})
        out = []
        for c in cands:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(c["R_delta"], dtype=np.float64) @ R0
            T[:3, 3] = t0
            out.append({"name": c["name"], "T": T})
        return out

    def _build_online_init_candidates(self, source_pcd, target_pcd) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        center_T = self._make_center_align_init(source_pcd, target_pcd)
        candidates.append({"name": "online_center_align", "T": center_T})
        pca_T = self._make_pca_align_init(source_pcd, target_pcd)
        for c in self._expand_rotation_flip_candidates(pca_T):
            candidates.append({"name": f"online_{c['name']}", "T": c["T"]})
        max_n = max(1, int(self.args.init_candidate_max))
        return candidates[:max_n]

    def _score_init_candidates(
        self,
        source_pcd,
        target_pcd,
        init_candidates: list[dict[str, Any]],
        *,
        eval_dist: float,
    ) -> list[dict[str, Any]]:
        reg = self.o3d.pipelines.registration
        scored: list[dict[str, Any]] = []
        for i, cand in enumerate(init_candidates):
            T = np.asarray(cand["T"], dtype=np.float64)
            ev = reg.evaluate_registration(
                source_pcd,
                target_pcd,
                max_correspondence_distance=float(eval_dist),
                transformation=T,
            )
            item = {
                "idx": i,
                "name": str(cand["name"]),
                "T": T,
                "fitness": float(ev.fitness),
                "rmse": float(ev.inlier_rmse),
                "dist": float(eval_dist),
            }
            scored.append(item)
            print(
                f"[icp][cand] #{i} src={item['name']} "
                f"fitness={item['fitness']:.4f} rmse={item['rmse']:.6f} dist={item['dist']:.5f}"
            )
        return scored

    def _select_best_init(self, scored_candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not scored_candidates:
            return None
        best = max(scored_candidates, key=lambda x: x["fitness"])
        print(
            f"[icp][init] best_candidate=#{int(best['idx'])} src={best['name']} "
            f"fitness={float(best['fitness']):.4f} rmse={float(best['rmse']):.6f}"
        )
        return best

    def _build_init_candidates(self, source_pcd, target_pcd) -> list[dict[str, Any]]:
        cands: list[dict[str, Any]] = []
        if (
            self._last_T_source_to_target is not None
            and self._consecutive_icp_failures < int(self.args.icp_fallback_after)
        ):
            cands.append(
                {
                    "name": "last_success",
                    "T": np.asarray(self._last_T_source_to_target, dtype=np.float64).copy(),
                }
            )
            print(
                f"[icp][init] priority=last_success "
                f"(failures={self._consecutive_icp_failures}/{int(self.args.icp_fallback_after)})"
            )
        if bool(self.args.online_init_enable):
            cands.extend(self._build_online_init_candidates(source_pcd, target_pcd))
        if self.T_source_to_target_manual_init is not None:
            cands.append(
                {
                    "name": "manual_fixed",
                    "T": np.asarray(self.T_source_to_target_manual_init, dtype=np.float64),
                }
            )
        return cands

    def _thin_source_main_surface(self, pcd, *, axis_mode_override: Optional[str] = None):
        if not bool(self.args.surface_thin_enable):
            return pcd
        if len(pcd.points) <= 0:
            raise RuntimeError("SKIP: 主曲面提纯输入为空")
        pts = np.asarray(pcd.points, dtype=np.float64)
        center = pts.mean(axis=0)
        axis_mode = str(axis_mode_override or self.args.surface_thin_axis).strip().lower()
        if axis_mode == "auto":
            basis = _pca_basis(pts)
            axis = basis[:, 2]
        elif axis_mode == "x":
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif axis_mode == "y":
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif axis_mode == "z":
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError(f"未知 surface_thin_axis: {axis_mode}")
        axis = _unit(axis)
        proj = (pts - center) @ axis
        mid = float(np.median(proj))
        band_m = max(1e-4, float(self.args.surface_thin_band_mm) * 0.001)
        half = band_m * 0.5
        keep_mask = np.abs(proj - mid) <= half
        keep_idx = np.flatnonzero(keep_mask)
        kept = int(keep_idx.size)
        total = int(pts.shape[0])
        keep_ratio = kept / max(1, total)
        print(
            f"[surface-thin] axis={axis_mode} axis_vec={axis.tolist()} "
            f"band_mm={float(self.args.surface_thin_band_mm):.2f} keep={kept}/{total} ({keep_ratio:.3f})"
        )
        if kept < int(self.args.surface_thin_min_points):
            raise RuntimeError(
                f"SKIP: 主曲面提纯后点数不足: {kept} < {int(self.args.surface_thin_min_points)}"
            )
        return pcd.select_by_index(keep_idx.tolist())

    def _transform_rot_deg(self, R: np.ndarray) -> float:
        r = np.asarray(R, dtype=np.float64).reshape(3, 3)
        cos_theta = (np.trace(r) - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_theta)))

    def _compute_transform_delta(self, T_delta: np.ndarray) -> tuple[float, float]:
        t = np.asarray(T_delta[:3, 3], dtype=np.float64)
        trans_mm = float(np.linalg.norm(t) * 1000.0)
        rot_deg = self._transform_rot_deg(np.asarray(T_delta[:3, :3], dtype=np.float64))
        return trans_mm, rot_deg

    def _should_fallback_to_coarse(
        self,
        coarse_fitness: float,
        refine_fitness: float,
        residual_trans_mm: float,
        residual_rot_deg: float,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        cfit = max(1e-9, float(coarse_fitness))
        ratio = float(refine_fitness) / cfit
        if ratio < float(self.args.refine_min_fitness_ratio):
            reasons.append(f"fitness_ratio={ratio:.3f}<{float(self.args.refine_min_fitness_ratio):.3f}")
        if residual_trans_mm > float(self.args.refine_max_residual_trans_mm):
            reasons.append(
                f"residual_trans_mm={residual_trans_mm:.2f}>{float(self.args.refine_max_residual_trans_mm):.2f}"
            )
        if residual_rot_deg > float(self.args.refine_max_residual_rot_deg):
            reasons.append(
                f"residual_rot_deg={residual_rot_deg:.2f}>{float(self.args.refine_max_residual_rot_deg):.2f}"
            )
        return (len(reasons) > 0), reasons

    def _select_final_icp_transform(
        self,
        T_coarse: np.ndarray,
        T_refine: np.ndarray,
        *,
        coarse_fitness: float,
        coarse_rmse: float,
        refine_fitness: float,
        refine_rmse: float,
        residual_T: np.ndarray,
    ) -> tuple[np.ndarray, str, list[str]]:
        residual_trans_mm, residual_rot_deg = self._compute_transform_delta(residual_T)
        print(
            f"[icp][quality] coarse_fit={float(coarse_fitness):.4f} coarse_rmse={float(coarse_rmse):.6f} "
            f"refine_fit={float(refine_fitness):.4f} refine_rmse={float(refine_rmse):.6f} "
            f"residual_trans_mm={residual_trans_mm:.2f} residual_rot_deg={residual_rot_deg:.2f}"
        )
        self._last_icp_quality = {
            "coarse_fitness": float(coarse_fitness),
            "coarse_rmse": float(coarse_rmse),
            "refine_fitness": float(refine_fitness),
            "refine_rmse": float(refine_rmse),
            "residual_trans_mm": float(residual_trans_mm),
            "residual_rot_deg": float(residual_rot_deg),
        }
        if not bool(self.args.refine_fallback_enable):
            return np.asarray(T_refine, dtype=np.float64), "refine", []
        do_fb, reasons = self._should_fallback_to_coarse(
            float(coarse_fitness),
            float(refine_fitness),
            float(residual_trans_mm),
            float(residual_rot_deg),
        )
        if do_fb:
            print(f"[icp][fallback] use=coarse reason={' | '.join(reasons)}")
            return np.asarray(T_coarse, dtype=np.float64), "coarse", reasons
        return np.asarray(T_refine, dtype=np.float64), "refine", []

    def estimate_pose_and_grasp(
        self,
        fused_source_pcd,
        T_base_cam: np.ndarray,
        T_region_to_grasp: np.ndarray,
        *,
        fused_frames: int,
    ) -> dict[str, Any]:
        raw_source = fused_source_pcd.voxel_down_sample(float(self.args.icp_voxel))
        if len(raw_source.points) < int(self.args.min_points):
            raise RuntimeError(f"SKIP: 点云点数不足: {len(raw_source.points)}")
        target = self.reg_target_pcd
        print(
            "[icp][pair] "
            f"source=fused_live_grasp_region, target={self.reg_target_name}, "
            f"use_local_template={self.use_local_region_template}, fused_frames={int(fused_frames)}"
        )
        print("[icp][pair] " + self._pcd_stats_str(raw_source, "source_raw_fused"))
        print("[icp][pair] " + self._pcd_stats_str(target, "target"))
        self._check_extent_consistency(raw_source, target)
        reg = self.o3d.pipelines.registration
        coarse_dist = float(self.args.icp_voxel) * float(self.args.coarse_icp_dist_mult)
        refine_dist = float(self.args.icp_voxel) * float(self.args.refine_icp_dist_mult)
        eval_dist = max(coarse_dist * 1.2, coarse_dist)
        init_candidates = self._build_init_candidates(raw_source, target)
        if not init_candidates:
            raise RuntimeError("SKIP: 缺少可用初始位姿（last_success/online/manual 均不可用）")
        scored = self._score_init_candidates(raw_source, target, init_candidates, eval_dist=eval_dist)
        non_manual = [x for x in scored if str(x["name"]) != "manual_fixed"]
        thr_stage0 = float(self.args.icp_coarse_fitness_thr)
        best_non_manual = self._select_best_init(non_manual) if non_manual else None
        if best_non_manual is not None and float(best_non_manual["fitness"]) >= thr_stage0:
            best_init = best_non_manual
        else:
            best_init = self._select_best_init(scored)
            if best_init is not None and str(best_init["name"]) == "manual_fixed":
                print("[fallback] online/track 初值不可用，回退 manual_fixed")
        if best_init is None:
            raise RuntimeError("SKIP: all init candidates failed: no valid candidate")
        if float(best_init["fitness"]) < thr_stage0:
            raise RuntimeError(
                f"SKIP: all init candidates failed: best_fitness={float(best_init['fitness']):.4f} "
                f"< thr={thr_stage0:.4f}"
            )
        T_init = np.asarray(best_init["T"], dtype=np.float64)
        print("[icp][init] T_source_to_target=" + np.array2string(T_init, precision=5, suppress_small=True))
        coarse = reg.registration_icp(
            raw_source,
            target,
            max_correspondence_distance=coarse_dist,
            init=T_init,
            estimation_method=reg.TransformationEstimationPointToPoint(),
            criteria=reg.ICPConvergenceCriteria(max_iteration=int(self.args.max_icp_stage1)),
        )
        print(
            f"[icp][coarse] fitness={float(coarse.fitness):.4f} rmse={float(coarse.inlier_rmse):.6f} "
            f"dist={coarse_dist:.5f}"
        )
        print(
            "[icp][coarse] T_source_to_target="
            + np.array2string(np.asarray(coarse.transformation, dtype=np.float64), precision=5, suppress_small=True)
        )
        if float(coarse.fitness) < float(self.args.icp_stage1_fitness_thr):
            raise RuntimeError(
                f"SKIP: ICP coarse fitness 过低: {float(coarse.fitness):.4f} < {float(self.args.icp_stage1_fitness_thr):.4f}"
            )
        if float(coarse.inlier_rmse) > float(self.args.icp_stage1_rmse_thr):
            raise RuntimeError(
                f"SKIP: ICP coarse rmse 过高: {float(coarse.inlier_rmse):.6f} > {float(self.args.icp_stage1_rmse_thr):.6f}"
            )
        source_coarse_aligned = self._clone_pcd(raw_source)
        source_coarse_aligned.transform(np.asarray(coarse.transformation, dtype=np.float64))
        axis_mode = str(self.args.surface_thin_axis).strip().lower()
        source_refine_aligned = self._thin_source_main_surface(source_coarse_aligned, axis_mode_override=axis_mode)
        if len(source_refine_aligned.points) < int(self.args.min_points):
            raise RuntimeError(
                f"SKIP: 主曲面提纯后点数不足: {len(source_refine_aligned.points)} < {int(self.args.min_points)}"
            )
        print("[icp][pair] " + self._pcd_stats_str(source_refine_aligned, "source_coarse_aligned_surface_thinned"))
        self._check_extent_consistency(source_refine_aligned, target)
        source_refine_aligned.estimate_normals(
            search_param=self.o3d.geometry.KDTreeSearchParamHybrid(radius=float(self.args.icp_voxel) * 2.5, max_nn=40)
        )
        source_refine_aligned.normalize_normals()
        refine = reg.registration_icp(
            source_refine_aligned,
            target,
            max_correspondence_distance=refine_dist,
            init=np.eye(4, dtype=np.float64),
            estimation_method=reg.TransformationEstimationPointToPlane(),
            criteria=reg.ICPConvergenceCriteria(max_iteration=int(self.args.max_icp_stage2)),
        )
        print(
            f"[icp][refine] fitness={float(refine.fitness):.4f} rmse={float(refine.inlier_rmse):.6f} "
            f"dist={refine_dist:.5f}"
        )
        print(
            "[icp][refine] T_residual="
            + np.array2string(np.asarray(refine.transformation, dtype=np.float64), precision=5, suppress_small=True)
        )
        T_source_to_region_refine = np.asarray(refine.transformation, dtype=np.float64) @ np.asarray(coarse.transformation, dtype=np.float64)
        print(
            "[icp][refine] T_source_to_target_final="
            + np.array2string(np.asarray(T_source_to_region_refine, dtype=np.float64), precision=5, suppress_small=True)
        )
        T_final, final_src, fb_reasons = self._select_final_icp_transform(
            np.asarray(coarse.transformation, dtype=np.float64),
            np.asarray(T_source_to_region_refine, dtype=np.float64),
            coarse_fitness=float(coarse.fitness),
            coarse_rmse=float(coarse.inlier_rmse),
            refine_fitness=float(refine.fitness),
            refine_rmse=float(refine.inlier_rmse),
            residual_T=np.asarray(refine.transformation, dtype=np.float64),
        )
        if final_src == "refine":
            if float(refine.fitness) < float(self.args.icp_fine_fitness_thr):
                if bool(self.args.keep_coarse_track_on_refine_fail):
                    self._last_T_source_to_target = np.asarray(coarse.transformation, dtype=np.float64).copy()
                    print("[fallback] refine 失败，使用 coarse 结果作为下一帧 warm start")
                raise RuntimeError(
                    f"SKIP: ICP refine fitness 过低: {float(refine.fitness):.4f} < {float(self.args.icp_fine_fitness_thr):.4f}"
                )
            if float(refine.inlier_rmse) > float(self.args.icp_rmse_thr):
                if bool(self.args.keep_coarse_track_on_refine_fail):
                    self._last_T_source_to_target = np.asarray(coarse.transformation, dtype=np.float64).copy()
                    print("[fallback] refine rmse 失败，使用 coarse 结果作为下一帧 warm start")
                raise RuntimeError(
                    f"SKIP: ICP refine rmse 过高: {float(refine.inlier_rmse):.6f} > {float(self.args.icp_rmse_thr):.6f}"
                )
        elif final_src == "coarse":
            if (float(coarse.fitness) < float(self.args.icp_stage1_fitness_thr)) or (
                float(coarse.inlier_rmse) > float(self.args.icp_stage1_rmse_thr)
            ):
                raise RuntimeError("SKIP: refine 回退到 coarse 后质量仍不满足阈值")
            print(f"[icp][fallback] final_pose_source=coarse reasons={' | '.join(fb_reasons) if fb_reasons else 'none'}")
        T_source_to_region = np.asarray(T_final, dtype=np.float64)
        self._last_icp_pose_source = str(final_src)
        print(f"[icp][quality] final_pose_source={self._last_icp_pose_source}")
        self._last_T_source_to_target = T_source_to_region.copy()
        self._consecutive_icp_failures = 0
        T_region_cam = np.linalg.inv(T_source_to_region)
        T_region_to_grasp = np.asarray(T_region_to_grasp, dtype=np.float64)
        T_grasp_cam = T_region_cam @ T_region_to_grasp
        T_grasp_base = np.asarray(T_base_cam, dtype=np.float64) @ T_grasp_cam
        pregrasp_xyz = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        T_base_pregrasp = T_grasp_base @ _make_T_from_xyz_m(pregrasp_xyz)
        T_base_grasp = T_grasp_base @ _make_T_from_xyz_m(grasp_xyz)
        print("[grasp-ref] T_region_to_camera=" + np.array2string(T_region_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_region_to_grasp=" + np.array2string(T_region_to_grasp, precision=5, suppress_small=True))
        print("[grasp-ref] T_grasp_to_camera=" + np.array2string(T_grasp_cam, precision=5, suppress_small=True))
        print("[grasp-ref] T_grasp_to_base=" + np.array2string(T_grasp_base, precision=5, suppress_small=True))
        print("[grasp-ref] T_base_pregrasp=" + np.array2string(T_base_pregrasp, precision=5, suppress_small=True))
        print("[grasp-ref] T_base_grasp=" + np.array2string(T_base_grasp, precision=5, suppress_small=True))
        return {
            "T_base_pregrasp": T_base_pregrasp,
            "T_base_grasp": T_base_grasp,
            "T_region_cam": T_region_cam,
            "T_grasp_cam": T_grasp_cam,
            "T_grasp_base": T_grasp_base,
            "T_source_to_region": T_source_to_region,
            "icp_final_pose_source": self._last_icp_pose_source,
            "icp_quality": dict(self._last_icp_quality),
        }
