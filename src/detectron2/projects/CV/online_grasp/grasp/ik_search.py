"""IK search runner scaffold."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from online_grasp.geometry.transforms import (
    _euler_xyz_deg_to_rot,
    _make_T_from_xyz_m,
    _parse_csv_float_list,
    _parse_csv_floats,
)


def _load_ik_candidates(args) -> list:
    """
    加载候选法兰姿态集合（ICP 稳定定位 + 候选姿态试 IK 模式）。
    返回 list[list[float]]，每个元素为 [rx, ry, rz] (deg, XYZ 欧拉角)。
    来源优先级：--ik-candidate-rpy-json > --ik-candidate-rpy-list
    """
    candidates: list[list[float]] = []
    source_type = str(getattr(args, "ik_candidate_source_type", "euler_rpy")).strip().lower()
    print(f"[ik-cand][type] source_type={source_type}")
    if source_type == "joint_guess":
        raise ValueError(
            "ik_candidate_source_type=joint_guess 不被接受："
            "当前候选接口仅支持 TCP 姿态角 RX/RY/RZ，不支持 J4/J5/J6。"
        )
    if source_type != "euler_rpy":
        raise ValueError(f"未知 ik_candidate_source_type: {source_type}")
    print("[ik-cand][warn] ik-candidate-rpy-* 语义为 TCP orientation (RX/RY/RZ)，不是关节角 J4/J5/J6")
    json_path = str(getattr(args, "ik_candidate_rpy_json", "") or "").strip()
    if json_path:
        data = json.loads(Path(json_path).expanduser().resolve().read_text(encoding="utf-8"))
        raw = data.get("candidates", data.get("ik_candidates", data.get("rpy_list", [])))
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                candidates.append([float(x) for x in item])
        if candidates:
            print(f"[ik-cand] 从 JSON 加载 {len(candidates)} 个候选法兰姿态")
            return candidates
    rpy_list = str(getattr(args, "ik_candidate_rpy_list", "") or "").strip()
    if rpy_list:
        for part in rpy_list.split(";"):
            vals = [float(x) for x in part.strip().split(",") if x.strip()]
            if len(vals) == 3:
                candidates.append(vals)
        if candidates:
            print(f"[ik-cand] 从命令行加载 {len(candidates)} 个候选法兰姿态")
            return candidates
    return candidates


class IKSearchRunner:
    """IK 检索与运行时执行链（从 legacy 实迁）。"""

    def __init__(self, args, executor, ctx=None):
        self.args = args
        self.executor = executor
        self.ctx = ctx
        self._fixed_template_rpy_cache = None

    def _format_pose_mm_deg(self, pose_mm_deg: np.ndarray) -> str:
        p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
        return f"[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f},{p[3]:.2f},{p[4]:.2f},{p[5]:.2f}]"

    def _format_pose_full_precision(self, pose_mm_deg: np.ndarray) -> str:
        p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
        return f"[{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{p[3]:.6f},{p[4]:.6f},{p[5]:.6f}]"

    def _cache_final_pose_and_log(self, pose_mm_deg: np.ndarray, *, context: str) -> np.ndarray:
        pose = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6).copy()
        if self.ctx is not None:
            self.ctx._last_final_pose_mm_deg = pose.copy()
        print(f"[robot][final-6d] context={context}")
        print(f"[robot][final-6d] rounded={self._format_pose_mm_deg(pose)}")
        print(f"[robot][final-6d] full={self._format_pose_full_precision(pose)}")
        return pose

    def _get_fixed_template_rpy(self) -> np.ndarray:
        if self._fixed_template_rpy_cache is not None:
            return np.asarray(self._fixed_template_rpy_cache, dtype=np.float64).reshape(3)
        source = str(getattr(self.args, "fixed_template_source", "cli")).strip().lower()
        if source == "cli":
            rpy = _parse_csv_floats(str(self.args.fixed_template_rpy), 3)
        elif source == "current_robot_pose":
            if self.executor is None:
                raise RuntimeError("runtime模板姿态来源为 current_robot_pose，但当前无可用机器人连接")
            pose_now = self.executor.get_current_pose_mm_deg()
            rpy = np.asarray(pose_now[3:6], dtype=np.float64)
        elif source == "json":
            cfg_path = str(getattr(self.args, "fixed_template_json", "") or "").strip()
            if not cfg_path:
                raise ValueError("fixed_template_source=json 时必须提供 --fixed-template-json")
            data = json.loads(Path(cfg_path).expanduser().resolve().read_text(encoding="utf-8"))
            raw = data.get("fixed_template_rpy", data.get("template_rpy", data.get("rpy", None)))
            if raw is None:
                raise KeyError("fixed-template json 未找到 fixed_template_rpy/template_rpy/rpy")
            rpy = np.asarray(raw, dtype=np.float64).reshape(-1)
            if rpy.size != 3:
                raise ValueError(f"fixed-template json rpy 维度错误: {rpy}")
        else:
            raise ValueError(f"未知 fixed_template_source: {source}")
        self._fixed_template_rpy_cache = np.asarray(rpy, dtype=np.float64).reshape(3)
        print(f"[ik-template] source={source}")
        print(f"[ik-template] rpy={self._fixed_template_rpy_cache.tolist()}")
        return np.asarray(self._fixed_template_rpy_cache, dtype=np.float64).reshape(3)

    def _build_fixed_template_pose(self, grasp_pos_base_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rpy = self._get_fixed_template_rpy()
        grasp_pos_m = np.asarray(grasp_pos_base_mm, dtype=np.float64).reshape(3) / 1000.0
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _euler_xyz_deg_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        T[:3, 3] = grasp_pos_m
        pregrasp_xyz_m = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz_m = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        T_pre = T @ _make_T_from_xyz_m(pregrasp_xyz_m)
        T_grasp = T @ _make_T_from_xyz_m(grasp_xyz_m)
        pose_pre = self.ctx._T_to_robot_pose_mm_deg(T_pre)
        pose_grasp = self.ctx._T_to_robot_pose_mm_deg(T_grasp)
        return pose_pre, pose_grasp

    def _ik_try_pose_with_reason(self, pose_mm_deg: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "joint": None, "err": "", "err_code": None, "raw": ""}
        if self.executor is None:
            out["err"] = "no_executor"
            return out
        try:
            j = self.executor.solve_ik_to_joint_deg(np.asarray(pose_mm_deg, dtype=np.float64))
            out["ok"] = True
            out["joint"] = j
            return out
        except Exception as e:
            msg = str(e)
            out["raw"] = msg
            out["err"] = msg
            m = re.search(r"ErrorID=([-]?\d+)", msg)
            if m:
                out["err_code"] = int(m.group(1))
            return out

    def _probe_pose_reachability_by_motion(self, pose_mm_deg: np.ndarray, stage_name: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ran": False,
            "ok": None,
            "mode": str(self.args.motion_probe_command),
            "err": "",
            "reason": "probe_not_executed",
        }
        if self.executor is None:
            out["reason"] = "no_executor"
            out["err"] = "no_executor"
            return out
        if not bool(self.args.motion_probe_execute):
            print(f"[ik-check][motion] dry-run only, skip actual probe for {stage_name}")
            return out
        try:
            if str(self.args.motion_probe_command) == "movl_pose":
                self.executor.movl_pose(np.asarray(pose_mm_deg, dtype=np.float64), f"probe_{stage_name}_movl")
            else:
                self.executor.movj_pose(np.asarray(pose_mm_deg, dtype=np.float64), f"probe_{stage_name}_movj")
            out["ran"] = True
            out["ok"] = True
            out["reason"] = "probe_success"
            return out
        except Exception as e:
            out["ran"] = True
            out["ok"] = False
            out["reason"] = "probe_failed"
            out["err"] = str(e)
            return out

    def _ik_check_pose(self, pose_mm_deg: np.ndarray, stage_name: str) -> dict[str, Any]:
        mode = str(self.args.ik_check_mode)
        out: dict[str, Any] = {
            "mode": mode,
            "api_ok": False,
            "motion_probe_ok": None,
            "motion_probe_ran": False,
            "final_ok": False,
            "joint": None,
            "error_code": None,
            "raw": "",
            "pose_mm_deg": np.asarray(pose_mm_deg, dtype=np.float64).reshape(6).tolist(),
            "reason": "",
        }
        print(f"[pose][rounded] {self._format_pose_mm_deg(pose_mm_deg)}")
        print(f"[pose][full]    {self._format_pose_full_precision(pose_mm_deg)}")
        if mode in ("api_only", "api_then_motion_probe"):
            api_ret = self._ik_try_pose_with_reason(pose_mm_deg)
            out["api_ok"] = bool(api_ret["ok"])
            out["joint"] = api_ret["joint"]
            out["error_code"] = api_ret.get("err_code", None)
            out["raw"] = str(api_ret.get("raw", api_ret.get("err", "")))
            if out["api_ok"]:
                out["final_ok"] = True
                out["reason"] = "api_ok"
                print(f"[ik-check][api] OK stage={stage_name}")
                return out
            print(f"[ik-check][api] FAIL stage={stage_name} code={out['error_code']} raw={out['raw'][:120]}")
            if mode == "api_only":
                out["final_ok"] = False
                out["reason"] = "api_fail"
                return out
        probe_ret = self._probe_pose_reachability_by_motion(pose_mm_deg, stage_name)
        out["motion_probe_ran"] = bool(probe_ret.get("ran", False))
        out["motion_probe_ok"] = probe_ret.get("ok", None)
        if probe_ret.get("ran", False):
            print(
                f"[ik-check][motion] stage={stage_name} "
                f"ok={probe_ret.get('ok', None)} mode={probe_ret.get('mode', '')} err={str(probe_ret.get('err', ''))[:120]}"
            )
        if bool(out["motion_probe_ok"]) and bool(self.args.allow_motion_probe_success_as_reachable):
            out["final_ok"] = True
            out["reason"] = "api_failed_but_motion_ok"
            print("[check-mismatch] API check failed but motion probe succeeded")
            print("[check-mismatch] InverseSolution result may be unreliable for this pose")
            print("[check-mismatch] do not directly classify this candidate as unreachable")
            return out
        out["final_ok"] = False
        out["reason"] = "motion_probe_failed" if probe_ret.get("ran", False) else "probe_not_run"
        return out

    def _dedup_rpy_candidates(self, cands: list[dict[str, Any]], thr_deg: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in cands:
            rpy = np.asarray(c["rpy"], dtype=np.float64)
            keep = True
            for e in out:
                r2 = np.asarray(e["rpy"], dtype=np.float64)
                if float(np.linalg.norm(rpy - r2)) <= float(thr_deg):
                    keep = False
                    break
            if keep:
                out.append(c)
        return out

    def _expand_ik_rpy_candidates(
        self,
        seed_rpys: list[list[float]],
        visual_rpy: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        seeds: list[dict[str, Any]] = []
        for i, r in enumerate(seed_rpys):
            seeds.append({"source": "seed", "parent_idx": i, "delta": [0.0, 0.0, 0.0], "rpy": [float(r[0]), float(r[1]), float(r[2])]})
            print(f"[ik-cand][seed] #{i} rpy={r}")
        if not seeds and visual_rpy is not None:
            vr = np.asarray(visual_rpy, dtype=np.float64).reshape(3)
            seeds.append({"source": "vision_seed", "parent_idx": -1, "delta": [0.0, 0.0, 0.0], "rpy": vr.tolist()})
            print(f"[ik-cand][seed] vision rpy={vr.tolist()}")
        if not bool(self.args.ik_expand_enable):
            return seeds
        d_roll = _parse_csv_float_list(self.args.ik_expand_roll_deltas)
        d_pitch = _parse_csv_float_list(self.args.ik_expand_pitch_deltas)
        d_yaw = _parse_csv_float_list(self.args.ik_expand_yaw_deltas)
        expanded: list[dict[str, Any]] = []
        for s in seeds:
            base = np.asarray(s["rpy"], dtype=np.float64)
            for dr in d_roll:
                for dp in d_pitch:
                    for dy in d_yaw:
                        rpy = (base + np.asarray([dr, dp, dy], dtype=np.float64)).tolist()
                        expanded.append(
                            {
                                "source": "expand",
                                "parent_idx": int(s["parent_idx"]),
                                "delta": [float(dr), float(dp), float(dy)],
                                "rpy": rpy,
                            }
                        )
        all_cands = seeds + expanded
        dedup = self._dedup_rpy_candidates(all_cands, thr_deg=float(self.args.ik_expand_dedup_thr_deg))
        max_n = max(1, int(self.args.ik_expand_max_candidates))
        dedup = dedup[:max_n]
        print(
            f"[ik-cand][expand] enable={bool(self.args.ik_expand_enable)} "
            f"seed={len(seeds)} expanded={len(expanded)} dedup={len(dedup)} max={max_n}"
        )
        return dedup

    def _sort_ik_candidate_priority(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def _key(c: dict[str, Any]):
            source_rank = 0 if str(c.get("source", "")) in ("seed", "vision_seed") else 1
            delta = np.asarray(c.get("delta", [0.0, 0.0, 0.0]), dtype=np.float64)
            delta_norm = float(np.linalg.norm(delta))
            return (source_rank, delta_norm)
        return sorted(candidates, key=_key)

    def _run_fixed_rpy_reachability_debug(self, grasp_pos_base_mm: np.ndarray) -> dict[str, Any]:
        if not bool(self.args.ik_debug_fixed_rpy_enable):
            return {"enabled": False, "ran": False, "ok": False, "err": "disabled"}
        fixed_rpy = _parse_csv_floats(self.args.ik_debug_fixed_rpy, 3)
        grasp_pos_m = np.asarray(grasp_pos_base_mm, dtype=np.float64) / 1000.0
        pregrasp_xyz_m = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _euler_xyz_deg_to_rot(fixed_rpy[0], fixed_rpy[1], fixed_rpy[2])
        T[:3, 3] = grasp_pos_m
        T_pre = T @ _make_T_from_xyz_m(pregrasp_xyz_m)
        pose_pre = self.ctx._T_to_robot_pose_mm_deg(T_pre)
        print(f"[ik-debug][fixed-rpy] rpy={fixed_rpy.tolist()} pregrasp_pose={self._format_pose_mm_deg(pose_pre)}")
        print(f"[ik-debug][fixed-rpy] pregrasp_pose_full={self._format_pose_full_precision(pose_pre)}")
        ret = self._ik_check_pose(pose_pre, "fixed_rpy_pregrasp")
        ok = bool(ret["final_ok"])
        if ok:
            print("[ik-debug][result] SUCCESS: 固定 rpy 可达，问题更可能来自候选姿态定义过窄")
        else:
            print("[ik-debug][result] FAIL: 固定 rpy 仍不可达，当前位置或 user/tool/TCP 设置可能异常")
        return {"enabled": True, "ran": True, "ok": ok, "detail": ret, "fixed_rpy": fixed_rpy.tolist()}

    def _estimate_workspace_risk(self, grasp_pos_base_mm: np.ndarray) -> str:
        p = np.asarray(grasp_pos_base_mm, dtype=np.float64).reshape(3)
        xy = float(np.linalg.norm(p[:2]))
        z = float(p[2])
        if xy > 850.0 or z < -50.0 or z > 900.0:
            return "high"
        if xy > 750.0 or z < 50.0 or z > 800.0:
            return "medium"
        return "low"

    def _summarize_ik_attempts(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(attempts)
        ok = sum(1 for a in attempts if bool(a.get("ok", False)))
        seed_total = sum(1 for a in attempts if str(a.get("source", "")) in ("seed", "vision_seed"))
        seed_ok = sum(1 for a in attempts if str(a.get("source", "")) in ("seed", "vision_seed") and bool(a.get("ok", False)))
        expand_total = sum(1 for a in attempts if str(a.get("source", "")) == "expand")
        expand_ok = sum(1 for a in attempts if str(a.get("source", "")) == "expand" and bool(a.get("ok", False)))
        return {
            "total": total,
            "ok": ok,
            "seed_total": seed_total,
            "seed_ok": seed_ok,
            "expand_total": expand_total,
            "expand_ok": expand_ok,
        }

    def _summarize_reachability_check(self, attempts: list[dict[str, Any]]) -> dict[str, int]:
        out = {
            "both_ok": 0,
            "pregrasp_failed_only": 0,
            "grasp_failed_only": 0,
            "both_failed": 0,
            "api_failed_but_motion_ok": 0,
        }
        for a in attempts:
            pre_ok = bool(a.get("pre_final_ok", False))
            grasp_ok = bool(a.get("grasp_final_ok", False))
            if pre_ok and grasp_ok:
                out["both_ok"] += 1
            elif (not pre_ok) and grasp_ok:
                out["pregrasp_failed_only"] += 1
            elif pre_ok and (not grasp_ok):
                out["grasp_failed_only"] += 1
            else:
                out["both_failed"] += 1
            pre_reason = str(a.get("pre_reason", ""))
            grasp_reason = str(a.get("grasp_reason", ""))
            if ("api_failed_but_motion_ok" in pre_reason) or ("api_failed_but_motion_ok" in grasp_reason):
                out["api_failed_but_motion_ok"] += 1
        return out

    def _diagnose_api_vs_motion_mismatch(self, attempts: list[dict[str, Any]]) -> str:
        s = self._summarize_reachability_check(attempts)
        if int(s["api_failed_but_motion_ok"]) > 0:
            return "api_inverse_solution_false_negative"
        if int(s["pregrasp_failed_only"]) > 0 and int(s["grasp_failed_only"]) == 0:
            return "pregrasp_only_unreachable"
        if int(s["grasp_failed_only"]) > 0 and int(s["pregrasp_failed_only"]) == 0:
            return "grasp_only_unreachable"
        if int(s["both_failed"]) > 0:
            return "true_pose_unreachable"
        return "mixed_check_results"

    def _diagnose_ik_failure(
        self,
        attempts: list[dict[str, Any]],
        grasp_pos_base_mm: np.ndarray,
        pregrasp_pos_base_mm: np.ndarray,
        fixed_debug: dict[str, Any],
    ) -> None:
        stats = self._summarize_ik_attempts(attempts)
        reach_stats = self._summarize_reachability_check(attempts)
        workspace_risk = self._estimate_workspace_risk(grasp_pos_base_mm)
        g_xy = float(np.linalg.norm(np.asarray(grasp_pos_base_mm, dtype=np.float64)[:2]))
        p_xy = float(np.linalg.norm(np.asarray(pregrasp_pos_base_mm, dtype=np.float64)[:2]))
        farther = "pregrasp" if p_xy > g_xy else "grasp"
        likely_reason = self._diagnose_api_vs_motion_mismatch(attempts)
        suggestion = "增加候选姿态或检查 TCP/user/tool。"
        if fixed_debug.get("ran", False) and bool(fixed_debug.get("ok", False)):
            if int(stats["seed_ok"]) == 0 and int(stats["expand_ok"]) > 0:
                likely_reason = "candidate_rpy_too_narrow"
                suggestion = "原始候选库过窄，建议补充示教姿态并保留自动扩增。"
            else:
                likely_reason = "candidate_rpy_too_narrow"
                suggestion = "固定 rpy 可达，建议优化候选姿态库。"
        elif fixed_debug.get("ran", False) and (not bool(fixed_debug.get("ok", False))):
            likely_reason = "position_out_of_workspace" if workspace_risk != "low" else "tcp_or_user_tool_mismatch"
            suggestion = "检查抓取点位置是否越界，并核对 user/tool/TCP 标定。"
        last_src = str(getattr(self.ctx, "_last_icp_pose_source", "unknown")) if self.ctx is not None else "unknown"
        last_quality = dict(getattr(self.ctx, "_last_icp_quality", {})) if self.ctx is not None else {}
        if last_src == "refine":
            q = last_quality or {}
            if float(q.get("refine_fitness", 1.0)) < float(q.get("coarse_fitness", 1.0)) * 0.6:
                likely_reason = "refine_pose_degraded"
                suggestion = "refine 相对 coarse 明显恶化，建议启用/收紧 refine 回退参数。"
        print(f"[ik-diagnose] grasp_xyz_mm={np.asarray(grasp_pos_base_mm, dtype=np.float64).tolist()}")
        print(f"[ik-diagnose] pregrasp_xyz_mm={np.asarray(pregrasp_pos_base_mm, dtype=np.float64).tolist()}")
        print(f"[ik-diagnose] workspace_risk={workspace_risk} farther_point={farther} pre_xy={p_xy:.1f} grasp_xy={g_xy:.1f}")
        print(
            f"[ik-diagnose] candidate_count={stats['total']} "
            f"seed_ok={stats['seed_ok']}/{stats['seed_total']} "
            f"expand_ok={stats['expand_ok']}/{stats['expand_total']}"
        )
        print(f"[ik-diagnose] fixed_rpy_debug={'SUCCESS' if bool(fixed_debug.get('ok', False)) else 'FAIL'}")
        print(
            f"[reachability] both_ok={reach_stats['both_ok']} "
            f"pregrasp_failed_only={reach_stats['pregrasp_failed_only']} "
            f"grasp_failed_only={reach_stats['grasp_failed_only']} "
            f"both_failed={reach_stats['both_failed']} "
            f"api_failed_but_motion_ok={reach_stats['api_failed_but_motion_ok']}"
        )
        print(f"[ik-diagnose] final_pose_source={last_src}")
        if last_quality:
            print(f"[ik-diagnose] icp_quality={json.dumps(last_quality, ensure_ascii=False)}")
        print(
            f"[ik-diagnose] config user={int(getattr(self.args, 'robot_user', 0))} "
            f"tool={int(getattr(self.args, 'robot_tool', 0))}"
        )
        print(f"[ik-diagnose] likely_reason={likely_reason}")
        print(f"[ik-diagnose] suggestion={suggestion}")

    def _try_all_candidate_ik(self, grasp_pos_base_mm: np.ndarray, visual_grasp_rpy_deg: Optional[np.ndarray] = None) -> list:
        pregrasp_xyz_m = _parse_csv_floats(self.args.pregrasp_offset_mm, 3) * 0.001
        grasp_xyz_m = _parse_csv_floats(self.args.grasp_offset_mm, 3) * 0.001
        grasp_pos_m = np.asarray(grasp_pos_base_mm, dtype=np.float64) / 1000.0
        seed = list(getattr(self.ctx, "_ik_candidates", [])) if self.ctx is not None else []
        pool = self._expand_ik_rpy_candidates(seed, visual_rpy=visual_grasp_rpy_deg)
        pool = self._sort_ik_candidate_priority(pool)
        results: list[dict[str, Any]] = []
        for i, c in enumerate(pool):
            rpy = np.asarray(c["rpy"], dtype=np.float64).reshape(3)
            T_cand = np.eye(4, dtype=np.float64)
            T_cand[:3, :3] = _euler_xyz_deg_to_rot(rpy[0], rpy[1], rpy[2])
            T_cand[:3, 3] = grasp_pos_m
            T_pre = T_cand @ _make_T_from_xyz_m(pregrasp_xyz_m)
            T_g = T_cand @ _make_T_from_xyz_m(grasp_xyz_m)
            pose_pre = self.ctx._T_to_robot_pose_mm_deg(T_pre)
            pose_grasp = self.ctx._T_to_robot_pose_mm_deg(T_g)
            pre_check = self._ik_check_pose(pose_pre, f"cand_{i}_pregrasp")
            grasp_check = self._ik_check_pose(pose_grasp, f"cand_{i}_grasp")
            pre_ok = bool(pre_check["final_ok"])
            grasp_ok = bool(grasp_check["final_ok"])
            ok = pre_ok and grasp_ok
            j_pre = pre_check.get("joint", None)
            j_grasp = grasp_check.get("joint", None)
            exec_mode = "joint_solution" if (j_pre is not None and j_grasp is not None) else "direct_pose_fallback"
            if pre_ok and grasp_ok:
                final_reason = "both_ok"
            elif (not pre_ok) and grasp_ok:
                final_reason = "pregrasp_failed_only"
            elif pre_ok and (not grasp_ok):
                final_reason = "grasp_failed_only"
            else:
                if ("api_failed_but_motion_ok" in str(pre_check.get("reason", ""))) or (
                    "api_failed_but_motion_ok" in str(grasp_check.get("reason", ""))
                ):
                    final_reason = "api_failed_but_motion_ok"
                else:
                    final_reason = "both_failed"
            r: dict[str, Any] = {
                "idx": i,
                "ok": ok,
                "source": str(c.get("source", "seed")),
                "parent_idx": int(c.get("parent_idx", -1)),
                "delta": list(c.get("delta", [0.0, 0.0, 0.0])),
                "rpy": rpy.tolist(),
                "pose_pre": pose_pre,
                "pose_grasp": pose_grasp,
                "j_pre": j_pre,
                "j_grasp": j_grasp,
                "pre_api_ok": bool(pre_check.get("api_ok", False)),
                "pre_motion_probe_ok": pre_check.get("motion_probe_ok", None),
                "pre_final_ok": pre_ok,
                "pre_reason": str(pre_check.get("reason", "")),
                "grasp_api_ok": bool(grasp_check.get("api_ok", False)),
                "grasp_motion_probe_ok": grasp_check.get("motion_probe_ok", None),
                "grasp_final_ok": grasp_ok,
                "grasp_reason": str(grasp_check.get("reason", "")),
                "err": "" if ok else (str(pre_check.get("raw", "")) or str(grasp_check.get("raw", ""))),
                "err_code": pre_check.get("error_code", grasp_check.get("error_code", None)),
                "exec_mode": exec_mode,
                "final_reason": final_reason,
            }
            status = "OK" if ok else f"FAIL({str(r['err'])[:80]})"
            print(
                f"[ik-cand][pre] #{i} api_ok={r['pre_api_ok']} motion_ok={r['pre_motion_probe_ok']} "
                f"final_ok={r['pre_final_ok']} reason={r['pre_reason']}"
            )
            print(
                f"[ik-cand][grasp] #{i} api_ok={r['grasp_api_ok']} motion_ok={r['grasp_motion_probe_ok']} "
                f"final_ok={r['grasp_final_ok']} reason={r['grasp_reason']}"
            )
            print(
                f"[ik-cand][summary] #{i} src={r['source']} parent={r['parent_idx']} delta={r['delta']} "
                f"rpy={r['rpy']} pose_pre={self._format_pose_mm_deg(pose_pre)} "
                f"pose_grasp={self._format_pose_mm_deg(pose_grasp)} final_reason={r['final_reason']} -> {status}"
            )
            results.append(r)
        return results

    def run_runtime_probe_and_execute(self, pose_pre, pose_grasp):
        pose_pre = np.asarray(pose_pre, dtype=np.float64).reshape(6)
        pose_grasp = np.asarray(pose_grasp, dtype=np.float64).reshape(6)
        print(f"[runtime][pre] rounded={self._format_pose_mm_deg(pose_pre)}")
        print(f"[runtime][pre] full={self._format_pose_full_precision(pose_pre)}")
        probe_pre = self._probe_pose_reachability_by_motion(pose_pre, "runtime_pregrasp")
        print(
            f"[runtime][probe] stage=pre ok={probe_pre.get('ok', None)} "
            f"ran={probe_pre.get('ran', False)} mode={probe_pre.get('mode', '')}"
        )
        if (not bool(probe_pre.get("ran", False))) or (not bool(probe_pre.get("ok", False))):
            print("[runtime][probe] pregrasp probe failed, skip current frame")
            return False
        do_grasp_probe = bool(getattr(self.args, "runtime_probe_grasp", False))
        if do_grasp_probe:
            print(f"[runtime][grasp] rounded={self._format_pose_mm_deg(pose_grasp)}")
            print(f"[runtime][grasp] full={self._format_pose_full_precision(pose_grasp)}")
            probe_grasp = self._probe_pose_reachability_by_motion(pose_grasp, "runtime_grasp")
            print(
                f"[runtime][probe] stage=grasp ok={probe_grasp.get('ok', None)} "
                f"ran={probe_grasp.get('ran', False)} mode={probe_grasp.get('mode', '')}"
            )
            if (not bool(probe_grasp.get("ran", False))) or (not bool(probe_grasp.get("ok", False))):
                print("[runtime][probe] grasp probe failed, skip current frame")
                return False
        if self.executor is None:
            print("[runtime][exec] auto_execute=False，仅输出位姿")
            return True
        if not bool(self.args.auto_execute):
            print("[runtime][exec] auto_execute=False，仅输出位姿")
            return True
        if bool(self.args.prefer_api_ik_joint_execution):
            pre_api = self._ik_try_pose_with_reason(pose_pre)
            grasp_api = self._ik_try_pose_with_reason(pose_grasp)
            if bool(pre_api.get("ok", False)) and bool(grasp_api.get("ok", False)):
                print("[runtime][exec] exec_mode=joint_solution")
                self.executor.movj_joint(np.asarray(pre_api["joint"], dtype=np.float64), "runtime_pregrasp_movj_joint")
                self.executor.movj_joint(np.asarray(grasp_api["joint"], dtype=np.float64), "runtime_grasp_movj_joint")
                if self.ctx is not None:
                    self.ctx._close_gripper_after_grasp(context="runtime_joint_solution")
                self._cache_final_pose_and_log(pose_grasp, context="runtime_joint_solution")
                print("[runtime][exec] done")
                return True
        print("[runtime][exec] exec_mode=direct_pose_fallback")
        self.executor.movj_pose(np.asarray(pose_pre, dtype=np.float64), "runtime_pregrasp_movj_pose")
        self.executor.movj_pose(np.asarray(pose_grasp, dtype=np.float64), "runtime_grasp_movj_pose")
        if self.ctx is not None:
            self.ctx._close_gripper_after_grasp(context="runtime_direct_pose_fallback")
        self._cache_final_pose_and_log(pose_grasp, context="runtime_direct_pose_fallback")
        print("[runtime][exec] done")
        return True


__all__ = ["IKSearchRunner", "_load_ik_candidates"]

