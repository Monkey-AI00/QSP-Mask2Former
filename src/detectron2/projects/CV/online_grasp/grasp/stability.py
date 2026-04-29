"""Stability checker migrated from legacy pipeline."""

from __future__ import annotations

import numpy as np

from online_grasp.geometry.transforms import _rot_to_euler_xyz_deg


class StabilityChecker:
    def __init__(self, stable_frames, pos_thr_mm, rot_thr_deg):
        self._stable_frames_required = int(stable_frames)
        self._stable_pos_thr_mm = float(pos_thr_mm)
        self._stable_rot_thr_deg = float(rot_thr_deg)
        self._stable_history: list[dict] = []

    def update(self, T_grasp_base, grasp_pos_base_mm, candidate_idx: int):
        rpy = np.asarray(_rot_to_euler_xyz_deg(np.asarray(T_grasp_base[:3, :3], dtype=np.float64)), dtype=np.float64)
        self._stable_history.append({
            "pos_mm": np.asarray(grasp_pos_base_mm, dtype=np.float64).copy(),
            "cand_idx": int(candidate_idx),
            "rpy_deg": rpy,
        })
        n = self._stable_frames_required
        if len(self._stable_history) > n * 3:
            self._stable_history = self._stable_history[-n:]
        if len(self._stable_history) < n:
            return False
        recent = self._stable_history[-n:]
        positions = np.array([h["pos_mm"] for h in recent])
        ref = positions[-1]
        max_dev = float(np.max(np.linalg.norm(positions - ref, axis=1)))
        if max_dev > self._stable_pos_thr_mm:
            print(f"[stable] 位置偏差 {max_dev:.2f}mm > 阈值 {self._stable_pos_thr_mm:.2f}mm")
            return False
        rpys = np.array([h["rpy_deg"] for h in recent], dtype=np.float64)
        rpy_ref = rpys[-1]
        max_rot_dev = float(np.max(np.linalg.norm(rpys - rpy_ref, axis=1)))
        if max_rot_dev > self._stable_rot_thr_deg:
            print(f"[stable] 姿态偏差 {max_rot_dev:.2f}deg > 阈值 {self._stable_rot_thr_deg:.2f}deg")
            return False
        cands = set(h["cand_idx"] for h in recent)
        if len(cands) > 1:
            print(f"[stable] 候选编号不一致: {cands}")
            return False
        return True

    def reset(self):
        self._stable_history.clear()


__all__ = ["StabilityChecker"]

