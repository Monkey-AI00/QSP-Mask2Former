"""Grasp frame builder migrated from legacy pipeline."""

from __future__ import annotations

import numpy as np

from online_grasp.geometry.transforms import _make_T_from_xyz_m, _parse_csv_floats


class GraspFrameBuilder:
    def __init__(self, args, T_region_to_grasp):
        self.args = args
        self.T_region_to_grasp = np.asarray(T_region_to_grasp, dtype=np.float64)

    def build_from_region_cam(self, T_region_cam, T_base_cam):
        T_region_cam = np.asarray(T_region_cam, dtype=np.float64)
        T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
        T_region_to_grasp = np.asarray(self.T_region_to_grasp, dtype=np.float64)
        T_grasp_cam = T_region_cam @ T_region_to_grasp
        T_grasp_base = T_base_cam @ T_grasp_cam
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
        return T_base_pregrasp, T_base_grasp, T_grasp_base, {
            "T_region_to_camera": T_region_cam,
            "T_region_to_grasp": T_region_to_grasp,
            "T_grasp_to_camera": T_grasp_cam,
            "T_grasp_to_base": T_grasp_base,
            "T_base_pregrasp": T_base_pregrasp,
            "T_base_grasp": T_base_grasp,
        }


__all__ = ["GraspFrameBuilder"]

