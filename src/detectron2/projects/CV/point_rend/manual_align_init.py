import json
import copy
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)


class ManualAlignTool:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        out_json: str,
        voxel: float = 0.0,
        trans_step_mm: float = 2.0,
        rot_step_deg: float = 2.0,
    ):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.out_json = Path(out_json)

        self.trans_step = trans_step_mm / 1000.0
        self.rot_step = rot_step_deg

        self.source_raw = o3d.io.read_point_cloud(str(self.source_path))
        self.target_raw = o3d.io.read_point_cloud(str(self.target_path))

        if len(self.source_raw.points) == 0:
            raise ValueError(f"source 点云为空: {self.source_path}")
        if len(self.target_raw.points) == 0:
            raise ValueError(f"target 点云为空: {self.target_path}")

        if voxel > 0:
            self.source_raw = self.source_raw.voxel_down_sample(voxel)
            self.target_raw = self.target_raw.voxel_down_sample(voxel)

        self.source_raw.paint_uniform_color([1.0, 0.85, 0.0])  # 黄
        self.target_raw.paint_uniform_color([0.0, 0.8, 0.0])   # 绿

        self.T = np.eye(4, dtype=np.float64)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Manual Align: yellow=source, green=target", width=1280, height=900)

        self.source_vis = copy.deepcopy(self.source_raw)
        self.target_vis = copy.deepcopy(self.target_raw)

        self.vis.add_geometry(self.source_vis)
        self.vis.add_geometry(self.target_vis)

        self._register_keys()
        self._refresh()

        print("\n===== 手工对齐说明 =====")
        print("平移:")
        print("  W/S : +Y / -Y")
        print("  A/D : -X / +X")
        print("  Q/E : +Z / -Z")
        print("旋转(绕 source 当前局部中心):")
        print("  J/L : Rx- / Rx+")
        print("  I/K : Ry+ / Ry-")
        print("  U/O : Rz- / Rz+")
        print("其他:")
        print("  R   : 重置位姿")
        print("  P   : 打印当前 4x4")
        print("  V   : 保存 grasp_region_init.json")
        print("  ESC : 退出")
        print("========================\n")

    def _refresh(self):
        pts = np.asarray(self.source_raw.points)
        src = copy.deepcopy(self.source_raw)
        src.transform(self.T)

        self.source_vis.points = src.points
        self.source_vis.colors = src.colors

        self.vis.update_geometry(self.source_vis)
        self.vis.update_geometry(self.target_vis)
        self.vis.poll_events()
        self.vis.update_renderer()

    def _apply_transform_local_center(self, R_delta: np.ndarray):
        pts = np.asarray(self.source_raw.points)
        center = pts.mean(axis=0)
        T1 = make_T(np.eye(3), -center)
        T2 = make_T(R_delta, np.zeros(3))
        T3 = make_T(np.eye(3), center)
        self.T = self.T @ (T3 @ T2 @ T1)
        self._refresh()
        return False

    def _apply_translation(self, dx: float, dy: float, dz: float):
        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, 3] = np.array([dx, dy, dz], dtype=np.float64)
        self.T = self.T @ T_delta
        self._refresh()
        return False

    def _print_T(self):
        print("\nCurrent T_source_to_target:")
        print(np.array2string(self.T, precision=6, suppress_small=False))
        print()
        return False

    def _save_json(self):
        payload = {
            "T_source_to_target": self.T.tolist()
        }
        self.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {self.out_json}")
        return False

    def _reset(self):
        self.T = np.eye(4, dtype=np.float64)
        self._refresh()
        print("[reset] T = I")
        return False

    def _register_keys(self):
        # 平移
        self.vis.register_key_callback(ord("W"), lambda vis: self._apply_translation(0, +self.trans_step, 0))
        self.vis.register_key_callback(ord("S"), lambda vis: self._apply_translation(0, -self.trans_step, 0))
        self.vis.register_key_callback(ord("A"), lambda vis: self._apply_translation(-self.trans_step, 0, 0))
        self.vis.register_key_callback(ord("D"), lambda vis: self._apply_translation(+self.trans_step, 0, 0))
        self.vis.register_key_callback(ord("Q"), lambda vis: self._apply_translation(0, 0, +self.trans_step))
        self.vis.register_key_callback(ord("E"), lambda vis: self._apply_translation(0, 0, -self.trans_step))

        # 旋转
        self.vis.register_key_callback(ord("J"), lambda vis: self._apply_transform_local_center(rot_x(-self.rot_step)))
        self.vis.register_key_callback(ord("L"), lambda vis: self._apply_transform_local_center(rot_x(+self.rot_step)))
        self.vis.register_key_callback(ord("I"), lambda vis: self._apply_transform_local_center(rot_y(+self.rot_step)))
        self.vis.register_key_callback(ord("K"), lambda vis: self._apply_transform_local_center(rot_y(-self.rot_step)))
        self.vis.register_key_callback(ord("U"), lambda vis: self._apply_transform_local_center(rot_z(-self.rot_step)))
        self.vis.register_key_callback(ord("O"), lambda vis: self._apply_transform_local_center(rot_z(+self.rot_step)))

        # 其他
        self.vis.register_key_callback(ord("P"), lambda vis: self._print_T())
        self.vis.register_key_callback(ord("V"), lambda vis: self._save_json())
        self.vis.register_key_callback(ord("R"), lambda vis: self._reset())

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Open3D 手工对齐 source 到 target，并导出 T_source_to_target")
    parser.add_argument("--source", required=True, help="实时抓取区域点云 ply")
    parser.add_argument("--target", required=True, help="局部 CAD 模板 ply")
    parser.add_argument("--out", default="grasp_region_init.json", help="输出 json 路径")
    parser.add_argument("--voxel", type=float, default=0.0, help="可选体素降采样，单位米")
    parser.add_argument("--trans-step-mm", type=float, default=2.0, help="每次平移步长，单位 mm")
    parser.add_argument("--rot-step-deg", type=float, default=2.0, help="每次旋转步长，单位 deg")
    args = parser.parse_args()

    tool = ManualAlignTool(
        source_path=args.source,
        target_path=args.target,
        out_json=args.out,
        voxel=args.voxel,
        trans_step_mm=args.trans_step_mm,
        rot_step_deg=args.rot_step_deg,
    )
    tool.run()


if __name__ == "__main__":
    main()