"""Point-cloud builder wrapper scaffold."""

from __future__ import annotations

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils
from postprocess_pointcloud import post_process_point_cloud


class PointCloudBuilder:
    """点云构建封装骨架，当前阶段由 legacy pipeline 执行完整流程。"""

    def __init__(self, args, o3d, intrinsics, depth_completer):
        self.args = args
        self.o3d = o3d
        self.intrinsics = intrinsics
        self.depth_completer = depth_completer
        self.use_local_region_template = True

    def build_target_pointcloud(self, depth_obj, mask_pc, color_bgr):
        depth_np = depth_obj.data()
        depth_m = self.depth_completer.maybe_complete_depth(depth_np, color_bgr)
        fx, fy, cx, cy = live_utils._get_depth_k_from_mecheye_intrinsics(self.intrinsics)
        xyz_m, rgb = live_utils._backproject_masked_xyzrgb(
            depth_m,
            color_bgr,
            mask_pc,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=max(1, int(self.args.pc_stride)),
        )
        pcd = self.o3d.geometry.PointCloud()
        if xyz_m.shape[0] == 0:
            return pcd
        pcd.points = self.o3d.utility.Vector3dVector(xyz_m.astype("float64"))
        pcd.colors = self.o3d.utility.Vector3dVector((rgb.astype("float64") / 255.0))
        return pcd

    def preprocess_target_pointcloud(self, pcd):
        if len(pcd.points) < int(self.args.min_points):
            return pcd
        if bool(self.args.grasp_region_preserve) and bool(self.use_local_region_template):
            p = pcd
            if float(self.args.pp_voxel) > 0:
                p = p.voxel_down_sample(float(self.args.pp_voxel))
            _, ind = p.remove_statistical_outlier(
                nb_neighbors=int(self.args.pp_sor_nb),
                std_ratio=float(self.args.pp_sor_std),
            )
            p = p.select_by_index(ind)
            if int(self.args.pp_ror_nb) > 0 and float(self.args.pp_ror_radius) > 0:
                _, ind = p.remove_radius_outlier(
                    nb_points=int(self.args.pp_ror_nb),
                    radius=float(self.args.pp_ror_radius),
                )
                p = p.select_by_index(ind)
            print(f"[pp][local] preserve grasp region -> points={len(p.points)}")
            return p
        pcd2 = post_process_point_cloud(
            pcd,
            voxel_size=float(self.args.pp_voxel),
            drop_zero_xyz=True,
            drop_zero_eps=1e-9,
            sor_nb_neighbors=int(self.args.pp_sor_nb),
            sor_std_ratio=float(self.args.pp_sor_std),
            ror_nb_points=int(self.args.pp_ror_nb),
            ror_radius=float(self.args.pp_ror_radius),
            dbscan_eps=float(self.args.pp_dbscan_eps),
            dbscan_min_points=int(self.args.pp_dbscan_min_points),
            keep_top_k=int(self.args.pp_keep_top_k),
            cluster_select=str(self.args.pp_cluster_select),
        )
        return pcd2


__all__ = ["PointCloudBuilder"]

