#!/usr/bin/env python3
"""
点云后处理（Open3D）：
- 统计滤波 SOR（Statistical Outlier Removal）：去稀疏噪声/离群点
- DBSCAN 聚类：保留最大的主体簇（通常是插头），丢弃周围碎片

新增：支持同时导出三种阶段效果，便于对比：
- 仅 SOR
- SOR + ROR
- SOR + ROR + DBSCAN（最终）

适用场景：
你用 Mech-Eye + PointRend 得到插头点云后，周围仍存在噪声碎片、离群点，会影响后续机器人抓取/位姿估计。

单位说明：
Mech-Eye 点云坐标常见以 mm 存储（样例里 depth/xyz 也以 mm 打印）。
因此 DBSCAN 的 eps 默认按“mm”给值（例如 eps=5 表示 5mm）。
如果你的点云单位是“米”，请把 eps/voxel 相应缩小 1000 倍。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def _drop_zero_xyz_points(pcd, *, eps: float = 0.0):
    """
    删除 (x,y,z) 全为 0 的占位点（很多相机/掩膜导出会用 0,0,0 表示无效点）。
    eps=0 表示严格等于 0；eps>0 表示近似为 0（容忍浮点误差）。
    """
    import numpy as np

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd, 0

    if float(eps) > 0:
        keep = np.any(np.abs(pts) > float(eps), axis=1)
    else:
        keep = np.any(pts != 0, axis=1)

    dropped = int((~keep).sum())
    if dropped <= 0:
        return pcd, 0

    idx = np.where(keep)[0].tolist()
    return pcd.select_by_index(idx), dropped


def _select_clusters_dbscan(
    pcd,
    *,
    dbscan_eps: float = 5.0,
    dbscan_min_points: int = 30,
    keep_top_k: int = 1,
    cluster_select: str = "largest",
    keep_ids: Optional[list[int]] = None,
    print_clusters: bool = False,
    dump_clusters_dir: str = "",
    dump_top_n: int = 0,
):
    """
    在输入点云 pcd 上执行 DBSCAN 聚类，并按策略保留主体簇。
    约定：pcd 已经过 SOR/ROR 等前置滤波（本函数不再做 SOR/ROR）。
    返回：筛选后的 pcd（Open3D PointCloud）。
    """
    import numpy as np
    import open3d as o3d  # type: ignore[import-not-found]

    if len(pcd.points) == 0:
        return pcd

    labels = np.array(pcd.cluster_dbscan(eps=float(dbscan_eps), min_points=int(dbscan_min_points), print_progress=False))
    if labels.size == 0:
        print("[dbs] no labels; skip clustering")
        return pcd

    # label=-1 为噪声
    valid = labels[labels >= 0]
    if valid.size == 0:
        print("[dbs] all points labeled as noise (-1). Try increasing --dbscan-eps or decreasing --dbscan-min-points.")
        return pcd

    unique = np.unique(valid)

    # 预计算每个簇的统计量：点数、质心、AABB 体积
    pts = np.asarray(pcd.points)
    stats = []
    for cid in unique.tolist():
        idx = np.where(labels == cid)[0]
        p = pts[idx]
        centroid = p.mean(axis=0)
        aabb_min = p.min(axis=0)
        aabb_max = p.max(axis=0)
        extent = aabb_max - aabb_min
        vol = float(extent[0] * extent[1] * extent[2])
        stats.append(
            {
                "id": int(cid),
                "count": int(idx.size),
                "centroid": centroid,
                "extent": extent,
                "vol": vol,
            }
        )

    if print_clusters:
        print(f"[dbs] clusters={len(unique)} (label=-1 is noise)")
        # 按点数从大到小打印，便于快速挑主体
        for s in sorted(stats, key=lambda x: x["count"], reverse=True):
            cx, cy, cz = s["centroid"].tolist()
            ex, ey, ez = s["extent"].tolist()
            print(
                f"  - id={s['id']:>3} count={s['count']:<7} "
                f"centroid=({cx:.1f},{cy:.1f},{cz:.1f}) "
                f"extent=({ex:.1f},{ey:.1f},{ez:.1f}) vol={s['vol']:.2e}"
            )

    # 可选：导出簇到目录，方便肉眼确认“插头是哪一簇”
    if dump_clusters_dir:
        out_dir = Path(dump_clusters_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 按点数从大到小导出，默认全导；也可只导 top N
        order_stats = sorted(stats, key=lambda x: x["count"], reverse=True)
        if dump_top_n and int(dump_top_n) > 0:
            order_stats = order_stats[: int(dump_top_n)]

        for s in order_stats:
            cid = int(s["id"])
            idx = np.where(labels == cid)[0].tolist()
            pc = pcd.select_by_index(idx)
            # 文件名带上 count/centroid/extent，便于快速筛选
            cx, cy, cz = s["centroid"].tolist()
            ex, ey, ez = s["extent"].tolist()
            fn = (
                f"cluster_id{cid:03d}_n{s['count']}_"
                f"c({cx:.1f},{cy:.1f},{cz:.1f})_"
                f"e({ex:.1f},{ey:.1f},{ez:.1f}).ply"
            )
            o3d.io.write_point_cloud(str(out_dir / fn), pc, write_ascii=False, compressed=False, print_progress=False)
        print(f"[dump] saved clusters to: {out_dir} (top_n={dump_top_n if dump_top_n else 'all'})")

    # 用户指定：直接保留这些簇
    if keep_ids:
        keep = set(int(x) for x in keep_ids)
    else:
        sel = str(cluster_select).strip().lower()
        if sel == "largest":
            order = [s["id"] for s in sorted(stats, key=lambda x: x["count"], reverse=True)]
        elif sel == "closest_z":
            # 以质心 z 最小为“最近”（常见相机坐标：z 为深度，越小越近）
            order = [s["id"] for s in sorted(stats, key=lambda x: float(x["centroid"][2]))]
        elif sel == "farthest_z":
            order = [s["id"] for s in sorted(stats, key=lambda x: float(x["centroid"][2]), reverse=True)]
        elif sel == "smallest_bbox":
            order = [s["id"] for s in sorted(stats, key=lambda x: x["vol"])]
        elif sel == "largest_bbox":
            order = [s["id"] for s in sorted(stats, key=lambda x: x["vol"], reverse=True)]
        else:
            raise ValueError("不支持的 --cluster-select。可选：largest/closest_z/farthest_z/smallest_bbox/largest_bbox")

        keep = set(order[: max(1, int(keep_top_k))])

    keep_idx = np.where(np.isin(labels, list(keep)))[0].tolist()
    pcd2 = pcd.select_by_index(keep_idx)
    print(
        f"[dbs] eps={dbscan_eps} min_pts={dbscan_min_points} clusters={len(unique)} "
        f"select={cluster_select} keep_top_k={keep_top_k} keep_ids={keep_ids} -> points={len(pcd2.points)}"
    )
    return pcd2


def post_process_point_cloud(
    pcd,
    *,
    voxel_size: float = 0.0,
    drop_zero_xyz: bool = False,
    drop_zero_eps: float = 0.0,
    sor_nb_neighbors: int = 50,
    sor_std_ratio: float = 1.0,
    ror_nb_points: int = 0,
    ror_radius: float = 0.0,
    dbscan_eps: float = 5.0,
    dbscan_min_points: int = 30,
    keep_top_k: int = 1,
    cluster_select: str = "largest",
    keep_ids: Optional[list[int]] = None,
    print_clusters: bool = False,
    dump_clusters_dir: str = "",
    dump_top_n: int = 0,
):
    """
    返回处理后的 pcd（Open3D PointCloud）。
    """
    print(f"[in ] points={len(pcd.points)}")

    if bool(drop_zero_xyz):
        pcd, dropped = _drop_zero_xyz_points(pcd, eps=float(drop_zero_eps))
        if dropped > 0:
            print(f"[pre] drop_zero_xyz eps={drop_zero_eps} -> dropped={dropped} points={len(pcd.points)}")

    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        print(f"[vox] voxel={voxel_size} -> points={len(pcd.points)}")

    # 1) SOR：去稀疏噪声
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=int(sor_nb_neighbors), std_ratio=float(sor_std_ratio))
    pcd = pcd.select_by_index(ind)
    print(f"[sor] nb={sor_nb_neighbors} std={sor_std_ratio} -> points={len(pcd.points)}")

    # 1.5) ROR：半径滤波（对 mask 边缘碎片/漂浮孤立片更敏感）
    # 语义：半径 r 内至少 nb_points 个邻居，否则删掉该点
    if ror_nb_points and ror_nb_points > 0 and ror_radius and ror_radius > 0:
        cl, ind = pcd.remove_radius_outlier(nb_points=int(ror_nb_points), radius=float(ror_radius))
        pcd = pcd.select_by_index(ind)
        print(f"[ror] nb_points={ror_nb_points} radius={ror_radius} -> points={len(pcd.points)}")

    if len(pcd.points) == 0:
        return pcd

    # 2) DBSCAN：聚类 + 选择主体簇
    return _select_clusters_dbscan(
        pcd,
        dbscan_eps=float(dbscan_eps),
        dbscan_min_points=int(dbscan_min_points),
        keep_top_k=int(keep_top_k),
        cluster_select=str(cluster_select),
        keep_ids=keep_ids,
        print_clusters=bool(print_clusters),
        dump_clusters_dir=str(dump_clusters_dir),
        dump_top_n=int(dump_top_n),
    )


def post_process_point_cloud_stages(
    pcd,
    *,
    voxel_size: float = 0.0,
    drop_zero_xyz: bool = False,
    drop_zero_eps: float = 0.0,
    sor_nb_neighbors: int = 50,
    sor_std_ratio: float = 1.0,
    ror_nb_points: int = 0,
    ror_radius: float = 0.0,
    dbscan_eps: float = 5.0,
    dbscan_min_points: int = 30,
    keep_top_k: int = 1,
    cluster_select: str = "largest",
    keep_ids: Optional[list[int]] = None,
    print_clusters: bool = False,
    dump_clusters_dir: str = "",
    dump_top_n: int = 0,
):
    """
    返回三个阶段的点云：
    - pcd_sor：仅 SOR
    - pcd_sor_ror：SOR + ROR（若 ROR 未启用则与 pcd_sor 相同）
    - pcd_final：SOR + ROR + DBSCAN（最终）
    """
    print(f"[in ] points={len(pcd.points)}")

    p = pcd
    if bool(drop_zero_xyz):
        p, dropped = _drop_zero_xyz_points(p, eps=float(drop_zero_eps))
        if dropped > 0:
            print(f"[pre] drop_zero_xyz eps={drop_zero_eps} -> dropped={dropped} points={len(p.points)}")

    if voxel_size and voxel_size > 0:
        p = p.voxel_down_sample(voxel_size=float(voxel_size))
        print(f"[vox] voxel={voxel_size} -> points={len(p.points)}")

    # 1) SOR
    _, ind = p.remove_statistical_outlier(nb_neighbors=int(sor_nb_neighbors), std_ratio=float(sor_std_ratio))
    pcd_sor = p.select_by_index(ind)
    print(f"[sor] nb={sor_nb_neighbors} std={sor_std_ratio} -> points={len(pcd_sor.points)}")

    # 2) ROR（可选）
    pcd_sor_ror = pcd_sor
    if ror_nb_points and ror_nb_points > 0 and ror_radius and ror_radius > 0:
        _, ind = pcd_sor.remove_radius_outlier(nb_points=int(ror_nb_points), radius=float(ror_radius))
        pcd_sor_ror = pcd_sor.select_by_index(ind)
        print(f"[ror] nb_points={ror_nb_points} radius={ror_radius} -> points={len(pcd_sor_ror.points)}")
    else:
        print("[ror] disabled (use --ror-nb and --ror-radius to enable)")

    # 3) DBSCAN（最终）
    if len(pcd_sor_ror.points) == 0:
        return pcd_sor, pcd_sor_ror, pcd_sor_ror

    pcd_final = _select_clusters_dbscan(
        pcd_sor_ror,
        dbscan_eps=float(dbscan_eps),
        dbscan_min_points=int(dbscan_min_points),
        keep_top_k=int(keep_top_k),
        cluster_select=str(cluster_select),
        keep_ids=keep_ids,
        print_clusters=bool(print_clusters),
        dump_clusters_dir=str(dump_clusters_dir),
        dump_top_n=int(dump_top_n),
    )
    return pcd_sor, pcd_sor_ror, pcd_final


def main():
    ap = argparse.ArgumentParser(description="Open3D 点云去噪（SOR + DBSCAN 保留主体簇）")
    ap.add_argument("--input", required=True, help="输入点云 .ply 路径")
    ap.add_argument("--output", default="", help="输出点云 .ply 路径；默认在 input 同目录生成 *_clean.ply")
    ap.add_argument("--vis", action="store_true", help="可视化前后点云（需要 GUI）")
    ap.add_argument(
        "--save-stages",
        action="store_true",
        help="同时输出三个阶段点云：*_sor.ply、*_sor_ror.ply、最终输出（默认仅输出最终）",
    )

    ap.add_argument("--voxel", type=float, default=0.0, help="体素下采样大小（单位同点云，mm 点云可设 1~3）")
    ap.add_argument("--drop-zero", action="store_true", help="删除 (0,0,0) 占位点（很多导出会把无效点写成 0,0,0）")
    ap.add_argument("--drop-zero-eps", type=float, default=0.0, help="drop-zero 的近似阈值（0=严格等于 0；可设 1e-6）")
    ap.add_argument("--sor-nb", type=int, default=50, help="SOR 邻域点数（典型 20~80）")
    ap.add_argument("--sor-std", type=float, default=1.0, help="SOR 标准差阈值（越小越严格，典型 0.8~2.0）")
    ap.add_argument("--ror-nb", type=int, default=0, help="ROR 半径滤波：邻居点数阈值（>0 启用，典型 10~30）")
    ap.add_argument("--ror-radius", type=float, default=0.0, help="ROR 半径（单位同点云；mm 点云典型 5~30）")
    ap.add_argument("--dbscan-eps", type=float, default=5.0, help="DBSCAN eps（单位同点云；mm 点云典型 2~15）")
    ap.add_argument("--dbscan-min-points", type=int, default=30, help="DBSCAN min_points（典型 10~80）")
    ap.add_argument("--keep-top-k", type=int, default=1, help="保留最大的 K 个簇（防止主体被分成多块时可设 2~3）")
    ap.add_argument(
        "--cluster-select",
        choices=["largest", "closest_z", "farthest_z", "smallest_bbox", "largest_bbox"],
        default="largest",
        help="主体簇选择策略。默认 largest（最大簇）在你的场景可能会选中托盘/背景。",
    )
    ap.add_argument("--keep-ids", default="", help="手动保留簇 id（逗号分隔），例如 '1' 或 '1,3'；优先级高于 --cluster-select")
    ap.add_argument("--print-clusters", action="store_true", help="打印每个簇的点数/质心/包围盒信息，方便选择主体簇")
    ap.add_argument("--dump-clusters", default="", help="导出每个簇为单独的 ply 到指定目录，方便快速确认插头簇 ID")
    ap.add_argument("--dump-top-n", type=int, default=0, help="仅导出点数最多的前 N 个簇（0=全部）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"未找到输入点云: {in_path}")
    out_path = Path(args.output) if str(args.output).strip() else in_path.with_name(in_path.stem + "_clean.ply")

    import open3d as o3d  # type: ignore[import-not-found]

    pcd = o3d.io.read_point_cloud(str(in_path))
    if pcd.is_empty():
        raise ValueError(f"输入点云为空: {in_path}")

    keep_ids = [int(x) for x in str(args.keep_ids).split(",") if x.strip().isdigit()] if str(args.keep_ids).strip() else None

    pcd_sor, pcd_sor_ror, pcd2 = post_process_point_cloud_stages(
        pcd,
        voxel_size=float(args.voxel),
        drop_zero_xyz=bool(args.drop_zero),
        drop_zero_eps=float(args.drop_zero_eps),
        sor_nb_neighbors=int(args.sor_nb),
        sor_std_ratio=float(args.sor_std),
        ror_nb_points=int(args.ror_nb),
        ror_radius=float(args.ror_radius),
        dbscan_eps=float(args.dbscan_eps),
        dbscan_min_points=int(args.dbscan_min_points),
        keep_top_k=int(args.keep_top_k),
        cluster_select=str(args.cluster_select),
        keep_ids=keep_ids,
        print_clusters=bool(args.print_clusters),
        dump_clusters_dir=str(args.dump_clusters).strip(),
        dump_top_n=int(args.dump_top_n),
    )

    o3d.io.write_point_cloud(str(out_path), pcd2, write_ascii=False, compressed=False, print_progress=False)
    print(f"[out] saved: {out_path}")

    if bool(args.save_stages):
        # 阶段输出文件与最终输出放同目录，使用最终输出的 stem 作为前缀
        prefix = out_path.stem
        sor_path = out_path.with_name(prefix + "_sor.ply")
        sor_ror_path = out_path.with_name(prefix + "_sor_ror.ply")
        o3d.io.write_point_cloud(str(sor_path), pcd_sor, write_ascii=False, compressed=False, print_progress=False)
        o3d.io.write_point_cloud(str(sor_ror_path), pcd_sor_ror, write_ascii=False, compressed=False, print_progress=False)
        print(f"[stage] saved: {sor_path}")
        print(f"[stage] saved: {sor_ror_path}")

    if bool(args.vis):
        # 颜色区分：原始灰、SOR 蓝、SOR+ROR 橙、最终红
        geoms = []
        pcd_vis = o3d.geometry.PointCloud(pcd)
        pcd_vis.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(pcd_vis)

        pcd_sor_vis = o3d.geometry.PointCloud(pcd_sor)
        pcd_sor_vis.paint_uniform_color([0.2, 0.6, 1.0])
        geoms.append(pcd_sor_vis)

        pcd_sor_ror_vis = o3d.geometry.PointCloud(pcd_sor_ror)
        pcd_sor_ror_vis.paint_uniform_color([1.0, 0.6, 0.2])
        geoms.append(pcd_sor_ror_vis)

        pcd2_vis = o3d.geometry.PointCloud(pcd2)
        pcd2_vis.paint_uniform_color([1.0, 0.2, 0.2])
        geoms.append(pcd2_vis)

        o3d.visualization.draw_geometries(geoms, window_name="pcd stages: raw(grey), sor(blue), sor+ror(orange), final(red)")


if __name__ == "__main__":
    main()


