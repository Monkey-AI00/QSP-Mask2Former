#!/usr/bin/env python3
"""
STL -> PLY 点云（用于 ICP 配准）

为什么不能直接用 STL？
- STL 只有三角面片，没有“均匀点采样”。ICP 更希望输入点云分布相对均匀。

本脚本做的事：
1) 读取 STL mesh
2) （可选）清理 mesh（去重复点/三角形、去退化）
3) 使用 Poisson-disk / Uniform 方式在表面采样得到高质量点云
4) （可选）缩放单位：常见 FreeCAD 导出为 mm，可用 --scale 0.001 转为 m
5) 输出 PLY 点云（默认带法向）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未找到 open3d。请先安装：pip install open3d\n"
            f"原始错误: {e}"
        )


def _clean_mesh(mesh):
    """
    Open3D 对 STL 读入后，可能包含重复点/退化三角形；这里做常规清理。
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description="将 STL 转换为均匀采样的 PLY 点云（ICP 用）")
    parser.add_argument("--stl", required=True, help="输入 STL 路径（ascii/binary 均可）")
    parser.add_argument("--out", default="", help="输出 PLY 路径（默认与 stl 同名 .ply）")
    parser.add_argument("--points", type=int, default=50000, help="目标采样点数（建议 5k~200k）")
    parser.add_argument(
        "--method",
        choices=["poisson", "uniform"],
        default="poisson",
        help="采样方法：poisson=更均匀（推荐）；uniform=更快",
    )
    parser.add_argument(
        "--init-factor",
        type=int,
        default=5,
        help="Poisson-disk 初始采样倍率（越大越均匀但越慢），仅 method=poisson 生效",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="整体缩放系数。常见：FreeCAD 导出为 mm，转 m 用 0.001",
    )
    parser.add_argument("--no-normals", action="store_true", help="不写入法向（默认会估计并写入法向）")
    parser.add_argument("--voxel-down", type=float, default=0.0, help="可选：对点云做体素下采样（单位同 scale 后）")
    args = parser.parse_args()

    o3d = _import_open3d()

    stl_path = Path(args.stl).expanduser().resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"未找到 STL：{stl_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else stl_path.with_suffix(".ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"读取 STL: {stl_path}")
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh is None or (not mesh.has_triangles()):
        raise RuntimeError("读取 STL 失败：mesh 为空或不包含三角形（请检查 STL 是否有效）")

    mesh = _clean_mesh(mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if float(args.scale) != 1.0:
        mesh.scale(float(args.scale), center=(0.0, 0.0, 0.0))
        print(f"已缩放 mesh：scale={args.scale}")

    n = int(args.points)
    if n <= 0:
        raise ValueError("--points 必须 > 0")

    print(f"表面采样：method={args.method} points={n}")
    if args.method == "poisson":
        init_factor = max(1, int(args.init_factor))
        pcd = mesh.sample_points_poisson_disk(number_of_points=n, init_factor=init_factor)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=n)

    if float(args.voxel_down) > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_down))
        print(f"已体素下采样：voxel={args.voxel_down} -> points={np.asarray(pcd.points).shape[0]}")

    if not args.no_normals:
        # ICP 不强依赖 normals，但很多后续（FPFH/点到面 ICP）会用到
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        pcd.normalize_normals()

    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False, print_progress=False)
    if not ok:
        raise RuntimeError(f"写入失败：{out_path}")

    print(f"✓ 输出 PLY 点云: {out_path}")
    print(f"  点数: {np.asarray(pcd.points).shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


