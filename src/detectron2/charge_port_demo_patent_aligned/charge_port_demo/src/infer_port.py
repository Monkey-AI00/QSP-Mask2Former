from __future__ import annotations

from typing import Any

import numpy as np

from geometry import invert_transform, make_transform, pixel_to_camera, project_camera, sample_depth, transform_point
from semantic_graph import SemanticGraph
from validation import validate_transform


def _heatmap_peaks(perception_result: dict[str, Any]) -> dict[str, float]:
    heatmaps = perception_result.get("heatmaps", {})
    if heatmaps:
        return {name: float(np.max(np.asarray(heatmap, dtype=float))) for name, heatmap in heatmaps.items()}
    return {name: float(score) for name, score in perception_result.get("keypoint_scores", {}).items()}


def infer_port_pose(
    perception_result: dict[str, Any],
    sample: dict[str, Any],
    graph: SemanticGraph,
) -> dict[str, Any]:
    camera = sample.get("camera", {})
    intrinsics = camera.get("intrinsics")
    if intrinsics is None:
        raise ValueError("sample camera is missing intrinsics")
    camera_to_robot = validate_transform(camera.get("camera_to_robot", np.eye(4)), "camera_to_robot")
    peaks = _heatmap_peaks(perception_result)
    candidates = []

    for part_name, xy in perception_result.get("keypoints_2d", {}).items():
        if part_name not in graph.relations:
            continue
        weight = float(peaks.get(part_name, 0.0))
        if weight <= 0.0:
            continue
        depth = sample_depth(sample["depth"], float(xy[0]), float(xy[1]))
        part_camera = pixel_to_camera(float(xy[0]), float(xy[1]), depth, intrinsics)
        part_robot = transform_point(camera_to_robot, part_camera)
        part_pose_robot = make_transform(np.eye(3), part_robot)
        port_pose_robot = part_pose_robot @ graph.get_part_to_port_transform(part_name)
        candidates.append(
            {
                "part_name": part_name,
                "weight": weight,
                "part_3d_camera": list(map(float, part_camera)),
                "part_3d_robot": part_robot.tolist(),
                "part_to_port_transform": graph.get_part_to_port_transform(part_name).tolist(),
                "port_candidate_3d_robot": port_pose_robot[:3, 3].tolist(),
            }
        )

    if not candidates:
        return {
            "port_2d": None,
            "port_3d": None,
            "port_pose_robot": None,
            "num_support_nodes": 0,
            "support_nodes": [],
            "method": "graph_rigid_transform_fusion",
            "coordinate_frame": "robot_base",
            "candidates": [],
        }

    weights = np.asarray([candidate["weight"] for candidate in candidates], dtype=float)
    weights /= weights.sum()
    positions = np.asarray([candidate["port_candidate_3d_robot"] for candidate in candidates], dtype=float)
    port_robot = np.sum(positions * weights[:, None], axis=0)
    port_pose_robot = make_transform(np.eye(3), port_robot)
    port_camera = transform_point(invert_transform(camera_to_robot), port_robot)
    pixel = project_camera(port_camera, intrinsics)
    return {
        "port_2d": list(pixel) if pixel is not None else None,
        "port_3d": port_robot.tolist(),
        "port_pose_robot": port_pose_robot.tolist(),
        "num_support_nodes": len(candidates),
        "support_nodes": [candidate["part_name"] for candidate in candidates],
        "method": "graph_rigid_transform_fusion",
        "coordinate_frame": "robot_base",
        "candidates": candidates,
    }


def infer_port_3d(
    perception_result: dict[str, Any],
    sample: dict[str, Any],
    graph: SemanticGraph,
    intrinsics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if intrinsics is not None:
        sample = dict(sample)
        camera = dict(sample.get("camera", {}))
        camera.setdefault("intrinsics", intrinsics)
        sample["camera"] = camera
    return infer_port_pose(perception_result, sample, graph)


def fuse_port_estimates(port_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not port_candidates:
        return {"port_2d": None, "port_3d": None, "num_support_nodes": 0, "support_nodes": []}
    weights = np.asarray([float(item.get("weight", 1.0)) for item in port_candidates], dtype=float)
    weights /= weights.sum()
    positions = np.asarray([item["port_3d"] for item in port_candidates], dtype=float)
    return {"port_3d": np.sum(positions * weights[:, None], axis=0).tolist(), "num_support_nodes": len(port_candidates)}
