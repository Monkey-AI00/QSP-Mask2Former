from typing import Any

from geometry import pixel_to_camera, sample_depth
from semantic_graph import SemanticGraph


def infer_port_2d(perception_result: dict[str, Any], graph: SemanticGraph) -> dict[str, Any]:
    true_port_2d = perception_result.get("true_port_2d")
    if true_port_2d is not None:
        return {
            "port_2d": [float(true_port_2d[0]), float(true_port_2d[1])],
            "num_support_nodes": 1,
            "support_nodes": ["true_port_2d"],
            "method": "direct_annotation",
        }

    keypoints = perception_result["keypoints_2d"]
    scores = perception_result["keypoint_scores"]
    weighted_x = 0.0
    weighted_y = 0.0
    total_w = 0.0
    support_nodes: list[str] = []

    for node_name, xy in keypoints.items():
        relation = graph.get_port_relation(node_name)
        if "to_charge_port_2d" not in relation:
            continue
        dx, dy = relation["to_charge_port_2d"]
        weight = float(scores.get(node_name, 0.0))
        weighted_x += (float(xy[0]) + float(dx)) * weight
        weighted_y += (float(xy[1]) + float(dy)) * weight
        total_w += weight
        support_nodes.append(node_name)

    if total_w <= 0.0:
        return {"port_2d": None, "num_support_nodes": 0, "support_nodes": []}

    return {
        "port_2d": [weighted_x / total_w, weighted_y / total_w],
        "num_support_nodes": len(support_nodes),
        "support_nodes": support_nodes,
        "method": "graph_weighted_average",
    }


def infer_port_3d(
    perception_result: dict[str, Any],
    sample: dict[str, Any],
    graph: SemanticGraph,
    intrinsics: dict[str, Any],
) -> dict[str, Any]:
    result_2d = infer_port_2d(perception_result, graph)
    if result_2d.get("port_2d") is not None and result_2d.get("method") == "direct_annotation":
        u, v = result_2d["port_2d"]
        depth = sample_depth(sample["depth"], u, v)
        result_2d["port_3d"] = pixel_to_camera(u, v, depth, intrinsics)
        return result_2d

    keypoints = perception_result["keypoints_2d"]
    scores = perception_result["keypoint_scores"]
    depth_map = sample["depth"]
    weighted = [0.0, 0.0, 0.0]
    total_w = 0.0

    for node_name, xy in keypoints.items():
        relation = graph.get_port_relation(node_name)
        if "to_charge_port_3d" not in relation:
            continue
        depth = sample_depth(depth_map, xy[0], xy[1])
        point_cam = pixel_to_camera(xy[0], xy[1], depth, intrinsics)
        delta = relation["to_charge_port_3d"]
        candidate = [point_cam[0] + delta[0], point_cam[1] + delta[1], point_cam[2] + delta[2]]
        weight = float(scores.get(node_name, 0.0))
        weighted[0] += candidate[0] * weight
        weighted[1] += candidate[1] * weight
        weighted[2] += candidate[2] * weight
        total_w += weight

    port_3d = None
    if total_w > 0.0:
        port_3d = [weighted[0] / total_w, weighted[1] / total_w, weighted[2] / total_w]

    result_2d["port_3d"] = port_3d
    return result_2d


def fuse_port_estimates(port_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not port_candidates:
        return {"port_2d": None, "port_3d": None, "num_support_nodes": 0, "support_nodes": []}
    return port_candidates[0]
