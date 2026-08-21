from __future__ import annotations

from typing import Any

import numpy as np

from geometry import invert_transform, project_camera, transform_point
from validation import validate_transform


DEFAULT_OBSERVABILITY_WEIGHTS = {"coverage": 1.0 / 3.0, "angle": 1.0 / 3.0, "occlusion": 1.0 / 3.0}
DEFAULT_CONFIDENCE_WEIGHTS = {"classification": 0.4, "localization": 0.6}


def generate_candidate_poses(current_pose: dict[str, float], grid_spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_x = float(current_pose.get("x", 0.0))
    current_y = float(current_pose.get("y", 0.0))
    current_yaw = float(current_pose.get("yaw_deg", 0.0))
    candidates = []
    for item in grid_spec:
        candidate = dict(item)
        candidate["robot_pose"] = {
            "x": current_x + float(item.get("dx", 0.0)),
            "y": current_y + float(item.get("dy", 0.0)),
            "yaw_deg": current_yaw + float(item.get("dyaw_deg", 0.0)),
        }
        candidates.append(candidate)
    return candidates


def _normalized_weights(weights: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    values = {name: float(weights[name]) for name in names}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("planning weights must be finite and non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("planning weights must have a positive sum")
    return {name: value / total for name, value in values.items()}


def predict_observability(
    candidate_pose: dict[str, Any],
    parts_robot: dict[str, list[float]],
    part_normals: dict[str, list[float]],
    camera: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if not parts_robot:
        return {"coverage": 0.0, "angle": 0.0, "occlusion": 0.0, "localization_confidence": 0.0}
    camera_to_robot = validate_transform(candidate_pose["camera_to_robot"], "candidate camera_to_robot")
    robot_to_camera = invert_transform(camera_to_robot)
    intrinsics = camera["intrinsics"]
    image_width = int(camera["image_width"])
    image_height = int(camera["image_height"])
    occlusion_prior = float(np.clip(candidate_pose.get("occlusion_factor", 1.0), 0.0, 1.0))
    observability_weights = _normalized_weights(
        weights or DEFAULT_OBSERVABILITY_WEIGHTS,
        ("coverage", "angle", "occlusion"),
    )

    coverage_scores = []
    angle_scores = []
    occlusion_scores = []
    camera_position_robot = camera_to_robot[:3, 3]
    for name, point_values in parts_robot.items():
        point_robot = np.asarray(point_values, dtype=float)
        point_camera = transform_point(robot_to_camera, point_robot)
        pixel = project_camera(point_camera, intrinsics)
        in_view = pixel is not None and 0.0 <= pixel[0] < image_width and 0.0 <= pixel[1] < image_height
        coverage = 1.0 if in_view else 0.0

        normal = np.asarray(part_normals.get(name, [0.0, 0.0, -1.0]), dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        to_camera = camera_position_robot - point_robot
        view_norm = float(np.linalg.norm(to_camera))
        if normal_norm <= 1e-12 or view_norm <= 1e-12:
            angle = 0.0
        else:
            angle = float(np.clip(np.dot(normal / normal_norm, to_camera / view_norm), 0.0, 1.0))

        coverage_scores.append(coverage)
        angle_scores.append(angle if in_view else 0.0)
        occlusion_scores.append(occlusion_prior if in_view else 0.0)

    coverage_mean = float(np.mean(coverage_scores))
    angle_mean = float(np.mean(angle_scores))
    occlusion_mean = float(np.mean(occlusion_scores))
    localization = (
        observability_weights["coverage"] * coverage_mean
        + observability_weights["angle"] * angle_mean
        + observability_weights["occlusion"] * occlusion_mean
    )
    return {
        "coverage": coverage_mean,
        "angle": angle_mean,
        "occlusion": occlusion_mean,
        "localization_confidence": float(localization),
    }


def rank_candidate_poses(
    *,
    current_view_id: str,
    current_confidence: float,
    cls_stability: float,
    candidate_poses: list[dict[str, Any]],
    parts_robot: dict[str, list[float]],
    part_normals: dict[str, list[float]],
    camera: dict[str, Any],
    visited_views: list[str] | set[str] | None = None,
    observability_weights: dict[str, float] | None = None,
    confidence_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    visited = set(visited_views or []) | {current_view_id}
    conf_weights = _normalized_weights(
        confidence_weights or DEFAULT_CONFIDENCE_WEIGHTS,
        ("classification", "localization"),
    )
    ranked = []
    for candidate in candidate_poses:
        view_id = str(candidate["view_id"])
        if view_id in visited:
            continue
        observability = predict_observability(candidate, parts_robot, part_normals, camera, observability_weights)
        predicted_confidence = (
            conf_weights["classification"] * float(np.clip(cls_stability, 0.0, 1.0))
            + conf_weights["localization"] * observability["localization_confidence"]
        )
        ranked.append(
            {
                "view_id": view_id,
                **observability,
                "classification_stability": float(np.clip(cls_stability, 0.0, 1.0)),
                "predicted_confidence": float(predicted_confidence),
                "confidence_gain": float(predicted_confidence - current_confidence),
            }
        )
    ranked.sort(key=lambda row: (-row["confidence_gain"], -row["predicted_confidence"], row["view_id"]))
    return ranked


def select_next_best_view(
    *,
    current_view_id: str,
    current_confidence: float,
    cls_stability: float,
    candidate_poses: list[dict[str, Any]],
    parts_robot: dict[str, list[float]],
    part_normals: dict[str, list[float]],
    camera: dict[str, Any],
    visited_views: list[str] | set[str] | None = None,
    observability_weights: dict[str, float] | None = None,
    confidence_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    ranked = rank_candidate_poses(
        current_view_id=current_view_id,
        current_confidence=current_confidence,
        cls_stability=cls_stability,
        candidate_poses=candidate_poses,
        parts_robot=parts_robot,
        part_normals=part_normals,
        camera=camera,
        visited_views=visited_views,
        observability_weights=observability_weights,
        confidence_weights=confidence_weights,
    )
    if not ranked or ranked[0]["confidence_gain"] <= 0.0:
        return {
            "need_explore": False,
            "next_view_id": current_view_id,
            "mode": "explore",
            "reason": "no_positive_confidence_gain",
            "confidence_gain": 0.0,
            "candidate_scores": ranked,
        }
    best = ranked[0]
    return {
        "need_explore": True,
        "next_view_id": best["view_id"],
        "mode": "explore",
        "reason": "maximum_predicted_confidence_gain",
        "confidence_gain": best["confidence_gain"],
        "predicted_confidence": best["predicted_confidence"],
        "candidate_scores": ranked,
    }
