import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from active_view import select_next_best_view
from confidence import compute_confidence
from config import ALPHA, BETA, CONF_THRESHOLD, DEFAULT_CAR_ID, DEFAULT_INIT_VIEW, MAX_EXPLORATION_STEPS, OUTPUT_DIR, RANDOM_SEED
from evaluation import evaluate_prediction
from geometry import pixel_to_camera, sample_depth, transform_point
from infer_port import infer_port_pose
from load_data import list_views, load_graph, load_meta, load_view
from perception import run_perception
from planner import build_explore_action, build_work_action
from semantic_graph import SemanticGraph
from visualize import build_video_from_frames, draw_action_panel, draw_confidence_panel, draw_keypoints, draw_port_result, save_frame


COORDINATE_FRAMES = {
    "keypoints_2d": "image_pixel",
    "depth_backprojection": "camera_optical",
    "semantic_graph_parts": "robot_base",
    "port_pose": "robot_base",
}


def _make_output_dirs(output_root: Path) -> tuple[Path, Path]:
    frame_dir = output_root / "frames"
    log_dir = output_root / "logs"
    frame_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir, log_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parts_in_robot(perception_result: dict[str, Any], sample: dict[str, Any], graph: SemanticGraph) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    intrinsics = sample["camera"]["intrinsics"]
    camera_to_robot = sample["camera"]["camera_to_robot"]
    parts_robot: dict[str, list[float]] = {}
    part_normals: dict[str, list[float]] = {}
    for name, xy in perception_result["keypoints_2d"].items():
        if name not in graph.relations:
            continue
        depth = sample_depth(sample["depth"], float(xy[0]), float(xy[1]))
        point_camera = pixel_to_camera(float(xy[0]), float(xy[1]), depth, intrinsics)
        parts_robot[name] = transform_point(camera_to_robot, point_camera).tolist()
        part_normals[name] = graph.get_part_normal(name).tolist()
    return parts_robot, part_normals


def _candidate_poses(meta: dict[str, Any], all_views: list[str]) -> list[dict[str, Any]]:
    poses = dict(meta.get("view_poses", {}))
    result = []
    for view_id in all_views:
        item = dict(poses.get(view_id, {}))
        item["view_id"] = view_id
        item.setdefault("camera_to_robot", np.eye(4).tolist())
        item.setdefault("robot_pose", {"x": 0.0, "y": 0.0, "yaw_deg": 0.0})
        item.setdefault("occlusion_factor", 1.0)
        result.append(item)
    return result


def _select_reference(sample: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if override is not None:
        reference = dict(override)
        reference.setdefault("source", "test_override")
        return reference
    return dict(sample.get("ground_truth", {}))


def run_demo(
    car_id: str,
    init_view_id: str,
    *,
    output_root: Path | None = None,
    ground_truth_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root) if output_root is not None else OUTPUT_DIR / f"run_{car_id}_{init_view_id}"
    frame_dir, log_dir = _make_output_dirs(root)
    meta = load_meta(car_id)
    graph = SemanticGraph(load_graph(car_id))
    all_views = list_views(car_id)
    if init_view_id not in all_views:
        raise ValueError(f"initial view {init_view_id!r} is not available for {car_id}")

    current_view_id = init_view_id
    visited: list[str] = []
    final_prediction: dict[str, Any] | None = None
    final_sample: dict[str, Any] | None = None
    run_log: dict[str, Any] = {
        "status": "running",
        "car_id": car_id,
        "init_view_id": init_view_id,
        "seed": RANDOM_SEED,
        "prototype_scope": "reproducible_simulation_engineering_prototype",
        "meta": meta,
        "steps": [],
    }

    for step_idx in range(1, MAX_EXPLORATION_STEPS + 1):
        visited.append(current_view_id)
        sample = load_view(car_id, current_view_id)
        perception_result = run_perception(sample)
        conf_result = compute_confidence(perception_result, alpha=ALPHA, beta=BETA)
        parts_robot, part_normals = _parts_in_robot(perception_result, sample, graph)

        frame = draw_keypoints(sample["rgb"].copy(), perception_result["keypoints_2d"], perception_result["keypoint_scores"])
        frame = draw_confidence_panel(frame, conf_result, perception_result)
        common_step = {
            "step_index": step_idx,
            "view_id": current_view_id,
            "confidence": conf_result,
            "feature_summary": perception_result["feature_summary"],
            "data_provenance": sample["data_provenance"],
            "coordinate_frames": COORDINATE_FRAMES,
            "transforms": {
                "camera_to_robot": sample["camera"]["camera_to_robot"],
                "candidate_camera_to_robot": {
                    pose["view_id"]: pose["camera_to_robot"] for pose in _candidate_poses(meta, all_views)
                },
            },
        }

        if conf_result["final_conf"] < CONF_THRESHOLD:
            explore_result = select_next_best_view(
                current_view_id=current_view_id,
                current_confidence=conf_result["final_conf"],
                cls_stability=conf_result["cls_conf"],
                candidate_poses=_candidate_poses(meta, all_views),
                parts_robot=parts_robot,
                part_normals=part_normals,
                camera=sample["camera"],
                visited_views=visited,
            )
            selected = next(
                (row for row in explore_result["candidate_scores"] if row["view_id"] == explore_result["next_view_id"]),
                {},
            )
            target_pose = next(
                (pose for pose in _candidate_poses(meta, all_views) if pose["view_id"] == explore_result["next_view_id"]),
                {},
            )
            action = build_explore_action(explore_result["next_view_id"], {**target_pose, **selected})
            frame = draw_action_panel(frame, action, title="explore")
            save_frame(frame, f"{step_idx:03d}_{current_view_id}_explore", frame_dir)
            run_log["steps"].append(
                {
                    **common_step,
                    "mode": "explore",
                    "active_view": explore_result,
                    "action": action,
                }
            )
            if not explore_result["need_explore"]:
                run_log["status"] = "stopped_no_positive_gain"
                final_sample = sample
                break
            current_view_id = explore_result["next_view_id"]
            continue

        final_prediction = infer_port_pose(perception_result, sample, graph)
        final_sample = sample
        action = build_work_action(final_prediction)
        frame = draw_port_result(frame, final_prediction)
        frame = draw_action_panel(frame, action, title="work")
        save_frame(frame, f"{step_idx:03d}_{current_view_id}_work", frame_dir)
        run_log["steps"].append(
            {
                **common_step,
                "mode": "work",
                "prediction": final_prediction,
                "action": action,
            }
        )
        run_log["status"] = "localized"
        break

    if run_log["status"] == "running":
        run_log["status"] = "stopped_step_limit"
    run_log["final_prediction"] = final_prediction or {
        "port_2d": None,
        "port_3d": None,
        "method": "not_localized",
        "coordinate_frame": "robot_base",
    }
    reference = _select_reference(final_sample or {}, ground_truth_override)
    run_log["evaluation"] = evaluate_prediction(run_log["final_prediction"], reference)
    run_log["evaluation"]["reference_present"] = bool(reference)
    animation_path = build_video_from_frames(frame_dir, root / "demo_video.mp4")
    run_log["artifacts"] = {
        "frames": str(frame_dir),
        "log": str(log_dir / f"run_{car_id}_{init_view_id}.json"),
        "animation": str(animation_path) if animation_path is not None else None,
    }

    serializable = _jsonable(run_log)
    with (log_dir / f"run_{car_id}_{init_view_id}.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic charging-port localization simulation prototype")
    parser.add_argument("--car_id", default=DEFAULT_CAR_ID)
    parser.add_argument("--init_view_id", default=DEFAULT_INIT_VIEW)
    parser.add_argument("--output_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(args.car_id, args.init_view_id, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
