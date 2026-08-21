from typing import Any


def build_explore_action(next_view_id: str, candidate_pose: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = dict(candidate_pose or {})
    robot_pose = dict(pose.get("robot_pose", {}))
    action = {
        "target_view": next_view_id,
        "target_x": float(robot_pose.get("x", 0.0)),
        "target_y": float(robot_pose.get("y", 0.0)),
        "target_yaw": float(robot_pose.get("yaw_deg", 0.0)),
        "mode": "explore",
        "coordinate_frame": "robot_base",
    }
    if "confidence_gain" in pose:
        action["predicted_confidence_gain"] = float(pose["confidence_gain"])
    return action


def build_work_action(port_result: dict[str, Any]) -> dict[str, Any]:
    port_3d = port_result.get("port_3d")
    if port_3d is not None:
        return {
            "target_x": float(port_3d[0]),
            "target_y": float(port_3d[1]),
            "target_z": float(port_3d[2]),
            "target_yaw": 0.0,
            "mode": "work",
            "coordinate_frame": "robot_base",
        }
    port_2d = port_result.get("port_2d") or [0.0, 0.0]
    return {
        "target_u": float(port_2d[0]),
        "target_v": float(port_2d[1]),
        "target_yaw": 0.0,
        "mode": "work",
        "coordinate_frame": "image_pixel",
    }


def build_action(mode: str, **kwargs) -> dict[str, Any]:
    if mode == "explore":
        return build_explore_action(kwargs["next_view_id"])
    return build_work_action(kwargs["port_result"])
