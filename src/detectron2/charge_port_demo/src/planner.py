from typing import Any


def build_explore_action(next_view_id: str) -> dict[str, Any]:
    return {
        "target_view": next_view_id,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_yaw": 0.0,
        "mode": "explore",
    }


def build_work_action(port_result: dict[str, Any]) -> dict[str, Any]:
    port_3d = port_result.get("port_3d")
    if port_3d is not None:
        return {
            "target_x": float(port_3d[0]),
            "target_y": float(port_3d[1]),
            "target_z": float(port_3d[2]),
            "target_yaw": 0.0,
            "mode": "work",
        }
    port_2d = port_result.get("port_2d") or [0.0, 0.0]
    return {
        "target_u": float(port_2d[0]),
        "target_v": float(port_2d[1]),
        "target_yaw": 0.0,
        "mode": "work",
    }


def build_action(mode: str, **kwargs) -> dict[str, Any]:
    if mode == "explore":
        return build_explore_action(kwargs["next_view_id"])
    return build_work_action(kwargs["port_result"])
