from typing import Any


def predict_vehicle_id(sample: dict[str, Any]) -> tuple[str, dict[str, float]]:
    ann = sample["kpts"]
    cls_probs = {str(k): float(v) for k, v in ann.get("cls_probs", {}).items()}
    vehicle_id = max(cls_probs, key=cls_probs.get) if cls_probs else sample["car_id"]
    return vehicle_id, cls_probs


def predict_keypoints(sample: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, float]]:
    ann = sample["kpts"]
    keypoints = {str(k): list(v) for k, v in ann.get("keypoints_2d", {}).items()}
    scores = {str(k): float(v) for k, v in ann.get("keypoint_scores", {}).items()}
    return keypoints, scores


def run_perception(sample: dict[str, Any]) -> dict[str, Any]:
    vehicle_id, cls_probs = predict_vehicle_id(sample)
    keypoints_2d, keypoint_scores = predict_keypoints(sample)
    ann = sample["kpts"]
    return {
        "vehicle_id": vehicle_id,
        "cls_probs": cls_probs,
        "keypoints_2d": keypoints_2d,
        "keypoint_scores": keypoint_scores,
        "true_port_2d": ann.get("true_port_2d"),
        "true_port_score": float(ann.get("true_port_score", 0.0)) if ann.get("true_port_2d") is not None else 0.0,
    }
