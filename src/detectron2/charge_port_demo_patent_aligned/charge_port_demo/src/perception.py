from typing import Any

from multimodal import extract_multimodal_features, global_semantic_head, local_discriminative_head


def _replay_output(sample: dict[str, Any]) -> dict[str, Any]:
    if "replay_output" in sample:
        return dict(sample["replay_output"])
    if "kpts" in sample:
        return dict(sample["kpts"])
    raise ValueError("sample is missing replay_output")


def predict_vehicle_id(sample: dict[str, Any]) -> tuple[str, dict[str, float]]:
    ann = _replay_output(sample)
    cls_probs = {str(k): float(v) for k, v in ann.get("cls_probs", {}).items()}
    vehicle_id = max(cls_probs, key=cls_probs.get) if cls_probs else sample["car_id"]
    return vehicle_id, cls_probs


def predict_keypoints(sample: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, float]]:
    ann = _replay_output(sample)
    keypoints = {str(k): list(v) for k, v in ann.get("keypoints_2d", {}).items()}
    scores = {str(k): float(v) for k, v in ann.get("keypoint_scores", {}).items()}
    return keypoints, scores


def run_perception(sample: dict[str, Any]) -> dict[str, Any]:
    _, replay_cls_probs = predict_vehicle_id(sample)
    keypoints_2d, keypoint_scores = predict_keypoints(sample)
    point_cloud = sample.get("point_cloud")
    if point_cloud is None:
        raise ValueError("sample is missing point_cloud")
    features = extract_multimodal_features(sample["rgb"], sample["depth"], point_cloud)
    cls_probs = global_semantic_head(features["fused_feature"], replay_cls_probs)
    vehicle_id = max(cls_probs, key=cls_probs.get) if cls_probs else sample["car_id"]
    heatmaps = local_discriminative_head(sample["rgb"].shape[:2], keypoints_2d, keypoint_scores)
    return {
        "vehicle_id": vehicle_id,
        "cls_probs": cls_probs,
        "keypoints_2d": keypoints_2d,
        "keypoint_scores": keypoint_scores,
        "heatmaps": heatmaps,
        "feature_summary": {
            "fused_feature": features["fused_feature"].tolist(),
            "attention_weights": features["attention_weights"].tolist(),
            "layer_shapes": features["layer_shapes"],
            "architecture_trace": features["architecture_trace"],
            "point_count": int(len(point_cloud)),
            "mode": "deterministic_multimodal_replay",
        },
    }
