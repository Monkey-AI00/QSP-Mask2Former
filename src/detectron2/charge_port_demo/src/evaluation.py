from typing import Any

import numpy as np


def _distance(left: Any, right: Any, expected_size: int) -> float | None:
    if left is None or right is None:
        return None
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != (expected_size,) or right_values.shape != (expected_size,):
        raise ValueError(f"evaluation coordinates must contain {expected_size} finite values")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("evaluation coordinates must be finite")
    return float(np.linalg.norm(left_values - right_values))


def evaluate_prediction(prediction: dict[str, Any], ground_truth: dict[str, Any] | None) -> dict[str, Any]:
    reference = dict(ground_truth or {})
    return {
        "pixel_error": _distance(prediction.get("port_2d"), reference.get("port_2d"), 2),
        "position_error_3d": _distance(prediction.get("port_3d"), reference.get("port_3d_robot"), 3),
        "reference_source": reference.get("source", "simulation_reference" if reference else None),
        "used_for_inference": False,
    }
