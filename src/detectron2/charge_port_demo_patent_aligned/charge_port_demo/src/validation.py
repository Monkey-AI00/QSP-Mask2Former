from typing import Any

import numpy as np


def validate_intrinsics(intrinsics: dict[str, Any]) -> dict[str, float]:
    required = ("fx", "fy", "cx", "cy")
    missing = [name for name in required if name not in intrinsics]
    if missing:
        raise ValueError(f"camera intrinsics missing: {', '.join(missing)}")
    result = {name: float(intrinsics[name]) for name in required}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("camera intrinsics must be finite")
    if result["fx"] <= 0.0 or result["fy"] <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    return result


def validate_probabilities(probabilities: dict[str, float]) -> tuple[list[str], np.ndarray]:
    labels = [str(label) for label in probabilities]
    values = np.asarray([probabilities[label] for label in probabilities], dtype=float)
    if values.size == 0:
        return labels, values
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("classification probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("classification probabilities must have a positive sum")
    return labels, values / total


def validate_transform(matrix: Any, name: str = "transform") -> np.ndarray:
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant must equal 1")
    return transform
