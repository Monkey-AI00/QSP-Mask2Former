from typing import Any

import numpy as np


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def classification_margin_loss(
    logits: np.ndarray | list[float],
    target_index: int,
    margin: float = 0.2,
    scale: float = 8.0,
) -> float:
    values = np.asarray(logits, dtype=float).copy()
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("classification logits must be a finite vector with at least two classes")
    if target_index < 0 or target_index >= values.size:
        raise ValueError("target class index is out of range")
    values[target_index] -= float(margin)
    scaled = values * float(scale)
    return float(_logsumexp(scaled) - scaled[target_index])


def heatmap_regression_loss(predicted: Any, target: Any) -> float:
    predicted_array = np.asarray(predicted, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if predicted_array.shape != target_array.shape or predicted_array.size == 0:
        raise ValueError("predicted and target heatmaps must have the same non-empty shape")
    if not np.all(np.isfinite(predicted_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("heatmaps must contain finite values")
    return float(np.mean((predicted_array - target_array) ** 2))


def uncertainty_distribution_loss(
    predicted_distribution: Any,
    target_distribution: Any,
    variance_weight: float = 1.0,
) -> dict[str, float]:
    predicted = np.asarray(predicted_distribution, dtype=float)
    target = np.asarray(target_distribution, dtype=float)
    if predicted.ndim != 1 or predicted.size < 2 or predicted.shape != target.shape:
        raise ValueError("uncertainty distributions must be equal-length vectors with at least two classes")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(target)) or np.any(predicted < 0) or np.any(target < 0):
        raise ValueError("uncertainty distributions must contain finite non-negative values")
    predicted /= predicted.sum()
    target /= target.sum()
    predicted = np.clip(predicted, 1e-9, 1.0)
    positive = target > 0.0
    kl_divergence = float(np.sum(target[positive] * np.log(target[positive] / predicted[positive])))
    variance_constraint = float(abs(np.var(predicted) - np.var(target)))
    return {
        "kl_divergence": kl_divergence,
        "variance_constraint": variance_constraint,
        "total": float(kl_divergence + float(variance_weight) * variance_constraint),
    }


def uncertainty_loss(predicted_confidence: float, correctness: float) -> float:
    """KL supervision plus a prediction-distribution variance constraint."""
    confidence = float(np.clip(predicted_confidence, 1e-9, 1.0 - 1e-9))
    target = float(correctness)
    if target < 0.0 or target > 1.0 or not np.isfinite(target):
        raise ValueError("correctness target must be in [0, 1]")
    return uncertainty_distribution_loss(
        [confidence, 1.0 - confidence],
        [target, 1.0 - target],
    )["total"]


def composite_loss(
    *,
    logits: Any,
    target_index: int,
    predicted_heatmap: Any,
    target_heatmap: Any,
    predicted_confidence: float,
    correctness: float,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    loss_weights = weights or {"classification": 1.0, "keypoint": 1.0, "uncertainty": 1.0}
    required = {"classification", "keypoint", "uncertainty"}
    if set(loss_weights) != required or any(float(loss_weights[name]) < 0.0 for name in required):
        raise ValueError("loss weights must define non-negative classification, keypoint and uncertainty values")
    classification = classification_margin_loss(logits, target_index)
    keypoint = heatmap_regression_loss(predicted_heatmap, target_heatmap)
    uncertainty = uncertainty_loss(predicted_confidence, correctness)
    total = (
        float(loss_weights["classification"]) * classification
        + float(loss_weights["keypoint"]) * keypoint
        + float(loss_weights["uncertainty"]) * uncertainty
    )
    return {
        "classification": classification,
        "keypoint": keypoint,
        "uncertainty": uncertainty,
        "total": float(total),
    }


def _average_calibration_loss(samples: list[dict[str, Any]], temperature: float) -> float:
    totals = []
    for sample in samples:
        result = composite_loss(
            logits=np.asarray(sample["logits"], dtype=float) / temperature,
            target_index=int(sample["target_index"]),
            predicted_heatmap=sample["predicted_heatmap"],
            target_heatmap=sample["target_heatmap"],
            predicted_confidence=float(sample["predicted_confidence"]),
            correctness=float(sample["correctness"]),
        )
        totals.append(result["total"])
    if not totals:
        raise ValueError("calibration requires at least one sample")
    return float(np.mean(totals))


def run_calibration(samples: list[dict[str, Any]], steps: int = 12, seed: int = 20260727) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("calibration steps must be positive")
    rng = np.random.default_rng(int(seed))
    temperature = float(1.0 + rng.uniform(-0.05, 0.05))
    current_loss = _average_calibration_loss(samples, temperature)
    loss_history = [current_loss]
    temperature_history = [temperature]
    for _ in range(steps):
        candidates = [max(0.2, temperature * 0.9), temperature, min(5.0, temperature * 1.1)]
        evaluated = [(candidate, _average_calibration_loss(samples, candidate)) for candidate in candidates]
        temperature, current_loss = min(evaluated, key=lambda item: (item[1], item[0]))
        temperature_history.append(float(temperature))
        loss_history.append(float(current_loss))
    return {
        "seed": int(seed),
        "steps": int(steps),
        "temperature": float(temperature),
        "temperature_history": temperature_history,
        "loss_history": loss_history,
    }
