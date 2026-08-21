import numpy as np

from validation import validate_probabilities


def calc_cls_confidence(cls_probs: dict[str, float]) -> float:
    _, probs = validate_probabilities(cls_probs)
    if probs.size == 0:
        return 0.0
    if probs.size == 1:
        return 1.0
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return float(np.clip(1.0 - entropy / np.log(probs.size), 0.0, 1.0))


def calc_kpt_confidence(keypoint_scores: dict[str, float]) -> float:
    scores = list(keypoint_scores.values())
    return float(sum(scores) / max(len(scores), 1))


def calc_heatmap_confidence(heatmaps: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    peaks = {str(name): float(np.max(np.asarray(heatmap, dtype=float))) for name, heatmap in heatmaps.items()}
    if not peaks:
        return 0.0, {}
    return float(np.mean(list(peaks.values()))), peaks


def calc_final_confidence(cls_conf: float, kpt_conf: float, alpha: float = 0.4, beta: float = 0.6) -> float:
    return float(alpha * cls_conf + beta * kpt_conf)


def compute_confidence(perception_result: dict, alpha: float = 0.4, beta: float = 0.6) -> dict[str, float]:
    cls_conf = calc_cls_confidence(perception_result["cls_probs"])
    if "heatmaps" in perception_result:
        kpt_conf, heatmap_peaks = calc_heatmap_confidence(perception_result["heatmaps"])
    else:
        kpt_conf = calc_kpt_confidence(perception_result["keypoint_scores"])
        heatmap_peaks = dict(perception_result["keypoint_scores"])
    final_conf = calc_final_confidence(cls_conf, kpt_conf, alpha, beta)
    return {
        "cls_conf": cls_conf,
        "kpt_conf": kpt_conf,
        "final_conf": final_conf,
        "heatmap_peaks": heatmap_peaks,
    }
