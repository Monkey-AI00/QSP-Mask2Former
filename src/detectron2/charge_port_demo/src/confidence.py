def calc_cls_confidence(cls_probs: dict[str, float]) -> float:
    probs = list(cls_probs.values())
    return float(max(probs)) if probs else 0.0


def calc_kpt_confidence(keypoint_scores: dict[str, float]) -> float:
    scores = list(keypoint_scores.values())
    return float(sum(scores) / max(len(scores), 1))


def calc_final_confidence(cls_conf: float, kpt_conf: float, alpha: float = 0.4, beta: float = 0.6) -> float:
    return float(alpha * cls_conf + beta * kpt_conf)


def compute_confidence(perception_result: dict, alpha: float = 0.4, beta: float = 0.6) -> dict[str, float]:
    cls_conf = calc_cls_confidence(perception_result["cls_probs"])
    kpt_conf = calc_kpt_confidence(perception_result["keypoint_scores"])
    final_conf = calc_final_confidence(cls_conf, kpt_conf, alpha, beta)
    return {
        "cls_conf": cls_conf,
        "kpt_conf": kpt_conf,
        "final_conf": final_conf,
    }
