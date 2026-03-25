from typing import Any


_DISTANCE_SCORE = {"close": 3.0, "mid": 2.0, "far": 1.0}
_ANGLE_SCORE = {"front": 3.0, "side": 2.0, "oblique": 1.0}


def rank_candidate_views(car_id: str, current_view_id: str, all_views: list[str], load_view_fn) -> list[dict[str, Any]]:
    ranked = []
    for view_id in all_views:
        if view_id == current_view_id:
            continue
        sample = load_view_fn(car_id, view_id)
        ann = sample["kpts"]
        quality = ann.get("view_quality", {})
        distance = str(quality.get("distance", "far"))
        angle = str(quality.get("angle", "oblique"))
        known_kpts = len(ann.get("keypoints_2d", {}))
        avg_score = sum(ann.get("keypoint_scores", {}).values()) / max(len(ann.get("keypoint_scores", {})), 1)
        score = _DISTANCE_SCORE.get(distance, 0.0) + _ANGLE_SCORE.get(angle, 0.0) + known_kpts * 0.1 + avg_score
        ranked.append({"view_id": view_id, "score": float(score), "distance": distance, "angle": angle})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def select_next_best_view(car_id: str, current_view_id: str, all_views: list[str], load_view_fn) -> dict[str, Any]:
    ranked = rank_candidate_views(car_id, current_view_id, all_views, load_view_fn)
    if ranked:
        best = ranked[0]
        return {
            "need_explore": True,
            "next_view_id": best["view_id"],
            "mode": "explore",
            "reason": "final_conf below threshold",
            "view_scores": ranked,
        }
    return {
        "need_explore": False,
        "next_view_id": current_view_id,
        "mode": "explore",
        "reason": "no better candidate",
        "view_scores": [],
    }
