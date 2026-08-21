import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import active_view  # noqa: E402


def _transform(x=0.0, y=0.0, z=0.0):
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = [x, y, z]
    return matrix.tolist()


class ActiveViewTests(unittest.TestCase):
    def _fixture(self):
        return {
            "current_view_id": "view_far",
            "current_confidence": 0.2,
            "cls_stability": 0.7,
            "candidate_poses": [
                {"view_id": "view_mid", "camera_to_robot": _transform(0.4, 0.0, 0.0), "occlusion_factor": 0.95},
                {"view_id": "view_close", "camera_to_robot": _transform(0.8, 0.0, 0.0), "occlusion_factor": 0.45},
            ],
            "parts_robot": {"logo": [0.0, 0.0, 2.0]},
            "part_normals": {"logo": [0.0, 0.0, -1.0]},
            "camera": {
                "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
                "image_width": 100,
                "image_height": 100,
            },
            "visited_views": ["view_far"],
        }

    def test_selects_largest_predicted_confidence_gain(self):
        try:
            result = active_view.select_next_best_view(**self._fixture())
        except TypeError as exc:
            self.fail(f"select_next_best_view must accept geometric planning inputs: {exc}")
        gains = [row["confidence_gain"] for row in result["candidate_scores"]]
        self.assertAlmostEqual(result["confidence_gain"], max(gains))
        self.assertEqual(result["next_view_id"], result["candidate_scores"][0]["view_id"])
        self.assertEqual(result["next_view_id"], "view_mid")

    def test_planner_does_not_accept_future_view_loader(self):
        parameters = inspect.signature(active_view.select_next_best_view).parameters
        self.assertNotIn("load_view_fn", parameters)

    def test_no_positive_gain_stops_exploration(self):
        fixture = self._fixture()
        fixture["current_confidence"] = 0.99
        try:
            result = active_view.select_next_best_view(**fixture)
        except TypeError as exc:
            self.fail(f"select_next_best_view must accept current confidence: {exc}")
        self.assertFalse(result["need_explore"])
        self.assertEqual(result["reason"], "no_positive_confidence_gain")


if __name__ == "__main__":
    unittest.main()
