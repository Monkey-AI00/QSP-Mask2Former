import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import confidence  # noqa: E402
import perception  # noqa: E402


class PerceptionConfidenceTests(unittest.TestCase):
    def test_entropy_confidence_extremes(self):
        self.assertAlmostEqual(confidence.calc_cls_confidence({"a": 1.0, "b": 0.0}), 1.0)
        self.assertAlmostEqual(confidence.calc_cls_confidence({"a": 0.5, "b": 0.5}), 0.0)

    def test_heatmap_peak_confidence(self):
        self.assertTrue(hasattr(confidence, "calc_heatmap_confidence"))
        heatmaps = {
            "logo": np.array([[0.1, 0.8]], dtype=float),
            "lamp": np.array([[0.7, 0.2]], dtype=float),
        }
        score, peaks = confidence.calc_heatmap_confidence(heatmaps)
        self.assertAlmostEqual(score, 0.75)
        self.assertAlmostEqual(peaks["logo"], 0.8)
        self.assertAlmostEqual(peaks["lamp"], 0.7)

    def test_perception_never_returns_ground_truth(self):
        sample = {
            "car_id": "model3",
            "rgb": np.zeros((32, 48, 3), dtype=np.uint8),
            "depth": np.full((32, 48), 1200.0, dtype=np.float32),
            "point_cloud": np.array([[0.0, 0.0, 1.2], [0.1, 0.0, 1.2]], dtype=float),
            "replay_output": {
                "vehicle_id": "model3",
                "cls_probs": {"model3": 0.8, "car_A": 0.2},
                "keypoints_2d": {"logo": [24.0, 16.0]},
                "keypoint_scores": {"logo": 0.9},
                "true_port_2d": [40.0, 20.0],
            },
            "ground_truth": {"port_2d": [40.0, 20.0]},
        }
        try:
            result = perception.run_perception(sample)
        except KeyError as exc:
            self.fail(f"run_perception must consume replay_output without legacy kpts: {exc}")
        self.assertNotIn("ground_truth", result)
        self.assertNotIn("true_port_2d", result)
        self.assertEqual(result["vehicle_id"], "model3")
        self.assertEqual(set(result["heatmaps"]), {"logo"})
        self.assertIn("fused_feature", result["feature_summary"])
        trace = result["feature_summary"]["architecture_trace"]
        self.assertEqual(trace["image_branch"], ["conv1", "pool1", "conv2", "pool2"])
        self.assertEqual(trace["point_branch"], ["mlp1", "max_pool", "mlp2"])
        self.assertEqual(trace["fusion"], ["concatenate", "self_attention", "conv1x1"])
        self.assertEqual(trace["global_head"], ["global_average_pool", "fully_connected", "softmax"])
        self.assertEqual(trace["local_head"], ["upsample", "convolution", "sigmoid"])


if __name__ == "__main__":
    unittest.main()
