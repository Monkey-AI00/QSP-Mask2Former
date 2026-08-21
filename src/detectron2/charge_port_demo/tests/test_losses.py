import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import losses
except ImportError:
    losses = None


class CompositeLossTests(unittest.TestCase):
    def test_uncertainty_distribution_loss_reports_kl_and_variance(self):
        self.assertTrue(hasattr(losses, "uncertainty_distribution_loss"))
        result = losses.uncertainty_distribution_loss([0.8, 0.2], [1.0, 0.0])
        self.assertEqual(set(result), {"kl_divergence", "variance_constraint", "total"})
        self.assertGreaterEqual(result["total"], result["kl_divergence"])

    def test_composite_loss_reports_weighted_components(self):
        self.assertIsNotNone(losses, "losses module must exist")
        result = losses.composite_loss(
            logits=np.array([2.0, 0.5]),
            target_index=0,
            predicted_heatmap=np.array([[0.2, 0.8]]),
            target_heatmap=np.array([[0.0, 1.0]]),
            predicted_confidence=0.8,
            correctness=1.0,
            weights={"classification": 1.0, "keypoint": 2.0, "uncertainty": 0.5},
        )
        self.assertEqual(set(result), {"classification", "keypoint", "uncertainty", "total"})
        expected = result["classification"] + 2.0 * result["keypoint"] + 0.5 * result["uncertainty"]
        self.assertAlmostEqual(result["total"], expected)

    def test_calibration_is_deterministic_and_non_increasing(self):
        self.assertIsNotNone(losses, "losses module must exist")
        samples = [
            {
                "logits": [2.0, 0.5, -0.3],
                "target_index": 0,
                "predicted_heatmap": [[0.1, 0.8]],
                "target_heatmap": [[0.0, 1.0]],
                "predicted_confidence": 0.75,
                "correctness": 1.0,
            },
            {
                "logits": [0.2, 1.2, 0.1],
                "target_index": 1,
                "predicted_heatmap": [[0.6, 0.2]],
                "target_heatmap": [[1.0, 0.0]],
                "predicted_confidence": 0.65,
                "correctness": 1.0,
            },
        ]
        first = losses.run_calibration(samples, steps=12, seed=20260727)
        second = losses.run_calibration(samples, steps=12, seed=20260727)
        self.assertEqual(first, second)
        self.assertLessEqual(first["loss_history"][-1], first["loss_history"][0])
        self.assertEqual(len(first["loss_history"]), 13)


if __name__ == "__main__":
    unittest.main()
