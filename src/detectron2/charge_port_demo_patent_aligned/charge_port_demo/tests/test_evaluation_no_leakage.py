import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import evaluation
except ImportError:
    evaluation = None

from main import run_demo  # noqa: E402


class EvaluationNoLeakageTests(unittest.TestCase):
    def test_evaluation_computes_pixel_and_3d_error(self):
        self.assertIsNotNone(evaluation, "evaluation module must exist")
        result = evaluation.evaluate_prediction(
            {"port_2d": [13.0, 14.0], "port_3d": [1.0, 2.0, 4.0]},
            {"port_2d": [10.0, 10.0], "port_3d_robot": [1.0, 2.0, 3.0]},
        )
        self.assertAlmostEqual(result["pixel_error"], 5.0)
        self.assertAlmostEqual(result["position_error_3d"], 1.0)

    def test_changing_ground_truth_does_not_change_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                first = run_demo("model3", "view_far", output_root=Path(temp_dir) / "first")
                second = run_demo(
                    "model3",
                    "view_far",
                    output_root=Path(temp_dir) / "second",
                    ground_truth_override={"port_2d": [10.0, 20.0], "port_3d_robot": [9.0, 9.0, 9.0]},
                )
            except TypeError as exc:
                self.fail(f"run_demo must isolate ground truth and support test output roots: {exc}")
        self.assertEqual(second["final_prediction"], first["final_prediction"])
        self.assertNotEqual(second["evaluation"], first["evaluation"])


if __name__ == "__main__":
    unittest.main()
