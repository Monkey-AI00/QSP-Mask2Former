import sys
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import run_demo  # noqa: E402

try:
    from run_validation import run_all_cases  # noqa: E402
except ImportError:
    run_all_cases = None


class IntegrationTests(unittest.TestCase):
    def test_three_cases_finish_with_structured_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for car_id in ("car_A", "car_B", "model3"):
                with self.subTest(car_id=car_id):
                    try:
                        result = run_demo(car_id, "view_far", output_root=Path(temp_dir) / car_id)
                    except TypeError as exc:
                        self.fail(f"run_demo must accept an isolated output root: {exc}")
                    self.assertEqual(result["status"], "localized")
                    self.assertTrue(result["steps"])
                    self.assertTrue(all("data_provenance" in step for step in result["steps"]))
                    self.assertTrue(all("coordinate_frames" in step for step in result["steps"]))
                    self.assertNotEqual(result["final_prediction"].get("method"), "direct_annotation")

    def test_validation_summary_contains_all_cases(self):
        self.assertIsNotNone(run_all_cases, "run_validation module must exist")
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_all_cases(output_root=Path(temp_dir))
        self.assertEqual(set(summary["cases"]), {"car_A", "car_B", "model3"})
        self.assertEqual(summary["seed"], 20260727)
        self.assertTrue(all("log_path" in row for row in summary["cases"].values()))

    def test_evidence_values_come_from_latest_summary(self):
        summary_path = ROOT / "outputs" / "validation_summary.json"
        evidence_path = ROOT.parent / "研发佐证材料_专利对齐仿真版.md"
        self.assertTrue(evidence_path.exists(), "aligned evidence Markdown must exist")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evidence = evidence_path.read_text(encoding="utf-8")
        for car_id, case in summary["cases"].items():
            self.assertIn(car_id, evidence)
            self.assertIn(f'{case["final_confidence"]:.3f}', evidence)
            self.assertIn(case["prediction_method"], evidence)


if __name__ == "__main__":
    unittest.main()
