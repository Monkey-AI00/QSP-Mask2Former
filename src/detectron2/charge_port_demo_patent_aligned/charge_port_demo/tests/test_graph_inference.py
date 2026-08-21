import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import infer_port  # noqa: E402
from semantic_graph import SemanticGraph  # noqa: E402


def _relation(tx, ty, tz):
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = [tx, ty, tz]
    return {
        "part_normal": [0.0, 0.0, -1.0],
        "part_to_port_transform": transform.tolist(),
    }


class GraphInferenceTests(unittest.TestCase):
    def _graph_data(self):
        return {
            "car_id": "model3",
            "root_node": "model3",
            "observable_nodes": ["left_tail_light"],
            "inference_nodes": ["charge_port_center"],
            "relations": {"left_tail_light": _relation(0.12, 0.0, 0.03)},
        }

    def test_graph_returns_rigid_transform(self):
        graph = SemanticGraph(self._graph_data())
        self.assertTrue(hasattr(graph, "get_part_to_port_transform"))
        transform = graph.get_part_to_port_transform("left_tail_light")
        self.assertEqual(transform.shape, (4, 4))
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])

    def test_inference_uses_hand_eye_and_graph_transform(self):
        self.assertTrue(hasattr(infer_port, "infer_port_pose"))
        perception_result = {
            "vehicle_id": "model3",
            "keypoints_2d": {"left_tail_light": [2.0, 2.0]},
            "keypoint_scores": {"left_tail_light": 0.9},
            "heatmaps": {"left_tail_light": np.array([[0.1, 0.9]], dtype=float)},
        }
        sample = {
            "depth": np.full((5, 5), 2000.0, dtype=np.float32),
            "camera": {
                "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0},
                "camera_to_robot": np.eye(4).tolist(),
            },
            "ground_truth": {"port_2d": [99.0, 99.0]},
        }
        result = infer_port.infer_port_pose(perception_result, sample, SemanticGraph(self._graph_data()))
        self.assertEqual(result["method"], "graph_rigid_transform_fusion")
        self.assertEqual(result["coordinate_frame"], "robot_base")
        self.assertGreaterEqual(result["num_support_nodes"], 1)
        self.assertNotIn("true_port_2d", result["support_nodes"])
        np.testing.assert_allclose(result["port_3d"], [0.12, 0.0, 2.03], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
