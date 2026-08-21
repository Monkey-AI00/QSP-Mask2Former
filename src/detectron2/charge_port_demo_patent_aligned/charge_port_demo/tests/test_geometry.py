import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geometry  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_transform_round_trip(self):
        self.assertTrue(hasattr(geometry, "make_transform"))
        self.assertTrue(hasattr(geometry, "invert_transform"))
        self.assertTrue(hasattr(geometry, "transform_point"))
        transform = geometry.make_transform(np.eye(3), [1.0, -2.0, 0.5])
        point = np.array([0.2, 0.3, 1.1])
        moved = geometry.transform_point(transform, point)
        restored = geometry.transform_point(geometry.invert_transform(transform), moved)
        np.testing.assert_allclose(restored, point, atol=1e-9)

    def test_rejects_non_homogeneous_last_row(self):
        self.assertTrue(hasattr(geometry, "validate_transform"))
        bad = np.eye(4)
        bad[3, 3] = 2.0
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            geometry.validate_transform(bad, "bad")

    def test_pixel_camera_projection_round_trip(self):
        self.assertTrue(hasattr(geometry, "project_camera"))
        intrinsics = {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0}
        point = geometry.pixel_to_camera(730.0, 405.0, 2.0, intrinsics)
        projected = geometry.project_camera(point, intrinsics)
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[0], 730.0, places=9)
        self.assertAlmostEqual(projected[1], 405.0, places=9)


if __name__ == "__main__":
    unittest.main()
