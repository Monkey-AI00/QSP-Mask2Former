"""Default constants shared by split modules."""

from __future__ import annotations

import numpy as np

_DEFAULT_T_CAM_TO_FLANGE = np.array(
    [
        [0.992923, 0.004783, -0.118660, 0.023239],
        [-0.003195, 0.999903, 0.013572, -0.112770],
        [0.118713, -0.013096, 0.992842, 0.118154],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

__all__ = ["_DEFAULT_T_CAM_TO_FLANGE"]

