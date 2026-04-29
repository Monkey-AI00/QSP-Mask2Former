"""Mech-Eye camera wrapper scaffold."""

from __future__ import annotations

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils


class MechEyeCameraWrapper:
    """
    相机封装骨架。
    当前无行为变化阶段由 legacy OnlineGraspPipeline 持有并调用 live_utils.Camera。
    """

    def __init__(self, args):
        self.args = args
        self.camera = live_utils.Camera()

    def capture(self):
        raise NotImplementedError("Capture flow is still handled by legacy pipeline in this phase.")

    @property
    def intrinsics(self):
        return None

    @property
    def K_3x3(self):
        return None


__all__ = ["MechEyeCameraWrapper"]

