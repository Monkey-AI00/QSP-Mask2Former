"""Pose backend base definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoseResult:
    ok: bool
    T_base_pregrasp: Any = None
    T_base_grasp: Any = None
    T_grasp_base: Any = None
    T_region_cam: Any = None
    T_grasp_cam: Any = None
    quality: dict = field(default_factory=dict)
    source: str = ""


class PoseBackendBase:
    def estimate_pose_and_grasp(self, *args, **kwargs):
        raise NotImplementedError


__all__ = ["PoseResult", "PoseBackendBase"]

