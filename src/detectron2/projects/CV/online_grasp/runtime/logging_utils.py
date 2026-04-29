"""Runtime logging utilities (no print behavior override)."""

from __future__ import annotations

import time


class Timer:
    def __init__(self):
        self._t0 = None

    def tic(self):
        self._t0 = time.time()

    def toc(self) -> float:
        if self._t0 is None:
            return 0.0
        return float(time.time() - self._t0)


__all__ = ["Timer"]

