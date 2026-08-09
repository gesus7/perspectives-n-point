"""PoseTracker — per-frame detect/predict orchestration around a PoseFilter.

Sits between the video loop and PoseFilter. Each frame the loop calls observe() with
the fresh PnP pose for the locked target, or (None, None) when YOLO produced no usable
pose this frame. PoseTracker decides:

  * fresh pose        -> correct the filter, return the measured pose (predicted=False)
  * miss, still alive -> coast on the filter's prediction (predicted=True)
  * miss, gave up     -> return None (target dropped; caller waits for re-detection)

Returned dict: {'R','t','predicted','lost_count'}. It carries no drawing concern; the
caller renders it (in a distinct colour when predicted=True).
"""
from __future__ import annotations

import numpy as np

from infer.pose_filter import PoseFilter


class PoseTracker:
    def __init__(self, max_lost: int = 15, vel_alpha: float = 0.5,
                 history_len: int = 30) -> None:
        self._filter = PoseFilter(max_lost=max_lost, vel_alpha=vel_alpha,
                                  history_len=history_len)
        self._seen = False

    def observe(self, R: np.ndarray | None, t: np.ndarray | None) -> dict | None:
        """Feed this frame's pose (or None on detection failure). See module docstring."""
        if R is not None and t is not None:
            self._filter.update(R, t)
            self._seen = True
            Rc, tc = self._filter.current()
            return {"R": Rc, "t": tc, "predicted": False,
                    "lost_count": self._filter.lost_count}

        # Detection failed this frame.
        if not self._seen or not self._filter.alive:
            return None
        Rp, tp = self._filter.predict()
        if not self._filter.alive:
            # This prediction pushed us past max_lost — drop the target.
            return None
        return {"R": Rp, "t": tp, "predicted": True,
                "lost_count": self._filter.lost_count}

    @property
    def alive(self) -> bool:
        return self._filter.alive

    @property
    def lost_count(self) -> int:
        return self._filter.lost_count
