"""Lightweight SE(3) constant-velocity pose filter for bridging YOLO drop-outs.

One PoseFilter tracks a single locked target's 6DoF pose across frames. It is pure
math (numpy + cv2.Rodrigues for the SO(3) exp/log map) with no drawing or I/O, so it
is cheap and fully unit-testable.

Model
-----
State is the current camera-frame pose (R (3,3), t (3,)) plus a constant-velocity
estimate:
  * linear velocity  v      (metres / frame), applied additively to t
  * angular velocity omega  (rotation-vector / frame), applied on the left: R <- Exp(omega) @ R

Usage in the tracker
--------------------
  * YOLO detected + PnP ok  -> update(R, t): snap state to the measurement, refresh the
    EMA-smoothed velocity from the pose delta, reset the lost counter.
  * YOLO failed / occluded  -> predict(): advance the pose by one step of constant
    velocity and increment the lost counter. Keep coasting while `alive`.

`predict()` never mutates the measured history; only `update()` does. This keeps the
velocity estimate anchored to real observations, not to its own extrapolations.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def _log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) log map: rotation matrix -> rotation vector (axis * angle)."""
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3)


def _exp_so3(rvec: np.ndarray) -> np.ndarray:
    """SO(3) exp map: rotation vector -> rotation matrix."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


class PoseFilter:
    """Constant-velocity 6DoF pose predictor for one tracked target.

    Parameters
    ----------
    max_lost : int
        Predict for at most this many consecutive frames; after that `alive` is False
        and the caller should drop the target and wait for a fresh detection.
    vel_alpha : float
        EMA weight for velocity updates in [0, 1]. Higher = trust the newest motion
        more (snappier but noisier); lower = smoother but laggier.
    history_len : int
        Number of recent measured (R, t) poses to retain (bounded ring buffer).
    """

    def __init__(self, max_lost: int = 15, vel_alpha: float = 0.5,
                 history_len: int = 30) -> None:
        self.max_lost = int(max_lost)
        self.vel_alpha = float(vel_alpha)
        self._R: np.ndarray | None = None
        self._t: np.ndarray | None = None
        self._v = np.zeros(3)          # linear velocity (m / frame)
        self._omega = np.zeros(3)      # angular velocity (rotvec / frame)
        self._has_vel = False
        self.lost_count = 0
        self.history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=history_len)

    # ── measurement / correction ────────────────────────────────────────────────

    def update(self, R: np.ndarray, t: np.ndarray) -> None:
        """Correct the filter with a fresh PnP pose (camera frame)."""
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)

        if self._R is not None:
            # Measure the motion between the last accepted pose and this one, then
            # EMA-smooth it into the velocity estimate.
            v_meas = t - self._t
            omega_meas = _log_so3(R @ self._R.T)   # left-multiplicative delta
            if self._has_vel:
                a = self.vel_alpha
                self._v = a * v_meas + (1 - a) * self._v
                self._omega = a * omega_meas + (1 - a) * self._omega
            else:
                self._v = v_meas
                self._omega = omega_meas
                self._has_vel = True

        self._R = R
        self._t = t
        self.lost_count = 0
        self.history.append((R.copy(), t.copy()))

    # ── prediction ───────────────────────────────────────────────────────────────

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame of constant velocity and return the predicted (R, t).

        Increments `lost_count`. Must not be called before the first update().
        """
        if self._R is None:
            raise RuntimeError("PoseFilter.predict() called before any update()")
        self._t = self._t + self._v
        self._R = _exp_so3(self._omega) @ self._R
        self.lost_count += 1
        return self.current()

    # ── accessors ────────────────────────────────────────────────────────────────

    def current(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current (R, t) estimate."""
        return self._R, self._t

    @property
    def alive(self) -> bool:
        """False once we've predicted past max_lost consecutive frames."""
        return self.lost_count <= self.max_lost
