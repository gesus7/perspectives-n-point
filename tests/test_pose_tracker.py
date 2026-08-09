"""Unit tests for PoseTracker — ties PoseFilter to per-frame detect/predict flow.

Wraps one PoseFilter for a locked target and decides, each frame, whether to correct
from a fresh PnP pose, coast on prediction, or declare the target dropped. Pure logic,
no OpenCV drawing.
"""
import numpy as np

from infer.pose_tracker import PoseTracker


def _pose(z=10.0):
    return np.eye(3), np.array([0.0, 0.0, z])


# ── detection path ───────────────────────────────────────────────────────────────

def test_observe_returns_measured_and_marks_not_predicted():
    tr = PoseTracker()
    R, t = _pose()
    out = tr.observe(R, t)
    assert out is not None
    assert not out["predicted"]
    assert np.allclose(out["t"], t)


# ── prediction path ──────────────────────────────────────────────────────────────

def test_miss_after_observation_returns_predicted_pose():
    tr = PoseTracker(max_lost=5)
    tr.observe(*_pose(10.0))
    out = tr.observe(None, None)   # YOLO failed this frame
    assert out is not None
    assert out["predicted"]
    assert np.allclose(out["R"], np.eye(3))


def test_miss_before_any_observation_returns_none():
    tr = PoseTracker()
    assert tr.observe(None, None) is None


# ── give-up path ─────────────────────────────────────────────────────────────────

def test_target_dropped_after_max_lost():
    tr = PoseTracker(max_lost=2)
    tr.observe(*_pose())
    assert tr.observe(None, None)["predicted"]   # lost 1
    assert tr.observe(None, None)["predicted"]   # lost 2
    assert tr.observe(None, None) is None         # lost 3 > max_lost -> dropped
    assert not tr.alive


def test_reacquire_resets_after_misses():
    tr = PoseTracker(max_lost=5)
    tr.observe(*_pose())
    tr.observe(None, None)
    tr.observe(None, None)
    out = tr.observe(*_pose())    # YOLO back
    assert not out["predicted"]
    assert tr.lost_count == 0
