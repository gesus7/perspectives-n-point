"""Unit tests for PoseFilter — SE(3) constant-velocity pose predictor.

Pure math (numpy + cv2.Rodrigues), no drawing or I/O.
"""
import numpy as np
import cv2
import pytest

from infer.pose_filter import PoseFilter


def _Rz(deg):
    """Rotation matrix about +Z by `deg` degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ── initialisation / correction ─────────────────────────────────────────────────

def test_update_sets_current_pose():
    f = PoseFilter()
    R, t = _Rz(10), np.array([1.0, 2.0, 30.0])
    f.update(R, t)
    Rc, tc = f.current()
    assert np.allclose(Rc, R)
    assert np.allclose(tc, t)


def test_update_resets_lost_count():
    f = PoseFilter(max_lost=5)
    f.update(np.eye(3), np.array([0.0, 0.0, 10.0]))
    f.predict()
    f.predict()
    assert f.lost_count == 2
    f.update(np.eye(3), np.array([0.0, 0.0, 10.0]))
    assert f.lost_count == 0


# ── constant-velocity translation ────────────────────────────────────────────────

def test_predict_zero_velocity_holds_pose():
    """With only one observation there is no measured velocity, so predict holds."""
    f = PoseFilter()
    t = np.array([0.0, 0.0, 20.0])
    f.update(np.eye(3), t)
    Rp, tp = f.predict()
    assert np.allclose(tp, t)
    assert np.allclose(Rp, np.eye(3))


def test_predict_extrapolates_constant_translation():
    f = PoseFilter()
    f.update(np.eye(3), np.array([0.0, 0.0, 30.0]))
    f.update(np.eye(3), np.array([1.0, 0.0, 30.0]))  # moved +1 in x
    _, tp = f.predict()
    assert np.allclose(tp, np.array([2.0, 0.0, 30.0]), atol=1e-6)


# ── constant-velocity rotation ───────────────────────────────────────────────────

def test_predict_extrapolates_constant_rotation():
    f = PoseFilter()
    t = np.array([0.0, 0.0, 30.0])
    f.update(_Rz(0), t)
    f.update(_Rz(5), t)   # rotated +5 deg about z
    Rp, _ = f.predict()
    assert np.allclose(Rp, _Rz(10), atol=1e-6)


# ── lost tracking / lifetime ─────────────────────────────────────────────────────

def test_predict_increments_lost_count():
    f = PoseFilter()
    f.update(np.eye(3), np.array([0.0, 0.0, 10.0]))
    assert f.lost_count == 0
    f.predict()
    assert f.lost_count == 1


def test_alive_until_max_lost_exceeded():
    f = PoseFilter(max_lost=3)
    f.update(np.eye(3), np.array([0.0, 0.0, 10.0]))
    for _ in range(3):
        f.predict()
        assert f.alive          # lost_count 1,2,3 still alive
    f.predict()                 # lost_count 4 > max_lost
    assert not f.alive


# ── history ──────────────────────────────────────────────────────────────────────

def test_history_records_updates_and_is_bounded():
    f = PoseFilter(history_len=3)
    for z in range(6):
        f.update(np.eye(3), np.array([0.0, 0.0, float(z)]))
    assert len(f.history) == 3
    # most recent kept
    assert np.allclose(f.history[-1][1], np.array([0.0, 0.0, 5.0]))
