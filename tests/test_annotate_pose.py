"""Tests for annotate_pose — draw a known (R,t) pose without re-solving PnP.

Used to render predicted poses during YOLO drop-outs, where no fresh 2D keypoints
exist. Reuses the visualize.py primitives (already covered by test_visualize.py).
"""
import numpy as np

from infer.pipeline import annotate_pose

K = np.array([[2828.3, 0.0, 960.0],
              [0.0, 2828.3, 540.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)


def _blank():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _kpts_3d():
    rng = np.random.default_rng(0)
    return rng.uniform(-5, 5, size=(12, 3))


def test_annotate_pose_draws_onto_frame():
    img = _blank()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    annotate_pose(img, R, t, K, DIST, _kpts_3d(), axis_len_m=5.0)
    assert img.sum() > 0   # something was drawn


def test_annotate_pose_returns_result_dict():
    img = _blank()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    kpts_3d = _kpts_3d()
    res = annotate_pose(img, R, t, K, DIST, kpts_3d,
                        class_id=1, class_name="TowerTop", track_id=7)
    assert np.allclose(res["R"], R)
    assert np.allclose(res["t"], t)
    assert res["class_id"] == 1
    assert res["class_name"] == "TowerTop"
    assert res["track_id"] == 7
