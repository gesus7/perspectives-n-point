import numpy as np
import cv2
from infer.visualize import draw_axes, draw_keypoints, draw_bbox3d, put_pose_text

K = np.array([[2828.3, 0.0, 960.0],
              [0.0, 2828.3, 540.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)


def _blank():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_draw_axes_changes_image():
    img = _blank()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    out = draw_axes(img.copy(), R, t, K, DIST, axis_len=5.0)
    assert out.shape == img.shape
    assert out.sum() > 0


def test_draw_keypoints_renders_dots():
    img = _blank()
    kpts = np.array([[960.0, 540.0], [100.0, 100.0]])
    conf = np.array([0.9, 0.2])
    out = draw_keypoints(img.copy(), kpts, conf, conf_thresh=0.5)
    assert out[540, 960].sum() > 0


def test_draw_bbox3d_returns_image():
    img = _blank()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    pts = np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]], dtype=np.float64)
    out = draw_bbox3d(img.copy(), R, t, K, DIST, pts)
    assert out.shape == img.shape


def test_put_pose_text_returns_image():
    img = _blank()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 30.0])
    out = put_pose_text(img.copy(), R, t)
    assert out.shape == img.shape
