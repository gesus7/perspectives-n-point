import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
import cv2

from infer.pipeline import run_on_frame


class _CpuArray(np.ndarray):
    """ndarray that also answers .cpu() -> self, so it mimics a torch tensor
    closely enough for the pipeline's `np.asarray(x.cpu())` calls."""
    def cpu(self):
        return self


def _t(arr, dtype=np.float32):
    a = np.asarray(arr, dtype=dtype)
    return a.view(_CpuArray)


def _fake_yolo_result(kpts_2d, kpts_conf, det_conf=0.9, cls=0):
    """Mimic ultralytics result.keypoints/.boxes structure (tensors expose .cpu())."""
    keypoints = SimpleNamespace(
        xy=_t([kpts_2d]),
        conf=_t([kpts_conf]),
    )
    boxes = SimpleNamespace(
        conf=_t([det_conf]),
        cls=_t([cls]),
        xyxy=_t([[100, 100, 800, 800]]),
        id=None,   # no track id in predict() mode
    )
    return [SimpleNamespace(keypoints=keypoints, boxes=boxes)]


def test_run_on_frame_returns_annotated_with_pose():
    rng = np.random.default_rng(0)
    kpts_3d = rng.uniform(-5, 5, size=(12, 3))
    K = np.array([[2828.3, 0, 960], [0, 2828.3, 540], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    R = np.eye(3); t = np.array([0.0, 0.0, 30.0])
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(kpts_3d, rvec, t, K, dist)
    kpts_2d = proj.reshape(-1, 2)
    model = MagicMock()
    model.predict = MagicMock(return_value=_fake_yolo_result(kpts_2d, np.ones(12)))

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    annotated, results = run_on_frame(
        model, frame, K, dist, {0: kpts_3d},
        conf_thresh=0.5, det_conf_thresh=0.25,
    )[:2]
    assert annotated.shape == frame.shape
    assert len(results) == 1
    assert results[0]["class_id"] == 0
    assert np.linalg.norm(results[0]["t"] - t) < 0.1


def test_run_on_frame_skips_low_det_conf():
    rng = np.random.default_rng(0)
    kpts_3d = rng.uniform(-5, 5, size=(12, 3))
    K = np.array([[2828.3, 0, 960], [0, 2828.3, 540], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    model = MagicMock()
    model.predict = MagicMock(
        return_value=_fake_yolo_result(np.zeros((12, 2)), np.ones(12), det_conf=0.1)
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, results = run_on_frame(model, frame, K, dist, {0: kpts_3d},
                              conf_thresh=0.5, det_conf_thresh=0.25)[:2]
    assert results == []
