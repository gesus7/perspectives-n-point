import cv2
import numpy as np
from infer.pose_solver import solve_pose

K = np.array([[2828.3, 0.0, 960.0],
              [0.0, 2828.3, 540.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)


def _project(R, t, pts3d):
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(pts3d, rvec, t, K, DIST)
    return pts2d.reshape(-1, 2)


def _make_kpts_3d():
    rng = np.random.default_rng(42)
    return rng.uniform(-5, 5, size=(12, 3)).astype(np.float64)


def test_solve_pose_recovers_known_pose():
    kpts_3d = _make_kpts_3d()
    R_gt, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))
    t_gt = np.array([0.5, -1.0, 40.0])
    kpts_2d = _project(R_gt, t_gt, kpts_3d)
    conf = np.ones(12)

    out = solve_pose(kpts_2d, conf, kpts_3d, K, DIST, conf_thresh=0.5)

    assert out is not None
    assert np.linalg.norm(out["t"] - t_gt) < 0.1
    R_err = cv2.Rodrigues(R_gt @ out["R"].T)[0]
    assert np.degrees(np.linalg.norm(R_err)) < 0.5


def test_solve_pose_robust_to_pixel_noise():
    kpts_3d = _make_kpts_3d()
    R_gt, _ = cv2.Rodrigues(np.array([0.0, 0.5, 0.0]))
    t_gt = np.array([0.0, 0.0, 50.0])
    rng = np.random.default_rng(0)
    kpts_2d = _project(R_gt, t_gt, kpts_3d) + rng.normal(0, 1.0, size=(12, 2))
    conf = np.ones(12)

    out = solve_pose(kpts_2d, conf, kpts_3d, K, DIST, conf_thresh=0.5)

    assert out is not None
    assert np.linalg.norm(out["t"] - t_gt) < 0.5
    R_err = cv2.Rodrigues(R_gt @ out["R"].T)[0]
    assert np.degrees(np.linalg.norm(R_err)) < 2.0


def test_solve_pose_rejects_outliers_via_ransac():
    kpts_3d = _make_kpts_3d()
    R_gt, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))
    t_gt = np.array([0.0, 0.0, 30.0])
    kpts_2d = _project(R_gt, t_gt, kpts_3d)
    kpts_2d[3] += np.array([200, -150])
    kpts_2d[7] += np.array([-180, 220])
    conf = np.ones(12)

    out = solve_pose(kpts_2d, conf, kpts_3d, K, DIST, conf_thresh=0.5)

    assert out is not None
    assert 3 not in out["inliers"]
    assert 7 not in out["inliers"]
    assert np.linalg.norm(out["t"] - t_gt) < 0.3


def test_solve_pose_returns_none_when_too_few_confident():
    kpts_3d = _make_kpts_3d()
    R_gt, _ = cv2.Rodrigues(np.zeros(3))
    t_gt = np.array([0.0, 0.0, 30.0])
    kpts_2d = _project(R_gt, t_gt, kpts_3d)
    conf = np.zeros(12)
    conf[:3] = 0.9

    out = solve_pose(kpts_2d, conf, kpts_3d, K, DIST, conf_thresh=0.5)

    assert out is None
