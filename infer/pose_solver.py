import cv2
import numpy as np


def solve_pose(
    kpts_2d: np.ndarray,
    kpts_conf: np.ndarray,
    kpts_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    conf_thresh: float = 0.5,
    ransac_reproj_err: float = 4.0,
    ransac_iters: int = 200,
) -> dict | None:
    """Solve 6DoF pose with RANSAC-SQPNP + LM refinement.

    Returns {'R','t','inliers','reproj_err'} or None if not enough confident
    keypoints / RANSAC fails / too few inliers.
    """
    kpts_2d = np.asarray(kpts_2d, dtype=np.float64).reshape(-1, 2)
    kpts_conf = np.asarray(kpts_conf, dtype=np.float64).reshape(-1)
    kpts_3d = np.asarray(kpts_3d, dtype=np.float64).reshape(-1, 3)

    mask = kpts_conf >= conf_thresh
    if mask.sum() < 4:
        return None

    obj_pts = kpts_3d[mask].astype(np.float64).reshape(-1, 1, 3)
    img_pts = kpts_2d[mask].astype(np.float64).reshape(-1, 1, 2)
    orig_idx = np.where(mask)[0]

    # SQPnP raises cv2.error (assertion: point_coordinate_variance >=
    # POINT_VARIANCE_THRESHOLD) when the filtered 3D points are nearly
    # collinear/coplanar — degenerate geometry that appears frequently
    # during tracking when only a few low-quality keypoints pass conf_thresh.
    # Treat it the same as RANSAC returning ok=False.
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K, dist,
            flags=cv2.SOLVEPNP_SQPNP,
            reprojectionError=ransac_reproj_err,
            iterationsCount=ransac_iters,
        )
    except cv2.error:
        return None

    if not ok or inliers is None or len(inliers) < 4:
        return None

    inlier_idx_local = inliers.reshape(-1)
    obj_in = obj_pts[inlier_idx_local]
    img_in = img_pts[inlier_idx_local]
    rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K, dist)
    reproj_err = float(
        np.linalg.norm(proj.reshape(-1, 2) - img_in.reshape(-1, 2), axis=1).mean()
    )

    R, _ = cv2.Rodrigues(rvec)
    return {
        "R": R,
        "t": tvec.reshape(3),
        "inliers": orig_idx[inlier_idx_local],
        "reproj_err": reproj_err,
    }
