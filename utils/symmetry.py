"""Compute proper-rotation symmetry permutations of a 3D keypoint set.

A symmetric object admits several physically-identical keypoint labelings. For
pose estimation only PROPER rotations (det +1) are valid — a reflection (det -1)
cannot be produced by any real camera and would make PnP unsolvable. This module
discovers the proper-rotation symmetries of the keypoints and returns them as
keypoint-index permutations, used by the symmetry-aware training loss.
"""
import itertools
import numpy as np

from utils.keypoints import load_keypoints_3d


def _orthogonal_fit(A, B):
    """Best-fit orthogonal matrix M with M @ A.T ~ B.T (Kabsch, no scaling)."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    return (U @ Vt).T


def compute_symmetry_perms(kpts_3d, tol=0.6, proper_only=True):
    """Find symmetry permutations of kpts_3d.

    Args:
        kpts_3d: (N, 3) array of object-frame keypoints.
        tol: max per-point distance (object units) to accept a symmetry.
        proper_only: keep only det(+1) rotations (drop reflections).

    Returns:
        list of permutations (each a length-N list) including identity first.
    """
    P = np.asarray(kpts_3d, dtype=np.float64)
    c = P.mean(0)
    Pc = P - c
    n = len(P)

    perms = []
    seen = set()
    # Candidate correspondences: for each symmetry, point i maps to some point j
    # at the same radius. Search over rotations that map the set onto itself by
    # testing all axis-aligned 90/180 deg rotations + their compositions is too
    # narrow; instead brute-force candidate permutations via nearest-neighbour
    # under a set of canonical rotation matrices about each principal axis.
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    axes = {
        "z": lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]),
        "x": lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]),
        "y": lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]),
    }
    candidates = []
    for axis_fn in axes.values():
        for a in angles:
            candidates.append(axis_fn(a))
    # also 180 deg about x and y already covered; include identity explicitly
    for R in candidates:
        Q = (R @ Pc.T).T
        perm = []
        maxd = 0.0
        for i in range(n):
            d = np.linalg.norm(Pc - Q[i], axis=1)
            j = int(d.argmin())
            perm.append(j)
            maxd = max(maxd, d[j])
        if sorted(perm) != list(range(n)) or maxd >= tol:
            continue
        if proper_only:
            M = _orthogonal_fit(Pc, Pc[perm])
            if np.linalg.det(M) < 0:
                continue
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            perms.append(perm)

    # Ensure identity is present and first
    identity = list(range(n))
    if tuple(identity) in seen:
        perms.remove(identity)
    perms.insert(0, identity)
    return perms


def load_symmetry_perms(keypoints_3d_path, tol=0.6):
    """Convenience: load keypoints from JSON and return proper-symmetry perms."""
    pts, _ = load_keypoints_3d(keypoints_3d_path)
    return compute_symmetry_perms(pts, tol=tol, proper_only=True)


def load_symmetry_perms_per_class(class_kpt_paths, tol=0.6):
    """Discover proper-rotation symmetry perms for several classes at once.

    Each class (e.g. 0=TowerBase, 1=TowerTop) has its OWN keypoint geometry, so
    its keypoints may only be permuted by ITS OWN proper symmetries. Applying one
    class's perms to another class's instance would produce a labeling PnP cannot
    solve. The symmetry-aware pose loss therefore needs a separate perm set per
    class (see train/symmetry_loss.py).

    Args:
        class_kpt_paths: {class_id: keypoints_json_path}.
        tol: per-point tolerance passed to compute_symmetry_perms.

    Returns:
        {class_id: list of permutations (identity first)}.
    """
    return {
        int(cid): load_symmetry_perms(path, tol=tol)
        for cid, path in class_kpt_paths.items()
    }

