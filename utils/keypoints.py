import json
import numpy as np


def load_keypoints_3d(path: str) -> tuple[np.ndarray, list[str]]:
    """Load 3D keypoints (object frame, meters). Returns (points (N,3), names (N,))."""
    with open(path, "r") as f:
        data = json.load(f)
    kpts = sorted(data["keypoints"], key=lambda k: k["id"])
    points = np.array([[k["x"], k["y"], k["z"]] for k in kpts], dtype=np.float64)
    names = [k["name"] for k in kpts]
    return points, names
