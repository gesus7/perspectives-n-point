import numpy as np
import yaml


def load_camera(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics from YAML. Returns (K (3,3), dist (5,))."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    if dist.size != 5:
        raise ValueError(f"dist must have 5 elements (k1,k2,p1,p2,k3), got {dist.size}")
    return K, dist
