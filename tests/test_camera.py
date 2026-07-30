import numpy as np
from utils.camera import load_camera


def test_load_camera_matches_yaml(tmp_path):
    yaml_text = """
K:
  - [2828.3, 0.0, 960.0]
  - [0.0, 2828.3, 540.0]
  - [0.0, 0.0, 1.0]
dist: [0.0, 0.0, 0.0, 0.0, 0.0]
"""
    p = tmp_path / "cam.yaml"
    p.write_text(yaml_text)
    K, dist = load_camera(str(p))
    assert K.shape == (3, 3)
    assert K.dtype == np.float64
    assert np.allclose(K[0, 0], 2828.3)
    assert np.allclose(K[1, 2], 540.0)
    assert dist.shape == (5,)
    assert np.allclose(dist, 0.0)
