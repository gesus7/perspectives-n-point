import json
import numpy as np
from utils.keypoints import load_keypoints_3d


def test_load_keypoints_sorted_by_id(tmp_path):
    data = {
        "keypoints": [
            {"id": 2, "x": 2.0, "y": 0.0, "z": 0.0, "name": "b"},
            {"id": 0, "x": 0.0, "y": 0.0, "z": 0.0, "name": "a"},
            {"id": 1, "x": 1.0, "y": 0.0, "z": 0.0, "name": "c"},
        ]
    }
    p = tmp_path / "k.json"
    p.write_text(json.dumps(data))
    points, names = load_keypoints_3d(str(p))
    assert points.shape == (3, 3)
    assert names == ["a", "c", "b"]
    assert np.allclose(points[:, 0], [0.0, 1.0, 2.0])


def test_load_real_keypoints():
    points, names = load_keypoints_3d("keypoints_3d_original.json")
    assert points.shape == (12, 3)
    assert len(names) == 12
    assert names[0] == "base_corner_0"
