# PnP Pose Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pose estimation system: render synthetic data with Blender → train YOLO26-pose → solve 6DoF pose with OpenCV PnP, for a transmission tower from aerial top-down imagery.

**Architecture:** Modular three-stage pipeline. Each stage has its own CLI entry point reading a shared `config.yaml`. Stage 1 (`data_synth/`) drives headless Blender to render images + YOLO-pose labels. Stage 2 (`train/`) wraps `ultralytics`. Stage 3 (`infer/`) runs YOLO inference, filters keypoints by confidence, and solves PnP with SQPNP+RANSAC+LM.

**Tech Stack:** Python 3.10+, ultralytics (YOLO26-pose), OpenCV, NumPy, PyYAML, tqdm, pytest, Blender 3.6+/4.x (headless via `blender --background --python`).

**Spec:** `docs/superpowers/specs/2026-05-16-pnp-pose-estimation-design.md`

---

## Task 1: Project skeleton & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.gitignore`
- Create: `README.md`
- Create empty dirs: `data_synth/`, `train/`, `infer/`, `utils/`, `tests/`, `backgrounds/`, `textures/`

- [ ] **Step 1: Create `requirements.txt`**

```
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
pytest>=7.4.0
```

- [ ] **Step 2: Create `config.yaml`** (verbatim from spec section 5)

```yaml
camera:
  path: camera.yaml
model:
  obj: Base.obj
  keypoints_3d: keypoints_3d.json
  num_keypoints: 12
render:
  num_train: 1000
  num_val: 200
  image_size: [1920, 1080]
  sensor_width_mm: 36.0
  camera_distance: [20.0, 80.0]
  pitch_deg: [60.0, 90.0]
  yaw_deg: [0.0, 360.0]
  roll_jitter_deg: 5.0
  sun_strength: [2.0, 6.0]
  hdri_strength: [0.3, 1.0]
  backgrounds_dir: backgrounds
  textures_dir: textures
  ground_size: 200.0
  occlusion_eps_m: 0.01
  min_visible_keypoints: 4
  min_bbox_area_ratio: 0.005
train:
  weights: yolo26n-pose.pt
  epochs: 100
  imgsz: 1280
  batch: 8
  device: 0
  project: runs/pose
  name: tower
infer:
  weights: runs/pose/tower/weights/best.pt
  conf_thresh: 0.5
  det_conf_thresh: 0.25
  ransac_reproj_err: 4.0
  ransac_iters: 200
  axis_len_m: 5.0
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
dataset/
runs/
*.pt
*.mp4
*.avi
*.png.tmp
.venv/
venv/
.idea/
.vscode/
```

- [ ] **Step 4: Create `README.md`** (short, just lists the stages and how to run)

```markdown
# Transmission Tower Pose Estimation (YOLO-pose + PnP)

## Stages
1. **Render**: `python data_synth/render_dataset.py --config config.yaml`
2. **Train**: `python train/train.py --config config.yaml`
3. **Infer image(s)**: `python infer/infer_image.py --config config.yaml --input <path>`
4. **Infer video**: `python infer/infer_video.py --config config.yaml --input <video|0>`

## Setup
```
pip install -r requirements.txt
# Blender must be on PATH for stage 1.
```

## Files
- `Base.obj` / `Base.mtl` – tower model (units: meters)
- `camera.yaml` – camera intrinsics K, distortion, image size
- `keypoints_3d.json` – 12 keypoints in object local frame
- `config.yaml` – all tunable parameters
```

- [ ] **Step 5: Create empty `__init__.py` and folder placeholders**

```bash
touch data_synth/__init__.py train/__init__.py infer/__init__.py utils/__init__.py tests/__init__.py
touch backgrounds/.gitkeep textures/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git init  # only if not already a repo
git add .
git commit -m "chore: project skeleton, config, requirements"
```

---

## Task 2: `utils/camera.py` — load camera intrinsics

**Files:**
- Create: `utils/camera.py`
- Create: `tests/test_camera.py`

- [ ] **Step 1: Write the failing test**

`tests/test_camera.py`:
```python
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
```

- [ ] **Step 2: Run the test, expect failure**

```bash
pytest tests/test_camera.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.camera'`.

- [ ] **Step 3: Implement `utils/camera.py`**

```python
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
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/test_camera.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/camera.py tests/test_camera.py
git commit -m "feat(utils): load_camera reads K and dist from YAML"
```

---

## Task 3: `utils/keypoints.py` — load 3D keypoints

**Files:**
- Create: `utils/keypoints.py`
- Create: `tests/test_keypoints.py`

- [ ] **Step 1: Write the failing test**

`tests/test_keypoints.py`:
```python
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
    points, names = load_keypoints_3d("keypoints_3d.json")
    assert points.shape == (12, 3)
    assert len(names) == 12
    assert names[0] == "base_corner_0"
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_keypoints.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `utils/keypoints.py`**

```python
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
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/test_keypoints.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/keypoints.py tests/test_keypoints.py
git commit -m "feat(utils): load_keypoints_3d returns (N,3) sorted by id"
```

---

## Task 4: `utils/config.py` — config loader

**Files:**
- Create: `utils/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

`tests/test_config.py`:
```python
from utils.config import load_config


def test_load_config_returns_nested_dict():
    cfg = load_config("config.yaml")
    assert "camera" in cfg
    assert "render" in cfg
    assert "train" in cfg
    assert "infer" in cfg
    assert cfg["render"]["image_size"] == [1920, 1080]
    assert cfg["model"]["num_keypoints"] == 12
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement `utils/config.py`**

```python
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load the project-wide config YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add utils/config.py tests/test_config.py
git commit -m "feat(utils): load_config loads global YAML"
```

---

## Task 5: `infer/pose_solver.py` — PnP core (TDD, before training!)

> Why before training: PnP is the deterministic core. We test it with synthetic data, no model required. If PnP works, the entire infer-side pipeline is de-risked.

**Files:**
- Create: `infer/pose_solver.py`
- Create: `tests/test_pose_solver.py`

- [ ] **Step 1: Write failing test (synthetic ground truth)**

`tests/test_pose_solver.py`:
```python
import cv2
import numpy as np
from infer.pose_solver import solve_pose

# Module-level fixtures
K = np.array([[2828.3, 0.0, 960.0],
              [0.0, 2828.3, 540.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)


def _project(R, t, pts3d):
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(pts3d, rvec, t, K, DIST)
    return pts2d.reshape(-1, 2)


def _make_kpts_3d():
    # Use a 12-point cube-ish set so PnP has plenty of geometry
    rng = np.random.default_rng(42)
    return rng.uniform(-5, 5, size=(12, 3)).astype(np.float64)


def test_solve_pose_recovers_known_pose():
    kpts_3d = _make_kpts_3d()
    R_gt, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))
    t_gt = np.array([0.5, -1.0, 40.0])  # 40m away, top-down-ish
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
    # Corrupt 2 keypoints with large offsets
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
    conf[:3] = 0.9  # only 3 confident

    out = solve_pose(kpts_2d, conf, kpts_3d, K, DIST, conf_thresh=0.5)

    assert out is None
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_pose_solver.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `infer/pose_solver.py`**

```python
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

    Args:
        kpts_2d: (N,2) pixel coordinates from detector.
        kpts_conf: (N,) per-keypoint confidence in [0,1].
        kpts_3d: (N,3) object-frame coordinates aligned to kpts_2d order.
        K: (3,3) camera intrinsics.
        dist: (5,) distortion coefficients.
        conf_thresh: drop keypoints below this confidence.
        ransac_reproj_err: RANSAC reprojection error threshold (pixels).
        ransac_iters: RANSAC iterations.

    Returns:
        {'R': (3,3), 't': (3,), 'inliers': np.ndarray of original indices,
         'reproj_err': float (px, mean over inliers)} or None.
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

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, dist,
        flags=cv2.SOLVEPNP_SQPNP,
        reprojectionError=ransac_reproj_err,
        iterationsCount=ransac_iters,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return None

    inlier_idx_local = inliers.reshape(-1)
    obj_in = obj_pts[inlier_idx_local]
    img_in = img_pts[inlier_idx_local]
    rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K, dist)
    reproj_err = float(np.linalg.norm(proj.reshape(-1, 2) - img_in.reshape(-1, 2), axis=1).mean())

    R, _ = cv2.Rodrigues(rvec)
    return {
        "R": R,
        "t": tvec.reshape(3),
        "inliers": orig_idx[inlier_idx_local],
        "reproj_err": reproj_err,
    }
```

- [ ] **Step 4: Run, expect all 4 tests pass**

```bash
pytest tests/test_pose_solver.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add infer/pose_solver.py tests/test_pose_solver.py
git commit -m "feat(infer): solve_pose with RANSAC-SQPNP + LM refinement"
```

---

## Task 6: `infer/visualize.py` — drawing helpers

**Files:**
- Create: `infer/visualize.py`
- Create: `tests/test_visualize.py`

- [ ] **Step 1: Write failing test**

`tests/test_visualize.py`:
```python
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
    assert out.sum() > 0  # something was drawn


def test_draw_keypoints_renders_dots():
    img = _blank()
    kpts = np.array([[960.0, 540.0], [100.0, 100.0]])
    conf = np.array([0.9, 0.2])
    out = draw_keypoints(img.copy(), kpts, conf, conf_thresh=0.5)
    # Center pixel should be drawn (above threshold)
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
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_visualize.py -v
```

- [ ] **Step 3: Implement `infer/visualize.py`**

```python
import cv2
import numpy as np


def draw_keypoints(
    img: np.ndarray,
    kpts_2d: np.ndarray,
    kpts_conf: np.ndarray,
    conf_thresh: float = 0.5,
    reproj_2d: np.ndarray | None = None,
) -> np.ndarray:
    """Draw detected keypoints (red) and optional reprojected keypoints (green)."""
    for (x, y), c in zip(kpts_2d, kpts_conf):
        if c >= conf_thresh:
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
    if reproj_2d is not None:
        for (x, y) in reproj_2d:
            cv2.circle(img, (int(round(x)), int(round(y))), 3, (0, 255, 0), 1)
    return img


def draw_axes(
    img: np.ndarray, R: np.ndarray, t: np.ndarray,
    K: np.ndarray, dist: np.ndarray, axis_len: float = 5.0,
) -> np.ndarray:
    """Draw object-frame X/Y/Z axes (red/green/blue) at object origin."""
    pts = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
                   dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts, rvec, t.reshape(3, 1), K, dist)
    o, x, y, z = [tuple(int(round(v)) for v in p.ravel()) for p in proj]
    cv2.line(img, o, x, (0, 0, 255), 2)
    cv2.line(img, o, y, (0, 255, 0), 2)
    cv2.line(img, o, z, (255, 0, 0), 2)
    return img


def draw_bbox3d(
    img: np.ndarray, R: np.ndarray, t: np.ndarray,
    K: np.ndarray, dist: np.ndarray, kpts_3d: np.ndarray,
) -> np.ndarray:
    """Draw an axis-aligned 3D bounding box derived from the keypoints."""
    mn = kpts_3d.min(axis=0)
    mx = kpts_3d.max(axis=0)
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(corners, rvec, t.reshape(3, 1), K, dist)
    p = [tuple(int(round(v)) for v in c.ravel()) for c in proj]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        cv2.line(img, p[a], p[b], (255, 255, 0), 1)
    return img


def put_pose_text(img: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Overlay distance and Euler angles (deg) in the top-left corner."""
    dist_m = float(np.linalg.norm(t))
    # Euler ZYX from R (yaw, pitch, roll)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    txt = f"d={dist_m:.2f}m  yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f}"
    cv2.putText(img, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/test_visualize.py -v
```

- [ ] **Step 5: Commit**

```bash
git add infer/visualize.py tests/test_visualize.py
git commit -m "feat(infer): visualization helpers (axes, kpts, 3D bbox, text)"
```

---

## Task 7: `infer/pipeline.py` — shared per-frame inference

**Files:**
- Create: `infer/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test (mock the YOLO model)**

`tests/test_pipeline.py`:
```python
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
import cv2

from infer.pipeline import run_on_frame


def _fake_yolo_result(kpts_2d, kpts_conf, det_conf=0.9):
    """Mimic ultralytics result.keypoints/.boxes structure."""
    keypoints = SimpleNamespace(
        xy=np.array([kpts_2d], dtype=np.float32),     # (1, N, 2)
        conf=np.array([kpts_conf], dtype=np.float32),  # (1, N)
    )
    boxes = SimpleNamespace(
        conf=np.array([det_conf], dtype=np.float32),
        xyxy=np.array([[100, 100, 800, 800]], dtype=np.float32),
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
    model = MagicMock(); model.predict = MagicMock(return_value=_fake_yolo_result(kpts_2d, np.ones(12)))

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    annotated, results = run_on_frame(
        model, frame, K, dist, kpts_3d,
        conf_thresh=0.5, det_conf_thresh=0.25,
    )
    assert annotated.shape == frame.shape
    assert len(results) == 1
    assert np.linalg.norm(results[0]["t"] - t) < 0.1


def test_run_on_frame_skips_low_det_conf():
    rng = np.random.default_rng(0)
    kpts_3d = rng.uniform(-5, 5, size=(12, 3))
    K = np.array([[2828.3, 0, 960], [0, 2828.3, 540], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    model = MagicMock(); model.predict = MagicMock(
        return_value=_fake_yolo_result(np.zeros((12, 2)), np.ones(12), det_conf=0.1)
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, results = run_on_frame(model, frame, K, dist, kpts_3d,
                              conf_thresh=0.5, det_conf_thresh=0.25)
    assert results == []
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement `infer/pipeline.py`**

```python
import cv2
import numpy as np

from infer.pose_solver import solve_pose
from infer.visualize import draw_axes, draw_bbox3d, draw_keypoints, put_pose_text


def run_on_frame(
    model,
    frame: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    kpts_3d: np.ndarray,
    conf_thresh: float = 0.5,
    det_conf_thresh: float = 0.25,
    axis_len_m: float = 5.0,
) -> tuple[np.ndarray, list[dict]]:
    """Run YOLO-pose inference + PnP on one frame.

    Returns (annotated frame BGR, list of per-detection result dicts).
    Each result dict has: 'R','t','reproj_err','inliers','kpts_2d','kpts_conf'.
    Detections below det_conf_thresh or with too few confident keypoints are skipped.
    """
    yolo_results = model.predict(frame, verbose=False)
    out_frame = frame.copy()
    out_results: list[dict] = []

    for res in yolo_results:
        boxes = res.boxes
        kps = res.keypoints
        if boxes is None or kps is None:
            continue
        det_conf = np.asarray(boxes.conf)
        kp_xy = np.asarray(kps.xy)        # (M, N, 2)
        kp_conf = np.asarray(kps.conf)    # (M, N)

        for i in range(len(det_conf)):
            if float(det_conf[i]) < det_conf_thresh:
                continue
            kpts_2d = kp_xy[i].astype(np.float64)
            kpts_conf = kp_conf[i].astype(np.float64)

            pose = solve_pose(kpts_2d, kpts_conf, kpts_3d, K, dist,
                              conf_thresh=conf_thresh)
            out_frame = draw_keypoints(out_frame, kpts_2d, kpts_conf, conf_thresh)
            if pose is None:
                continue

            rvec, _ = cv2.Rodrigues(pose["R"])
            reproj, _ = cv2.projectPoints(kpts_3d, rvec, pose["t"].reshape(3, 1), K, dist)
            out_frame = draw_keypoints(
                out_frame, kpts_2d, kpts_conf, conf_thresh,
                reproj_2d=reproj.reshape(-1, 2),
            )
            out_frame = draw_bbox3d(out_frame, pose["R"], pose["t"], K, dist, kpts_3d)
            out_frame = draw_axes(out_frame, pose["R"], pose["t"], K, dist, axis_len_m)
            out_frame = put_pose_text(out_frame, pose["R"], pose["t"])

            out_results.append({
                **pose,
                "kpts_2d": kpts_2d,
                "kpts_conf": kpts_conf,
            })

    return out_frame, out_results
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add infer/pipeline.py tests/test_pipeline.py
git commit -m "feat(infer): per-frame pipeline glues YOLO + PnP + visualization"
```

---

## Task 8: `infer/infer_image.py` — single image / folder CLI

**Files:**
- Create: `infer/infer_image.py`

- [ ] **Step 1: Implement the CLI**

```python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Allow running as: python infer/infer_image.py ...
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from infer.pipeline import run_on_frame
from utils.camera import load_camera
from utils.config import load_config
from utils.keypoints import load_keypoints_3d


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--output", default="infer_out", help="output directory")
    parser.add_argument("--weights", default=None,
                        help="override config.infer.weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    K, dist = load_camera(cfg["camera"]["path"])
    kpts_3d, _ = load_keypoints_3d(cfg["model"]["keypoints_3d"])
    weights = args.weights or cfg["infer"]["weights"]
    model = YOLO(weights)

    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"[skip] unreadable: {f}")
            continue
        annotated, results = run_on_frame(
            model, frame, K, dist, kpts_3d,
            conf_thresh=cfg["infer"]["conf_thresh"],
            det_conf_thresh=cfg["infer"]["det_conf_thresh"],
            axis_len_m=cfg["infer"]["axis_len_m"],
        )
        cv2.imwrite(str(out_dir / f"{f.stem}_annot.jpg"), annotated)

        pose_json = []
        for r in results:
            pose_json.append({
                "R": r["R"].tolist(),
                "t": r["t"].tolist(),
                "reproj_err_px": r["reproj_err"],
                "inliers": [int(x) for x in r["inliers"]],
            })
        (out_dir / f"{f.stem}_pose.json").write_text(json.dumps(pose_json, indent=2))
        print(f"[ok] {f.name}: {len(results)} detection(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (no weights yet, just argparse)**

```bash
python infer/infer_image.py --help
```
Expected: argparse usage prints, no error.

- [ ] **Step 3: Commit**

```bash
git add infer/infer_image.py
git commit -m "feat(infer): CLI for single image / folder inference"
```

---

## Task 9: `infer/infer_video.py` — video / camera CLI

**Files:**
- Create: `infer/infer_video.py`

- [ ] **Step 1: Implement**

```python
import argparse
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from infer.pipeline import run_on_frame
from utils.camera import load_camera
from utils.config import load_config
from utils.keypoints import load_keypoints_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True,
                        help="video path or camera index (e.g. 0)")
    parser.add_argument("--output", default=None, help="output mp4 path (optional)")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-show", action="store_true",
                        help="don't open a display window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    K, dist = load_camera(cfg["camera"]["path"])
    kpts_3d, _ = load_keypoints_3d(cfg["model"]["keypoints_3d"])
    weights = args.weights or cfg["infer"]["weights"]
    model = YOLO(weights)

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {args.input}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, _ = run_on_frame(
                model, frame, K, dist, kpts_3d,
                conf_thresh=cfg["infer"]["conf_thresh"],
                det_conf_thresh=cfg["infer"]["det_conf_thresh"],
                axis_len_m=cfg["infer"]["axis_len_m"],
            )
            if writer is not None:
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("pose", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
python infer/infer_video.py --help
```

- [ ] **Step 3: Commit**

```bash
git add infer/infer_video.py
git commit -m "feat(infer): CLI for video file / camera inference"
```

---

## Task 10: `data_synth/blender_render.py` — Blender-side renderer

> This script runs **inside Blender's Python**, not the project's venv. It must not import project modules. All inputs come from CLI args. The orchestrator (Task 11) passes them.

**Files:**
- Create: `data_synth/blender_render.py`

- [ ] **Step 1: Implement the file in full**

```python
"""Run inside Blender via: blender --background --python data_synth/blender_render.py -- <args>

Renders `count` images of an OBJ model with randomized aerial camera poses,
writing YOLO-pose labels (12 keypoints, vis flag) and images to a split dir.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Euler


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--keypoints", required=True)
    p.add_argument("--out-images", required=True)
    p.add_argument("--out-labels", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fx", type=float, required=True)
    p.add_argument("--sensor-width-mm", type=float, default=36.0)
    p.add_argument("--dist-min", type=float, default=20.0)
    p.add_argument("--dist-max", type=float, default=80.0)
    p.add_argument("--pitch-min-deg", type=float, default=60.0)
    p.add_argument("--pitch-max-deg", type=float, default=90.0)
    p.add_argument("--roll-jitter-deg", type=float, default=5.0)
    p.add_argument("--sun-min", type=float, default=2.0)
    p.add_argument("--sun-max", type=float, default=6.0)
    p.add_argument("--ground-size", type=float, default=200.0)
    p.add_argument("--backgrounds-dir", default="")
    p.add_argument("--textures-dir", default="")
    p.add_argument("--occlusion-eps-m", type=float, default=0.01)
    p.add_argument("--min-visible", type=int, default=4)
    p.add_argument("--min-bbox-ratio", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(
        bpy.context.scene.render, "engine") and "BLENDER_EEVEE_NEXT" in [
        i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ] else "BLENDER_EEVEE"


def load_obj(path: str):
    # Blender 4.x: bpy.ops.wm.obj_import (legacy: import_scene.obj)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)
    # Collect imported mesh objects and join them
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh imported from OBJ")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    return obj


def add_ground(size: float, image_dir: str, texture_dir: str):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, -9.6))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    candidates = []
    for d in (image_dir, texture_dir):
        if d and Path(d).is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                candidates.extend(sorted(Path(d).glob(ext)))
    if candidates:
        img_path = random.choice(candidates)
        tex_img = bpy.data.images.load(str(img_path))
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = tex_img
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.2, 0.6),
            random.uniform(0.3, 0.6),
            random.uniform(0.2, 0.5), 1.0,
        )
    plane.data.materials.append(mat)
    return plane


def add_sun(strength: float):
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 100))
    sun = bpy.context.active_object
    sun.data.energy = strength
    sun.rotation_euler = (
        math.radians(random.uniform(-30, 30)),
        math.radians(random.uniform(-30, 30)),
        math.radians(random.uniform(0, 360)),
    )
    return sun


def setup_camera(width: int, height: int, fx_px: float, sensor_width_mm: float):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens_unit = "MILLIMETERS"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = sensor_width_mm
    cam.data.lens = fx_px * sensor_width_mm / width
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0
    bpy.context.scene.camera = cam
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    return cam


def sample_camera_pose(cam, target: Vector, dist_min, dist_max,
                       pitch_min_deg, pitch_max_deg, roll_jitter_deg):
    d = random.uniform(dist_min, dist_max)
    pitch = math.radians(random.uniform(pitch_min_deg, pitch_max_deg))
    yaw = math.radians(random.uniform(0, 360))
    # Spherical -> position above target, pitch from horizontal:
    # pitch=90 deg => directly above (z+d), pitch=0 => same plane.
    x = d * math.cos(pitch) * math.cos(yaw)
    y = d * math.cos(pitch) * math.sin(yaw)
    z = d * math.sin(pitch)
    cam.location = target + Vector((x, y, z))

    direction = (target - cam.location).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    # Apply small roll perturbation
    roll = math.radians(random.uniform(-roll_jitter_deg, roll_jitter_deg))
    eul = cam.rotation_euler
    cam.rotation_euler = Euler((eul.x, eul.y, eul.z + roll), "XYZ")


def project_world_to_pixel(world_co: Vector, cam, scene) -> tuple[float, float, bool]:
    """Returns (px, py, in_image)."""
    from bpy_extras.object_utils import world_to_camera_view
    co_ndc = world_to_camera_view(scene, cam, world_co)
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    px = co_ndc.x * w
    py = (1.0 - co_ndc.y) * h
    in_img = (0.0 <= px < w) and (0.0 <= py < h) and (co_ndc.z > 0)
    return px, py, in_img


def keypoint_visibility(world_co: Vector, cam, scene, depsgraph, eps_m: float) -> int:
    px, py, in_img = project_world_to_pixel(world_co, cam, scene)
    if not in_img:
        return 0
    cam_loc = cam.matrix_world.translation
    direction = (world_co - cam_loc).normalized()
    hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, cam_loc, direction)
    if not hit:
        return 2
    dist_hit = (loc - cam_loc).length
    dist_kp = (world_co - cam_loc).length
    if dist_hit + eps_m < dist_kp:
        return 1
    return 2


def load_keypoints(path: str) -> list[Vector]:
    with open(path, "r") as f:
        data = json.load(f)
    kpts = sorted(data["keypoints"], key=lambda k: k["id"])
    return [Vector((k["x"], k["y"], k["z"])) for k in kpts]


def render_one(scene, cam, depsgraph, obj_world_matrix, kpts_local, args,
               out_img: str) -> dict | None:
    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)

    w = scene.render.resolution_x
    h = scene.render.resolution_y
    annotations = []
    for kp in kpts_local:
        world_co = obj_world_matrix @ kp
        v = keypoint_visibility(world_co, cam, scene, depsgraph, args.occlusion_eps_m)
        px, py, _ = project_world_to_pixel(world_co, cam, scene)
        annotations.append((px, py, v))

    visible = [a for a in annotations if a[2] >= 1]
    if len(visible) < args.min_visible:
        return None

    xs = [a[0] for a in visible]
    ys = [a[1] for a in visible]
    x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
    y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
    # 5% margin
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(x_min - 0.05 * bw, 0)
    x_max = min(x_max + 0.05 * bw, w - 1)
    y_min = max(y_min - 0.05 * bh, 0)
    y_max = min(y_max + 0.05 * bh, h - 1)
    bw = x_max - x_min
    bh = y_max - y_min
    if bw * bh < args.min_bbox_ratio * w * h:
        return None

    cx = (x_min + x_max) / 2.0 / w
    cy = (y_min + y_max) / 2.0 / h
    nw = bw / w
    nh = bh / h
    parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
    for (px, py, v) in annotations:
        parts.append(f"{px / w:.6f} {py / h:.6f} {v}")
    return {"label_line": " ".join(parts)}


def main():
    args = parse_args()
    random.seed(args.seed)

    reset_scene()
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True

    obj = load_obj(args.obj)
    add_ground(args.ground_size, args.backgrounds_dir, args.textures_dir)
    sun_strength = random.uniform(args.sun_min, args.sun_max)
    add_sun(sun_strength)
    cam = setup_camera(args.width, args.height, args.fx, args.sensor_width_mm)

    kpts_local = load_keypoints(args.keypoints)
    obj_target_world = obj.matrix_world @ Vector((0, 0, 1.0))  # aim slightly above base

    Path(args.out_images).mkdir(parents=True, exist_ok=True)
    Path(args.out_labels).mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    i = args.start_index
    produced = 0
    attempts = 0
    max_attempts = args.count * 5
    while produced < args.count and attempts < max_attempts:
        attempts += 1
        sample_camera_pose(
            cam, obj_target_world,
            args.dist_min, args.dist_max,
            args.pitch_min_deg, args.pitch_max_deg,
            args.roll_jitter_deg,
        )
        # randomize sun each frame too
        sun = next(o for o in bpy.context.scene.objects if o.type == "LIGHT")
        sun.data.energy = random.uniform(args.sun_min, args.sun_max)

        out_img = str(Path(args.out_images) / f"img_{i:06d}.png")
        result = render_one(scene, cam, depsgraph, obj.matrix_world, kpts_local,
                            args, out_img)
        if result is None:
            try:
                os.remove(out_img)
            except OSError:
                pass
            continue
        with open(Path(args.out_labels) / f"img_{i:06d}.txt", "w") as f:
            f.write(result["label_line"] + "\n")
        produced += 1
        i += 1
        print(f"[render] produced {produced}/{args.count} (idx={i - 1})")

    if produced < args.count:
        print(f"[render] WARNING: only produced {produced}/{args.count} after "
              f"{attempts} attempts.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add data_synth/blender_render.py
git commit -m "feat(data_synth): Blender-side renderer with visibility & label output"
```

---

## Task 11: `data_synth/render_dataset.py` — orchestrator

**Files:**
- Create: `data_synth/render_dataset.py`

- [ ] **Step 1: Implement**

```python
"""Run via the project venv (not Blender's). Spawns blender headless processes."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.camera import load_camera
from utils.config import load_config


def _blender_executable() -> str:
    exe = shutil.which("blender")
    if exe:
        return exe
    raise RuntimeError(
        "`blender` not found on PATH. Install Blender 3.6+/4.x and ensure "
        "`blender --version` works in your shell."
    )


def _run_split(blender_exe: str, cfg: dict, K, split: str, count: int, out_root: Path):
    images_dir = out_root / "images" / split
    labels_dir = out_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    render = cfg["render"]
    cmd = [
        blender_exe, "--background", "--python",
        "data_synth/blender_render.py", "--",
        "--obj", cfg["model"]["obj"],
        "--keypoints", cfg["model"]["keypoints_3d"],
        "--out-images", str(images_dir),
        "--out-labels", str(labels_dir),
        "--count", str(count),
        "--width", str(render["image_size"][0]),
        "--height", str(render["image_size"][1]),
        "--fx", f"{float(K[0, 0]):.6f}",
        "--sensor-width-mm", str(render["sensor_width_mm"]),
        "--dist-min", str(render["camera_distance"][0]),
        "--dist-max", str(render["camera_distance"][1]),
        "--pitch-min-deg", str(render["pitch_deg"][0]),
        "--pitch-max-deg", str(render["pitch_deg"][1]),
        "--roll-jitter-deg", str(render["roll_jitter_deg"]),
        "--sun-min", str(render["sun_strength"][0]),
        "--sun-max", str(render["sun_strength"][1]),
        "--ground-size", str(render["ground_size"]),
        "--backgrounds-dir", render["backgrounds_dir"],
        "--textures-dir", render["textures_dir"],
        "--occlusion-eps-m", str(render["occlusion_eps_m"]),
        "--min-visible", str(render["min_visible_keypoints"]),
        "--min-bbox-ratio", str(render["min_bbox_area_ratio"]),
        "--seed", "0" if split == "train" else "1",
        "--start-index", "0",
    ]
    print(f"[orchestrator] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def write_data_yaml(out_root: Path, num_kpts: int):
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "transmission_tower"},
        "kpt_shape": [num_kpts, 3],
        "flip_idx": [],
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"[orchestrator] wrote {out_root / 'data.yaml'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="dataset")
    p.add_argument("--num-train", type=int, default=None)
    p.add_argument("--num-val", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    K, _ = load_camera(cfg["camera"]["path"])

    expected_w = cfg["render"]["image_size"][0]
    expected_h = cfg["render"]["image_size"][1]
    # cx,cy sanity: must match image center for the alignment to be valid
    if abs(K[0, 2] - expected_w / 2) > 1.0 or abs(K[1, 2] - expected_h / 2) > 1.0:
        raise SystemExit(
            f"camera.yaml principal point ({K[0,2]}, {K[1,2]}) does not match "
            f"image center ({expected_w/2}, {expected_h/2}). Update config.render.image_size "
            f"or your camera.yaml."
        )

    n_train = args.num_train or cfg["render"]["num_train"]
    n_val = args.num_val or cfg["render"]["num_val"]

    out_root = Path(args.out)
    blender = _blender_executable()
    _run_split(blender, cfg, K, "train", n_train, out_root)
    _run_split(blender, cfg, K, "val", n_val, out_root)
    write_data_yaml(out_root, cfg["model"]["num_keypoints"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (no blender execution)**

```bash
python data_synth/render_dataset.py --help
```
Expected: usage prints.

- [ ] **Step 3: Commit**

```bash
git add data_synth/render_dataset.py
git commit -m "feat(data_synth): orchestrator spawns blender headless and writes data.yaml"
```

---

## Task 12: Render-side integration test (small)

**Files:**
- Create: `tests/test_render_integration.py`

> Marked as integration and skipped automatically if `blender` is not on PATH.

- [ ] **Step 1: Write the test**

`tests/test_render_integration.py`:
```python
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("blender") is None, reason="blender not on PATH"
)


def test_render_5_frames_produces_labels(tmp_path):
    out_root = tmp_path / "ds"
    cmd = [
        sys.executable, "data_synth/render_dataset.py",
        "--config", "config.yaml",
        "--out", str(out_root),
        "--num-train", "5",
        "--num-val", "2",
    ]
    subprocess.run(cmd, check=True)
    train_images = list((out_root / "images" / "train").glob("*.png"))
    train_labels = list((out_root / "labels" / "train").glob("*.txt"))
    assert len(train_images) == 5
    assert len(train_labels) == 5
    # Sanity check first label
    line = train_labels[0].read_text().strip().split()
    # 5 (cls + bbox) + 12*3 (kpts) = 41 tokens
    assert len(line) == 41
    cls = int(line[0])
    assert cls == 0
    # bbox in [0,1]
    for v in line[1:5]:
        assert 0.0 <= float(v) <= 1.0
    # kpt visibility tokens are integers 0/1/2
    for i in range(12):
        v = line[5 + i * 3 + 2]
        assert v in {"0", "1", "2"}, v
    # data.yaml present
    assert (out_root / "data.yaml").exists()
```

- [ ] **Step 2: Run (skips if no blender, that's fine)**

```bash
pytest tests/test_render_integration.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_render_integration.py
git commit -m "test(data_synth): integration test for 5+2 frame render"
```

---

## Task 13: `train/train.py` — training entry

**Files:**
- Create: `train/train.py`

- [ ] **Step 1: Implement**

```python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", default="dataset/data.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    t = cfg["train"]
    model = YOLO(t["weights"])
    model.train(
        data=args.data,
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        device=t["device"],
        project=t["project"],
        name=t["name"],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
python train/train.py --help
```

- [ ] **Step 3: Commit**

```bash
git add train/train.py
git commit -m "feat(train): YOLO26-pose training entry"
```

---

## Task 14: End-to-end sanity script — `scripts/sanity_pnp_from_gt.py`

> Goal: take any rendered label file's keypoints (treat them as a perfect detection) and run PnP through `infer/pipeline.py` → confirm reprojection error is tiny. Validates the data + PnP loop without needing a trained model. Run this once after Task 12 produces 5 frames.

**Files:**
- Create: `scripts/sanity_pnp_from_gt.py`

- [ ] **Step 1: Implement**

```python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer.pose_solver import solve_pose
from utils.camera import load_camera
from utils.config import load_config
from utils.keypoints import load_keypoints_3d


def parse_label(line: str, w: int, h: int):
    toks = line.strip().split()
    kpts = np.array(toks[5:], dtype=np.float64).reshape(-1, 3)
    kpts_2d = kpts[:, :2].copy()
    kpts_2d[:, 0] *= w
    kpts_2d[:, 1] *= h
    vis = kpts[:, 2]
    conf = np.where(vis >= 1, 1.0, 0.0)
    return kpts_2d, conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--labels", required=True, help="path to a labels/ split dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    K, dist = load_camera(cfg["camera"]["path"])
    kpts_3d, _ = load_keypoints_3d(cfg["model"]["keypoints_3d"])
    w, h = cfg["render"]["image_size"]

    errs = []
    n_ok = 0
    n_fail = 0
    for lab in sorted(Path(args.labels).glob("*.txt")):
        line = lab.read_text().splitlines()[0]
        kpts_2d, conf = parse_label(line, w, h)
        res = solve_pose(kpts_2d, conf, kpts_3d, K, dist, conf_thresh=0.5)
        if res is None:
            n_fail += 1
            continue
        errs.append(res["reproj_err"])
        n_ok += 1
    if not errs:
        raise SystemExit("All frames failed — labels or PnP broken.")
    print(f"frames ok={n_ok} fail={n_fail}")
    print(f"reproj_err px: median={np.median(errs):.3f} p95={np.percentile(errs,95):.3f} max={np.max(errs):.3f}")
    # Sanity bound: with GT keypoints, reprojection error should be sub-pixel.
    if np.median(errs) > 1.0:
        raise SystemExit("Median reproj error > 1px on GT — alignment problem.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/sanity_pnp_from_gt.py
git commit -m "feat(scripts): sanity check PnP against GT labels"
```

---

## Task 15: Top-level smoke-test runner

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append to `README.md`**

```markdown

## Recommended end-to-end smoke test (small)

```bash
# 1. Quick render (needs Blender on PATH)
python data_synth/render_dataset.py --num-train 10 --num-val 4

# 2. Sanity check: PnP on GT labels should yield sub-pixel reprojection
python scripts/sanity_pnp_from_gt.py --labels dataset/labels/train

# 3. Run unit tests
pytest -v

# 4. Train (full size)
python data_synth/render_dataset.py            # 1000+200
python train/train.py

# 5. Inference
python infer/infer_image.py --input some_image.jpg
python infer/infer_video.py --input some_video.mp4 --output out.mp4
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add end-to-end smoke test instructions"
```

---

## Self-Review

**Spec coverage check (against `docs/superpowers/specs/2026-05-16-pnp-pose-estimation-design.md`):**

| Spec section | Task |
|---|---|
| §3.1 `utils/camera.py` | Task 2 |
| §3.2 `utils/keypoints.py` | Task 3 |
| §3.3 `render_dataset.py` orchestrator | Task 11 |
| §3.4 `blender_render.py` (intrinsics, scene, sampling, labels, occlusion) | Task 10 |
| §3.5 `dataset/data.yaml` | Task 11 (`write_data_yaml`) |
| §3.6 `train/train.py` | Task 13 |
| §3.7 `infer/pose_solver.py` (RANSAC-SQPNP + LM) | Task 5 |
| §3.8 `infer/visualize.py` | Task 6 |
| §3.9 `infer_image.py` / `infer_video.py` | Tasks 8, 9 |
| §5 `config.yaml` | Task 1 |
| §6 error handling | covered in Tasks 5 (PnP None paths), 10 (drop frame), 11 (K/image-size assert) |
| §7 testing strategy | Tasks 2, 3, 5, 6, 7, 12, 14 |

**Placeholder scan:** No `TBD`/`TODO`. Every code block is concrete.

**Type consistency:** `solve_pose` signature in Task 5 matches its call sites in Tasks 7 and 14. `run_on_frame` signature in Task 7 matches calls in Tasks 8, 9. `load_camera` returns `(K, dist)` everywhere. `load_keypoints_3d` returns `(points, names)` everywhere. `load_config` returns dict everywhere.

**Quirks worth flagging to the executor at run time, not plan changes:**
- Task 13 uses `weights: yolo26n-pose.pt`. If ultralytics can't fetch that exact tag, fall back to a current pose weight (e.g. `yolo11n-pose.pt`); only one line in `config.yaml` needs to change.
- Task 10 uses Blender 4.x's `bpy.ops.wm.obj_import`; legacy `import_scene.obj` is the fallback. The script handles both.

