# Dual-segment Transmission Tower Pose Estimation (YOLO-pose + PnP)

End-to-end 6DoF pose estimation for **two segments** of a symmetric transmission tower
from helicopter-mounted down-looking aerial views:

- **TowerBase** (class 0) — the base section fixed to the ground.
- **TowerTop** (class 1) — the top section suspended on ropes below the helicopter.

Both segments are detected in a single YOLO26-pose model (two-class, 12 shared keypoint
slots) and each one's pose is solved independently via OpenCV RANSAC-PnP against its own
3D keypoints. The relative pose (Top relative to Base) can then be computed.

```
synthetic two-segment data (Blender)  →  YOLO26-pose keypoint detection  →  per-class OpenCV RANSAC-PnP  →  6DoF pose (R, t)
```

The objects are geometrically symmetric, which makes naive keypoint training fail.
The core of this project is a **per-class symmetry-aware training loss** that lets the
network converge while keeping fixed physical keypoint ids per class, so PnP stays solvable
at inference. See [The symmetry problem](#the-symmetry-problem) below.

---

## Pipeline stages

| Stage | Command |
|------|---------|
| 1. Render dataset | `python data_synth/render_dataset_batch.py --num-train 5000 --num-val 500` |
| 2. Train | `python train/train.py --config config.yaml` |
| 3. Evaluate | `python eval_dataset.py --dataset dataset --split val --weights <trained.pt>` |
| 4. Infer image(s) | `python infer/infer_image.py --input <path> --output infer_out` |
| 5. Infer video / camera | `python infer/infer_video.py --input <video\|0>` |

## Setup

```bash
pip install -r requirements.txt
# Blender 3.6+ / 4.x must be on PATH for stage 1 (data rendering).
#   e.g.  export PATH="/e/Apps/Blender:$PATH"
# Use the `yolo` conda env (Ultralytics 8.4.x, torch + CUDA).
```

---

## The symmetry problem

A symmetric object has several physically-identical keypoint labelings. Two naive
approaches both fail:

- **Fixed-id labels** (keypoint 1 is always the same physical corner): the same
  visual corner gets contradictory labels across viewpoints, so `pose_loss` never
  converges.
- **Position-canonicalized labels** (e.g. "leftmost corner = id 1"): training
  converges, but the 2D↔3D correspondence is destroyed, so **PnP can no longer
  solve the pose**.

### The fix: symmetry-aware min-loss

Labels keep **fixed physical ids** (PnP-safe). At training time the keypoint loss
is computed against **every proper-rotation symmetry** of the object and only the
**minimum** is back-propagated. The network is then free to converge to whichever
symmetric labeling is easiest for a given view, and its output always corresponds
to a real rigid pose that PnP can solve.

```
Loss = min over s in symmetries  of  keypoint_loss(pred, GT[permutation_s])
```

**Only proper rotations (det +1) are valid hypotheses.** Reflections (det −1, e.g.
a pure mirror) cannot be produced by any real camera, so including them would let
the network output a labeling PnP cannot solve. The valid symmetries are
auto-discovered from the 3D keypoints.

### Per-class symmetry (two segments)

The dataset has **two classes** sharing one YOLO-pose head (kpt_shape `[12, 3]`).
Each class has its own geometry and therefore its own symmetry permutations. The
loss recovers each foreground anchor's class from the batch, then minimizes over
**that class's** permutations — a TowerBase instance is never scored against
TowerTop's symmetries and vice versa.

Implementation:
- `utils/symmetry.py` — `compute_symmetry_perms(keypoints_3d)` auto-discovers a
  given object's proper-rotation symmetries; `load_symmetry_perms_per_class({cls:
  json_path})` returns a per-class dict.
- `train/symmetry_loss.py` — `patch_symmetry_aware_pose_loss(perms_by_class)`
  monkey-patches **both** Ultralytics keypoint-loss implementations (`v8PoseLoss`
  and `PoseLoss26`, the latter used by end-to-end YOLO26-pose) and wraps the `loss()`
  method to stash `batch["cls"]` so the patched keypoint loss can route each
  instance to its class's perms. Patches are **idempotent** (safe to call twice).
- `train/train.py` reads `model.keypoints_3d` (base) and `model.keypoints_3d_top`,
  builds the per-class dict, and applies the patch automatically before
  `model.train()`. Disable with `--no-symmetry` for debugging.

> **Metric note:** because the model may output any valid symmetric labeling, a
> naive keypoint-i-vs-GT-i metric (and Ultralytics' built-in pose mAP) reports
> large error even when the pose is perfect. `eval_dataset.py` minimizes over the
> same permutations the loss uses — per class — so it reports the **true** geometric
> accuracy.
>
> On our trained model (towerv63): symmetry-**ON** RMSE is 19 px (base) / 45 px (top)
> with OKS 0.98; symmetry-**OFF** RMSE inflates to ~250 px — exactly the artifact.

---

## Synthetic data rendering

`data_synth/render_dataset_batch.py` spawns Blender processes (one per batch) that
render randomized aerial views and write two-class YOLO-pose labels (12 keypoints +
visibility flag per object, 0-2 instances per image).

**Two-segment scene composition** (per-frame, weighted by `render.scene_weights`):

| Scene type | Weight | Description |
|-----------|--------|-------------|
| base_only | 0.25 | Only the ground-fixed base segment. Camera orbits base at 20–80 m. |
| top_only | 0.25 | Only the floating top segment. Camera orbits top at ≤20 m. |
| both | 0.50 | Base on ground, top suspended on the camera→base view ray at ≤20 m from camera (helicopter carrying the top past the camera down to the base). |

- Every 50th frame is a **pure-background** image (no objects, empty label) for
  hard-negative training.
- Top placement reserves the object's bbox radius as margin so the **entire** top
  (farthest keypoint) stays within `top_camera_distance` max (20 m).
- The top segment is oriented upright (native +Y → world +Z) with random yaw and
  a small rope-sway tilt (`render.top_tilt_jitter_deg`).

**Domain randomization** (all PnP-safe — camera intrinsics and object geometry are
never perturbed):

- Camera: distance, pitch, yaw, roll jitter (from `config.yaml`).
- Sun: direction, energy, and a warm/cool color tint, re-randomized per frame.
- World ambient/sky fill: strength + slight tint (`hdri_strength` range) — softens
  hard shadows so shadowed keypoints stay visible.
- Ground: background texture, randomized UV mapping (scale / planar rotation /
  offset) and roughness, so the same texture never looks identical twice.

**Background ordering:**

- Backgrounds are drawn **without replacement** from a shuffled deck — every
  background is used once before any repeats, giving even coverage.
- By default each run reshuffles from fresh OS entropy, so **every run gets a
  different background order**. Pass `--bg-seed N` for a reproducible order.

```bash
# Full render (reads num_train/num_val from config.yaml if flags omitted)
python data_synth/render_dataset_batch.py --num-train 5000 --num-val 500

# Small smoke render with a reproducible background order
python data_synth/render_dataset_batch.py --num-train 20 --num-val 5 --bg-seed 42
```

Output: `dataset/images/{train,val}/*.png`, `dataset/labels/{train,val}/*.txt`
(multi-instance, one line per visible object), and `dataset/data.yaml`
(`names: {0: TowerBase, 1: TowerTop}`).

---

## Evaluation

`eval_dataset.py` runs inference over a labeled dataset and reports **per-class**
detection rate, bbox IoU, keypoint OKS, and keypoint RMSE (px). It is
**symmetry-aware by default** and uses **per-class** symmetry permutations so
each object is scored against its own valid labelings. Matching is restricted to
same-class predictions (TowerBase GT ↔ TowerBase pred only).

```bash
# Evaluate on the rendered val split with symmetry-aware matching
python eval_dataset.py --dataset dataset --split val \
    --weights runs/pose/perspectives-n-point/towerv63/weights/best.pt \
    --iou-thresh 0.5

# --no-symmetry  raw keypoint-i vs GT-i matching (shows the symmetry artifact)
```

Metrics for our trained model (500 val images):

| Class | GT | Det | Rate | IoU | OKS | RMSE(px) |
|-------|----|-----|------|-----|-----|----------|
| TowerBase | 386 | 376 | 97.4% | 0.904 | 0.984 | 19.2 |
| TowerTop | 324 | 247 | 76.2% | 0.806 | 0.972 | 45.0 |
| OVERALL | 710 | 623 | 87.7% | 0.863 | 0.979 | 29.4 |

---

## Inference

Inference is **class-aware**: each detection is routed to its own class's 3D
keypoints for PnP pose solving. The annotated image draws the two classes in
different colours (base = red keypoints, top = orange), labels each object with
its class name + 6DoF pose text, and the JSON output carries `class_id` +
`class_name` per detection.

```bash
# Single image or a folder of images
python infer/infer_image.py --input some.jpg  --output infer_out
python infer/infer_image.py --input some_dir/ --output infer_out
#   -> infer_out/<name>_annot.jpg  (annotated)  +  <name>_pose.json (R, t, reproj_err, class)

# Video file / webcam (press q to quit)
python infer/infer_video.py --input some.mp4 --output out.mp4
python infer/infer_video.py --input 0

# Use a specific checkpoint
python infer/infer_image.py --input some.jpg --weights path/to/best.pt
```

Pose solving (`infer/pose_solver.py`): RANSAC + `SOLVEPNP_SQPNP`, then LM
refinement on the inliers. Keypoints below `infer.conf_thresh` are not fed to PnP;
needs ≥4 confident keypoints.

### Relative pose & flight instruction

When the frame contains **exactly one TowerBase and one TowerTop**, the inference
pipeline also computes the Top-relative-to-Base 6DoF pose and a **helicopter flight
instruction** — a directional text telling the operator where to move so the suspended
Top aligns over the ground-fixed Base.

```
T_base_top = T_cam_base⁻¹ × T_cam_top    (Top expressed in Base's frame)
move_cam   = t_base − t_top              (helicopter displacement, camera frame)
```

The move vector is interpreted in the camera's coordinate frame (helicopter operator
view, camera looking straight down):

| Axis | + direction | − direction | Meaning |
|------|------------|------------|---------|
| X | 向右 (right) | 向左 (left) | image-right / image-left |
| Y | 向后 (back) | 向前 (forward) | image-down / image-up |
| Z | 向下 (down) | 向上 (up) | deeper (descend) / shallower (climb) |

A small deadzone (0.5 m) suppresses negligible axes. The instruction is drawn as a
banner on the annotated image (`draw_text` via PIL + TrueType CJK font) and written
to `_pose.json` under a `relative_pose` key.

Example output (img_000002, both segments detected):
```
请向下41.6m、向右15.0m、向后0.6m
move down 41.6m, right 15.0m, back 0.6m
```

Implementation: `infer/relative_pose.py` (`compute_relative_pose` +
`flight_instruction`).

---

## Project layout

```
config.yaml              all tunable parameters (camera/model/render/train/infer)
camera.yaml              camera intrinsics K, distortion, image size
mesh/Base.obj            tower base segment model (units: meters)
mesh/TowerTop.obj        tower top segment model (units: meters)
keypoints_3d_original.json  12 keypoints for TowerBase (object-local frame)
keypoints_3d_top.json   12 keypoints for TowerTop (object-local frame)

data_synth/
  blender_render.py        runs inside Blender: render + label one batch (two-class)
  render_dataset_batch.py  spawns batched Blender processes (main entry)
  render_dataset.py        single-process renderer (small jobs)
train/
  train.py                 training entry; applies per-class symmetry-aware loss
  symmetry_loss.py         patches Ultralytics keypoint loss (v8PoseLoss + PoseLoss26)
utils/
  symmetry.py              auto-discover proper-rotation keypoint permutations
  camera.py / config.py / keypoints.py   loaders
infer/
  pipeline.py              class-aware YOLO inference + per-class PnP
  pose_solver.py           RANSAC-SQPNP + LM pose solve
  relative_pose.py         Top-relative-to-Base 6DoF + helicopter flight instruction
  visualize.py             draw keypoints / 3D bbox / axes / per-class pose text
  infer_image.py / infer_video.py        CLI entries
eval_dataset.py           per-class symmetry-aware accuracy evaluation
scripts/                  sanity checks (e.g. PnP from GT labels)
tests/                    pytest unit + Blender integration tests (15 passing)
```

## Key config parameters (`config.yaml`)

| Parameter | Meaning |
|-----------|---------|
| `model.obj` / `model.obj_top` | base / top segment OBJ paths |
| `model.keypoints_3d` / `keypoints_3d_top` | 3D keypoints JSON per class; also the source for symmetry discovery |
| `model.classes` | class-id → name mapping (`{0: TowerBase, 1: TowerTop}`) |
| `render.num_train` / `num_val` | number of images to render |
| `render.scene_weights` | `[base_only, top_only, both]` per-frame sampling weights |
| `render.camera_distance` | camera distance range from tower (m) for base-only scenes |
| `render.top_camera_distance` | camera→top distance range (m); hard cap so top stays ≤20 m |
| `render.top_tilt_jitter_deg` | rope-sway tilt of the suspended top segment |
| `render.pitch_deg` | pitch range; 90° = straight down |
| `render.sun_strength` / `hdri_strength` | sun energy / ambient-fill ranges |
| `train.epochs` / `imgsz` / `batch` | training hyperparameters |
| `train.pose` | pose-loss gain |
| `infer.weights` | default model checkpoint |
| `infer.conf_thresh` | keypoint confidence threshold (below → not fed to PnP) |
| `infer.det_conf_thresh` | detection box confidence threshold |
| `infer.ransac_reproj_err` / `ransac_iters` | RANSAC-PnP settings |

**RTX 3060 Laptop (6 GB) OOM?** Lower `train.batch` (16 → 8 → 4) or `train.imgsz`
(1088 → 960). If `yolo26n-pose.pt` can't be auto-downloaded, set `train.weights`
to a locally available checkpoint (e.g. `yolo11n-pose.pt`).

## Tests

```bash
pytest -v          # PnP unit tests + Blender integration tests (if Blender on PATH)
```
