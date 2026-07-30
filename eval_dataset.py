"""Evaluate a two-class YOLO-pose model (0=TowerBase, 1=TowerTop) on a labeled
YOLO-format dataset, per class.

Metrics (reported per class + overall):
  - Detection rate (best same-class pred with IoU > thresh)
  - Mean bbox IoU
  - Per-keypoint OKS (Object Keypoint Similarity), mean across visible GT kpts
  - Keypoint RMSE (pixels, visible GT kpts only)

Each class is matched only against same-class predictions and scored with ITS OWN
proper-rotation symmetry permutations (symmetry-aware, like the training loss), so
the reported keypoint accuracy is the TRUE geometric accuracy — not the artifact
the built-in keypoint-i-vs-GT-i metric reports for symmetric objects.

Usage:
    python eval_dataset.py --dataset dataset --split val \
        --weights runs/pose/perspectives-n-point/towerv63/weights/best.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.camera import load_camera  # noqa: F401  (kept for parity / future use)
from utils.config import load_config
from utils.symmetry import load_symmetry_perms_per_class


def parse_label(txt: Path, w: int, h: int):
    """Return list of (cls, bbox_xyxy, kpts_px) per instance. Skip blank lines."""
    lines = txt.read_text().strip().splitlines()
    instances = []
    for line in lines:
        vals = list(map(float, line.split()))
        if len(vals) < 5:
            continue
        cls = int(vals[0])
        cx, cy, bw, bh = vals[1] * w, vals[2] * h, vals[3] * w, vals[4] * h
        bbox = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        kp_raw = vals[5:]
        n = len(kp_raw) // 3
        kpts = np.array(kp_raw[:n * 3]).reshape(n, 3)
        kpts[:, 0] *= w
        kpts[:, 1] *= h
        instances.append((cls, bbox, kpts))
    return instances


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def oks_single(pred_kpts, gt_kpts, bbox_area, sigma=0.1):
    """OKS for one instance. gt_kpts shape (N,3) with vis in col 2."""
    visible = gt_kpts[:, 2] >= 1
    if not visible.any():
        return np.nan
    d2 = np.sum((pred_kpts[visible, :2] - gt_kpts[visible, :2]) ** 2, axis=1)
    s2 = (2 * sigma) ** 2 * bbox_area * 2
    return float(np.mean(np.exp(-d2 / s2)))


def best_symmetry_gt(pred_kpts, gt_kpts, perms):
    """Pick the symmetry-permuted GT labeling closest to the prediction.

    The symmetry-aware training loss lets the model converge to ANY proper-rotation
    labeling of the symmetric object, all of which are valid rigid poses PnP solves
    correctly. To measure true geometric accuracy we minimize over the same
    permutations the loss uses. Returns the reordered gt_kpts (N,3).
    """
    visible = gt_kpts[:, 2] >= 1
    if not visible.any():
        return gt_kpts
    best_gt, best_err = gt_kpts, np.inf
    for perm in perms:
        cand = gt_kpts[perm]
        vis = cand[:, 2] >= 1
        d = np.linalg.norm(pred_kpts[vis, :2] - cand[vis, :2], axis=1)
        err = float(np.mean(d)) if d.size else np.inf
        if err < best_err:
            best_err, best_gt = err, cand
    return best_gt


def _resolve_split(dataset: Path, split: str):
    """Support both flat (images/*.png) and split (images/<split>/*.png) layouts."""
    split_img = dataset / "images" / split
    if split_img.is_dir():
        return split_img, dataset / "labels" / split
    return dataset / "images", dataset / "labels"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="dataset")
    p.add_argument("--split",      default="val", help="images/<split> subdir if present")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--weights",    default=None,
                   help="model weights (default: cfg['infer']['weights'])")
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--no-symmetry", action="store_true",
                   help="disable symmetry-aware GT matching (raw keypoint-i vs GT-i)")
    args = p.parse_args()

    cfg = load_config(args.config)
    weights = args.weights or cfg["infer"]["weights"]
    model = YOLO(weights)
    print(f"Model: {weights}")
    conf_thresh = cfg["infer"]["det_conf_thresh"]
    class_names = cfg["model"].get("classes", {0: "TowerBase", 1: "TowerTop"})

    # Per-class symmetry permutations (each class scored against its own symmetries).
    class_kpts = {0: cfg["model"]["keypoints_3d"]}
    if cfg["model"].get("keypoints_3d_top"):
        class_kpts[1] = cfg["model"]["keypoints_3d_top"]
    if args.no_symmetry:
        n_k = cfg["model"]["num_keypoints"]
        perms_by_class = {c: [list(range(n_k))] for c in class_kpts}
        print("Symmetry-aware matching: OFF (raw)\n")
    else:
        perms_by_class = load_symmetry_perms_per_class(class_kpts)
        counts = {class_names.get(c, c): len(p) for c, p in perms_by_class.items()}
        print(f"Symmetry-aware matching: ON, per-class labelings: {counts}\n")

    img_dir, lbl_dir = _resolve_split(Path(args.dataset), args.split)
    images = sorted(img_dir.glob("*.png"))
    print(f"Found {len(images)} images in {img_dir}\n")

    # Per-class accumulators.
    iou_scores = defaultdict(list)
    oks_scores = defaultdict(list)
    rmse_list = defaultdict(list)
    n_gt = defaultdict(int)
    n_detected = defaultdict(int)
    n_background = 0
    n_false_pos = 0

    for img_path in images:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        gt_instances = parse_label(lbl_path, w, h) if lbl_path.exists() else []
        res = model.predict(frame, verbose=False)[0]

        # Background frame (no GT): any confident detection is a false positive.
        if not gt_instances:
            n_background += 1
            detected = res.boxes is not None and len(res.boxes) > 0 and \
                float(res.boxes.conf.max()) >= conf_thresh
            if detected:
                n_false_pos += 1
            continue

        if res.boxes is None or len(res.boxes) == 0:
            for (cls, _, _) in gt_instances:
                n_gt[cls] += 1
            continue

        pred_boxes = res.boxes.xyxy.cpu().numpy()
        pred_confs = res.boxes.conf.cpu().numpy()
        pred_cls   = res.boxes.cls.cpu().numpy().astype(int)
        pred_kpts  = res.keypoints.xy.cpu().numpy()   # (N, K, 2)

        for (cls, gt_bbox, gt_kpts) in gt_instances:
            n_gt[cls] += 1
            # Only same-class, above-conf predictions are candidates.
            cand = [i for i in range(len(pred_boxes))
                    if pred_cls[i] == cls and pred_confs[i] >= conf_thresh]
            if not cand:
                continue
            best_idx = max(cand, key=lambda i: iou(pred_boxes[i], gt_bbox))
            best_iou = iou(pred_boxes[best_idx], gt_bbox)
            iou_scores[cls].append(best_iou)
            if best_iou < args.iou_thresh:
                continue

            n_detected[cls] += 1
            pk = pred_kpts[best_idx]
            gk = best_symmetry_gt(pk, gt_kpts, perms_by_class.get(cls, [list(range(len(gt_kpts)))]))
            visible = gk[:, 2] >= 1
            bbox_area = (gt_bbox[2]-gt_bbox[0]) * (gt_bbox[3]-gt_bbox[1])
            oks_scores[cls].append(oks_single(pk, gk, bbox_area))
            dists = np.linalg.norm(pk[visible, :2] - gk[visible, :2], axis=1)
            rmse_list[cls].append(float(np.sqrt(np.mean(dists ** 2))))

    # ---- Report ----
    classes = sorted(set(n_gt) | set(class_kpts))
    print("=" * 72)
    print(f"{'Class':<14} {'GT':>5} {'Det':>5} {'DetRate':>8} {'IoU':>7} {'OKS':>7} {'RMSE(px)':>9}")
    print("-" * 72)
    for c in classes:
        name = str(class_names.get(c, c))
        g, d = n_gt[c], n_detected[c]
        rate = f"{100*d/g:.1f}%" if g else "—"
        miou = f"{np.mean(iou_scores[c]):.3f}" if iou_scores[c] else "—"
        moks = f"{np.nanmean(oks_scores[c]):.3f}" if oks_scores[c] else "—"
        mrmse = f"{np.mean(rmse_list[c]):.2f}" if rmse_list[c] else "—"
        print(f"{name:<14} {g:>5} {d:>5} {rate:>8} {miou:>7} {moks:>7} {mrmse:>9}")

    # Overall (pooled).
    all_iou = [v for c in classes for v in iou_scores[c]]
    all_oks = [v for c in classes for v in oks_scores[c]]
    all_rmse = [v for c in classes for v in rmse_list[c]]
    tot_gt = sum(n_gt.values())
    tot_det = sum(n_detected.values())
    print("-" * 72)
    rate = f"{100*tot_det/tot_gt:.1f}%" if tot_gt else "—"
    miou = f"{np.mean(all_iou):.3f}" if all_iou else "—"
    moks = f"{np.nanmean(all_oks):.3f}" if all_oks else "—"
    mrmse = f"{np.mean(all_rmse):.2f}" if all_rmse else "—"
    print(f"{'OVERALL':<14} {tot_gt:>5} {tot_det:>5} {rate:>8} {miou:>7} {moks:>7} {mrmse:>9}")
    print("=" * 72)
    print(f"Images: {len(images)}  |  background: {n_background}  |  "
          f"false-positive bg frames: {n_false_pos}")


if __name__ == "__main__":
    main()
