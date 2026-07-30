"""Visualize predicted vs GT keypoints on val images.

Usage:
    python scripts/viz_pred_vs_gt.py --weights runs/.../best.pt --data dataset/data.yaml --n 12
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_gt(label_path: Path, img_w: int, img_h: int, n_kpts: int):
    """Return list of (bbox_xyxy, kpts_px) from a YOLO-pose label file."""
    results = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 + n_kpts * 3:
            continue
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        kpts = []
        for i in range(n_kpts):
            kx = float(parts[5 + i * 3]) * img_w
            ky = float(parts[5 + i * 3 + 1]) * img_h
            vis = int(parts[5 + i * 3 + 2])
            kpts.append((kx, ky, vis))
        results.append(((x1, y1, x2, y2), kpts))
    return results


def draw(img, gt_instances, pred_result, n_kpts):
    out = img.copy()

    # GT: green bbox + cyan keypoints
    for (x1, y1, x2, y2), kpts in gt_instances:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i, (kx, ky, vis) in enumerate(kpts):
            if vis >= 1:
                cv2.circle(out, (int(kx), int(ky)), 6, (255, 255, 0), -1)
                cv2.putText(out, str(i), (int(kx) + 5, int(ky) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Predictions: red bbox + red keypoints
    if pred_result and pred_result.keypoints is not None:
        boxes = pred_result.boxes.xyxy.cpu().numpy()
        kpts_all = pred_result.keypoints.xy.cpu().numpy()  # (N, K, 2)
        conf_all = pred_result.keypoints.conf
        if conf_all is not None:
            conf_all = conf_all.cpu().numpy()
        for box, kpts in zip(boxes, kpts_all):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for i, (kx, ky) in enumerate(kpts):
                if kx > 0 or ky > 0:
                    cv2.circle(out, (int(kx), int(ky)), 4, (0, 0, 255), -1)
                    cv2.putText(out, str(i), (int(kx) + 5, int(ky) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    # Legend
    cv2.putText(out, "GT (cyan) | Pred (red)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--n", type=int, default=20, help="number of images to show")
    p.add_argument("--out", default="scripts/viz_out", help="output directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    n_kpts = data_cfg["kpt_shape"][0]
    data_root = Path(data_cfg["path"])
    val_img_dir = data_root / data_cfg["val"]
    val_lbl_dir = data_root / "labels" / "val"

    model = YOLO(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(val_img_dir.glob("*.png")) + sorted(val_img_dir.glob("*.jpg"))
    random.seed(args.seed)
    sample = random.sample(img_paths, min(args.n, len(img_paths)))

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        gt = load_gt(lbl_path, w, h, n_kpts) if lbl_path.exists() else []

        results = model(img_path, verbose=False)
        pred = results[0] if results else None

        out_img = draw(img, gt, pred, n_kpts)
        # Downscale for display if too large
        if w > 1280:
            scale = 1280 / w
            out_img = cv2.resize(out_img, (1280, int(h * scale)))

        cv2.imwrite(str(out_dir / img_path.name), out_img)
        print(f"saved {out_dir / img_path.name}")

    print(f"\nDone. Open {out_dir}/ to inspect results.")


if __name__ == "__main__":
    main()
