import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from utils.config import load_config
from utils.symmetry import load_symmetry_perms_per_class
from train.symmetry_loss import patch_symmetry_aware_pose_loss


def _class_keypoint_paths(cfg) -> dict:
    """Map each class id to its keypoints JSON. Base (0) -> keypoints_3d,
    Top (1) -> keypoints_3d_top; falls back to base-only for single-class configs."""
    model = cfg["model"]
    paths = {0: model["keypoints_3d"]}
    if model.get("keypoints_3d_top"):
        paths[1] = model["keypoints_3d_top"]
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--no-symmetry", action="store_true",
                   help="disable symmetry-aware pose loss (debug)")
    args = p.parse_args()

    cfg = load_config(args.config)
    t = cfg["train"]

    # Enable per-class symmetry-aware pose loss so the network can converge on the
    # symmetric base AND top segments while keeping fixed physical keypoint ids
    # (PnP-solvable at inference). Each class is scored against its own symmetries.
    if not args.no_symmetry:
        perms_by_class = load_symmetry_perms_per_class(_class_keypoint_paths(cfg))
        if any(len(p) > 1 for p in perms_by_class.values()):
            patch_symmetry_aware_pose_loss(perms_by_class)
        else:
            print("[train] no non-trivial symmetry found; using standard pose loss")

    model = YOLO(t["weights"])
    model.train(
        data=args.data,
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        device=t["device"],
        project=t["project"],
        name=t["name"],
        warmup_epochs=t.get("warmup_epochs", 5),
        pose=t.get("pose", 12.0),
    )


if __name__ == "__main__":
    main()
