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
    n_empty = 0
    n_fail = 0
    for lab in sorted(Path(args.labels).glob("*.txt")):
        #line = lab.read_text().splitlines()[0]
        lines = lab.read_text().splitlines()
        if not lines:
            n_empty += 1
            continue
        line = lines[0]
        kpts_2d, conf = parse_label(line, w, h)
        res = solve_pose(kpts_2d, conf, kpts_3d, K, dist, conf_thresh=0.5)
        if res is None:
            n_fail += 1
            continue
        errs.append(res["reproj_err"])
        n_ok += 1
    if not errs:
        raise SystemExit("All frames failed — labels or PnP broken.")
    print(f"frames ok={n_ok} empty={n_empty}")
    print(f"reproj_err px: median={np.median(errs):.3f} "
          f"p95={np.percentile(errs,95):.3f} max={np.max(errs):.3f}")
    if np.median(errs) > 1.0:
        raise SystemExit("Median reproj error > 1px on GT — alignment problem.")


if __name__ == "__main__":
    main()
