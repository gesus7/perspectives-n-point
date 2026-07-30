import argparse
import json
import sys
from pathlib import Path
import numpy as np

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from infer.pipeline import run_on_frame
from utils.camera import load_camera
from utils.config import load_config
from utils.keypoints import load_keypoints_3d


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_keypoints_by_class(cfg) -> dict:
    """Map each class id to its 3D keypoints array. Base (0) -> keypoints_3d,
    Top (1) -> keypoints_3d_top (if configured)."""
    model = cfg["model"]
    by_cls = {0: load_keypoints_3d(model["keypoints_3d"])[0]}
    if model.get("keypoints_3d_top"):
        by_cls[1] = load_keypoints_3d(model["keypoints_3d_top"])[0]
    return by_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--output", default="infer_out", help="output directory")
    parser.add_argument("--weights",
                        help="override config.infer.weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    K, dist = load_camera(cfg["camera"]["path"])
    kpts_3d_by_class = load_keypoints_by_class(cfg)
    class_names = cfg["model"].get("classes", {0: "TowerBase", 1: "TowerTop"})
    weights = args.weights or cfg["infer"]["weights"]
    print(f"Using weights: {weights}")
    model = YOLO(weights)

    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ob_in_cam_dir = out_dir / "ob_in_cam"
    ob_in_cam_dir.mkdir(parents=True, exist_ok=True)

    # 统计信息
    total_frames = 0
    missed_frames = 0

    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"[skip] unreadable: {f}")
            continue

        total_frames += 1

        annotated, results, relative = run_on_frame(
            model, frame, K, dist, kpts_3d_by_class,
            conf_thresh=cfg["infer"]["conf_thresh"],
            det_conf_thresh=cfg["infer"]["det_conf_thresh"],
            axis_len_m=cfg["infer"]["axis_len_m"],
            class_names=class_names,
        )
        cv2.imwrite(str(out_dir / f"{f.stem}_annot.jpg"), annotated)

        pose_json = {"detections": [], "relative_pose": None}

        txt_path = ob_in_cam_dir / f"{f.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            for i, r in enumerate(results):
                pose_json["detections"].append({
                    "class_id": r["class_id"],
                    "class_name": r["class_name"],
                    "R": r["R"].tolist(),
                    "t": r["t"].tolist(),
                    "reproj_err_px": r["reproj_err"],
                    "inliers": [int(x) for x in r["inliers"]],
                })

                # 构建 4x4 位姿变换矩阵 (6DoF): T = [R|t; 0 0 0 1]
                R = r["R"]
                t = r["t"].flatten()
                T_matrix = np.eye(4)
                T_matrix[:3, :3] = R
                T_matrix[:3, 3] = t

                # 每个目标写入区分头（类别名 + 序号）
                txt_file.write(f"# Object {i} {r['class_name']}\n")
                np.savetxt(txt_file, T_matrix, fmt="%.6f", delimiter=" ")
                txt_file.write("\n")

            # TowerTop 相对 TowerBase 的位姿 + 直升机对齐飞行指令
            if relative is not None:
                instr = relative["instruction"]
                pose_json["relative_pose"] = {
                    "R_top_in_base": relative["R_rel"].tolist(),
                    "t_top_in_base": relative["t_rel"].tolist(),
                    "dist_m": relative["dist_m"],
                    "instruction_zh": instr["zh"],
                    "instruction_en": instr["en"],
                    "move_cam_m": relative["move_cam"].tolist(),
                }
                T_rel = np.eye(4)
                T_rel[:3, :3] = relative["R_rel"]
                T_rel[:3, 3] = relative["t_rel"]
                txt_file.write("# TowerTop relative to TowerBase\n")
                np.savetxt(txt_file, T_rel, fmt="%.6f", delimiter=" ")
                txt_file.write(f"# instruction: {instr['zh']}  ({instr['en']})\n")

        (out_dir / f"{f.stem}_pose.json").write_text(
            json.dumps(pose_json, indent=2, ensure_ascii=False), encoding="utf-8")

        # 统计漏检
        if len(results) == 0:
            missed_frames += 1
            print(f"[miss] {f.name}: 0 detection(s)")
        else:
            extra = f"  -> {relative['instruction']['zh']}" if relative else ""
            print(f"[ok] {f.name}: {len(results)} detection(s){extra}")

    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"Total frames: {total_frames}")
    print(f"Missed frames: {missed_frames}")
    if total_frames > 0:
        miss_rate = (missed_frames / total_frames) * 100
        print(f"Miss rate: {miss_rate:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
