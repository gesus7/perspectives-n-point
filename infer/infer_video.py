import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from infer.pipeline import run_on_frame
from infer.infer_image import load_keypoints_by_class
from utils.camera import load_camera
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True,
                        help="video path or camera index (e.g. 0)")
    parser.add_argument("--output", default=r"F:\PoseEstimation\perspectives-n-point\infer_out", help="output mp4 path (optional)")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-show", action="store_true",
                        help="don't open a display window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    K, dist = load_camera(cfg["camera"]["path"])
    kpts_3d_by_class = load_keypoints_by_class(cfg)
    class_names = cfg["model"].get("classes", {0: "TowerBase", 1: "TowerTop"})
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
            annotated, _, _ = run_on_frame(
                model, frame, K, dist, kpts_3d_by_class,
                conf_thresh=cfg["infer"]["conf_thresh"],
                det_conf_thresh=cfg["infer"]["det_conf_thresh"],
                axis_len_m=cfg["infer"]["axis_len_m"],
                class_names=class_names,
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
