import cv2
import numpy as np

from infer.pose_solver import solve_pose
from infer.relative_pose import compute_relative_pose, flight_instruction
from infer.visualize import (draw_axes, draw_bbox3d, draw_keypoints, draw_text,
                             put_pose_text)

# Per-class keypoint draw colors (BGR): base = red, top = orange.
_CLASS_COLORS = {0: (0, 0, 255), 1: (0, 165, 255)}
_BASE_CLASS = 0
_TOP_CLASS = 1


def run_on_frame(
    model,
    frame: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    kpts_3d_by_class: dict[int, np.ndarray],
    conf_thresh: float = 0.5,
    det_conf_thresh: float = 0.25,
    axis_len_m: float = 5.0,
    class_names: dict[int, str] | None = None,
) -> tuple[np.ndarray, list[dict], dict | None]:
    """Run YOLO-pose inference + PnP on one frame, class-aware.

    Each detection is solved against ITS OWN class's 3D keypoints
    (kpts_3d_by_class[class_id]) — TowerBase (0) and TowerTop (1) have different
    geometry, so using the wrong set would break PnP.

    Returns (annotated frame BGR, list of per-detection result dicts, relative dict).
    Per-detection dicts have keys 'R','t','reproj_err','inliers','kpts_2d',
    'kpts_conf','class_id','class_name'. The relative dict is None unless exactly one
    TowerBase and one TowerTop were solved, in which case it holds the Top-relative-to-
    Base pose and a helicopter flight instruction (see infer/relative_pose.py).
    Detections below det_conf_thresh, of an unknown class, or with too few confident
    keypoints are skipped.
    """
    class_names = class_names or {}
    yolo_results = model.predict(frame, verbose=False)
    out_frame = frame.copy()
    out_results: list[dict] = []

    for res in yolo_results:
        for det in parse_detections(res, det_conf_thresh, kpts_3d_by_class, class_names):
            r = solve_and_annotate(out_frame, det, K, dist, conf_thresh, axis_len_m,
                                   slot=len(out_results))
            if r is not None:
                out_results.append(r)

    relative = _relative_pose_and_banner(out_frame, out_results)
    return out_frame, out_results, relative


def parse_detections(res, det_conf_thresh, kpts_3d_by_class, class_names):
    """Extract per-detection dicts from one Ultralytics result.

    Yields dicts with 'class_id','class_name','kpts_2d','kpts_conf','box_xyxy',
    'det_conf','track_id' (track_id is None unless the result came from model.track).
    Detections below det_conf_thresh or of an unknown class are skipped.
    """
    boxes = res.boxes
    kps = res.keypoints
    if boxes is None or kps is None:
        return
    det_conf = np.asarray(boxes.conf.cpu())
    det_cls = np.asarray(boxes.cls.cpu()).astype(int)
    box_xyxy = np.asarray(boxes.xyxy.cpu())
    kp_xy = np.asarray(kps.xy.cpu())
    kp_conf = np.asarray(kps.conf.cpu())
    # boxes.id is present only when tracking (model.track); None for predict().
    track_ids = None if boxes.id is None else np.asarray(boxes.id.cpu()).astype(int)

    for i in range(len(det_conf)):
        if float(det_conf[i]) < det_conf_thresh:
            continue
        cls_id = int(det_cls[i])
        if kpts_3d_by_class.get(cls_id) is None:
            continue
        yield {
            "class_id": cls_id,
            "class_name": class_names.get(cls_id, str(cls_id)),
            "kpts_2d": kp_xy[i].astype(np.float64),
            "kpts_conf": kp_conf[i].astype(np.float64),
            "box_xyxy": box_xyxy[i].astype(np.float64),
            "det_conf": float(det_conf[i]),
            "track_id": None if track_ids is None else int(track_ids[i]),
        }


def solve_and_annotate(out_frame, det, K, dist, conf_thresh, axis_len_m, slot=0,
                       kpts_3d_by_class=None, kpts_3d=None):
    """Solve PnP for one detection dict (from parse_detections) and draw it onto
    out_frame. Returns a result dict (pose + detection metadata) or None if PnP fails.

    Pass either kpts_3d (the class's 3D keypoints) or kpts_3d_by_class to look it up.
    """
    cls_id = det["class_id"]
    if kpts_3d is None:
        kpts_3d = (kpts_3d_by_class or {}).get(cls_id)
    if kpts_3d is None:
        return None
    color = _CLASS_COLORS.get(cls_id, (0, 0, 255))
    kpts_2d, kpts_conf = det["kpts_2d"], det["kpts_conf"]

    pose = solve_pose(kpts_2d, kpts_conf, kpts_3d, K, dist, conf_thresh=conf_thresh)
    draw_keypoints(out_frame, kpts_2d, kpts_conf, conf_thresh, color=color)
    if pose is None:
        return None

    rvec, _ = cv2.Rodrigues(pose["R"])
    reproj, _ = cv2.projectPoints(kpts_3d, rvec, pose["t"].reshape(3, 1), K, dist)
    draw_keypoints(out_frame, kpts_2d, kpts_conf, conf_thresh,
                   reproj_2d=reproj.reshape(-1, 2), color=color)
    draw_bbox3d(out_frame, pose["R"], pose["t"], K, dist, kpts_3d)
    draw_axes(out_frame, pose["R"], pose["t"], K, dist, axis_len_m)
    label = det["class_name"]
    if det.get("track_id") is not None:
        label = f"{label}#{det['track_id']}"
    put_pose_text(out_frame, pose["R"], pose["t"], label=label, slot=slot)

    return {
        **pose,
        "kpts_2d": kpts_2d,
        "kpts_conf": kpts_conf,
        "class_id": cls_id,
        "class_name": det["class_name"],
        "track_id": det.get("track_id"),
        "box_xyxy": det.get("box_xyxy"),
    }


def _relative_pose_and_banner(frame: np.ndarray, results: list[dict]) -> dict | None:
    """If exactly one TowerBase and one TowerTop were solved, compute the Top-relative-
    to-Base pose + a flight instruction and draw a banner at the bottom of the frame."""
    bases = [r for r in results if r["class_id"] == _BASE_CLASS]
    tops = [r for r in results if r["class_id"] == _TOP_CLASS]
    if len(bases) != 1 or len(tops) != 1:
        return None

    rel = compute_relative_pose(bases[0], tops[0])
    instr = flight_instruction(rel["move_cam"])
    rel["instruction"] = instr

    h = frame.shape[0]
    banner = f"对齐指令: {instr['zh']}  (间距 {rel['dist_m']:.1f}m)"
    draw_text(frame, banner, (12, h - 48), color=(0, 255, 255), font_size=30,
              bg=(40, 40, 40))
    return rel

