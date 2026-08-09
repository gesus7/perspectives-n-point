"""Video inference with ByteTrack tracking + pilot target selection.

Usage
-----
python infer/track_video.py --input some.mp4 --output out.mp4 \\
    --weights runs/pose/.../best.pt

Behaviour
---------
* Uses model.track(persist=True, tracker="bytetrack.yaml") — Ultralytics handles
  Kalman-filter motion prediction between frames, which uses the previous frame's
  result as a prior and keeps identities stable across occlusions.
* State machine (infer/tracker_state.py) drives locking:
    - 0 base targets  → keep searching, draw all detections grey
    - 1 base target   → auto-lock, full pose annotation
    - > 1 base targets, first time → PAUSE video, show numbered boxes, wait for
      pilot to CLICK one → lock that track_id, NEVER show popup again this run
    - > 1 base targets, popup already used → auto-lock highest-confidence base
    - locked target lost > patience frames → re-enter searching
* While locked, ONLY the locked TowerBase (and the single TowerTop if visible) are
  fully annotated and sent to PnP.  All other detections are drawn as faint grey
  boxes so the pilot sees what else is around but is not confused by extra pose text.
* Relative pose (TowerTop relative to TowerBase) + flight instruction banner are
  drawn whenever both a locked base and a single TowerTop are solved, same as before.

Interaction
-----------
* During SELECTING: video pauses on the frozen frame.  Candidate bases are numbered
  1, 2, 3 … in bright yellow.  Click any numbered box to lock it.  Press ESC or 'q'
  to cancel (auto-lock highest-confidence instead).
* During normal playback: press 'r' to RESET (release lock, re-enter searching),
  press 'q' or ESC to quit.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from infer.infer_image import load_keypoints_by_class
from infer.pipeline import (parse_detections, solve_and_annotate, annotate_pose,
                            _relative_pose_and_banner)
from infer.pose_tracker import PoseTracker
from infer.tracker_state import TrackSelector
from infer.visualize import draw_text
from utils.camera import load_camera
from utils.config import load_config

# ── constants ──────────────────────────────────────────────────────────────────
_BASE_CLASS   = 0
_TOP_CLASS    = 1
_GREY         = (160, 160, 160)
_YELLOW       = (0, 230, 230)
_CYAN         = (255, 230, 0)
_GREEN        = (0, 220, 80)
_AMBER        = (0, 200, 255)   # predicted (coasted) pose colour
_WIN          = "Transmission Tower Tracker"


# ── helpers ────────────────────────────────────────────────────────────────────

def _draw_ghost(frame, box_xyxy, label: str = ""):
    """Draw a faint grey box (non-locked detection) so it is visible but muted."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), _GREY, 1)
    if label:
        cv2.putText(frame, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREY, 1, cv2.LINE_AA)


def _draw_candidate(frame, box_xyxy, number: int):
    """Draw a bright candidate box with a big number for the selection popup."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), _YELLOW, 3)
    label = str(number)
    font_scale = 2.2
    thick = 3
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    cx = (x1 + x2) // 2 - tw // 2
    cy = (y1 + y2) // 2 + th // 2
    cv2.putText(frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _YELLOW, thick, cv2.LINE_AA)


def _draw_locked_border(frame):
    """Green corner brackets to show a target is locked."""
    h, w = frame.shape[:2]
    L = 60
    T = 4
    c = _GREEN
    for (x, y, dx, dy) in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
        cv2.line(frame, (x, y), (x + dx * L, y), c, T)
        cv2.line(frame, (x, y), (x, y + dy * L), c, T)


def _draw_status(frame, label: str):
    h = frame.shape[0]
    draw_text(frame, label, (12, h - 90), color=_CYAN, font_size=26, bg=(30, 30, 30))


def _draw_selecting_prompt(frame):
    draw_text(frame, "请点击目标编号框来锁定  |  Click a numbered box to lock",
              (12, 12), color=_YELLOW, font_size=28, bg=(0, 0, 100))


def _box_contains(box_xyxy, px: int, py: int) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


# ── mouse callback for selection ──────────────────────────────────────────────

class _MouseState:
    def __init__(self):
        self.clicked_xy = None

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_xy = (x, y)


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       default="config.yaml")
    ap.add_argument("--input",        required=True, help="video path or camera index")
    ap.add_argument("--output",       default=None, help="output mp4 path (optional)")
    ap.add_argument("--weights",      default=None)
    ap.add_argument("--tracker",      default="bytetrack.yaml",
                    help="Ultralytics tracker config (bytetrack.yaml or botsort.yaml)")
    ap.add_argument("--lost-patience",type=int, default=30,
                    help="frames to wait before releasing a lost lock")
    ap.add_argument("--no-show",      action="store_true")
    args = ap.parse_args()

    cfg           = load_config(args.config)
    K, dist       = load_camera(cfg["camera"]["path"])
    kpts_by_cls   = load_keypoints_by_class(cfg)
    class_names   = cfg["model"].get("classes", {0: "TowerBase", 1: "TowerTop"})
    conf_thresh   = cfg["infer"]["conf_thresh"]
    det_thresh    = cfg["infer"]["det_conf_thresh"]
    axis_len      = cfg["infer"]["axis_len_m"]
    weights       = args.weights or cfg["infer"]["weights"]

    model = YOLO(weights)

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.input}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    if not args.no_show:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    mouse = _MouseState()
    if not args.no_show:
        cv2.setMouseCallback(_WIN, mouse.callback)

    selector = TrackSelector(tracked_class=_BASE_CLASS, lost_patience=args.lost_patience)

    # Pose filter for the locked base — bridges YOLO drop-outs by predicting R,t from
    # constant-velocity motion while the detection is momentarily lost. Rebuilt each
    # time a new target is locked; None while searching. max_lost caps how many frames
    # we coast before giving up on prediction (tied to the same patience budget).
    base_tracker: PoseTracker | None = None
    tracked_lock_id: int | None = None

    # Frozen frame kept during SELECTING popup
    frozen_display: np.ndarray | None = None
    # Candidate bases for the current popup
    candidates: list[dict] = []

    print(f"[track] weights={weights}  tracker={args.tracker}  input={args.input}")

    try:
        while True:
            # ── read frame ─────────────────────────────────────────────
            ok, frame = cap.read()
            if not ok:
                break
            out = frame.copy()

            # ── run tracking ───────────────────────────────────────────
            track_results = model.track(frame, persist=True,
                                        tracker=args.tracker, verbose=False)
            all_dets = []
            for res in track_results:
                for det in parse_detections(res, det_thresh, kpts_by_cls, class_names):
                    all_dets.append(det)

            # ── advance state machine ──────────────────────────────────
            selector.update(all_dets)

            # ── check if popup should fire ─────────────────────────────
            if selector.needs_selection():
                candidates = [d for d in all_dets
                              if d["class_id"] == _BASE_CLASS
                              and d.get("track_id") is not None]
                # Build the frozen display frame with numbered boxes
                frozen_display = out.copy()
                for num, cand in enumerate(candidates, 1):
                    _draw_candidate(frozen_display, cand["box_xyxy"], num)
                _draw_selecting_prompt(frozen_display)

            # ── SELECTING: pause and wait for click ────────────────────
            if selector.state.name == "SELECTING":
                display = frozen_display.copy()
                _draw_status(display, f"状态: 选择目标  [{len(candidates)}个地面塔]")

                if not args.no_show:
                    cv2.imshow(_WIN, display)

                # Process mouse click
                if mouse.clicked_xy is not None:
                    cx, cy = mouse.clicked_xy
                    mouse.clicked_xy = None
                    for cand in candidates:
                        if _box_contains(cand["box_xyxy"], cx, cy):
                            selector.select(cand["class_id"], cand["track_id"])
                            print(f"[track] Pilot selected cls={cand['class_id']} "
                                  f"id={cand['track_id']}")
                            break

                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):  # q or ESC: auto-lock best and continue
                    if candidates:
                        best = max(candidates, key=lambda d: d.get("det_conf", 0.0))
                        selector.select(best["class_id"], best["track_id"])
                        print(f"[track] ESC: auto-locked cls={best['class_id']} "
                              f"id={best['track_id']}")
                    else:
                        break
                continue  # don't render anything else this frame

            # ── Normal frame rendering ─────────────────────────────────
            locked_base_id  = selector.locked_id()
            solved_results  = []

            # Separate base candidates from tops
            base_dets = [d for d in all_dets if d["class_id"] == _BASE_CLASS]
            top_dets  = [d for d in all_dets if d["class_id"] == _TOP_CLASS]

            # ── LOCKED base: full pose on the one locked track ──────────
            locked_base_result = None
            if locked_base_id is not None:
                # (Re)create the pose filter whenever a new id is locked.
                if base_tracker is None or tracked_lock_id != locked_base_id:
                    base_tracker = PoseTracker(max_lost=args.lost_patience)
                    tracked_lock_id = locked_base_id

                locked_dets = [d for d in base_dets if d["track_id"] == locked_base_id]
                r = None
                if locked_dets:
                    r = solve_and_annotate(out, locked_dets[0], K, dist,
                                           conf_thresh, axis_len, slot=0,
                                           kpts_3d=kpts_by_cls[_BASE_CLASS])

                if r is not None:
                    # Good PnP this frame — correct the filter and use the detection.
                    base_tracker.observe(r["R"], r["t"])
                    locked_base_result = r
                    solved_results.append(r)
                else:
                    # YOLO missed the locked base (occlusion / detection gap). Coast on
                    # the filter's prediction and draw it in amber, if still alive.
                    pred = base_tracker.observe(None, None)
                    if pred is not None:
                        pr = annotate_pose(out, pred["R"], pred["t"], K, dist,
                                           kpts_by_cls[_BASE_CLASS], axis_len_m=axis_len,
                                           slot=0, class_id=_BASE_CLASS,
                                           class_name=class_names.get(_BASE_CLASS, "Base"),
                                           track_id=locked_base_id, color=_AMBER)
                        locked_base_result = pr
                        solved_results.append(pr)

                # Ghost-draw other unlocked bases
                for d in base_dets:
                    if d["track_id"] != locked_base_id:
                        lbl = f"Base#{d['track_id']}" if d["track_id"] else "Base"
                        _draw_ghost(out, d["box_xyxy"], lbl)
            else:
                # SEARCHING / LOST: no active lock, drop any stale filter.
                base_tracker = None
                tracked_lock_id = None
                for d in base_dets:
                    lbl = f"Base#{d['track_id']}" if d["track_id"] else "Base"
                    _draw_ghost(out, d["box_xyxy"], lbl)

            # ── TowerTop: full pose if exactly one visible ──────────────
            locked_top_result = None
            if len(top_dets) == 1:
                r = solve_and_annotate(out, top_dets[0], K, dist,
                                       conf_thresh, axis_len, slot=len(solved_results),
                                       kpts_3d=kpts_by_cls[_TOP_CLASS])
                if r is not None:
                    locked_top_result = r
                    solved_results.append(r)
            elif len(top_dets) > 1:
                for d in top_dets:
                    _draw_ghost(out, d["box_xyxy"],
                                f"Top#{d['track_id']}" if d["track_id"] else "Top")

            # ── Relative pose banner (needs exactly 1 locked base + 1 top) ──
            if locked_base_result and locked_top_result:
                _relative_pose_and_banner(out, [locked_base_result, locked_top_result])

            # ── UI decoration ──────────────────────────────────────────
            if selector.state.name == "LOCKED":
                _draw_locked_border(out)
            _draw_status(out, f"状态: {selector.status_label()}  |  r=重置  q=退出")

            if writer is not None:
                writer.write(out)
            if not args.no_show:
                cv2.imshow(_WIN, out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    selector = TrackSelector(tracked_class=_BASE_CLASS,
                                             lost_patience=args.lost_patience)
                    model.track(None, persist=False)   # reset tracker state
                    print("[track] Reset — searching for new target")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
