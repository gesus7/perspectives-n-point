"""Relative pose between the two tower segments + helicopter flight instruction.

Both segment poses are solved independently against the camera (pose["R"], pose["t"]
map object-frame points to the CAMERA frame: X_cam = R @ X_obj + t). From the two
camera-frame poses we recover the rigid transform of TowerTop relative to TowerBase,
and turn the camera-frame offset into a flight command for the helicopter operator.

Camera convention (OpenCV): +X = image-right, +Y = image-down, +Z = forward along the
optical axis (toward the ground, since the camera looks straight down). The helicopter
carries the suspended TowerTop, so to align it over the ground-fixed TowerBase it must
move by (t_base - t_top) expressed in the camera frame:
  +X -> 向右 (right) , -X -> 向左 (left)
  +Y -> 向后/下沉视角下为 image-down; we map the horizontal image axes to 左右/前后 and
        the optical axis (depth, +Z) to 向下/向上 (descend/climb), matching a top-down
        operator view where "forward/back" and "left/right" are the two image axes and
        "down" means closer to the ground.
"""
import numpy as np


# Camera-frame axis -> (positive token zh/en, negative token zh/en).
# Operator view, camera looking straight down:
#   X (image right/left)  -> 向右 / 向左
#   Y (image down/up)     -> 向后 / 向前   (image-down reads as "behind" the operator)
#   Z (optical axis depth)-> 向下 / 向上   (deeper = closer to ground = descend)
_AXIS_TOKENS = {
    "x": (("向右", "right"), ("向左", "left")),
    "y": (("向后", "back"), ("向前", "forward")),
    "z": (("向下", "down"), ("向上", "up")),
}


def pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from (R (3,3), t (3,))."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def compute_relative_pose(base_pose: dict, top_pose: dict) -> dict:
    """TowerTop pose relative to TowerBase.

    base_pose/top_pose: dicts with 'R' (3,3) and 't' (3,) in CAMERA frame.

    Returns dict with:
      'R_rel','t_rel'   : Top expressed in Base's frame (T_base_top = T_cam_base^-1 @ T_cam_top)
      'dist_m'          : distance between the two object origins (m)
      'move_cam'        : (3,) vector the helicopter must move (camera frame) to align
                          Top's origin onto Base's origin = t_base - t_top
    """
    T_cb = pose_to_matrix(base_pose["R"], base_pose["t"])   # cam <- base
    T_ct = pose_to_matrix(top_pose["R"], top_pose["t"])     # cam <- top
    T_bt = np.linalg.inv(T_cb) @ T_ct                       # base <- top
    t_base = np.asarray(base_pose["t"]).reshape(3)
    t_top = np.asarray(top_pose["t"]).reshape(3)
    move_cam = t_base - t_top
    return {
        "R_rel": T_bt[:3, :3],
        "t_rel": T_bt[:3, 3],
        "dist_m": float(np.linalg.norm(t_top - t_base)),
        "move_cam": move_cam,
    }


def flight_instruction(move_cam: np.ndarray, deadzone_m: float = 0.5) -> dict:
    """Turn the camera-frame move vector into a helicopter flight command.

    Names each axis whose magnitude exceeds `deadzone_m`, ordered by magnitude
    (largest first). Returns {'zh','en','components'} where components is a list of
    (axis, signed_value_m).
    """
    move = np.asarray(move_cam, dtype=float).reshape(3)
    axes = ["x", "y", "z"]
    order = sorted(range(3), key=lambda i: -abs(move[i]))

    zh_parts, en_parts, comps = [], [], []
    for i in order:
        v = move[i]
        if abs(v) < deadzone_m:
            continue
        pos, neg = _AXIS_TOKENS[axes[i]]
        zh_tok, en_tok = (pos if v >= 0 else neg)
        zh_parts.append(f"{zh_tok}{abs(v):.1f}m")
        en_parts.append(f"{en_tok} {abs(v):.1f}m")
        comps.append((axes[i], float(v)))

    if not zh_parts:
        return {"zh": "已对齐", "en": "aligned", "components": []}
    return {
        "zh": "请" + "、".join(zh_parts),
        "en": "move " + ", ".join(en_parts),
        "components": comps,
    }
