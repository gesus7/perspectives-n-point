import cv2
import numpy as np

# Optional CJK text rendering: OpenCV's Hershey fonts can't draw Chinese, so we use
# a TrueType font via PIL when available. Falls back to ASCII-only cv2.putText.
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:  # pragma: no cover
    _PIL_OK = False

_CJK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",    # Microsoft YaHei
    "C:/Windows/Fonts/simhei.ttf",  # SimHei
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont"] = {}


def _cjk_font(size: int):
    """Return a cached CJK TrueType font of `size`, or None if none is installed."""
    if not _PIL_OK:
        return None
    if size not in _FONT_CACHE:
        import os
        font = None
        for path in _CJK_FONT_CANDIDATES:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


def draw_text(img: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] = (255, 255, 255),
              font_size: int = 26, bg: tuple[int, int, int] | None = None) -> np.ndarray:
    """Draw `text` (may contain CJK) at pixel `org` (top-left). Uses a TrueType font
    via PIL when available; otherwise falls back to cv2.putText (ASCII only).

    `color`/`bg` are BGR (OpenCV convention). `bg` draws a filled background box."""
    font = _cjk_font(font_size)
    if font is None:
        # ASCII fallback — strip non-latin so cv2 doesn't draw '?' boxes.
        ascii_text = text.encode("ascii", "ignore").decode() or text
        if bg is not None:
            (tw, th), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (org[0] - 4, org[1] - 4),
                          (org[0] + tw + 4, org[1] + th + 8), bg, -1)
        cv2.putText(img, ascii_text, (org[0], org[1] + font_size), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        return img

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if bg is not None:
        l, t, r, b = draw.textbbox(org, text, font=font)
        draw.rectangle((l - 4, t - 4, r + 4, b + 4), fill=(bg[2], bg[1], bg[0]))
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def draw_keypoints(
    img: np.ndarray,
    kpts_2d: np.ndarray,
    kpts_conf: np.ndarray,
    conf_thresh: float = 0.5,
    reproj_2d: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw detected keypoints (filled, `color`) and optional reprojected ones (green ring)."""
    for (x, y), c in zip(kpts_2d, kpts_conf):
        if c >= conf_thresh:
            cv2.circle(img, (int(round(x)), int(round(y))), 4, color, -1)
    if reproj_2d is not None:
        for (x, y) in reproj_2d:
            cv2.circle(img, (int(round(x)), int(round(y))), 3, (0, 255, 0), 1)
    return img


def draw_axes(
    img: np.ndarray, R: np.ndarray, t: np.ndarray,
    K: np.ndarray, dist: np.ndarray, axis_len: float = 5.0,
) -> np.ndarray:
    """Draw object-frame X/Y/Z axes (red/green/blue) at object origin."""
    pts = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
                   dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts, rvec, t.reshape(3, 1), K, dist)
    o, x, y, z = [tuple(int(round(v)) for v in p.ravel()) for p in proj]
    cv2.line(img, o, x, (0, 0, 255), 2)
    cv2.line(img, o, y, (0, 255, 0), 2)
    cv2.line(img, o, z, (255, 0, 0), 2)
    return img


def draw_bbox3d(
    img: np.ndarray, R: np.ndarray, t: np.ndarray,
    K: np.ndarray, dist: np.ndarray, kpts_3d: np.ndarray,
) -> np.ndarray:
    """Draw object-frame AABB derived from kpts_3d."""
    mn = kpts_3d.min(axis=0)
    mx = kpts_3d.max(axis=0)
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(corners, rvec, t.reshape(3, 1), K, dist)
    p = [tuple(int(round(v)) for v in c.ravel()) for c in proj]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        cv2.line(img, p[a], p[b], (255, 255, 0), 1)
    return img


def put_pose_text(img: np.ndarray, R: np.ndarray, t: np.ndarray,
                  label: str | None = None, slot: int = 0) -> np.ndarray:
    """Overlay class label, distance ||t|| and ZYX Euler angles (deg).

    `slot` stacks multiple detections vertically so per-object lines don't overlap."""
    dist_m = float(np.linalg.norm(t))
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    prefix = f"{label}: " if label else ""
    txt = f"{prefix}d={dist_m:.2f}m  yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f}"
    y = 32 + slot * 30
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img
