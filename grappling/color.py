"""Gi-colour identity cue: HSV histogram of the torso region.

A = white gi, B = blue gi (read off the first frames of each shot, see report.md).
"""
import cv2
import numpy as np

SHOULDERS, HIPS = (5, 6), (11, 12)
KP_CONF = 0.3
HS_BINS = (12, 6, 3)  # hue, sat, val


def torso_box(det):
    """Torso rectangle (x1,y1,x2,y2,from_keypoints) from shoulder/hip keypoints, padded and
    given a minimum size relative to the person box (profile / lying poses collapse the
    keypoints into a sliver), else a box-relative guess."""
    x1, y1, x2, y2 = [float(v) for v in det[:4]]
    bw, bh = x2 - x1, y2 - y1
    kp = det[5:].reshape(17, 3)
    pts = np.array([kp[i, :2] for i in SHOULDERS + HIPS if kp[i, 2] > KP_CONF])
    if len(pts) >= 3:
        a, b = pts.min(0), pts.max(0)
        a, b = a - [.06 * bw, .06 * bh], b + [.06 * bw, .06 * bh]
        cx, cy = (a + b) / 2
        w, h = max(b[0] - a[0], .25 * bw), max(b[1] - a[1], .25 * bh)
        return (max(x1, cx - w / 2), max(y1, cy - h / 2), min(x2, cx + w / 2), min(y2, cy + h / 2), True)
    return x1 + .3 * bw, y1 + .2 * bh, x2 - .3 * bw, y1 + .55 * bh, False


def torso_hsv(frame, det):
    x1, y1, x2, y2, _ = torso_box(det)
    H, W = frame.shape[:2]
    x1, x2 = int(max(0, x1)), int(min(W, x2))
    y1, y2 = int(max(0, y1)), int(min(H, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3)


def color_fractions(hsv):
    """Fraction of torso pixels that look white / blue / dark."""
    if hsv is None or len(hsv) == 0:
        return dict(white=0., blue=0., dark=0.)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    return dict(
        white=float(np.mean((s < 60) & (v > 140))),
        blue=float(np.mean((h >= 95) & (h <= 135) & (s > 90) & (v > 40))),
        dark=float(np.mean(v < 70)),
    )


def hist(hsv):
    if hsv is None or len(hsv) == 0:
        return None
    hh = cv2.calcHist([hsv.reshape(-1, 1, 3)], [0, 1, 2], None, HS_BINS, [0, 180, 0, 256, 0, 256]).ravel()
    return hh / max(hh.sum(), 1e-9)


def bhatta(p, q):
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))  # 0 = identical
