"""Recover per-athlete boxes and pose ownership when the detector merges both athletes into one box.

Boxes: bounding box of each athlete's gi-coloured pixels (white = A, blue = B) near the merged box.
Pose ownership: colour sampled around the pose's confident keypoints decides which athlete the pose is on.
These are weaker than direct detections and are labelled 'color_split' / low confidence accordingly.
"""
import cv2
import numpy as np

EXPAND = 0.25
MIN_AREA_FRAC = 0.10  # component kept if >= this fraction of the largest same-colour component
MIN_MASK_PX = 400
KERNEL = np.ones((7, 7), np.uint8)
KP_SAMPLE = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # no face: skin/hair carry no gi colour


def colour_masks(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = ((s < 55) & (v > 150)).astype(np.uint8)
    blue = ((h >= 95) & (h <= 135) & (s > 90) & (v > 35)).astype(np.uint8)
    return white, blue


def _bbox(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [k + 1 for k, a in enumerate(areas) if a >= max(MIN_MASK_PX, MIN_AREA_FRAC * areas.max())]
    if not keep:
        return None, 0
    x1 = min(stats[k, cv2.CC_STAT_LEFT] for k in keep)
    y1 = min(stats[k, cv2.CC_STAT_TOP] for k in keep)
    x2 = max(stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] for k in keep)
    y2 = max(stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] for k in keep)
    return (x1, y1, x2, y2), int(sum(stats[k, cv2.CC_STAT_AREA] for k in keep))


def region_of(box, shape, y_floor_frac=0.30):
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (int(max(0, x1 - EXPAND * w)), int(max(y_floor_frac * H, y1 - EXPAND * h)),
            int(min(W, x2 + EXPAND * w)), int(min(H, y2 + EXPAND * h)))


def split_by_colour(frame, merged_box):
    """-> {'A': (box|None, px), 'B': (box|None, px)} in full-frame coordinates."""
    rx1, ry1, rx2, ry2 = region_of(merged_box, frame.shape)
    hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
    white, blue = colour_masks(hsv)
    out = {}
    for ident, m in (("A", white), ("B", blue)):
        b, px = _bbox(m)
        if b is not None:
            b = (b[0] + rx1, b[1] + ry1, b[2] + rx1, b[3] + ry1)
        out[ident] = (b, px)
    return out


def pose_owner(frame, kp, patch=4):
    """Which athlete does this pose sit on? Returns (id|None, white_share, n_samples).

    Samples a small patch at each confident limb/torso keypoint and counts white vs blue gi pixels.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white, blue = colour_masks(hsv)
    H, W = white.shape
    w = b = 0
    n = 0
    for i in KP_SAMPLE:
        x, y, c = kp[i]
        if c < 0.3:
            continue
        x, y = int(round(x)), int(round(y))
        x1, x2, y1, y2 = max(0, x - patch), min(W, x + patch + 1), max(0, y - patch), min(H, y + patch + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        cw, cb = int(white[y1:y2, x1:x2].sum()), int(blue[y1:y2, x1:x2].sum())
        if cw + cb == 0:
            continue
        n += 1
        w += cw
        b += cb
    if n < 3 or w + b == 0:
        return None, 0.5, n
    share = w / (w + b)
    return ("A" if share > 0.65 else "B" if share < 0.35 else None), float(share), n
