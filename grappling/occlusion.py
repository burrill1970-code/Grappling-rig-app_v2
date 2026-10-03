"""Recover per-athlete boxes and pose ownership when the detector merges both athletes into one box.

Boxes: bounding box of each athlete's gi-coloured pixels (white = A, blue = B) near the merged box.
Pose ownership: colour sampled around the pose's confident keypoints decides which athlete the pose is on.
These are weaker than direct detections and are labelled 'color_split' / low confidence accordingly.
"""
import cv2
import numpy as np

EXPAND = 0.15
SHADOW_WHITE_V = 105  # shadowed white gi; safe only inside a person mask (no ad boards)
MIN_AREA_FRAC = 0.10  # component kept if >= this fraction of the largest same-colour component
MIN_MASK_PX = 400
KERNEL = np.ones((7, 7), np.uint8)
KP_SAMPLE = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # no face: skin/hair carry no gi colour


def colour_masks(hsv, white_v=150, white_s=55):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = ((s < white_s) & (v > white_v)).astype(np.uint8)
    blue = ((h >= 95) & (h <= 135) & (s > 90) & (v > 35)).astype(np.uint8)
    return white, blue


GAP_PX = 60  # blobs farther than this from the main body are separate things (scoreboard, spectators)


def _bbox(mask, union_all=False):
    """Bounding box of the largest same-colour blob plus nearby blobs (jacket/trousers, split by a belt).
    union_all (fallback tier): also join blobs farther away, e.g. an athlete's legs on both sides of the other athlete."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = int(np.argmax(areas)) + 1
    cand = [k + 1 for k, a in enumerate(areas) if a >= max(MIN_MASK_PX, MIN_AREA_FRAC * areas.max())]
    if big not in cand:
        return None, 0

    def bb(k):
        return (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH],
                stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT])

    keep, union = [big], list(bb(big))
    changed = True
    while changed:
        changed = False
        for k in cand:
            if k in keep:
                continue
            x1, y1, x2, y2 = bb(k)
            gap_x = max(0, max(x1 - union[2], union[0] - x2))
            gap_y = max(0, max(y1 - union[3], union[1] - y2))
            if union_all or max(gap_x, gap_y) <= GAP_PX:
                keep.append(k)
                union = [min(union[0], x1), min(union[1], y1), max(union[2], x2), max(union[3], y2)]
                changed = True
    return tuple(int(v) for v in union), int(sum(stats[k, cv2.CC_STAT_AREA] for k in keep))


def region_of(box, shape, y_floor_frac=0.30):
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (int(max(0, x1 - EXPAND * w)), int(max(y_floor_frac * H, y1 - EXPAND * h)),
            int(min(W, x2 + EXPAND * w)), int(min(H, y2 + EXPAND * h)))


def split_by_colour(frame, merged_box, person=None):
    """-> {'A': (box|None, px), 'B': (box|None, px)} in full-frame coordinates."""
    rx1, ry1, rx2, ry2 = region_of(merged_box, frame.shape)
    hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
    white, blue = colour_masks(hsv, white_v=SHADOW_WHITE_V if person is not None else 150)
    if person is not None:  # drop same-coloured background (ad boards, scoreboard)
        pm = person[ry1:ry2, rx1:rx2].astype(np.uint8)
        white, blue = white & pm, blue & pm
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


MIN_PX = 600
MIN_BOTTOM_FRAC = 0.5


LOOSE_WHITE_S = 80  # motion blur / shadow desaturates white towards the other gi's colour; used only as a fallback


def _drop_enclosed(white, blue):
    """Remove white blobs that sit wholly inside the blue silhouette: a white patch on a blue gi (collar, back label,
    sponsor panel) is not the white athlete, whose visible parts touch the outside of the blue athlete."""
    if int(blue.sum()) < 1500:
        return white
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(blue)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    area_blue = max(int(filled.sum()), 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    out = white.copy()
    for k in range(1, n):
        comp = (lab == k).astype(np.uint8)
        grown = cv2.dilate(comp, np.ones((7, 7), np.uint8))
        if int((grown & (1 - filled)).sum()) == 0 and stats[k, cv2.CC_STAT_AREA] < 0.5 * area_blue:
            out[lab == k] = 0
    return out


def locate(frame, person, ident, region, min_px=MIN_PX, exclude=(), union_fallback=False):
    """Bounding box of `ident`'s gi colour (A=white, B=blue) among person pixels inside `region`.
    -> (box|None, pixel_count) in full-frame coordinates. For A inside a person mask a looser white
    (S<80) is tried when the strict one yields no valid box, and white blobs enclosed by the blue gi are dropped.
    union_fallback: last tier, join all blobs in the region (an athlete mostly hidden behind the other)."""
    rx1, ry1, rx2, ry2 = region
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None, 0
    hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
    tries = [(55, False), (LOOSE_WHITE_S, False)] if (ident == "A" and person is not None) else [(55, False)]
    if union_fallback:
        tries = tries + [(ws, True) for ws, _ in tries]
    px = 0
    for ws, union_all in tries:
        white, blue = colour_masks(hsv, white_v=SHADOW_WHITE_V if person is not None else 150, white_s=ws)
        if ident == "A" and person is not None:
            pmk = person[ry1:ry2, rx1:rx2].astype(np.uint8)
            white = _drop_enclosed(white & pmk, blue & pmk)
        m = white if ident == "A" else blue
        if person is not None:
            m = m & person[ry1:ry2, rx1:rx2].astype(np.uint8)
        for ex1, ey1, ex2, ey2 in exclude:  # e.g. referee boxes
            m[max(0, int(ey1) - ry1):max(0, int(ey2) - ry1), max(0, int(ex1) - rx1):max(0, int(ex2) - rx1)] = 0
        b, px = _bbox(m, union_all=union_all)
        if b is None or px < min_px:
            continue
        box = (b[0] + rx1, b[1] + ry1, b[2] + rx1, b[3] + ry1)
        if box[3] < MIN_BOTTOM_FRAC * frame.shape[0] or box[3] - box[1] < 30 or box[2] - box[0] < 40:
            continue  # athletes are on the mat, not the crowd/scoreboard
        return box, px
    return None, px


def grow(box, shape, frac, y_floor_frac=0.30):
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (int(max(0, x1 - frac * w)), int(max(y_floor_frac * H, y1 - frac * h)),
            int(min(W, x2 + frac * w)), int(min(H, y2 + frac * h)))


def keypoint_owners(frame, kp, person, patch=5, min_share=0.75):
    """Per-keypoint gi-colour owner: 'A' (white), 'B' (blue) or None, from a small patch around each keypoint.

    Only person pixels count. Skin keypoints (head, hands, feet) carry no gi colour and come back None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white, blue = colour_masks(hsv, white_v=SHADOW_WHITE_V)
    if person is not None:
        white, blue = white & person.astype(np.uint8), blue & person.astype(np.uint8)
    H, W = white.shape
    owners = []
    for x, y, c in kp:
        if c < 0.3:
            owners.append(None)
            continue
        x, y = int(round(x)), int(round(y))
        x1, x2, y1, y2 = max(0, x - patch), min(W, x + patch + 1), max(0, y - patch), min(H, y + patch + 1)
        if x2 <= x1 or y2 <= y1:
            owners.append(None)
            continue
        cw, cb = int(white[y1:y2, x1:x2].sum()), int(blue[y1:y2, x1:x2].sum())
        tot = cw + cb
        owners.append(None if tot < 8 else "A" if cw / tot >= min_share else "B" if cb / tot >= min_share else None)
    return owners
