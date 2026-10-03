"""Pick athlete candidates out of all detected people (drop crowd, referee, edge slivers)."""
import cv2
import numpy as np
from .color import torso_hsv, color_fractions, hist

H_REF, W_REF = 608, 1080
MIN_BOTTOM = 0.50   # box bottom must be on the mat (lower half of the picture); crowd sits above
MIN_WIDTH = 80      # px; narrower boxes are edge slivers / far spectators
MIN_CONF = 0.12     # low on purpose; position/size/colour filters below reject crowd false positives
BOX_MIN_PX = 400


def classify(fr):
    """white / blue / dark(referee-like) / other from torso colour fractions."""
    if fr["blue"] >= 0.30:
        return "blue"
    if fr["white"] >= 0.25 and fr["white"] > fr["blue"]:
        return "white"
    if fr["dark"] >= 0.70:
        return "dark"
    return "other"


def box_colour_class(frame, box, person):
    """Dominant gi colour over ALL person pixels in the box (more robust than one torso patch):
    'white' / 'blue' / 'mixed' (both gi colours present = a pair) / None (no gi pixels)."""
    from .occlusion import colour_masks, SHADOW_WHITE_V
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    white, blue = colour_masks(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV), white_v=SHADOW_WHITE_V)
    pm = person[y1:y2, x1:x2].astype(np.uint8)
    w, b = int((white & pm).sum()), int((blue & pm).sum())
    if w < BOX_MIN_PX and b < BOX_MIN_PX:
        return None
    share = w / (w + b)
    if share >= 0.8:
        return "white"
    if share <= 0.2:
        return "blue"
    return "mixed" if min(w, b) >= 300 else ("white" if share > 0.5 else "blue")


def referee_boxes(frame, dets):
    """Boxes of black-suited people on the mat (referee): excluded from athletes AND from colour recovery."""
    H, W = frame.shape[:2]
    out = []
    for d in dets:
        x1, y1, x2, y2, c = [float(v) for v in d[:5]]
        if c < 0.3 or y2 < MIN_BOTTOM * H or (x2 - x1) < 40:
            continue
        if classify(color_fractions(torso_hsv(frame, d))) == "dark":
            out.append((x1, y1, x2, y2))
    return out


def candidates(frame, dets, person=None):
    """Return list of dicts for plausible athletes in one frame; each keeps the raw detection.
    With a person mask, the colour class comes from all gi pixels in the box (torso class is the fallback)."""
    H, W = frame.shape[:2]
    out = []
    for j, d in enumerate(dets):
        x1, y1, x2, y2, c = [float(v) for v in d[:5]]
        if c < MIN_CONF or y2 < MIN_BOTTOM * H or (x2 - x1) < MIN_WIDTH:
            continue
        hsv = torso_hsv(frame, d)
        fr = color_fractions(hsv)
        cls = classify(fr)
        if cls == "dark":  # referee (black suit, no blue/white gi pixels)
            continue
        torso_cls = cls
        if person is not None:
            bc = box_colour_class(frame, (x1, y1, x2, y2), person)
            if bc is not None:
                cls = bc
        out.append(dict(idx=j, det=d, box=np.array([x1, y1, x2, y2]), conf=c,
                        frac=fr, cls=cls, torso_cls=torso_cls, hist=hist(hsv)))
    return out
