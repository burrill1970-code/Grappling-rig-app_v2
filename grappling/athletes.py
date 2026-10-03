"""Pick athlete candidates out of all detected people (drop crowd, referee, edge slivers)."""
import numpy as np
from .color import torso_hsv, color_fractions, hist

H_REF, W_REF = 608, 1080
MIN_BOTTOM = 0.50   # box bottom must be on the mat (lower half of the picture); crowd sits above
MIN_WIDTH = 80      # px; narrower boxes are edge slivers / far spectators
MIN_CONF = 0.20


def classify(fr):
    """white / blue / dark(referee-like) / other from torso colour fractions."""
    if fr["blue"] >= 0.30:
        return "blue"
    if fr["white"] >= 0.25 and fr["white"] > fr["blue"]:
        return "white"
    if fr["dark"] >= 0.70:
        return "dark"
    return "other"


def candidates(frame, dets):
    """Return list of dicts for plausible athletes in one frame; each keeps the raw detection."""
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
        out.append(dict(idx=j, det=d, box=np.array([x1, y1, x2, y2]), conf=c,
                        frac=fr, cls=cls, hist=hist(hsv)))
    return out
