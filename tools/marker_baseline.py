import cv2, numpy as np
from tools.marker_eval import evaluate

def hsv_torso(d, bins=(12, 6, 3)):
    """BASELINE (what the tracker uses today): HSV histogram of the torso patch."""
    from grappling.color import torso_box
    x1, y1, x2, y2 = d["box"]
    full = np.concatenate([d["box"], [d["conf"]], d["kp"].ravel()])
    tx1, ty1, tx2, ty2, _ = torso_box(full)
    c, m = d["crop"]; s = c.shape[0] / (y2 - y1)
    a, b, e, f = [int(max(0, v)) for v in ((tx1 - x1) * s, (ty1 - y1) * s, (tx2 - x1) * s, (ty2 - y1) * s)]
    p = c[b:f, a:e]
    if p.size == 0: return None
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1e-9)

if __name__ == "__main__":
    evaluate(hsv_torso, "hsv_torso_baseline", metric="chi2")
    evaluate(hsv_torso, "hsv_torso_baseline", metric="chi2", mode="nn")


def hs_only(d):
    """Hue/saturation histogram only (no brightness): tests whether the baseline leaned on darkness."""
    v = hsv_torso(d, bins=(12, 6, 1))
    return v
