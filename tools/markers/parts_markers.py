"""parts_markers family: PART-BASED colour and geometric markers placed with the COCO keypoints.

Each person crop (192 px high, tight around the person box) is cut into body parts using the 17 keypoints of that
detection. Part rectangles are expressed in units of the person's OWN body (crop height H, shoulder width, torso
length T = hip line - shoulder line), so they are scale free; nothing uses absolute box position, frame number, crop
width or any image position.  When a keypoint is missing (conf < 0.3) a body-proportion fallback is used
(shoulder line 0.20H, hip line 0.52H, ankles 0.95H, shoulder width 0.25H ...), so the layout is always defined.

Parts (all restricted to the person-segmentation mask, eroded 1 px to drop the halo; < MIN_PX pixels -> part missing):

  head    the box top down to the shoulder line, narrow strip around the face keypoints   (hair + face + neck skin)
  chest   centre strip from the neck down 70 % of the torso   (collar / lapels / shirt-and-tie)
  jacket  the torso left and right of that strip, 12-85 % of the torso   (gi jacket / suit jacket, the "back patch" side)
  belt    band just above the hip line   (obi / waistband)
  legs    from 30 % of the torso below the hips to just above the ankles   (gi trousers / suit trousers)
  feet    from the ankles to the box bottom   (bare feet in a gi, shoes for the referee)

Axis-sampled variants of the same parts (aimed at contact, where a rectangle picks up the other body):
  legs_axis   thin strips (0.05H wide) along hip->knee and knee->ankle of each leg (falls back to `legs` if no limb is visible)
  torso_quad  the shoulder-shoulder-hip-hip quadrilateral shrunk to 55 % around its centre (falls back to `jacket`)

Descriptors
  hsv       per-part HSV histogram (8 hue x 4 sat x 3 val); the vector is the concatenation of the chosen parts, each part
            histogram normalised to sum 1/n_parts (so chi2 on the concatenation = mean of per-part chi2). A missing part is
            all zeros, which adds the same constant to every class distance (neutral).  Uses brightness (V) -> a navy suit and
            a blue gi are separated partly by darkness. R was labelled by exclusion (not colour) so this is not circular.
  sem       per-part fractions of fixed semantic colour classes (dark / grey / white / blue / warm-skin / other); thresholds
            are fixed a priori (see _sem) and never tuned. Also uses darkness.
  contrast  inter-part colour DIFFERENCES in Lab (chest-jacket, belt-jacket, legs-jacket, head-jacket): drops the absolute
            colour and keeps the "gradient" between parts (illumination-robust). Signed -> use l2.
  geom      keypoint body proportions (shoulder/hip width, torso, thigh, shin lengths relative to the torso); no colour.

No model is trained; the harness' nearest-prototype classifier does the rest.  FEATURES maps name -> feature_fn and
METRIC maps name -> the metric used in the study.
"""
import cv2
import numpy as np

KC = 0.3          # keypoint confidence threshold (same as grappling.color.KP_CONF)
MIN_PX = 15       # a part needs at least this many mask pixels, otherwise it is "missing"
BINS = (8, 4, 3)
PARTS_ALL = ("head", "chest", "jacket", "belt", "legs", "feet")
_CACHE = {}


def _key(d):
    return (d["clip"], d["frame"], d["label"], tuple(np.round(d["box"], 1)))


# ---------------------------------------------------------------- layout from keypoints
def _layout(d):
    """dict part -> list of (x1, y1, x2, y2) int rectangles in crop coordinates."""
    c, _ = d["crop"]
    Hc, W = c.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in d["box"]]
    s = Hc / (y2 - y1)
    kp = d["kp"]
    xy = (kp[:, :2] - [x1, y1]) * s
    ok = kp[:, 2] > KC

    def ym(ids):
        v = [xy[i, 1] for i in ids if ok[i]]
        return float(np.mean(v)) if v else None

    def xs(ids):
        return [float(xy[i, 0]) for i in ids if ok[i]]

    ysh, yhip = ym((5, 6)), ym((11, 12))
    if ysh is None and yhip is None:
        ysh, yhip = 0.20 * Hc, 0.52 * Hc
    elif ysh is None:
        ysh = yhip - 0.32 * Hc
    elif yhip is None:
        yhip = ysh + 0.32 * Hc
    T = max(yhip - ysh, 0.15 * Hc)
    yhip = ysh + T

    sh_x, hip_x = xs((5, 6)), xs((11, 12))
    wsh = max(abs(sh_x[0] - sh_x[1]) if len(sh_x) == 2 else 0.25 * Hc, 0.14 * Hc)
    whp = max(abs(hip_x[0] - hip_x[1]) if len(hip_x) == 2 else 0.17 * Hc, 0.10 * Hc)
    allx = sh_x + hip_x
    cx = float(np.mean(allx)) if allx else W / 2
    cxs = float(np.mean(sh_x)) if sh_x else cx
    cxh = float(np.mean(hip_x)) if hip_x else cx
    face = xs((0, 1, 2, 3, 4))
    hcx = float(np.mean(face)) if face else cxs

    yank = ym((15, 16))
    if yank is None:
        yk = ym((13, 14))
        yank = (yk + 0.9 * (yk - yhip)) if yk is not None else 0.95 * Hc
    yank = float(np.clip(yank, yhip + 0.2 * Hc, Hc))
    legx = xs((11, 12, 13, 14, 15, 16))
    lx1, lx2 = (min(legx), max(legx)) if legx else (cxh - 0.5 * whp, cxh + 0.5 * whp)
    pad = 0.07 * Hc
    ax = xs((15, 16))
    fx1, fx2 = (min(ax), max(ax)) if ax else (lx1, lx2)

    def R(a, b, c_, d_):
        return (int(round(a)), int(round(b)), int(round(c_)), int(round(d_)))

    hw = max(0.22 * wsh, 0.045 * Hc)
    return dict(
        head=[R(hcx - hw, 0, hcx + hw, ysh)],
        chest=[R(cxs - 0.14 * wsh, ysh - 0.02 * Hc, cxs + 0.14 * wsh, ysh + 0.70 * T)],
        jacket=[R(cxs - 0.46 * wsh, ysh + 0.12 * T, cxs - 0.16 * wsh, ysh + 0.85 * T),
                R(cxs + 0.16 * wsh, ysh + 0.12 * T, cxs + 0.46 * wsh, ysh + 0.85 * T)],
        belt=[R(cxh - 0.40 * wsh, yhip - 0.28 * T, cxh + 0.40 * wsh, yhip - 0.03 * T)],
        legs=[R(lx1 - pad, yhip + 0.30 * T, lx2 + pad, yank - 0.04 * Hc)],
        feet=[R(fx1 - pad, yank - 0.02 * Hc, fx2 + pad, Hc)],
    )


def _parts(d):
    """dict part -> (hsv pixels Nx3 uint8, lab pixels Nx3 float32) or None when too few mask pixels. Cached."""
    k = _key(d)
    if k in _CACHE:
        return _CACHE[k]
    c, m = d["crop"]
    Hc, W = c.shape[:2]
    m8 = m.astype(np.uint8)
    me = cv2.erode(m8, np.ones((3, 3), np.uint8))
    if me.sum() >= 200:
        m8 = me
    m = m8.astype(bool)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    out = {}
    for name, rects in _layout(d).items():
        hs, ls = [], []
        for (a, b, e, f) in rects:
            a, b, e, f = max(0, a), max(0, b), min(W, e), min(Hc, f)
            if e - a < 2 or f - b < 2:
                continue
            mm = m[b:f, a:e]
            hs.append(hsv[b:f, a:e][mm])
            ls.append(lab[b:f, a:e][mm])
        h = np.concatenate(hs) if hs else np.zeros((0, 3), np.uint8)
        l_ = np.concatenate(ls) if ls else np.zeros((0, 3), np.float32)
        out[name] = (h, l_) if len(h) >= MIN_PX else None
    # axis-sampled parts (polygons / limb strips from the keypoints), fall back to the rectangle versions
    x1, y1, x2, y2 = [float(v) for v in d["box"]]
    sc = Hc / (y2 - y1)
    xy = (d["kp"][:, :2] - [x1, y1]) * sc
    ok = d["kp"][:, 2] > KC
    legs_m = np.zeros((Hc, W), np.uint8)
    th = max(3, int(round(0.05 * Hc)))
    n_seg = 0
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        if ok[a] and ok[b]:
            cv2.line(legs_m, tuple(int(round(v)) for v in xy[a]), tuple(int(round(v)) for v in xy[b]), 1, th)
            n_seg += 1
    quad_m = np.zeros((Hc, W), np.uint8)
    if all(ok[i] for i in (5, 6, 11, 12)):
        q = xy[[5, 6, 12, 11]]
        q = q.mean(0) + 0.55 * (q - q.mean(0))
        cv2.fillPoly(quad_m, [np.round(q).astype(np.int32)], 1)
    for name, pm_, fb in (("legs_axis", legs_m, "legs"), ("torso_quad", quad_m, "jacket")):
        mm = (pm_ > 0) & m
        if mm.sum() >= MIN_PX:
            out[name] = (hsv[mm], lab[mm])
        else:
            out[name] = out[fb]
    _CACHE[k] = out
    return out


# ---------------------------------------------------------------- descriptors
def _hist(h):
    v = cv2.calcHist([h.reshape(-1, 1, 3)], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256]).ravel()
    return v / max(v.sum(), 1e-9)


def _sem(h):
    """Fixed semantic colour classes (OpenCV HSV: H 0-179, S/V 0-255): dark, grey, white, blue, warm/skin, other."""
    H, S, V = [h[:, i].astype(int) for i in range(3)]
    dark = V < 60
    low = (S < 50) & ~dark
    white = low & (V >= 110)
    grey = low & (V < 110)
    col = ~dark & ~low
    blue = col & (H >= 90) & (H <= 135)
    warm = col & ((H <= 30) | (H >= 165))
    other = col & ~blue & ~warm
    v = np.array([dark.sum(), grey.sum(), white.sum(), blue.sum(), warm.sum(), other.sum()], float)
    return v / v.sum()


def _concat(d, parts, desc):
    p = _parts(d)
    vs, any_ok = [], False
    for name in parts:
        e = p[name]
        if e is None:
            vs.append(np.zeros(BINS[0] * BINS[1] * BINS[2] if desc == "hsv" else 6))
        else:
            any_ok = True
            vs.append((_hist(e[0]) if desc == "hsv" else _sem(e[0])) / len(parts))
    return np.concatenate(vs) if any_ok else None


def _med_lab(e):
    return np.median(e[1], 0) * np.array([0.5, 1.0, 1.0])


def contrast(d):
    p = _parts(d)
    if p["jacket"] is None:
        return None
    ref = _med_lab(p["jacket"])
    vs = []
    for name in ("chest", "belt", "legs", "head"):
        vs.append(_med_lab(p[name]) - ref if p[name] is not None else np.zeros(3))
    return np.concatenate(vs)


def geom(d):
    """Scale-free keypoint proportions relative to torso length (needs shoulders, hips, knees and ankles)."""
    kp = d["kp"]
    need = (5, 6, 11, 12, 13, 14, 15, 16)
    if any(kp[i, 2] <= KC for i in need):
        return None
    P = kp[:, :2].astype(float)
    sh, hp = (P[5] + P[6]) / 2, (P[11] + P[12]) / 2
    T = np.linalg.norm(hp - sh)
    if T < 1e-3:
        return None
    wsh, whp = np.linalg.norm(P[5] - P[6]), np.linalg.norm(P[11] - P[12])
    thigh = (np.linalg.norm(P[11] - P[13]) + np.linalg.norm(P[12] - P[14])) / 2
    shin = (np.linalg.norm(P[13] - P[15]) + np.linalg.norm(P[14] - P[16])) / 2
    # typical standing values: shoulder 0.85T, hip 0.6T, thigh 1.1T, shin 1.0T; features are (value / typical - 1)
    return np.array([wsh / T / 0.85, whp / T / 0.6, thigh / T / 1.1, shin / T / 1.0]) - 1.0


def _mk(parts, desc):
    return lambda d: _concat(d, parts, desc)


FEATURES = {
    # --- single parts (hsv histogram)
    "head_hsv": _mk(("head",), "hsv"),
    "chest_hsv": _mk(("chest",), "hsv"),
    "jacket_hsv": _mk(("jacket",), "hsv"),
    "belt_hsv": _mk(("belt",), "hsv"),
    "legs_hsv": _mk(("legs",), "hsv"),
    "feet_hsv": _mk(("feet",), "hsv"),
    # --- part combinations
    "jacket_legs_hsv": _mk(("jacket", "legs"), "hsv"),
    "jacket_chest_legs_hsv": _mk(("jacket", "chest", "legs"), "hsv"),
    "all6_hsv": _mk(PARTS_ALL, "hsv"),
    # --- parts chosen from the single-part clip1 numbers (jacket 0.889, belt 0.968, legs 0.966 clip1 block bal_acc)
    "belt_legs_jacket_hsv": _mk(("jacket", "belt", "legs"), "hsv"),
    # --- same three parts, axis-sampled (limb strips + shrunk torso quad) to resist contact contamination
    "axis_quad_belt_legs_hsv": _mk(("torso_quad", "belt", "legs_axis"), "hsv"),
    # --- other markers
    "all6_sem": _mk(PARTS_ALL, "sem"),
    "contrast_lab": contrast,
    "geom_proportions": geom,
}
METRIC = {k: "chi2" for k in FEATURES}
METRIC.update(contrast_lab="l2", geom_proportions="l2")

# prototype mode used in the study (all 'mean' except the final nn variant); BEST = winner on CLIP1 block bal_acc
MODE = {k: "mean" for k in FEATURES}
BEST = ("axis_quad_belt_legs_hsv", "chi2", "nn")
