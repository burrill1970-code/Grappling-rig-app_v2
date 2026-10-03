"""colour_profile marker family: vertical colour profile / band-to-band colour gradient down the body.

For each instance the person-mask pixels of the 192-px-high crop are split into K equal horizontal bands (top = head,
bottom = feet; heights are normalised by the crop, nothing about absolute box position, frame, or crop width is used).
Per band a robust colour statistic (median, or mean) is taken over mask pixels only, in one of several colour spaces:

  ab      CIE Lab a*,b* only (brightness-free chroma)
  Lab     Lab L,a,b with L scaled by 0.5 so its range matches a*,b*   (USES DARKNESS: the navy suit is dark)
  hs      saturation-weighted hue vector (S cos 2H, S sin 2H) from HSV   (brightness-free)
  chroma  rg chromaticity R/(R+G+B), G/(R+G+B)                              (brightness-free up to noise)

Optionally the vector of band-to-band differences (the "gradient" down the body) is appended or used alone.
Bands with too few mask pixels are filled with the whole-body statistic. Instances with a tiny mask abstain (None).

No model is trained; the harness's nearest-prototype classifier does the rest.  Features are signed, so chi2 is not
applicable; l2 / l1 / cos are used (see METRIC).
"""
import cv2
import numpy as np

MIN_BAND_PX = 20
MIN_MASK_PX = 200
_CACHE = {}


def _key(d):
    return (d["clip"], d["frame"], d["label"], tuple(np.round(d["box"], 1)))


def _planes(d, space, erode):
    """(H*W x C feature planes as float32 [192,W,C], bool mask). Cached per (instance, space, erode)."""
    k = (_key(d), space, erode)
    if k in _CACHE:
        return _CACHE[k]
    c, m = d["crop"]
    m = m.astype(np.uint8)
    if erode:
        me = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode)
        if me.sum() >= MIN_MASK_PX:
            m = me
    m = m.astype(bool)
    if space in ("ab", "Lab"):
        lab = cv2.cvtColor(c.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        if space == "ab":
            pl = lab[..., 1:3]
        else:
            pl = np.concatenate([0.5 * lab[..., :1], lab[..., 1:3]], -1)
    elif space == "hs":
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
        ang = hsv[..., 0] * (2 * np.pi / 180.0)
        s = hsv[..., 1] / 255.0 * 100.0           # scaled so magnitudes are comparable to Lab a*,b*
        pl = np.stack([s * np.cos(ang), s * np.sin(ang)], -1)
    elif space == "chroma":
        f = c.astype(np.float32) + 1.0
        t = f.sum(-1, keepdims=True)
        pl = (f[..., [2, 1]] / t) * 100.0         # r, g chromaticity (x100)
    else:
        raise ValueError(space)
    if len(_CACHE) > 4000:
        _CACHE.clear()
    _CACHE[k] = (pl, m)
    return pl, m


def profile(d, space="ab", K=6, stat="median", grad="none", erode=2):
    """grad: 'none' (bands only), 'both' (bands + adjacent-band differences), 'only' (differences + global level)."""
    pl, m = _planes(d, space, erode)
    if m.sum() < MIN_MASK_PX:
        return None
    red = np.median if stat == "median" else np.mean
    glob = red(pl[m], 0)
    edges = np.linspace(0, pl.shape[0], K + 1).round().astype(int)
    rows = []
    for k in range(K):
        mk = m[edges[k]:edges[k + 1]]
        if mk.sum() < MIN_BAND_PX:
            rows.append(glob)
        else:
            rows.append(red(pl[edges[k]:edges[k + 1]][mk], 0))
    P = np.asarray(rows)                          # K x C
    if grad == "none":
        return P.ravel()
    G = np.diff(P, axis=0)
    if grad == "both":
        return np.concatenate([P.ravel(), G.ravel()])
    return np.concatenate([glob, G.ravel()])      # 'only'


def _mk(**kw):
    return lambda d: profile(d, **kw)


# name -> feature_fn ; METRIC[name] = (metric, mode) used with tools.marker_eval.evaluate
FEATURES = {}
METRIC = {}


def _reg(name, metric="l2", mode="mean", **kw):
    FEATURES[name] = _mk(**kw)
    METRIC[name] = (metric, mode)


# ---- the 13 variants that were actually evaluated (budget: 15), in the order they were run -------------------------
# stage 1: colour space (K=6, median, l2, prototype mean)
_reg("ab_K6_med", space="ab", K=6, stat="median", grad="none")
_reg("Lab_K6_med", space="Lab", K=6, stat="median", grad="none")
_reg("hs_K6_med", space="hs", K=6, stat="median", grad="none")
_reg("chroma_K6_med", space="chroma", K=6, stat="median", grad="none")
# stage 2 (on the clip1-leading Lab space): band count, statistic, gradient
_reg("Lab_K3_med", space="Lab", K=3, stat="median", grad="none")
_reg("Lab_K12_med", space="Lab", K=12, stat="median", grad="none")
_reg("Lab_K6_mean", space="Lab", K=6, stat="mean", grad="none")
_reg("Lab_K6_med_gradboth", space="Lab", K=6, stat="median", grad="both")
# stage 3: gradient + band count / prototype mode / metric / brightness-free
_reg("Lab_K12_med_gradboth", space="Lab", K=12, stat="median", grad="both")
_reg("Lab_K6_med_gradboth_nn", mode="nn", space="Lab", K=6, stat="median", grad="both")
_reg("Lab_K6_med_gradboth_l1", metric="l1", space="Lab", K=6, stat="median", grad="both")
_reg("Lab_K6_med_gradboth_cos", metric="cos", space="Lab", K=6, stat="median", grad="both")
_reg("ab_K6_med_gradboth", space="ab", K=6, stat="median", grad="both")

# Selected by clip1 block balanced accuracy only (3-way tie at 1.000 with l1/cos; tie broken by clip1 start, then by
# the pre-set default metric l2).  clip2 numbers were NOT used for selection.
BEST = "Lab_K6_med_gradboth"

if __name__ == "__main__":
    import sys
    from tools.marker_eval import evaluate
    for n in (sys.argv[1:] or [BEST]):
        evaluate(FEATURES[n], n, metric=METRIC[n][0], mode=METRIC[n][1])
