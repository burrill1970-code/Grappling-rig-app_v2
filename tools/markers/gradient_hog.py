"""Family: gradient_hog -- image-gradient markers for telling A (white gi), B (blue gi) and R (referee) apart.

Everything here is computed from the 192-px-high person crop and its person mask only.  Nothing uses absolute box
position, frame number, crop width or image position.  Gradients are computed on grey (or log-grey, which makes the
gradient a *relative* contrast and removes the "dark cloth => small gradient" confound), and only INTERIOR gradients
are counted (person mask eroded by ~3 px) so the silhouette edge (pose/shape, not appearance) does not leak in.
Zones are row fractions of the crop (head / torso / legs), not keypoints, so the markers do not depend on keypoint
quality.  No darkness threshold or colour is used by any pure-gradient variant (the only colour in this file is the
optional HSV-baseline fusion variant `fuse_hsv_plus_grad`, used purely to measure what gradients ADD).

FEATURES maps name -> feature_fn(inst); META maps name -> (metric, mode) chosen a priori (cos for HOG vectors, chi2 for
L1-normalised histograms, l2 for small statistic vectors).
"""
import numpy as np
import cv2

EPS_LOG = 16.0          # log(gray + EPS_LOG): soft floor so near-black pixels do not explode
MIN_INTERIOR = 150      # fewer interior mask pixels than this -> abstain
ZONES = ((0.0, 0.20), (0.20, 0.55), (0.55, 1.0))   # head, torso, legs (row fractions of the crop)

_CACHE = {}


def clear_cache():
    _CACHE.clear()


def _key(d, tag):
    return (tag, d["clip"], d["frame"], d["label"], tuple(np.round(d["box"], 1)))


def _interior(mask, k=7):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), ker).astype(np.float32)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _grad(g, log=True, sigma=1.0):
    """Scharr gradient (per-pixel derivative units) of (log-)grey; returns gx, gy, mag, unsigned angle in [0, pi)."""
    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma)
    if log:
        g = np.log(g + EPS_LOG)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1) / 32.0
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    return gx, gy, mag, ang


def _native(d, log=True, sigma=1.0):
    """Gradient maps at the native 192xW crop resolution (aspect preserved), cached per instance."""
    k = _key(d, ("nat", log, sigma))
    if k not in _CACHE:
        img, m = d["crop"]
        gx, gy, mag, ang = _grad(_gray(img), log, sigma)
        _CACHE[k] = (gx, gy, mag, ang, _interior(m))
    return _CACHE[k]


def _zone_slices(H):
    return [slice(int(round(a * H)), int(round(b * H))) for a, b in ZONES]


def _hog(mag, ang, wt, cell=16, nb=9, bn=2, clipv=0.2, rel_eps=0.2):
    """Plain-numpy HOG with a pixel weight map (interior mask): soft orientation binning (unsigned), cells of
    cell x cell px, overlapping bn x bn block L2-Hys normalisation.  The block epsilon is relative to the mean block
    norm so that cells with (almost) no interior pixels stay near zero instead of being amplified."""
    H, W = mag.shape
    ch, cw = H // cell, W // cell
    mag, ang, wt = mag[:ch * cell, :cw * cell], ang[:ch * cell, :cw * cell], wt[:ch * cell, :cw * cell]
    pos = ang / np.pi * nb
    b0f = np.floor(pos)
    f = pos - b0f
    b0 = b0f.astype(np.int64) % nb
    b1 = (b0 + 1) % nb
    cid = ((np.arange(ch * cell) // cell)[:, None] * cw + (np.arange(cw * cell) // cell)[None, :])
    w = mag * wt
    n = ch * cw * nb
    hist = (np.bincount((cid * nb + b0).ravel(), (w * (1 - f)).ravel(), minlength=n)
            + np.bincount((cid * nb + b1).ravel(), (w * f).ravel(), minlength=n)).reshape(ch, cw, nb)
    nbi, nbj = ch - bn + 1, cw - bn + 1
    blocks = np.stack([hist[i:i + bn, j:j + bn].ravel() for i in range(nbi) for j in range(nbj)])
    nrm = np.sqrt((blocks ** 2).sum(1))
    eps = rel_eps * (nrm[nrm > 0].mean() if (nrm > 0).any() else 1.0)
    v = blocks / np.sqrt(nrm[:, None] ** 2 + eps ** 2)
    v = np.minimum(v, clipv)
    v = v / np.sqrt((v ** 2).sum(1, keepdims=True) + 1e-6)
    return v.ravel()


def _resized(d, size, log, sigma=1.0):
    """gradient maps of the crop resized to a canonical (H, W) -- removes the crop width entirely."""
    img, m = d["crop"]
    H, W = size
    g = cv2.resize(_gray(img), (W, H), interpolation=cv2.INTER_AREA)
    mm = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) > 0.5
    _, _, mag, ang = _grad(g, log, sigma)
    return mag, ang, _interior(mm)


def _ok(wt):
    return wt.sum() >= MIN_INTERIOR * (wt.size / (192 * 64))


# ---------------------------------------------------------------------------------------------------------------
# 1-4: HOG family
# ---------------------------------------------------------------------------------------------------------------
def hog_body(d, log=False, cell=16, nb=9):
    """Whole-body HOG on the person crop resized to 128x64, interior-masked."""
    mag, ang, wt = _resized(d, (128, 64), log)
    if not _ok(wt):
        return None
    return _hog(mag, ang, wt, cell=cell, nb=nb)


def hog_torso_legs(d, log=True, cell=16, nb=9):
    """Separate HOG on the torso zone (rows .20-.55 -> 64x64) and the legs zone (rows .55-1.0 -> 96x64)."""
    img, m = d["crop"]
    H = img.shape[0]
    out = []
    for (a, b), size in zip(ZONES[1:], ((64, 64), (96, 64))):
        sl = slice(int(round(a * H)), int(round(b * H)))
        g = cv2.resize(_gray(img[sl]), (size[1], size[0]), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(m[sl].astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_AREA) > 0.5
        _, _, mag, ang = _grad(g, log)
        wt = _interior(mm)
        if wt.sum() < 0.5 * MIN_INTERIOR:
            return None
        h = _hog(mag, ang, wt, cell=cell, nb=nb)
        out.append(h / (np.linalg.norm(h) + 1e-9))
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------------------------
# 5-10: pooled gradient statistics (pose-robust: no spatial grid inside a zone)
# ---------------------------------------------------------------------------------------------------------------
def _orient_hist(mag, ang, wt, nb):
    pos = ang / np.pi * nb
    b0f = np.floor(pos)
    f = pos - b0f
    b0 = b0f.astype(np.int64) % nb
    b1 = (b0 + 1) % nb
    w = mag * wt
    h = np.bincount(b0.ravel(), (w * (1 - f)).ravel(), minlength=nb) + np.bincount(b1.ravel(), (w * f).ravel(), minlength=nb)
    s = h.sum()
    return h / s if s > 1e-9 else np.full(nb, 1.0 / nb)


def orient_hist_global(d, nb=18, log=True):
    """Magnitude-weighted unsigned gradient-orientation histogram over the whole interior (L1-normalised)."""
    _, _, mag, ang, wt = _native(d, log)
    if wt.sum() < MIN_INTERIOR:
        return None
    return _orient_hist(mag, ang, wt, nb)


def orient_hist_zones(d, nb=12, log=True):
    """Orientation histograms for head / torso / legs zones, concatenated (each L1-normalised, total sums to 1)."""
    _, _, mag, ang, wt = _native(d, log)
    if wt.sum() < MIN_INTERIOR:
        return None
    return np.concatenate([_orient_hist(mag[s], ang[s], wt[s], nb) for s in _zone_slices(mag.shape[0])]) / 3.0


def mag_hist_zones(d, nb=10, log=True):
    """Distribution of gradient magnitude (log-spaced bins) in torso + legs: how much contrast / fold structure the
    cloth carries (a smooth suit has a very different distribution from a creased gi).  Interior pixels only."""
    _, _, mag, ang, wt = _native(d, log)
    edges = np.concatenate([[0], np.geomspace(0.004, 0.25, nb - 1), [10]])
    out = []
    for s in _zone_slices(mag.shape[0])[1:]:
        w = wt[s] > 0
        if w.sum() < 0.5 * MIN_INTERIOR:
            return None
        h = np.histogram(mag[s][w], bins=edges)[0].astype(float)
        out.append(h / h.sum())
    return np.concatenate(out) / 2.0


def edge_density_grid(d, log=True, rows=6, cols=2, thr=(0.04, 0.10)):
    """Fraction of interior pixels whose (log-grey) gradient exceeds each threshold, in a rows x cols grid, plus the
    mean gradient magnitude per cell.  Grid cells with no interior pixels get the global value."""
    _, _, mag, ang, wt = _native(d, log)
    if wt.sum() < MIN_INTERIOR:
        return None
    H, W = mag.shape
    g_dens = [((mag > t) * wt).sum() / wt.sum() for t in thr] + [(mag * wt).sum() / wt.sum()]
    out = []
    for i in range(rows):
        for j in range(cols):
            sl = (slice(i * H // rows, (i + 1) * H // rows), slice(j * W // cols, (j + 1) * W // cols))
            w = wt[sl]
            if w.sum() < 20:
                out.extend(g_dens)
            else:
                out.extend([((mag[sl] > t) * w).sum() / w.sum() for t in thr] + [(mag[sl] * w).sum() / w.sum()])
    return np.array(out)


def structure_tensor_zones(d, log=True):
    """Per zone: log gradient energy, and the doubled-angle orientation coherence vector ((Jxx-Jyy)/E, 2Jxy/E).  Captures
    'smooth versus creased' (energy), and 'dominant straight edges vs isotropic folds' (coherence)."""
    gx, gy, mag, ang, wt = _native(d, log)
    if wt.sum() < MIN_INTERIOR:
        return None
    out = []
    for s in _zone_slices(mag.shape[0]):
        w = wt[s]
        n = max(w.sum(), 1.0)
        jxx, jyy, jxy = (gx[s] ** 2 * w).sum() / n, (gy[s] ** 2 * w).sum() / n, (gx[s] * gy[s] * w).sum() / n
        e = jxx + jyy + 1e-8
        out.extend([0.35 * np.log(e + 1e-6), (jxx - jyy) / e, 2 * jxy / e])
    return np.array(out)


def multiscale_energy(d, log=True, sigmas=(1.0, 2.0, 4.0, 8.0)):
    """Band-pass (difference-of-Gaussian) energy of log-grey per zone and scale: how fold/texture energy is spread
    across spatial scales (log-grey => brightness invariant).  Interior pixels only."""
    img, m = d["crop"]
    wt = _interior(m)
    if wt.sum() < MIN_INTERIOR:
        return None
    g = _gray(img)
    if log:
        g = np.log(g + EPS_LOG)
    blur = [g] + [cv2.GaussianBlur(g, (0, 0), s) for s in sigmas]
    out = []
    for lo, hi in zip(blur[:-1], blur[1:]):
        band = np.abs(lo - hi)
        for s in _zone_slices(g.shape[0]):
            w = wt[s]
            out.append(np.log(1e-3 + (band[s] * w).sum() / max(w.sum(), 1.0)))
    return np.array(out) * 0.3


# ---------------------------------------------------------------------------------------------------------------
# 11-12: combinations
# ---------------------------------------------------------------------------------------------------------------
def _l1(v):
    return v / max(v.sum(), 1e-9)


def combo_hist_stats(d):
    """Orientation-zone histogram (chi2-style, L1) concatenated with the magnitude-zone histogram, half weight each."""
    a, b = orient_hist_zones(d), mag_hist_zones(d)
    if a is None or b is None:
        return None
    return np.concatenate([0.5 * _l1(a), 0.5 * _l1(b)])


def fuse_hsv_plus_grad(d, w_grad=0.5):
    """FUSION TEST (not a pure gradient marker): baseline torso HSV histogram (L1) at weight 1-w and zone orientation +
    magnitude histograms at weight w, concatenated; evaluated with chi2 (a sum of chi2 over blocks).  Used only to see
    what gradients ADD on top of colour."""
    from tools.marker_baseline import hsv_torso
    h = hsv_torso(d)
    g = combo_hist_stats(d)
    if h is None or g is None:
        return None
    return np.concatenate([(1 - w_grad) * _l1(h), w_grad * _l1(g)])


FEATURES = {
    "hog_body_gray": lambda d: hog_body(d, log=False),
    "hog_body_log": lambda d: hog_body(d, log=True),
    "hog_torso_legs": lambda d: hog_torso_legs(d, log=True),
    "hog_body_cell32": lambda d: hog_body(d, log=True, cell=32),
    "orient_hist_global": orient_hist_global,
    "orient_hist_zones": orient_hist_zones,
    "mag_hist_zones": mag_hist_zones,
    "edge_density_grid": edge_density_grid,
    "structure_tensor_zones": structure_tensor_zones,
    "multiscale_energy": multiscale_energy,
    "combo_hist_stats": combo_hist_stats,
    "fuse_hsv_plus_grad": fuse_hsv_plus_grad,
}

META = {  # a-priori (metric, mode); never tuned on any clip
    "hog_body_gray": ("cos", "mean"),
    "hog_body_log": ("cos", "mean"),
    "hog_torso_legs": ("cos", "mean"),
    "hog_body_cell32": ("cos", "mean"),
    "orient_hist_global": ("chi2", "mean"),
    "orient_hist_zones": ("chi2", "mean"),
    "mag_hist_zones": ("chi2", "mean"),
    "edge_density_grid": ("l2", "mean"),
    "structure_tensor_zones": ("l2", "mean"),
    "multiscale_energy": ("l2", "mean"),
    "combo_hist_stats": ("chi2", "mean"),
    "fuse_hsv_plus_grad": ("chi2", "mean"),
}


# ---------------------------------------------------------------------------------------------------------------
# Variants 13-15 (added after round 1, all decided from the round-1 clip1 block numbers only)
# ---------------------------------------------------------------------------------------------------------------
def fuse_hsv_plus_grad_light(d):
    """Same fusion test as fuse_hsv_plus_grad but with gradients at weight 0.2 instead of 0.5."""
    return fuse_hsv_plus_grad(d, w_grad=0.2)


FEATURES["multiscale_energy_nn"] = multiscale_energy            # best round-1 clip1 variant, exemplar (nn) mode
FEATURES["hog_body_log_nn"] = FEATURES["hog_body_log"]          # HOG with exemplar matching instead of a pose-blurring mean
FEATURES["fuse_hsv_plus_grad_w0.2"] = fuse_hsv_plus_grad_light
META["multiscale_energy_nn"] = ("l2", "nn")
META["hog_body_log_nn"] = ("cos", "nn")
META["fuse_hsv_plus_grad_w0.2"] = ("chi2", "mean")
