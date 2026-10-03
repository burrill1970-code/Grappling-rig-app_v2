"""Family: texture -- cloth-texture markers for telling A (white gi), B (blue gi) and R (referee suit) apart.

Hypothesis: a judogi has weave / folds / lapel and grip creases, a suit is smooth.  So *how the grey-level surface is
structured* might separate bodies independently of hue.  Everything is computed from the 192-px-high person crop and its
person mask (plus, for two variants, the COCO keypoints to place body parts).  Nothing uses absolute box position, frame
number, crop width/size or image position.

Common conventions
  * Texture is measured on grey or log-grey (log(g + 16): a *relative* contrast, so "dark cloth => small gradient" is
    partly normalised away) over the person mask ERODED a few px, so the silhouette edge (pose / shape) is not counted.
  * Zones are row fractions of the crop (head / torso / legs) -- same convention as gradient_hog -- so no keypoint is needed.
  * No colour and no darkness threshold is used by any pure-texture variant.  NOTE: texture amplitude is nevertheless
    correlated with darkness (dark pixels have little absolute contrast, 8-bit noise/compression floor), so a dark navy
    suit and a smooth white gi can still differ for "darkness" reasons even when the maths is colour-free.  R was
    labelled by exclusion (no colour), so this is not circular, but it is not pure "weave" either.
  * The person mask comes from a whole-frame segmenter, so in contact it also contains the OTHER athlete's pixels
    (texture of the opponent leaks in).  `lbp_torso_kp` / `gradpart_kp` place a patch with the person's own keypoints
    to test whether that helps.
  * The only colour in this file is in the fusion variants `fuse_hsv_glcm`, `fuse_hsv_glcm_w03` (HSV-baseline histogram +
    texture, to measure what texture ADDS) and the colour-only control `ctrl_hsv_hellinger` (same colour block, no texture).
  * `glcm_torso_native` re-reads the clip's full-resolution frame (frame index used only to look up pixels).

FEATURES maps name -> feature_fn(inst); META maps name -> (metric, mode) chosen a priori (chi2 for L1-normalised
histograms, l1/l2 for small statistic vectors).  Per-instance results are cached; call clear_cache() between runs.

The 15 harness evaluations made for this family (V1..V15, all of them, bad ones included) are listed in VARIANTS_EVALUATED;
selection was by clip1 block-protocol balanced accuracy only.  Winner: glcm_torso / l2 / mean.
"""
import numpy as np
import cv2

EPS_LOG = 16.0
MIN_INTERIOR = 150          # fewer interior mask pixels than this -> abstain
MIN_ZONE = 40               # fewer valid px in a zone -> zone falls back to the whole-body statistic
ZONES = ((0.0, 0.20), (0.20, 0.55), (0.55, 1.0))   # head, torso, legs (row fractions of the crop)
KP_CONF = 0.3

_CACHE = {}


def clear_cache():
    _CACHE.clear()


def _key(d, tag):
    return (tag, d["clip"], d["frame"], d["label"], tuple(np.round(d["box"], 1)))


def _cached(d, tag, fn):
    k = _key(d, tag)
    if k not in _CACHE:
        _CACHE[k] = fn()
    return _CACHE[k]


def _erode(mask, r):
    if r <= 0:
        return mask.astype(bool)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask.astype(np.uint8), ker).astype(bool)


def _gray(d):
    return cv2.cvtColor(d["crop"][0], cv2.COLOR_BGR2GRAY).astype(np.float32)


def _loggray(d):
    return np.log(_gray(d) + EPS_LOG)


def _zone_rows(H):
    return [(int(round(a * H)), int(round(b * H))) for a, b in ZONES]


# ----------------------------------------------------------------------------------------------------------------------
# LBP (rotation-invariant uniform, "riu2"): P+2 bins, implemented with vectorised bit planes (no skimage here)
# ----------------------------------------------------------------------------------------------------------------------
def lbp_riu2(g, P, R):
    """Return int code map (H,W) in [0..P+1] for the grey image g (float32).  Neighbours are bilinear-sampled on a circle."""
    H, W = g.shape
    gp = cv2.copyMakeBorder(g, R + 1, R + 1, R + 1, R + 1, cv2.BORDER_REFLECT)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    bits = np.zeros((P, H, W), bool)
    for k in range(P):
        a = 2 * np.pi * k / P
        mx = xs + (R + 1) + R * np.cos(a)
        my = ys + (R + 1) - R * np.sin(a)
        nb = cv2.remap(gp, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        bits[k] = nb >= g
    trans = (bits != np.roll(bits, 1, axis=0)).sum(0)
    pc = bits.sum(0)
    return np.where(trans <= 2, pc, P + 1).astype(np.int32)


def _lbp_hist(codes, valid, P):
    h = np.bincount(codes[valid], minlength=P + 2).astype(np.float64)
    return h


def _lbp_maps(d, radii):
    def f():
        g = _gray(d)
        g = cv2.GaussianBlur(g, (0, 0), 0.7)       # tiny blur: suppress 8-bit / macroblock noise
        out = {}
        for P, R in radii:
            out[(P, R)] = lbp_riu2(g, P, R)
        return out
    return _cached(d, ("lbpmap", tuple(radii)), f)


def _zone_hists(codes, valid, P, nz=len(ZONES)):
    """Per-zone L1-normalised hist (+ whole-body hist for fallback)."""
    H = codes.shape[0]
    whole = _lbp_hist(codes, valid, P)
    out = []
    for r0, r1 in _zone_rows(H):
        v = valid.copy()
        v[:r0] = False
        v[r1:] = False
        if v.sum() >= MIN_ZONE:
            h = _lbp_hist(codes, v, P)
        else:
            h = whole
        out.append(h / max(h.sum(), 1e-9))
    return out, whole / max(whole.sum(), 1e-9)


RADII_1 = ((8, 1),)
RADII_MS = ((8, 1), (16, 2), (24, 3))


def lbp8_whole(d):
    """V1: single-scale riu2 LBP (P=8,R=1) histogram over the whole eroded mask (10 bins)."""
    valid = _erode(d["crop"][1], 3)
    if valid.sum() < MIN_INTERIOR:
        return None
    codes = _lbp_maps(d, RADII_1)[(8, 1)]
    h = _lbp_hist(codes, valid, 8)
    return h / max(h.sum(), 1e-9)


def lbp_ms_zones(d, radii=RADII_MS, zones=True):
    """V2: multi-scale riu2 LBP (R=1,2,3) x 3 row-zones; every (scale,zone) histogram L1-normalised then the whole
    vector divided by the number of blocks (so it sums to 1; chi2 is then well scaled)."""
    valid = _erode(d["crop"][1], 5)
    if valid.sum() < MIN_INTERIOR:
        return None
    maps = _lbp_maps(d, radii)
    parts = []
    for (P, R) in radii:
        if zones:
            hs, _ = _zone_hists(maps[(P, R)], valid, P)
            parts.extend(hs)
        else:
            h = _lbp_hist(maps[(P, R)], valid, P)
            parts.append(h / max(h.sum(), 1e-9))
    v = np.concatenate(parts)
    return v / v.sum()


def _kp_patch_mask(d, shape):
    """Boolean map (crop frame) of the person's OWN torso rectangle from shoulder/hip keypoints (expanded a little),
    falling back to the central rows of the crop.  Crop coordinates: scale = crop_h / box_h (as the baseline does)."""
    H, W = shape
    x1, y1, x2, y2 = [float(v) for v in d["box"]]
    s = H / max(y2 - y1, 1.0)
    kp = d["kp"]
    idx = [5, 6, 11, 12]
    pts = np.array([[(kp[i, 0] - x1) * s, (kp[i, 1] - y1) * s] for i in idx if kp[i, 2] > KP_CONF])
    m = np.zeros((H, W), bool)
    if len(pts) >= 3:
        a, b = pts.min(0), pts.max(0)
        w, h = b - a
        a = a - [0.1 * max(w, 0.25 * H * 0.5), 0.05 * H]
        b = b + [0.1 * max(w, 0.25 * H * 0.5), 0.05 * H]
        ax, ay, bx, by = [int(round(v)) for v in (a[0], a[1], b[0], b[1])]
        ax, ay = max(ax, 0), max(ay, 0)
        bx, by = min(bx, W), min(by, H)
        if bx - ax >= 8 and by - ay >= 8:
            m[ay:by, ax:bx] = True
            return m
    m[int(0.2 * H):int(0.55 * H), int(0.3 * W):int(0.7 * W)] = True
    return m


def lbp_torso_kp(d, radii=RADII_MS):
    """V7: same multi-scale LBP but ONLY inside the person's own keypoint torso rectangle (one zone, no row zones)."""
    mask = d["crop"][1]
    valid = _erode(mask, 5) & _kp_patch_mask(d, mask.shape)
    if valid.sum() < MIN_ZONE:
        valid = _erode(mask, 3) & _kp_patch_mask(d, mask.shape)
    if valid.sum() < MIN_ZONE:
        return None
    maps = _lbp_maps(d, radii)
    parts = []
    for (P, R) in radii:
        h = _lbp_hist(maps[(P, R)], valid, P)
        parts.append(h / max(h.sum(), 1e-9))
    v = np.concatenate(parts)
    return v / v.sum()


# ----------------------------------------------------------------------------------------------------------------------
# Gabor bank energies
# ----------------------------------------------------------------------------------------------------------------------
GAB_LAMBDAS = (4.0, 8.0, 16.0)       # wavelengths in crop px (crop is 192 px high)
GAB_NORI = 6


def _gabor_bank():
    if "bank" not in _CACHE:
        bank = []
        for lam in GAB_LAMBDAS:
            sig = 0.56 * lam
            ks = int(2 * np.ceil(2.5 * sig) + 1)
            row = []
            for o in range(GAB_NORI):
                th = np.pi * o / GAB_NORI
                ke = cv2.getGaborKernel((ks, ks), sig, th, lam, 1.0, 0, ktype=cv2.CV_32F)
                ko = cv2.getGaborKernel((ks, ks), sig, th, lam, 1.0, np.pi / 2, ktype=cv2.CV_32F)
                ke -= ke.mean()                      # zero DC for the even filter
                row.append((ke / np.abs(ke).sum(), ko / np.abs(ko).sum()))
            bank.append(row)
        _CACHE["bank"] = bank
    return _CACHE["bank"]


def _gabor_energy_maps(d):
    """(nscale, nori, H, W) local Gabor energy sqrt(even^2+odd^2) of log-grey, outside-mask filled with the inside mean."""
    def f():
        lg = _loggray(d)
        m = d["crop"][1].astype(bool)
        if m.sum() < 10:
            return None
        lg = lg.copy()
        lg[~m] = lg[m].mean()
        out = np.zeros((len(GAB_LAMBDAS), GAB_NORI) + lg.shape, np.float32)
        for si, row in enumerate(_gabor_bank()):
            for oi, (ke, ko) in enumerate(row):
                re = cv2.filter2D(lg, cv2.CV_32F, ke, borderType=cv2.BORDER_REFLECT)
                im = cv2.filter2D(lg, cv2.CV_32F, ko, borderType=cv2.BORDER_REFLECT)
                out[si, oi] = np.sqrt(re * re + im * im)
        return out
    return _cached(d, "gabor", f)


def gabor_zones(d):
    """V3: Gabor energies (3 wavelengths x 6 orientations) of log-grey, mean over eroded mask in each of 3 row zones
    -> 54 numbers, log(1 + 100 e) transformed."""
    valid = _erode(d["crop"][1], 5)
    if valid.sum() < MIN_INTERIOR:
        return None
    E = _gabor_energy_maps(d)
    if E is None:
        return None
    H = valid.shape[0]
    whole = E[:, :, valid].mean(-1)                   # (ns, no)
    out = []
    for r0, r1 in _zone_rows(H):
        v = valid.copy()
        v[:r0] = False
        v[r1:] = False
        out.append(E[:, :, v].mean(-1) if v.sum() >= MIN_ZONE else whole)
    x = np.stack(out)                                 # (3 zones, ns, no)
    return np.log1p(100.0 * x).ravel()


# ----------------------------------------------------------------------------------------------------------------------
# Local variance / contrast distribution
# ----------------------------------------------------------------------------------------------------------------------
STD_EDGES = np.array([0.0, 0.01, 0.02, 0.035, 0.06, 0.10, 0.17, 0.30, 10.0])    # in log-grey units
STD_WIN = (3, 7, 15)


def _localstd_maps(d):
    def f():
        lg = _loggray(d)
        m = d["crop"][1].astype(np.float32)
        out = []
        for w in STD_WIN:
            k = (w, w)
            sm = cv2.boxFilter(m, cv2.CV_32F, k, normalize=True, borderType=cv2.BORDER_CONSTANT)
            s1 = cv2.boxFilter(lg * m, cv2.CV_32F, k, normalize=True, borderType=cv2.BORDER_CONSTANT)
            s2 = cv2.boxFilter(lg * lg * m, cv2.CV_32F, k, normalize=True, borderType=cv2.BORDER_CONSTANT)
            mu = s1 / np.maximum(sm, 1e-6)
            var = np.maximum(s2 / np.maximum(sm, 1e-6) - mu * mu, 0)
            std = np.sqrt(var)
            std[sm < 0.9] = np.nan                    # window must lie (almost) fully inside the person mask
            out.append(std)
        return out
    return _cached(d, "localstd", f)


def localstd_hist(d):
    """V4: histogram of local standard deviation of log-grey (3 window sizes x 3 row zones x 8 bins), L1-normalised.
    Windows must lie >=90% inside the person mask (no silhouette edge).  Describes the *distribution* of local contrast."""
    valid = _erode(d["crop"][1], 2)
    if valid.sum() < MIN_INTERIOR:
        return None
    maps = _localstd_maps(d)
    H = valid.shape[0]
    parts = []
    for std in maps:
        ok = valid & np.isfinite(std)
        whole = np.histogram(std[ok], STD_EDGES)[0].astype(float) if ok.sum() else np.zeros(len(STD_EDGES) - 1)
        for r0, r1 in _zone_rows(H):
            v = ok.copy()
            v[:r0] = False
            v[r1:] = False
            h = np.histogram(std[v], STD_EDGES)[0].astype(float) if v.sum() >= MIN_ZONE else whole
            parts.append(h / max(h.sum(), 1e-9) if h.sum() > 0 else np.full(len(h), 1.0 / len(h)))
    v = np.concatenate(parts)
    return v / v.sum()


# ----------------------------------------------------------------------------------------------------------------------
# GLCM (co-occurrence) statistics -- numpy implementation (skimage not installed)
# ----------------------------------------------------------------------------------------------------------------------
GLCM_L = 16
GLCM_OFFS = [(dy, dx) for dist in (1, 2, 4) for (dy, dx) in ((0, dist), (dist, dist), (dist, 0), (dist, -dist))]


def _glcm_stats(q, valid, L=GLCM_L):
    """Haralick stats averaged over 4 angles for each of distances 1,2,4 -> 18 numbers in ~[0,1]."""
    H, W = q.shape
    feats = []
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    for di in range(3):
        P = np.zeros((L, L))
        for (dy, dx) in GLCM_OFFS[4 * di:4 * di + 4]:
            ya, yb = max(0, -dy), H - max(0, dy)
            xa, xb = max(0, -dx), W - max(0, dx)
            a = q[ya:yb, xa:xb]
            b = q[ya + dy:yb + dy, xa + dx:xb + dx]
            ok = valid[ya:yb, xa:xb] & valid[ya + dy:yb + dy, xa + dx:xb + dx]
            if ok.sum() == 0:
                continue
            c = np.bincount((a[ok] * L + b[ok]).ravel(), minlength=L * L).reshape(L, L).astype(float)
            P += c + c.T
        s = P.sum()
        if s == 0:
            feats.extend([0, 0, 1, 1, 0, 1])
            continue
        P /= s
        mu_i = (ii * P).sum()
        sd = np.sqrt(max(((ii - mu_i) ** 2 * P).sum(), 1e-9))
        contrast = ((ii - jj) ** 2 * P).sum() / (L - 1) ** 2
        dissim = (np.abs(ii - jj) * P).sum() / (L - 1)
        homog = (P / (1.0 + (ii - jj) ** 2)).sum()
        asm = (P ** 2).sum()
        ent = -(P[P > 0] * np.log(P[P > 0])).sum() / np.log(L * L)
        corr = (((ii - mu_i) * (jj - mu_i) * P).sum()) / (sd * sd)
        feats.extend([contrast * 4, dissim * 2, homog, asm * 4, ent, 0.5 * (corr + 1)])
    return np.array(feats)


def _glcm_region(d, v):
    """Haralick stats of the grey crop inside boolean region v: 2-98 percentile stretch inside v to 16 levels (so the
    statistic describes the *pattern*, independent of absolute brightness/contrast), pairs counted only when both px in v."""
    g = cv2.GaussianBlur(_gray(d), (0, 0), 0.7)
    lo, hi = np.percentile(g[v], [2, 98])
    q = np.clip((g - lo) / max(hi - lo, 8.0), 0, 1)           # >= 8 grey levels span: flat patches stay flat
    q = np.minimum((q * GLCM_L).astype(np.int32), GLCM_L - 1)
    return _glcm_stats(q, v)


def _zone_valid(d, zi, erode=3):
    valid = _erode(d["crop"][1], erode)
    r0, r1 = _zone_rows(valid.shape[0])[zi]
    v = valid.copy()
    v[:r0] = False
    v[r1:] = False
    return v, valid


def glcm_torso(d):
    """V5: GLCM (Haralick) statistics on the torso row-zone (20-55% of the crop height), 18 numbers (6 stats x 3 distances)."""
    v, valid = _zone_valid(d, 1)
    if v.sum() < MIN_INTERIOR:
        v = valid
        if v.sum() < MIN_INTERIOR:
            return None
    return _glcm_region(d, v)


def glcm_zones(d):
    """V8: GLCM stats for the torso zone AND the legs zone (55-100%), concatenated (36 numbers)."""
    parts = []
    whole = _erode(d["crop"][1], 3)
    if whole.sum() < MIN_INTERIOR:
        return None
    for zi in (1, 2):
        v, valid = _zone_valid(d, zi)
        parts.append(_glcm_region(d, v if v.sum() >= MIN_INTERIOR else valid))
    return np.concatenate(parts)


def glcm_kp(d):
    """V9: GLCM stats inside the person's own keypoint torso rectangle (instead of the torso row zone)."""
    mask = d["crop"][1]
    valid = _erode(mask, 3)
    v = valid & _kp_patch_mask(d, mask.shape)
    if v.sum() < MIN_INTERIOR:
        v = valid
        if v.sum() < MIN_INTERIOR:
            return None
    return _glcm_region(d, v)


# ----------------------------------------------------------------------------------------------------------------------
# Native-resolution variant: same GLCM torso statistic but computed on the box cut from the FULL-RES video frame
# (crops are 192 px high, boxes are 300-500 px high, so the crop discards ~2x of the resolution where a weave would live)
# ----------------------------------------------------------------------------------------------------------------------
VIDEOS = {"clip1": "test_clips/clip.mp4", "clip2": "test_clips/clip2.mp4"}


def _native_frames(clip):
    """Lazily read the clip once; keep only the frames that the dataset references (the frame index is used only to
    LOOK UP pixels, never as a feature)."""
    key = ("frames", clip)
    if key not in _CACHE:
        from grappling.video import read_frames
        from tools.marker_eval import load
        need = {d["frame"] for d in load() if d["clip"] == clip}
        _CACHE[key] = {i: f for i, f in enumerate(read_frames(VIDEOS[clip])) if i in need}
    return _CACHE[key]


def glcm_torso_native(d):
    """V15: GLCM torso statistic (identical recipe to glcm_torso) on the person box cut from the full-resolution frame;
    the crop's person mask is resized back (nearest) to the native box size and the erosion radius is scaled with it."""
    fr = _native_frames(d["clip"]).get(d["frame"])
    if fr is None:
        return None
    H, W = fr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    c = fr[y1:y2, x1:x2]
    if c.shape[0] < 32 or c.shape[1] < 8:
        return None
    m = cv2.resize(d["crop"][1].astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    r = max(3, int(round(3 * c.shape[0] / 192.0)))
    valid = _erode(m, r)
    r0, r1 = [int(round(a * c.shape[0])) for a in ZONES[1]]
    v = valid.copy()
    v[:r0] = False
    v[r1:] = False
    if v.sum() < MIN_INTERIOR:
        v = valid
        if v.sum() < MIN_INTERIOR:
            return None
    g = cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.7)
    lo, hi = np.percentile(g[v], [2, 98])
    q = np.clip((g - lo) / max(hi - lo, 8.0), 0, 1)
    q = np.minimum((q * GLCM_L).astype(np.int32), GLCM_L - 1)
    return _glcm_stats(q, v)


# ----------------------------------------------------------------------------------------------------------------------
# Gradient energy per body part (keypoint-placed limb strips)
# ----------------------------------------------------------------------------------------------------------------------
PARTS = {   # name -> list of (joint a, joint b) segments pooled left+right
    "upper_arm": [(5, 7), (6, 8)],
    "forearm": [(7, 9), (8, 10)],
    "thigh": [(11, 13), (12, 14)],
    "shin": [(13, 15), (14, 16)],
}


def _grad_mag(d, sigma):
    def f():
        lg = cv2.GaussianBlur(_loggray(d), (0, 0), sigma)
        gx = cv2.Scharr(lg, cv2.CV_32F, 1, 0) / 32.0
        gy = cv2.Scharr(lg, cv2.CV_32F, 0, 1) / 32.0
        return np.sqrt(gx * gx + gy * gy)
    return _cached(d, ("gmag", sigma), f)


def _part_masks(d, shape):
    """Boolean masks (crop frame) for torso + 4 pooled limb parts, from the person's own keypoints; None if unavailable."""
    H, W = shape
    x1, y1, x2, y2 = [float(v) for v in d["box"]]
    s = H / max(y2 - y1, 1.0)
    kp = d["kp"]
    pt = lambda i: ((kp[i, 0] - x1) * s, (kp[i, 1] - y1) * s)
    out = {}
    # torso quad
    if all(kp[i, 2] > KP_CONF for i in (5, 6, 11, 12)):
        quad = np.array([pt(5), pt(6), pt(12), pt(11)], np.float32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
        out["torso"] = m.astype(bool)
    for name, segs in PARTS.items():
        m = np.zeros((H, W), np.uint8)
        ok = False
        for a, b in segs:
            if kp[a, 2] > KP_CONF and kp[b, 2] > KP_CONF:
                pa, pb = pt(a), pt(b)
                ln = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
                th = int(max(5, round(0.3 * ln)))
                cv2.line(m, tuple(int(round(v)) for v in pa), tuple(int(round(v)) for v in pb), 1, th)
                ok = True
        if ok:
            out[name] = m.astype(bool)
    return out


def gradpart_kp(d):
    """V6: mean gradient magnitude of log-grey (sigma 1 and 2) in 5 keypoint-placed body parts (torso, upper arms,
    forearms, thighs, shins), restricted to the eroded person mask.  Missing parts take the mean of the available ones
    (no information).  10 numbers, log(g + 0.01)."""
    mask = d["crop"][1]
    valid = _erode(mask, 4)
    if valid.sum() < MIN_INTERIOR:
        return None
    pm = _part_masks(d, mask.shape)
    names = ["torso"] + list(PARTS)
    vals = np.full((len(names), 2), np.nan)
    for si, sg in enumerate((1.0, 2.0)):
        gm = _grad_mag(d, sg)
        for pi, n in enumerate(names):
            if n in pm:
                v = pm[n] & valid
                if v.sum() >= MIN_ZONE:
                    vals[pi, si] = gm[v].mean()
    if np.isnan(vals).all():
        # no usable keypoints: fall back to whole interior mean
        vals[:] = np.array([[_grad_mag(d, 1.0)[valid].mean(), _grad_mag(d, 2.0)[valid].mean()]] * len(names))
    col = np.nanmean(vals, 0)
    vals = np.where(np.isnan(vals), col[None, :], vals)
    return np.log(vals.ravel() + 0.01)


# ----------------------------------------------------------------------------------------------------------------------
# Combinations (heterogeneous blocks are made comparable by Hellinger mapping: sqrt of an L1-normalised histogram is a unit
# L2 vector; GLCM stats are already O(1).  Metric l2 on the concatenation.  Fixed a-priori weights 1:1.)
# ----------------------------------------------------------------------------------------------------------------------
def _hsv_baseline(d):
    from tools.marker_baseline import hsv_torso
    return hsv_torso(d)


def glcm_lbp_concat(d):
    """V11: [GLCM torso stats (18), Hellinger(LBP multi-scale inside keypoint torso) (54, unit norm)]; l2."""
    a, b = glcm_torso(d), lbp_torso_kp(d)
    if a is None or b is None:
        return None
    return np.concatenate([a, np.sqrt(b)])


def fuse_hsv_glcm(d):
    """V12: baseline HSV torso histogram (Hellinger, unit norm) + GLCM torso stats (18); l2.  Measures what texture ADDS."""
    a, b = _hsv_baseline(d), glcm_torso(d)
    if a is None or b is None:
        return None
    return np.concatenate([np.sqrt(a), b])


def fuse_hsv_glcm_w03(d):
    """V13: as fuse_hsv_glcm but the GLCM block is down-weighted to 0.3 (texture as a tie-breaker, not an equal partner)."""
    a, b = _hsv_baseline(d), glcm_torso(d)
    if a is None or b is None:
        return None
    return np.concatenate([np.sqrt(a), 0.3 * b])


def ctrl_hsv_hellinger(d):
    """V14 (CONTROL, not texture): baseline HSV torso histogram alone, Hellinger-mapped, l2 -- same representation as the
    colour block in the fusions, so the effect of adding texture is not confounded with the chi2 -> Hellinger/l2 switch."""
    a = _hsv_baseline(d)
    return None if a is None else np.sqrt(a)


FEATURES = {
    "lbp8_whole": lbp8_whole,
    "lbp_ms_zones": lbp_ms_zones,
    "gabor_zones": gabor_zones,
    "localstd_hist": localstd_hist,
    "glcm_torso": glcm_torso,
    "gradpart_kp": gradpart_kp,
    "lbp_torso_kp": lbp_torso_kp,
    "glcm_zones": glcm_zones,
    "glcm_kp": glcm_kp,
    "glcm_lbp_concat": glcm_lbp_concat,
    "fuse_hsv_glcm": fuse_hsv_glcm,
    "fuse_hsv_glcm_w03": fuse_hsv_glcm_w03,
    "ctrl_hsv_hellinger": ctrl_hsv_hellinger,
    "glcm_torso_native": glcm_torso_native,
}

META = {
    "lbp8_whole": ("chi2", "mean"),
    "lbp_ms_zones": ("chi2", "mean"),
    "gabor_zones": ("l1", "mean"),
    "localstd_hist": ("chi2", "mean"),
    "glcm_torso": ("l2", "mean"),
    "gradpart_kp": ("l2", "mean"),
    "lbp_torso_kp": ("chi2", "mean"),
    "glcm_zones": ("l2", "mean"),
    "glcm_kp": ("l2", "mean"),
    "glcm_lbp_concat": ("l2", "mean"),
    "fuse_hsv_glcm": ("l2", "mean"),
    "fuse_hsv_glcm_w03": ("l2", "mean"),
    "ctrl_hsv_hellinger": ("l2", "mean"),
    "glcm_torso_native": ("l2", "mean"),
}


VARIANTS_EVALUATED = (   # (feature, metric, mode) -- in the order they were run
    ("lbp8_whole", "chi2", "mean"), ("lbp_ms_zones", "chi2", "mean"), ("gabor_zones", "l1", "mean"),
    ("localstd_hist", "chi2", "mean"), ("glcm_torso", "l2", "mean"), ("gradpart_kp", "l2", "mean"),
    ("lbp_torso_kp", "chi2", "mean"), ("glcm_zones", "l2", "mean"), ("glcm_kp", "l2", "mean"),
    ("glcm_torso", "l2", "nn"), ("glcm_lbp_concat", "l2", "mean"), ("fuse_hsv_glcm", "l2", "mean"),
    ("fuse_hsv_glcm_w03", "l2", "mean"), ("ctrl_hsv_hellinger", "l2", "mean"), ("glcm_torso_native", "l2", "mean"),
)
BEST = "glcm_torso"


if __name__ == "__main__":
    from tools.marker_eval import evaluate
    for n, f in FEATURES.items():
        evaluate(f, n, metric=META[n][0], mode=META[n][1])
