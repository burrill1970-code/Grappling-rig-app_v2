"""Family: context_motion -- NON-appearance markers (motion / stationarity / geometry) for telling A, B and R apart.

Nothing here looks at pixels.  Everything comes from the raw per-frame detections (outputs/raw_dets.npz for clip1,
outputs/clip2/raw_dets.npz for clip2: box, conf, 17 keypoints) of the instance's own clip, linked over neighbouring frames
by IoU, plus the instance's own box / keypoints.  The feature functions only get `inst`; the clip + frame in it are used
ONLY to look up the neighbouring raw detections of that frame, never as a feature.

Invariance rules (the clips are broadcast footage: cuts, pans, zooms; per-clip scale and camera position differ):
  * every length is divided by the instance's own box height  -> invariant to zoom / distance to the camera
  * no absolute image position, no absolute time, no frame index, no crop size, is ever put into a vector
  * velocity DIRECTION is never used (it is an absolute image direction); only speeds (magnitudes)
  * camera pans add a common velocity to everybody: the `articulation` motion is measured relative to the box centre
    (pose change, invariant to a pure pan) and `cc_` variants subtract the median velocity of ALL tracked boxes
  * "frame height" is only used as a divisor for the box height in the `hrel_frame` quantity (FRAME_H = 608, the
    cropped picture height of both clips); the preferred height quantity is relative to the other persons of the SAME
    frame (so it is also invariant to zoom)
Honesty notes
  * The R label was made by EXCLUSION: R = a confident on-mat person that does NOT overlap (IoU > 0.3) the two verified
    athlete boxes.  Any feature of "overlap / distance to the nearest other person" therefore partly reproduces the
    labelling rule (A and B overlap each other when in contact, R by construction does not).  Those variants are tagged
    `_pair` and reported as such; the other variants do not use inter-person geometry.
  * No colour, no darkness.  No model is trained.  Prototype scaling is hard-coded a priori (SCALE_*), not fitted.
FEATURES maps name -> feature_fn(inst); META maps name -> (metric, mode), fixed a priori (l2 + mean prototypes for all
but one nn variant; the vectors are log-speeds / log-aspects, not histograms, so chi2 is not used).
Variants are the 14 entries of FEATURES (all of them were run; none was dropped).  Best = highest CLIP1 block-protocol
balanced accuracy among the non-diagnostic ones (BEST); clip2 was never used to choose anything.
Abstention: motion variants return None when the person cannot be linked to ANY neighbouring frame (10 of 324 instances).
"""
import numpy as np

FRAME_H = 608.0
IOU_LINK = 0.3
CONF_MIN = 0.3          # a detection counts as "a person in the frame" above this confidence
KP_CONF = 0.3

_RAW = {}
_PATHS = {"clip1": "outputs/raw_dets.npz", "clip2": "outputs/clip2/raw_dets.npz"}


def _raw(clip):
    if clip not in _RAW:
        _RAW[clip] = list(np.load(_PATHS[clip], allow_pickle=True)["dets"])
    return _RAW[clip]


def _iou_mat(a, b):
    """IoU between every box in a (n,4+) and every box in b (m,4+)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _link(clip, f, i, step):
    """Index of the detection in frame f+step that is the mutual-best IoU match of det i of frame f, or None.
    Mutual best + IoU >= IOU_LINK + similar box height (<= 1.5x) so that a scene cut does not create a link."""
    raw = _raw(clip)
    g = f + step
    if g < 0 or g >= len(raw) or len(raw[g]) == 0:
        return None
    m = _iou_mat(raw[f][:, :4], raw[g][:, :4])
    j = int(np.argmax(m[i]))
    if m[i, j] < IOU_LINK or int(np.argmax(m[:, j])) != i:
        return None
    h0 = raw[f][i, 3] - raw[f][i, 1]
    h1 = raw[g][j, 3] - raw[g][j, 1]
    if max(h0, h1) > 1.5 * min(h0, h1):
        return None
    return j


def _own_index(inst):
    raw = _raw(inst["clip"])[inst["frame"]]
    d = np.abs(raw[:, :4] - inst["box"][None]).max(1)
    return int(np.argmin(d))


def _chain(inst, W):
    """{offset: det_row} for offsets -W..W reached by contiguous links from the instance (offset 0 = the instance)."""
    clip, f = inst["clip"], inst["frame"]
    raw = _raw(clip)
    i0 = _own_index(inst)
    out = {0: raw[f][i0]}
    for sgn in (1, -1):
        i, cur = i0, f
        for k in range(1, W + 1):
            j = _link(clip, cur, i, sgn)
            if j is None:
                break
            cur += sgn
            i = j
            out[sgn * k] = raw[cur][j]
    return out


def _cen(d):
    return np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])


def _height(d):
    return d[3] - d[1]


def _kp_rel(d):
    """keypoints (x,y,conf) -> positions relative to the box centre in box-height units, with visibility mask"""
    kp = d[5:].reshape(17, 3)
    h = max(_height(d), 1.0)
    return (kp[:, :2] - _cen(d)[None]) / h, kp[:, 2] > KP_CONF


def _context_people(inst):
    """Other confident persons of the same frame (conf >= 0.5, height >= 25% of the tallest one: drops tiny background
    spectators without using any absolute pixel size or position)."""
    raw = _raw(inst["clip"])[inst["frame"]]
    i0 = _own_index(inst)
    ok = [k for k, d in enumerate(raw) if d[4] >= 0.5 or k == i0]
    if not ok:
        return []
    hmax = max(_height(raw[k]) for k in ok)
    return [raw[k] for k in ok if k != i0 and _height(raw[k]) >= 0.25 * hmax]


EPS_V = 0.005     # floor of the log-speed transforms, in box-heights per frame (a standing person jitters ~0.003-0.01)
_Q = {}


def _quantities(inst, W):
    """All scalar quantities of one instance (cached by clip/frame/box/W).  Returns a dict; motion entries are NaN when
    the instance cannot be linked to a neighbouring frame at all."""
    key = (inst["clip"], inst["frame"], tuple(np.round(inst["box"], 2)), W)
    if key in _Q:
        return _Q[key]
    ch = _chain(inst, W)
    offs = sorted(ch)
    d0 = ch[0]
    h0 = _height(d0)
    sp, art, dh, da, path = [], [], [], [], 0.0
    for a, b in zip(offs[:-1], offs[1:]):
        if b - a != 1:
            continue
        pa, pb = ch[a], ch[b]
        step = np.linalg.norm(_cen(pb) - _cen(pa)) / h0
        sp.append(step)
        ka, ma = _kp_rel(pa)
        kb, mb = _kp_rel(pb)
        m = ma & mb
        if m.sum() >= 3:
            art.append(np.linalg.norm(kb[m] - ka[m], axis=1).mean())
        dh.append(abs(np.log(_height(pb) / _height(pa))))
        wa, wb = pa[2] - pa[0], pb[2] - pb[0]
        da.append(abs(np.log((wb / _height(pb)) / (wa / _height(pa)))))
    nan = float("nan")
    q = dict(
        sp=float(np.mean(sp)) if sp else nan,
        art=float(np.mean(art)) if art else nan,
        dh=float(np.mean(dh)) if dh else nan,
        da=float(np.mean(da)) if da else nan,
        path=float(np.sum(sp)) if sp else nan,
        net=float(np.linalg.norm(_cen(ch[offs[-1]]) - _cen(ch[offs[0]])) / h0) if len(offs) > 1 else nan,
        frac=len(ch) / (2 * W + 1),
        asp=float(np.log((d0[2] - d0[0]) / h0)),
    )
    oth = _context_people(inst)
    q["hrel"] = float(np.log(h0 / np.median([_height(o) for o in oth]))) if oth else 0.0
    q["hfr"] = float(h0 / FRAME_H)
    if oth:
        c0 = _cen(d0)
        dist = [np.linalg.norm(_cen(o) - c0) / h0 for o in oth]
        k = int(np.argmin(dist))
        q["nd"] = float(dist[k])
        q["iou"] = float(_iou_mat(d0[None, :4], np.array([o[:4] for o in oth])).max())
        q["dx"] = float((c0[0] - _cen(oth[k])[0]) / h0)
    else:
        q["nd"], q["iou"], q["dx"] = 3.0, 0.0, 0.0     # nobody else in the frame: "far", no overlap, no side
    _Q[key] = q
    return q


def _lg(x):
    return np.log(x + EPS_V)


def _scal(q, n):
    if n == "lsp":
        return _lg(q["sp"])
    if n == "lart":
        return _lg(q["art"])
    if n == "ldh":
        return _lg(q["dh"])
    if n == "lda":
        return _lg(q["da"])
    if n == "lnet":
        return _lg(q["net"])
    if n == "lpath":
        return _lg(q["path"])
    if n == "tort":
        return q["net"] / (q["path"] + EPS_V)
    if n == "frac":
        return q["frac"]
    if n == "asp":
        return q["asp"]
    if n == "hrel":
        return q["hrel"]
    if n == "lhfr":
        return np.log(q["hfr"])
    if n == "lnd":
        return np.log(q["nd"] + 0.1)
    if n == "iou":
        return q["iou"]
    if n == "side":
        return np.tanh(q["dx"])
    raise KeyError(n)


def _vec(inst, W, names):
    q = _quantities(inst, W)
    v = np.array([_scal(q, n) for n in names], float)
    return None if np.isnan(v).any() else v


def _rel_vec(inst, W, names):
    """Each quantity minus the mean of the same quantity over the OTHER confident persons of the same frame (their own
    chains are computed the same way).  Standardises away what all people in the shot share (camera pan / blur / zoom,
    a still or a busy moment); NaN-valued others are ignored; nobody else valid -> 0 (neutral)."""
    q = _quantities(inst, W)
    qo = [_quantities(dict(clip=inst["clip"], frame=inst["frame"], box=o[:4]), W) for o in _context_people(inst)]
    out = []
    for n in names:
        a = _scal(q, n)
        b = [_scal(x, n) for x in qo]
        b = [x for x in b if not np.isnan(x)]
        out.append(a - np.mean(b) if b else 0.0)
    v = np.array(out, float)
    return None if np.isnan(v).any() else v


def make(W, names):
    return lambda inst: _vec(inst, W, names)


MOT = ("lsp", "lart", "ldh")
SHP = ("asp", "hrel")
PAIR = ("lnd", "iou")

# name -> feature_fn.  `_pair` = uses inter-person geometry that partly mirrors the R-by-exclusion labelling rule;
# `side_diag` = diagnostic only (left/right order relative to the nearest other person persists in time but is not an
# identity cue), excluded from best-variant selection.
FEATURES = {
    "speed_artic_W4": make(4, MOT),
    "artic_only_W4": make(4, ("lart", "ldh", "lda")),
    "shape_ctx": make(4, SHP),
    "shape_frame": make(4, ("asp", "lhfr")),
    "motion_shape_W4": make(4, MOT + SHP),
    "pair_geom_pair": make(4, PAIR),
    "all_ctx_pair": make(4, MOT + SHP + PAIR),
    "speed_artic_W2": make(2, MOT),
    "speed_artic_W8": make(8, MOT),
    "speed_artic_W4_nn": make(4, MOT),
    "track_stat_W8": make(8, ("lnet", "lpath", "tort", "frac")),
    "side_diag": make(4, ("side",)),
    "rel_ctx_W4": lambda inst: _rel_vec(inst, 4, ("lsp", "lart", "asp")),
    "still_narrow_W4": make(4, ("lart", "asp")),
}
META = {n: ("l2", "mean") for n in FEATURES}
META["speed_artic_W4_nn"] = ("l2", "nn")
DIAGNOSTIC = {"side_diag"}
BEST = "artic_only_W4"


if __name__ == "__main__":
    import sys
    from tools.marker_eval import evaluate
    names = sys.argv[1:] or list(FEATURES)
    for n in names:
        metric, mode = META[n]
        evaluate(FEATURES[n], n, metric=metric, mode=mode)
