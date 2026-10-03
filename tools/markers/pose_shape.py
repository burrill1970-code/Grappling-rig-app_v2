"""Family: pose_shape -- body-shape biometrics from the 17 COCO keypoints (+ box) for telling A, B and R apart.

Everything is derived from inst["kp"] (x, y, conf) and the box SHAPE.  Only keypoints with conf > CONF (0.3) are used.
Nothing uses absolute box position, absolute box size, frame number, crop size or image position: every quantity is
either a ratio between lengths (scale free), an angle, or a coordinate expressed relative to the box (unit = box
height).  No colour and no darkness is used at all, so there is no overlap with how R was labelled (by exclusion).
No model is trained.  Missing keypoints are handled in two ways: the instance abstains (None) when the torso cannot be
measured at all or fewer than two limb bones can; otherwise a missing feature is filled with a hard-coded generic value
(REF_*; label-free medians of the pooled clip1 instances, written down once, never re-fitted) so that it does not
contribute a difference between prototypes.

Quantities
  bones   torso (mid-shoulder -> mid-hip), shoulder width, hip width, upper arm, forearm, thigh, shin (left/right
          averaged when both are visible), head-neck length (nose -> mid-shoulder).
  prop_*  proportion features: log bone lengths relative to torso length (prop_logT) or relative to the geometric mean
          of all measurable bones (prop_clr, "centred log-ratio"; no single bone is privileged) or, without the torso
          at all, relative to the geometric mean of the 7 non-torso bones (prop_free; immune to torso foreshortening).
  posture box aspect (log w/h), torso axis (|ux|, uy), leg / arm straightness, vertical leg extension: crouched vs
          upright, grappling vs standing.  These are POSE, not identity, so they separate roles, not people.
  pose_box  the 13 head/limb joints as coordinates inside the box (unit = box height).

FEATURES maps name -> feature_fn(inst); META maps name -> (metric, mode) fixed a priori (l2 for dense real-valued
vectors, none of the vectors is a histogram so chi2 is not used).
"""
import numpy as np

CONF = 0.3
BONES = ("tor", "shw", "hipw", "uarm", "farm", "thigh", "shin", "head")
# generic log(bone / torso) used to fill unmeasurable bones (label-free pooled clip1 medians)
REF_T = np.array([0.0, -1.02, -1.42, -0.63, -0.84, -0.21, -0.24, -0.78])
REF_CLR = REF_T - REF_T.mean()
REF_FREE = REF_T[1:] - REF_T[1:].mean()
REF_POSTURE = np.array([0.0, 0.0, -1.0, 0.9, 0.8, 0.8])        # generic upright person
# generic joint positions inside the box, x/h (centre) and y/h, for [nose, sh, sh, elb, elb, wri, wri, hip, hip, kn, kn, an, an]
JOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
REF_Y = np.array([0.07, 0.20, 0.20, 0.35, 0.35, 0.50, 0.50, 0.50, 0.50, 0.72, 0.72, 0.95, 0.95])
MIN_LIMB_BONES = 2

_CACHE = {}


def clear_cache():
    _CACHE.clear()


def _key(d, tag):
    return (tag, d["clip"], d["frame"], d["label"], tuple(np.round(d["box"], 1)))


def _geom(d, conf=CONF):
    """Basic measurements, or None when the torso / limbs cannot be measured."""
    kp = d["kp"]
    xy, ok = kp[:, :2], kp[:, 2] > conf

    def seg(i, j):
        return float(np.hypot(*(xy[i] - xy[j]))) if ok[i] and ok[j] else np.nan

    def side(pairs):
        v = [x for x in (seg(i, j) for i, j in pairs) if x == x]
        return float(np.mean(v)) if v else np.nan

    sh = [xy[i] for i in (5, 6) if ok[i]]
    hp = [xy[i] for i in (11, 12) if ok[i]]
    if not sh or not hp:
        return None
    ms, mh = np.mean(sh, 0), np.mean(hp, 0)
    tor = float(np.hypot(*(ms - mh)))
    x1, y1, x2, y2 = [float(v) for v in d["box"]]
    h = max(y2 - y1, 1.0)
    if tor < 0.02 * h:                         # degenerate torso (point-like): cannot normalise
        return None
    bones = np.array([
        tor,
        seg(5, 6),
        seg(11, 12),
        side([(5, 7), (6, 8)]),
        side([(7, 9), (8, 10)]),
        side([(11, 13), (12, 14)]),
        side([(13, 15), (14, 16)]),
        float(np.hypot(*(xy[0] - ms))) if ok[0] else np.nan,
    ])
    if np.sum(~np.isnan(bones[3:7])) < MIN_LIMB_BONES:
        return None
    u = (ms - mh) / tor                         # unit torso axis, hips -> shoulders (image y points down)
    # leg / arm straightness and vertical leg extension, per side then averaged
    def straight(a, b, c):
        if ok[a] and ok[b] and ok[c]:
            s = seg(a, b) + seg(b, c)
            return seg(a, c) / s if s > 1e-6 else np.nan
        return np.nan
    leg_s = np.nanmean([straight(11, 13, 15), straight(12, 14, 16)]) if (ok[11] and ok[13] and ok[15]) or (ok[12] and ok[14] and ok[16]) else np.nan
    arm_s = np.nanmean([straight(5, 7, 9), straight(6, 8, 10)]) if (ok[5] and ok[7] and ok[9]) or (ok[6] and ok[8] and ok[10]) else np.nan
    lv = []
    for hi, kn, an in ((11, 13, 15), (12, 14, 16)):
        if ok[hi] and ok[kn] and ok[an]:
            ll = seg(hi, kn) + seg(kn, an)
            if ll > 1e-6:
                lv.append((xy[an][1] - xy[hi][1]) / ll)
    leg_v = float(np.mean(lv)) if lv else np.nan
    return dict(bones=bones, u=u, h=h, w=max(x2 - x1, 1.0), leg_s=leg_s, arm_s=arm_s, leg_v=leg_v,
                xy=xy, ok=ok, box=(x1, y1, x2, y2))


def _g(d, conf=CONF):
    k = _key(d, f"g{conf}")
    if k not in _CACHE:
        if len(_CACHE) > 6000:
            _CACHE.clear()
        _CACHE[k] = _geom(d, conf)
    return _CACHE[k]


def _log(b, floor):
    return np.log(np.maximum(b, floor))


def _fill(v, ref):
    v = np.array(v, float)
    m = np.isnan(v)
    v[m] = ref[m]
    return v


# ---------------------------------------------------------------- feature blocks
def _prop_logT(g):
    b = g["bones"]
    return _fill(_log(b, 0.02 * g["h"]) - np.log(max(b[0], 0.02 * g["h"])), REF_T)[1:]       # 7 dims (torso == 0 dropped)


def _prop_clr(g):
    b = g["bones"]
    l = _log(b, 0.02 * g["h"])
    l = np.where(np.isnan(b), np.nan, l)
    return _fill(l - np.nanmean(l), REF_CLR)                                                  # 8 dims


def _prop_free(g):
    b = g["bones"][1:]
    l = np.where(np.isnan(b), np.nan, _log(b, 0.02 * g["h"]))
    return _fill(l - np.nanmean(l), REF_FREE)                                                 # 7 dims, no torso


def _posture(g):
    u = g["u"]
    v = [np.log(g["w"] / g["h"]), abs(u[0]), u[1], g["leg_s"], g["arm_s"], g["leg_v"]]
    return _fill(v, REF_POSTURE)                                                              # 6 dims


def _pose_box(g):
    x1, y1, x2, y2 = g["box"]
    xy, ok, h = g["xy"], g["ok"], g["h"]
    cx = 0.5 * (x2 - x1) / h
    out = []
    for n, j in enumerate(JOINTS):
        if ok[j]:
            out += [(xy[j][0] - x1) / h - cx, (xy[j][1] - y1) / h]     # x centred on the box centre
        else:
            out += [0.0, REF_Y[n]]
    return np.clip(out, -2, 2)                                                                # 26 dims


# ---------------------------------------------------------------- features
def _mk(*blocks, conf=CONF):
    def fn(d):
        g = _g(d, conf)
        return None if g is None else np.concatenate([b(g) for b in blocks])
    return fn


def box_aspect(d):
    x1, y1, x2, y2 = d["box"]
    return np.array([np.log((x2 - x1) / max(y2 - y1, 1.0))])


def _aspect_tilt(g):
    return np.array([np.log(g["w"] / g["h"]), abs(g["u"][0])])                                # 2 dims


FEATURES = {
    "box_aspect": box_aspect,
    "aspect_tilt": _mk(_aspect_tilt),
    "posture": _mk(_posture),
    "prop_logT": _mk(_prop_logT),
    "prop_clr": _mk(_prop_clr),
    "prop_free": _mk(_prop_free),
    "prop_clr+posture": _mk(_prop_clr, _posture),
    "pose_box": _mk(_pose_box),
    "prop_clr+posture+pose_box": _mk(_prop_clr, _posture, _pose_box),
}
META = {k: ("l2", "mean") for k in FEATURES}      # default; the nn runs are listed explicitly in RUNS
META["pose_box"] = ("l2", "nn")                   # chosen by CLIP1 block bal_acc (0.757) among the 15 RUNS
BEST = "pose_box"

# Every (feature, metric, mode) that was run through tools.marker_eval.evaluate -- all 15, none omitted.
RUNS = [
    ("box_aspect", "l2", "mean"), ("posture", "l2", "mean"), ("prop_logT", "l2", "mean"), ("prop_clr", "l2", "mean"),
    ("prop_free", "l2", "mean"), ("prop_clr+posture", "l2", "mean"), ("pose_box", "l2", "mean"),
    ("prop_clr+posture+pose_box", "l2", "mean"),
    ("box_aspect", "l2", "nn"), ("aspect_tilt", "l2", "mean"), ("aspect_tilt", "l2", "nn"), ("posture", "l2", "nn"),
    ("prop_clr+posture+pose_box", "l2", "nn"), ("pose_box", "l2", "nn"), ("prop_clr+posture", "l2", "nn"),
]

if __name__ == "__main__":
    from tools.marker_eval import evaluate
    for name, metric, mode in RUNS:
        evaluate(FEATURES[name], name, metric=metric, mode=mode)
