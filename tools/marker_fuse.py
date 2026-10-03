"""Distance-level fusion of several markers under the same block / start protocols as tools.marker_eval.

fused_eval([(name, feature_fn, metric, mode, weight), ...])
Each marker's distance to a class prototype is divided by the median test-vs-train distance of that marker in the fold
(so weights are comparable), then summed with the weights.
"""
import numpy as np
from tools.marker_eval import load, _dist, BLOCK


def _feats(fn, insts):
    out = []
    for d in insts:
        try:
            v = fn(d)
        except Exception:
            v = None
        out.append(None if v is None else np.asarray(v, float).ravel())
    return out


def fused_eval(markers, label="fused", verbose=True):
    insts = load()
    F = [_feats(fn, insts) for _, fn, _, _, _ in markers]
    res = {}
    for proto in ("block", "start"):
        rows = []
        for clip in sorted({d["clip"] for d in insts}):
            idx = [i for i, d in enumerate(insts) if d["clip"] == clip]
            nmax = max(insts[i]["frame"] for i in idx) + 1
            for i in idx:
                d = insts[i]
                if proto == "block":
                    b = d["frame"] // BLOCK
                    tr = [j for j in idx if abs(insts[j]["frame"] // BLOCK - b) > 1]
                else:
                    if d["frame"] < 0.3 * nmax:
                        continue
                    tr = [j for j in idx if insts[j]["frame"] < 0.3 * nmax]
                classes = sorted({insts[j]["label"] for j in tr})
                if d["label"] not in classes:
                    continue
                # per-marker distances to each class
                dist = {c: 0.0 for c in classes}
                used = 0
                for (name, fn, metric, mode, w), feats in zip(markers, F):
                    if feats[i] is None:
                        continue
                    per = {}
                    allD = []
                    for c in classes:
                        vs = [feats[j] for j in tr if insts[j]["label"] == c and feats[j] is not None]
                        if not vs:
                            continue
                        if mode == "mean":
                            dd = _dist(feats[i], np.mean(vs, 0), metric)
                            allD.append(dd)
                        else:
                            dd = min(_dist(feats[i], q, metric) for q in vs)
                            allD.append(dd)
                        per[c] = dd
                    if len(per) < len(classes):
                        continue
                    scale = np.median(allD) + 1e-9
                    for c in classes:
                        dist[c] += w * per[c] / scale
                    used += 1
                if used == 0:
                    continue
                pred = min(dist, key=dist.get)
                two = lambda s: min((c for c in classes if c in s), key=dist.get)
                rows.append((clip, d["label"], pred, d["contact"],
                             two({"A", "B"}) if d["label"] in "AB" else None,
                             two({"B", "R"}) if d["label"] in "BR" and "R" in classes else None))
        def bal(rs, kt=1, kp=2):
            accs = [np.mean([r[kt] == r[kp] for r in rs if r[kt] == lab and r[kp] is not None])
                    for lab in "ABR" if any(r[kt] == lab and r[kp] is not None for r in rs)]
            return float(np.mean(accs)) if accs else float("nan")
        r = dict(n=len(rows), bal=bal(rows), contact=bal([x for x in rows if x[3]]),
                 clip1=bal([x for x in rows if x[0] == "clip1"]), clip2=bal([x for x in rows if x[0] == "clip2"]),
                 A_vs_B=float(np.mean([x[1] == x[4] for x in rows if x[1] in "AB" and x[4] is not None])),
                 B_vs_R=bal([(x[0], x[1], x[5]) for x in rows if x[1] in "BR" and x[5] is not None and x[0] == "clip2"], 1, 2))
        res[proto] = r
    if verbose:
        print(f"== {label}")
        for p, r in res.items():
            print(f"  {p:5s} bal {r['bal']:.3f} contact {r['contact']:.3f} clip1 {r['clip1']:.3f} clip2 {r['clip2']:.3f} A-vs-B {r['A_vs_B']:.3f} B-vs-R {r['B_vs_R']:.3f} n={r['n']}")
    return res
