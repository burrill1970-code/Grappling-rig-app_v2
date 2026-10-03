"""Evaluation harness for appearance markers.

    from tools.marker_eval import load, evaluate
    insts = load()
    evaluate(lambda inst: vector_or_None, "my_marker", metric="cos")

Each instance dict: clip, frame, label in {A,B,R}, box, kp (17x3), conf, crop=(bgr_192xW, personmask_192xW),
segment, contact (overlaps another labelled box). A feature function returns a 1-D float vector (or None if the
marker cannot be computed on that instance; None counts as an abstention and is reported).

Protocols (per clip, nearest-prototype; classes are those present in the clip):
  block  : leave-one-time-block-out. Blocks of 24 frames; prototypes from all instances at least one block away.
  start  : gallery from the first 30% of the clip's frames, test on the rest (how a tracker would use it online).
Reported: balanced accuracy over classes, accuracy on 'contact' instances (bodies overlapping), abstention rate,
and the key pair accuracies A-vs-B and B-vs-R (restricted two-class decisions).
"""
import pickle
import numpy as np

PATH = "outputs/marker_study/dataset.pkl"
BLOCK = 24


def load():
    return pickle.load(open(PATH, "rb"))


def _dist(a, b, metric):
    if metric == "cos":
        return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    if metric == "l1":
        return float(np.abs(a - b).sum())
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric == "chi2":
        return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))
    raise ValueError(metric)


def _protos(train, metric, mode):
    out = {}
    for lab in "ABR":
        v = [t[1] for t in train if t[0] == lab]
        if v:
            out[lab] = np.mean(v, 0) if mode == "mean" else v
    return out


def _classify(x, protos, metric, mode, allowed):
    best = None
    for lab, p in protos.items():
        if lab not in allowed:
            continue
        d = _dist(x, p, metric) if mode == "mean" else min(_dist(x, q, metric) for q in p)
        if best is None or d < best[0]:
            best = (d, lab)
    return None if best is None else best[1]


def evaluate(feature_fn, name="marker", metric="cos", mode="mean", verbose=True):
    insts = load()
    feats = []
    for d in insts:
        try:
            v = feature_fn(d)
        except Exception:
            v = None
        feats.append(None if v is None else np.asarray(v, float).ravel())
    res = {}
    for proto in ("block", "start"):
        rows = []  # (clip, true, pred, contact)
        abstain = 0
        for clip in sorted({d["clip"] for d in insts}):
            idx = [i for i, d in enumerate(insts) if d["clip"] == clip]
            nmax = max(insts[i]["frame"] for i in idx) + 1
            for i in idx:
                d = insts[i]
                if feats[i] is None:
                    abstain += 1
                    continue
                if proto == "block":
                    b = d["frame"] // BLOCK
                    train = [(insts[j]["label"], feats[j]) for j in idx if feats[j] is not None and abs(insts[j]["frame"] // BLOCK - b) > 1]
                else:
                    if d["frame"] < 0.3 * nmax:
                        continue
                    train = [(insts[j]["label"], feats[j]) for j in idx if feats[j] is not None and insts[j]["frame"] < 0.3 * nmax]
                protos = _protos(train, metric, mode)
                if d["label"] not in protos:
                    continue
                present = set(protos)
                pred = _classify(feats[i], protos, metric, mode, present)
                rows.append((clip, d["label"], pred, d["contact"], _classify(feats[i], protos, metric, mode, {"A", "B"} & present) if d["label"] in "AB" else None,
                             _classify(feats[i], protos, metric, mode, {"B", "R"} & present) if d["label"] in "BR" else None))
        def bal(rs, key_true=1, key_pred=2):
            accs = []
            for lab in "ABR":
                x = [r for r in rs if r[key_true] == lab and r[key_pred] is not None]
                if x:
                    accs.append(np.mean([r[key_true] == r[key_pred] for r in x]))
            return float(np.mean(accs)) if accs else float("nan")
        r = dict(n=len(rows), abstain=abstain, balanced_acc=bal(rows),
                 contact_acc=bal([r for r in rows if r[3]]), n_contact=sum(1 for r in rows if r[3]))
        for clip in sorted({x[0] for x in rows}):
            r[f"{clip}_acc"] = bal([x for x in rows if x[0] == clip])
        ab = [x for x in rows if x[1] in "AB" and x[4] is not None]
        br = [x for x in rows if x[1] in "BR" and x[5] is not None and x[0] == "clip2"]
        r["A_vs_B"] = float(np.mean([x[1] == x[4] for x in ab])) if ab else float("nan")
        r["B_vs_R_clip2"] = bal([(x[0], x[1], x[5]) for x in br], 1, 2) if br else float("nan")
        res[proto] = r
    if verbose:
        print(f"== {name} (metric={metric}, mode={mode}) ==")
        for proto, r in res.items():
            print(f"  {proto:5s} bal_acc {r['balanced_acc']:.3f} | contact {r['contact_acc']:.3f} (n={r['n_contact']}) | "
                  f"clip1 {r.get('clip1_acc', float('nan')):.3f} clip2 {r.get('clip2_acc', float('nan')):.3f} | "
                  f"A-vs-B {r['A_vs_B']:.3f} B-vs-R(c2) {r['B_vs_R_clip2']:.3f} | n={r['n']} abstain={r['abstain']}")
    return res
