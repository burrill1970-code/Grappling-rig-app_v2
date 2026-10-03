"""Failure analysis computed from the tracking output (nothing here is hand-entered)."""
import numpy as np
from .tracker import IDS, iou

N_MAX_GAP = 5          # a lost run longer than this is a headline failure
LOW_CONF = 0.5


def runs(mask):
    out, s = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def analyse(results, frames, raw_dets, cuts):
    from .color import torso_hsv, color_fractions
    n = len(results)
    st = {i: np.array([r[i]["status"] for r in results]) for i in IDS}
    out = dict(n=n, cuts=list(cuts))
    out["lost"] = {i: runs(st[i] == "lost") for i in IDS}
    out["lost_long"] = {i: [r for r in out["lost"][i] if r[1] - r[0] + 1 > N_MAX_GAP] for i in IDS}
    out["colour_only"] = {i: runs(np.isin(st[i], ["color_split", "color_recovered"])) for i in IDS}
    out["low_conf"] = {i: runs(np.array([(r[i]["status"] == "detected" and r[i]["conf"] < LOW_CONF) for r in results]))
                       for i in IDS}
    out["counts"] = {i: {s: int((st[i] == s).sum()) for s in ("detected", "color_split", "color_recovered", "lost")}
                     for i in IDS}
    out["pose"] = {i: int(sum(r[i]["kp"] is not None for r in results)) for i in IDS}
    out["pose_shared"] = {i: int(sum(r[i]["pose_source"] == "colour_assigned_shared_detection" for r in results))
                          for i in IDS}
    both_det = np.array([results[f]["A"]["status"] == results[f]["B"]["status"] == "detected" for f in range(n)])
    clear = np.array([bool(both_det[f] and iou(results[f]["A"]["box"], results[f]["B"]["box"]) < 0.05)
                      for f in range(n)])
    out["clear"] = runs(clear)
    out["clear_n"] = int(clear.sum())
    out["contact"] = runs(~clear)
    # identity risk = identity rests on colour/overlap, not on separated detections
    out["id_risk"] = runs(~clear)
    # independent colour check on every directly-detected athlete (keypoint torso)
    want = {"A": "white", "B": "blue"}
    conflict = {i: np.zeros(n, bool) for i in IDS}
    for f in range(n):
        for i in IDS:
            r = results[f][i]
            if r["status"] == "detected":
                fr = color_fractions(torso_hsv(frames[f], raw_dets[f][r["cand"]]))
                other = "blue" if i == "A" else "white"
                conflict[i][f] = fr[want[i]] < fr[other] - 0.05
    out["colour_conflict"] = {i: runs(conflict[i]) for i in IDS}
    out["colour_conflict_n"] = {i: int(conflict[i].sum()) for i in IDS}
    return out
