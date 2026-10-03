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


def analyse(results, frames, raw_dets, cuts, person_masks=None):
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
    # measured quality numbers quoted in report.md (also regression floors in the tests)
    from .athletes import candidates
    out["two_athlete_frac"] = float(np.mean([len(candidates(frames[f], raw_dets[f])) >= 2 for f in range(n)]))
    if person_masks is not None:
        import cv2
        from .occlusion import colour_masks, SHADOW_WHITE_V
        bad = tot = 0
        for f in range(0, n, 5):
            for i in IDS:
                r = results[f][i]
                if r["status"] not in ("color_split", "color_recovered"):
                    continue
                x1, y1, x2, y2 = [int(v) for v in r["box"]]
                w, b = colour_masks(cv2.cvtColor(frames[f][y1:y2, x1:x2], cv2.COLOR_BGR2HSV), white_v=SHADOW_WHITE_V)
                pm = person_masks[f][y1:y2, x1:x2]
                nw, nb = int((w.astype(bool) & pm).sum()), int((b.astype(bool) & pm).sum())
                own, other = (nw, nb) if i == "A" else (nb, nw)
                tot += 1
                bad += own < other
        out["colour_box_bad"], out["colour_box_total"] = bad, tot
    return out
