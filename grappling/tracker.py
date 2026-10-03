"""Two-target tracker: A = white gi, B = blue gi.

Per frame: Hungarian assignment of athlete candidates to the two identities.
Primary cost = gi-colour (torso HSV histogram vs the identity's reference + colour-class prior).
Secondary cost = distance to the constant-velocity predicted box.
Motion state is reset at shot cuts; colour references persist (and re-seed per shot).

Statuses per athlete per frame:
  detected  own box, matched by colour+motion
  color_split      one detection covers both athletes; box = that athlete's gi-colour pixels (low conf)
  color_recovered  detector missed this athlete; box = gi-colour pixels near its last position (low conf)
  lost             nothing usable for this identity
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .athletes import candidates, referee_boxes
from .color import bhatta
from .occlusion import locate, grow, region_of, MIN_PX

IDS = ("A", "B")
CLS_OF = {"A": "white", "B": "blue"}
W_MOTION = 0.25
GATE = 0.95
EMA = 0.1


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def containment(inner, outer):
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


class Track:
    def __init__(self, ident):
        self.id, self.box, self.vel, self.ref = ident, None, np.zeros(4), None
        self.last_seen = -1

    def predict(self):
        return None if self.box is None else self.box + self.vel

    def reset_motion(self):
        self.box, self.vel = None, np.zeros(4)

    def update(self, box, frame_i):
        if self.box is not None:
            self.vel = 0.5 * self.vel + 0.5 * (box - self.box)
        self.box, self.last_seen = box.copy(), frame_i

    def update_ref(self, h):
        self.ref = h.copy() if self.ref is None else (1 - EMA) * self.ref + EMA * h


def _cost(cand, tr, use_motion):
    col = 1.0 if tr.ref is None or cand["hist"] is None else bhatta(cand["hist"], tr.ref)
    prior = 0.0
    if cand["cls"] in ("white", "blue"):
        prior = -0.25 if cand["cls"] == CLS_OF[tr.id] else 0.4
    mot = 0.0
    p = tr.predict()
    if use_motion and p is not None:
        cb = cand["box"]
        diag = np.hypot(cb[2] - cb[0], cb[3] - cb[1]) + 1e-6
        mot = min(1.0, np.hypot((cb[0] + cb[2]) / 2 - (p[0] + p[2]) / 2,
                                (cb[1] + cb[3]) / 2 - (p[1] + p[3]) / 2) / diag)
    return col + prior + W_MOTION * mot


def _patch_mask(mask, boxes, min_cov=0.15):
    """OR a candidate's box into the person mask where the mask covers < min_cov of it."""
    out = mask
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        area = max((x2 - x1) * (y2 - y1), 1)
        if mask[y1:y2, x1:x2].sum() / area < min_cov:
            if out is mask:
                out = mask.copy()
            out[y1:y2, x1:x2] = True
    return out


def _without_boxes(mask, boxes):
    if mask is None or not boxes:
        return mask
    m = mask.copy()
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = False
    return m


def track_video(frames, raw_dets, cuts=(), person_masks=None):
    tracks = {i: Track(i) for i in IDS}
    cutset = set(cuts)
    results = []
    for fi, (frame, dets) in enumerate(zip(frames, raw_dets)):
        new_shot = fi == 0 or fi in cutset
        if new_shot:
            for t in tracks.values():
                t.reset_motion()
        pmask = None if person_masks is None else person_masks[fi]
        cands = candidates(frame, dets, pmask)
        if pmask is not None:
            patched = _patch_mask(pmask, [c["box"] for c in cands])  # segmenter missed an athlete: use the detector box
            if patched is not pmask:
                pmask = patched
                cands = candidates(frame, dets, pmask)
        refs = referee_boxes(frame, dets)
        # remove only the referee's upper body (shirt/tie carry blue and white); legs may overlap a lying athlete
        pm_nr = _without_boxes(pmask, [(x1, y1, x2, y1 + 0.6 * (y2 - y1)) for x1, y1, x2, y2 in refs])
        rec = {i: dict(status="lost", box=None, conf=0.0, cand=None) for i in IDS}

        # seed / re-seed colour references from rule classes when a clean pair is visible
        whites = [c for c in cands if c["cls"] == "white"]
        blues = [c for c in cands if c["cls"] == "blue"]
        if whites and blues and (new_shot or any(t.ref is None for t in tracks.values())):
            w = max(whites, key=lambda c: c["conf"]); b = max(blues, key=lambda c: c["conf"])
            if iou(w["box"], b["box"]) < 0.3:
                tracks["A"].ref, tracks["B"].ref = w["hist"].copy(), b["hist"].copy()

        # merged: a single candidate swallows both athletes' last known boxes
        merged = None
        if tracks["A"].box is not None and tracks["B"].box is not None:
            for c in sorted(cands, key=lambda c: -c["conf"]):
                standalone = [o for o in cands if o is not c and containment(o["box"], c["box"]) < 0.6
                              and iou(o["box"], c["box"]) < 0.5]
                if (not standalone and (
                        (containment(tracks["A"].box, c["box"]) > 0.6 and containment(tracks["B"].box, c["box"]) > 0.6) or
                        (c["cls"] == "mixed" and (containment(tracks["A"].box, c["box"]) > 0.3 or
                                                  containment(tracks["B"].box, c["box"]) > 0.3)))):
                    merged = c
                    break
        # single candidate when we were not overlapping before: still ambiguous if it is
        # much wider than either athlete was (two bodies in one box)
        if merged is None and len(cands) == 1 and tracks["A"].box is not None and tracks["B"].box is not None:
            c = cands[0]
            wA = tracks["A"].box[2] - tracks["A"].box[0]; wB = tracks["B"].box[2] - tracks["B"].box[0]
            if (c["box"][2] - c["box"][0]) > 1.3 * max(wA, wB) and \
                    containment(tracks["A"].box, c["box"]) > 0.3 and containment(tracks["B"].box, c["box"]) > 0.3:
                merged = c
        if merged is not None:
            for i in IDS:
                box, px = locate(frame, pm_nr, i, region_of(merged["box"], frame.shape))
                if box is None:
                    rec[i] = dict(status="lost", box=None, conf=0.0, cand=merged["idx"], note="occluded in merged box")
                    continue
                rec[i] = dict(status="color_split", box=[float(v) for v in box],
                              conf=0.4 * min(1.0, px / 3000), cand=merged["idx"], px=px)
                tracks[i].update(np.array(box, float), fi)
        elif cands:
            use_motion = not new_shot
            C = np.array([[_cost(c, tracks[i], use_motion) for c in cands] for i in IDS])
            for r, k in zip(*linear_sum_assignment(C)):
                if C[r, k] > GATE:
                    continue
                i, c = IDS[r], cands[k]
                rec[i] = dict(status="detected", box=c["box"].tolist(), conf=float(c["conf"]), cand=c["idx"])
                tracks[i].update(c["box"], fi)
                if c["cls"] == CLS_OF[i] and c["hist"] is not None and len(cands) >= 2 and \
                        all(iou(c["box"], o["box"]) < 0.1 for o in cands if o is not c):
                    tracks[i].update_ref(c["hist"])
        if merged is None and person_masks is not None:
            for i in IDS:
                t = tracks[i]
                if rec[i]["status"] == "lost" and t.box is not None and fi - t.last_seen <= 30:
                    # search near the last box and inside any candidate box that overlaps it
                    regions = [grow(t.box, frame.shape, 1.0)]
                    regions += [grow(c["box"], frame.shape, 0.1) for c in cands if iou(c["box"], t.box) > 0.05
                                or containment(t.box, c["box"]) > 0.3]
                    region = (min(r[0] for r in regions), min(r[1] for r in regions),
                              max(r[2] for r in regions), max(r[3] for r in regions))
                    box, px = locate(frame, pm_nr, i, region)
                    note = None
                    if box is None:
                        # stale/tiny last box: widen to its surroundings plus the other athlete's neighbourhood
                        other = rec["B" if i == "A" else "A"]
                        wide = [grow(t.box, frame.shape, 2.0)]
                        if other["box"] is not None:
                            wide.append(grow(np.array(other["box"]), frame.shape, 1.0))
                        region = (min(r[0] for r in wide), min(r[1] for r in wide),
                                  max(r[2] for r in wide), max(r[3] for r in wide))
                        box, px = locate(frame, pm_nr, i, region)
                        note = "widened search" if box is not None else None
                    if box is None:
                        # last resort inside the person mask: the whole mat (still needs >= 1.5x the usual pixels)
                        box, px = locate(frame, pm_nr, i, (0, int(0.3 * frame.shape[0]), frame.shape[1], frame.shape[0]),
                                         min_px=int(1.5 * MIN_PX))
                        note = "mat-wide search" if box is not None else None
                    if box is None:
                        # person mask may be missing this athlete entirely: retry on the mat without it
                        box, px = locate(frame, None, i, (region[0], max(region[1], int(0.5 * frame.shape[0])),
                                                           region[2], region[3]), min_px=1000, exclude=[(x1, y1, x2, y1 + 0.6 * (y2 - y1)) for x1, y1, x2, y2 in refs])
                        note = "no person mask: mat-only colour search"
                    if box is not None:
                        rec[i] = dict(status="color_recovered", box=[float(v) for v in box],
                                      conf=(0.4 if note is None else 0.3 if note == "widened search" else 0.25 if note == "mat-wide search" else 0.2) * min(1.0, px / 3000), cand=None, px=px, note=note)
                        t.update(np.array(box, float), fi)
        results.append(rec)
    return results

