import numpy as np
from grappling.tracker import iou, IDS


def _runs(res, ident, status):
    runs, s = [], None
    for i, r in enumerate(res + [dict.fromkeys(IDS, dict(status="x"))]):
        if r[ident]["status"] == status and s is None:
            s = i
        if r[ident]["status"] != status and s is not None:
            runs.append((s, i - 1))
            s = None
    return runs


def test_occlusion_frames_are_labelled_not_hidden(tracked):
    res, _ = tracked
    counts = {st: sum(r[i]["status"] == st for r in res for i in IDS)
              for st in ("detected", "color_split", "color_recovered", "lost")}
    assert counts["color_split"] > 300, counts  # the detector merges the pair for most of the ground phase
    assert sum(counts.values()) == 2 * len(res)


def test_colour_boxes_sit_on_the_right_gi_colour(tracked, frames, person_masks):
    """A colour-derived box must contain mainly that athlete's gi colour among person pixels."""
    import cv2
    from grappling.occlusion import colour_masks
    res, _ = tracked
    bad = tot = 0
    for fi in range(0, len(res), 5):
        for i in IDS:
            r = res[fi][i]
            if r["status"] not in ("color_split", "color_recovered"):
                continue
            x1, y1, x2, y2 = [int(v) for v in r["box"]]
            hsv = cv2.cvtColor(frames[fi][y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            w, b = colour_masks(hsv)
            pm = person_masks[fi][y1:y2, x1:x2]
            nw, nb = int((w.astype(bool) & pm).sum()), int((b.astype(bool) & pm).sum())
            own, other = (nw, nb) if i == "A" else (nb, nw)
            tot += 1
            bad += own < other
    assert tot > 40 and bad / tot < 0.10, (bad, tot)


def test_identity_recovers_by_colour_after_occlusion(tracked, frames, raw_dets):
    """Frames where an athlete is detected again right after a colour-derived/lost stretch: the detection's
    keypoint torso colour must match the ID (A white / B blue). Only 76 frames in this clip are 'clear'
    (non-overlapping), so this is the check that covers re-identification during contact."""
    from grappling.color import torso_hsv, color_fractions
    res, _ = tracked
    want = {"A": "white", "B": "blue"}
    checked = wrong = 0
    for fi in range(1, len(res)):
        for i in IDS:
            if res[fi][i]["status"] == "detected" and res[fi - 1][i]["status"] != "detected":
                fr = color_fractions(torso_hsv(frames[fi], raw_dets[fi][res[fi][i]["cand"]]))
                other = "blue" if i == "A" else "white"
                checked += 1
                wrong += fr[want[i]] < fr[other] - 0.05
    assert checked >= 10
    assert wrong / checked <= 0.10, (wrong, checked)


def test_pose_only_when_owned(tracked):
    res, _ = tracked
    for r in res:
        for a in r.values():
            if a["kp"] is not None:
                assert a["pose_source"] in ("own_detection", "colour_assigned_shared_detection")
                assert a["status"] != "lost"
