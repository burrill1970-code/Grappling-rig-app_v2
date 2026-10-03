import numpy as np
from grappling.tracker import iou, IDS

MAX_UNFLAGGED_GAP = 0  # a missing track must always carry status 'lost' (never silently absent)


def _clear(rec):
    a, b = rec["A"], rec["B"]
    return a["status"] == b["status"] == "detected" and iou(a["box"], b["box"]) < 0.05


def test_exactly_two_tracks_every_frame(tracked):
    res, _ = tracked
    assert len(res) == 535
    assert all(set(r) == set(IDS) for r in res)


def test_every_missing_track_is_flagged(tracked):
    res, _ = tracked
    for fi, r in enumerate(res):
        for i in IDS:
            if r[i]["box"] is None:
                assert r[i]["status"] == "lost", (fi, i)
            else:
                assert r[i]["status"] in ("detected", "merged"), (fi, i)


def test_no_swaps_in_clear_segments(tracked, frames, raw_dets):
    """Clear = both athletes detected and non-overlapping. Within a shot, consecutive clear frames
    must keep each ID on the same body: small centre jump AND torso colour matching the ID."""
    from grappling.color import torso_hsv, color_fractions
    res, cuts = tracked
    shot = np.searchsorted(cuts, np.arange(len(res)), side="right")
    clear = [fi for fi, r in enumerate(res) if _clear(r)]
    assert len(clear) >= 60, "too few clear frames to test anything"
    prev = None
    for fi in clear:
        for i, want in (("A", "white"), ("B", "blue")):
            det = raw_dets[fi][res[fi][i]["cand"]]  # real detection, keypoint-based torso
            fr = color_fractions(torso_hsv(frames[fi], det))
            other = "blue" if want == "white" else "white"
            assert fr[want] >= fr[other] - 0.05, f"frame {fi}: {i} torso looks {other}: {fr}"
        if prev is not None and prev[0] == fi - 1 and shot[prev[0]] == shot[fi]:
            for i in IDS:
                c0 = np.array(res[fi - 1][i]["box"]).reshape(2, 2).mean(0)
                c1 = np.array(res[fi][i]["box"]).reshape(2, 2).mean(0)
                w = res[fi][i]["box"][2] - res[fi][i]["box"][0]
                assert np.linalg.norm(c1 - c0) < 0.6 * w, f"frame {fi}: {i} jumped (swap?)"
        prev = (fi, None)


def test_shot_cut_resets_do_not_break_ids(tracked):
    res, cuts = tracked
    assert cuts == [12]
    assert res[12]["A"]["status"] != "lost" or res[13]["A"]["status"] != "lost"
