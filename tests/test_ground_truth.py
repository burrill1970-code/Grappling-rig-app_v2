"""Ground truth: the scoring rules always, and each clip's hand labels once they exist.

There are no pass/fail floors on the scores yet: they are the first real measurement of the tracker, and a floor
set before seeing them would be a guess. Once the labels exist, pin floors here from the measured numbers.
"""
import json
from pathlib import Path

import pytest

from tools.eval_ground_truth import evaluate, iou, verdict, wilson
from tools.gt_label import CLIPS, task_frames

A_BOX, B_BOX = [0, 0, 100, 200], [150, 0, 250, 200]


def test_verdict_rules():
    assert verdict([0, 0, 100, 200], A_BOX, B_BOX) == "correct"
    assert verdict([0, 0, 40, 100], A_BOX, B_BOX) == "partial"
    assert verdict([150, 0, 250, 200], A_BOX, B_BOX) == "wrong_body"
    assert verdict([400, 0, 500, 100], A_BOX, B_BOX) == "off_target"
    assert verdict(None, A_BOX, B_BOX) == "lost_but_visible"
    assert verdict(None, None, B_BOX) == "hidden_ok"
    assert verdict([0, 0, 10, 10], None, B_BOX) == "box_on_hidden"
    # a box straddling both athletes about equally: IoU cannot decide, a person's identity check does
    straddle = [60, 0, 240, 200]
    assert verdict(straddle, A_BOX, B_BOX) == "undecided"
    assert verdict(straddle, A_BOX, B_BOX, "right") == "partial"
    assert verdict(straddle, A_BOX, B_BOX, "wrong") == "wrong_body"
    assert verdict(straddle, A_BOX, B_BOX, "both") == "covers_both"
    # a clear margin needs no check, and an answer does not override it
    assert verdict([0, 0, 120, 200], A_BOX, B_BOX, "wrong") == "correct"


def test_iou_and_wilson():
    assert iou(A_BOX, A_BOX) == 1.0 and iou(A_BOX, B_BOX) == 0.0 and iou(None, A_BOX) == 0.0
    lo, hi = wilson(95, 100)
    assert lo < 0.95 < hi and 0.88 < lo and hi < 0.98
    assert wilson(0, 0) is None


@pytest.mark.parametrize("video,out,targeted", CLIPS, ids=[c[1] for c in CLIPS])
def test_labels(video, out, targeted):
    gt = Path(out) / "ground_truth.json"
    if not gt.exists():
        pytest.skip(f"{gt} not present: no ground truth for this clip (reported as skipped, not passed)")
    meta = json.load(open(Path(out) / "tracks.json"))["meta"]
    w, h = 1080, meta["crop_y"][1] - meta["crop_y"][0]
    labels = json.load(open(gt))["frames"]
    expected = {f for f, _ in task_frames(meta["frames"], targeted)}
    missing = sorted(expected - {int(k) for k in labels})
    assert not missing, f"labelling not finished; unlabelled frames: {missing}"
    for k, lab in labels.items():
        if lab["unusable"]:
            continue
        for who in "AB":
            box, hidden = lab[who]["box"], lab[who]["hidden"]
            assert (box is None) == hidden, (k, who)
            if box:
                assert 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h, (k, who, box)

    # every box IoU cannot decide has a person's identity answer
    ev = evaluate(out, write=False)
    pending = ev["summary"]["id_checks_pending"]
    assert not pending, f"identity check not finished ({len(pending)} left): run tools/gt_identity.py"

    # the committed scores are the scores of the committed labels, answers and tracks
    committed = Path(out) / "gt_eval.json"
    assert committed.exists(), f"run tools/eval_ground_truth.py --out {out}"
    before = json.loads(committed.read_text())
    assert ev == before, f"{committed} is stale: rerun tools/eval_ground_truth.py --out {out}"


# Floors measured from the first hand labels + identity check (2026-10-12): a change that scores below them on the
# regular sample has made the tracker worse. Mean IoU gets 0.02 of slack for small box changes.
FLOORS = {
    "outputs": {"A": dict(right_person=76, mean_iou=0.4887), "B": dict(right_person=75, mean_iou=0.5223)},
    "outputs/clip2": {"A": dict(right_person=61, mean_iou=0.5837), "B": dict(right_person=63, mean_iou=0.5419)},
}


@pytest.mark.parametrize("out", sorted(FLOORS))
def test_ground_truth_floors(out):
    if not (Path(out) / "ground_truth.json").exists():
        pytest.skip(f"{out}/ground_truth.json not present")
    s = evaluate(out, write=False)["summary"]
    for who, fl in FLOORS[out].items():
        a = s["athletes"][who]
        assert a["right_person"] >= fl["right_person"], (who, a["right_person"], fl)
        assert a["mean_iou_visible"] >= fl["mean_iou"] - 0.02, (who, a["mean_iou_visible"], fl)
