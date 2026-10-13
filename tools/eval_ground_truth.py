"""Score a clip's tracks against hand-labelled ground truth (made with tools/gt_label.py).

    python tools/eval_ground_truth.py --out outputs/clip2

Reads <out>/ground_truth.json, <out>/tracks.json and, if present, <out>/identity_check.json; writes
<out>/gt_eval.json. Each athlete on each labelled frame gets one verdict, using IoU of the tracker's box with that
athlete's labelled box ("own") and with the other athlete's labelled box ("other"):

    correct           own IoU >= 0.5
    partial           0 < own IoU < 0.5 and own IoU >= other IoU (right person, poor box)
    wrong_body        other IoU > own IoU (box sits more on the other athlete)
    off_target        box overlaps neither labelled athlete
    covers_both       (identity check only) the box covers both athletes
    undecided         identity needs a check (see below) and none has been answered
    lost_but_visible  labelled visible, tracker has no box
    hidden_ok         labelled hidden, tracker has no box
    box_on_hidden     labelled hidden, tracker has a box

Labelled boxes cover the whole visible body, colour-derived tracker boxes cover gi pixels only, so colour boxes
score lower IoU even when on the right person; `partial` is the expected verdict for many of them.
Identity check: when the athletes are tangled their labelled boxes overlap heavily, and a track box then overlaps
both about equally, so IoU cannot say which athlete it is on (the wrong-body count swung from 12 to 1 on clip 2 as
an arbitrary margin went from 0 to 0.15). Wherever own IoU minus other IoU is below ID_MARGIN and the box touches
both, the identity comes from a person instead: tools/gt_identity.py shows the track box and records right / wrong /
both in <out>/identity_check.json. "right" keeps the IoU box-quality verdict (correct or partial).

Rates are quoted only for the "regular" frames (every 7th frame, an unbiased sample), with Wilson 95% intervals.
"Targeted" frames were chosen because they failed before, so they are listed, not turned into rates.
"""
import argparse
import json
import math
from pathlib import Path

IOU_OK = 0.5
ID_MARGIN = 0.15
VERDICTS = ("correct", "partial", "wrong_body", "off_target", "covers_both", "undecided", "lost_but_visible", "hidden_ok",
            "box_on_hidden")
ID_ANSWERS = {"right", "wrong", "both"}


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def needs_id_check(io, ix):
    """IoU alone cannot tell which athlete the box is on: it touches both and favours its own by less than ID_MARGIN."""
    return io > 0 and ix > 0 and io - ix < ID_MARGIN


def verdict(track_box, own, other, answer=None):
    """own/other: labelled box or None if that athlete is labelled hidden. answer: the identity check, if any."""
    if own is None:
        return "hidden_ok" if track_box is None else "box_on_hidden"
    if track_box is None:
        return "lost_but_visible"
    io, ix = iou(track_box, own), iou(track_box, other)
    if needs_id_check(io, ix):
        if answer is None:
            return "undecided"
        if answer == "wrong":
            return "wrong_body"
        if answer == "both":
            return "covers_both"
        return "correct" if io >= IOU_OK else "partial"
    if io >= IOU_OK:
        return "correct"
    if ix > io:
        return "wrong_body"
    if io > 0:
        return "partial"
    return "off_target"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def status_group(status):
    return "detected" if status == "detected" else ("lost" if status == "lost" else "colour")


def evaluate(out, write=True):
    out = Path(out)
    gt = json.load(open(out / "ground_truth.json"))
    frames = json.load(open(out / "tracks.json"))["frames"]
    idp = out / "identity_check.json"
    answers = json.load(open(idp))["frames"] if idp.exists() else {}
    rows = []
    for key, lab in sorted(gt["frames"].items(), key=lambda kv: int(kv[0])):
        f = int(key)
        if lab.get("unusable"):
            rows.append(dict(frame=f, kind=lab.get("kind", "regular"), unusable=True))
            continue
        trk = frames[f]["athletes"]
        row = dict(frame=f, kind=lab.get("kind", "regular"), unusable=False)
        for who, oth in (("A", "B"), ("B", "A")):
            own, other = lab[who]["box"], lab[oth]["box"]
            tb = trk[who]["box"]
            ans = answers.get(key, {}).get(who)
            row[who] = dict(verdict=verdict(tb, own, other, ans), iou=round(iou(tb, own), 4),
                            iou_other=round(iou(tb, other), 4), status=trk[who]["status"],
                            unsure=bool(lab[who].get("unsure")), id_checked=ans)
        # both boxes sitting more on the other athlete than on their own = an identity swap
        row["swap"] = row["A"]["verdict"] == "wrong_body" and row["B"]["verdict"] == "wrong_body"
        rows.append(row)

    usable = [r for r in rows if not r["unusable"]]
    regular = [r for r in usable if r["kind"] == "regular"]
    summary = {"labelled_frames": len(rows), "unusable_frames": [r["frame"] for r in rows if r["unusable"]],
               "regular_frames": len(regular), "targeted_frames": len(usable) - len(regular), "athletes": {}}
    for who in "AB":
        reg = [r[who] for r in regular]
        counts = {v: sum(x["verdict"] == v for x in reg) for v in VERDICTS}
        visible = [x for x in reg if x["verdict"] not in ("hidden_ok", "box_on_hidden")]
        decided = [x for x in visible if x["verdict"] != "undecided"]   # rates use these; undecided is counted
        right = sum(x["verdict"] in ("correct", "partial") for x in decided)
        wrong = sum(x["verdict"] == "wrong_body" for x in decided)
        by_status = {}
        for g in ("detected", "colour"):
            xs = [x for x in decided if status_group(x["status"]) == g]
            by_status[g] = dict(n=len(xs), correct=sum(x["verdict"] == "correct" for x in xs),
                                right_person=sum(x["verdict"] in ("correct", "partial") for x in xs),
                                mean_iou=round(sum(x["iou"] for x in xs) / len(xs), 4) if xs else None)
        summary["athletes"][who] = dict(
            counts=counts, visible=len(visible), decided=len(decided),
            right_person=right, right_person_rate=round(right / len(decided), 4) if decided else None,
            right_person_ci95=wilson(right, len(decided)),
            correct_rate=round(counts["correct"] / len(decided), 4) if decided else None,
            correct_ci95=wilson(counts["correct"], len(decided)),
            wrong_body_rate=round(wrong / len(decided), 4) if decided else None,
            wrong_body_ci95=wilson(wrong, len(decided)),
            covers_both_rate=round(counts["covers_both"] / len(decided), 4) if decided else None,
            id_checked=sum(x["id_checked"] is not None for x in reg),
            mean_iou_visible=round(sum(x["iou"] for x in visible) / len(visible), 4) if visible else None,
            by_status=by_status,
            unsure=sum(x["unsure"] for x in reg))
    summary["swap_frames"] = [r["frame"] for r in usable if r["swap"]]
    summary["errors"] = {who: [dict(frame=r["frame"], kind=r["kind"], **r[who]) for r in usable
                               if r[who]["verdict"] in ("wrong_body", "covers_both", "off_target", "lost_but_visible", "box_on_hidden")]
                         for who in "AB"}
    summary["id_checks_pending"] = [dict(frame=r["frame"], athlete=w) for r in usable for w in "AB"
                                    if r[w]["verdict"] == "undecided"]
    summary["targeted"] = [dict(frame=r["frame"], A=r["A"]["verdict"], B=r["B"]["verdict"])
                           for r in usable if r["kind"] == "targeted"]
    result = {"summary": summary, "per_frame": rows}
    if write:
        (out / "gt_eval.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    out = ap.parse_args().out
    if not (Path(out) / "ground_truth.json").exists():
        print(f"{out}/ground_truth.json not found: nothing to score (label it with tools/gt_label.py)")
        raise SystemExit(0)
    s = evaluate(out)["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "errors"}, indent=1))
