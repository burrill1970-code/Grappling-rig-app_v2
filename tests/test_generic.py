"""Clip-independent checks on a tracking output directory (tracks.json + annotated.mp4).

Run for every clip that has outputs. These use only the delivered JSON/video, so they work for any clip.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

CASES = [("test_clips/clip.mp4", "outputs"), ("test_clips/clip2.mp4", "outputs/clip2")]
CASES = [c for c in CASES if (Path(c[1]) / "tracks.json").exists()]
STATUSES = {"detected", "color_split", "color_recovered", "lost"}


def _frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


@pytest.fixture(scope="module", params=CASES, ids=[c[1] for c in CASES])
def case(request):
    clip, out = request.param
    return clip, out, json.load(open(Path(out) / "tracks.json"))


def test_two_tracks_every_frame_and_frame_counts_match(case):
    clip, out, d = case
    n = _frames(clip)
    assert len(d["frames"]) == n == d["meta"]["frames"]
    assert _frames(str(Path(out) / "annotated.mp4")) == n
    for f in d["frames"]:
        assert set(f["athletes"]) == {"A", "B"}


def test_lost_is_always_flagged_and_json_is_consistent(case):
    _, _, d = case
    for f in d["frames"]:
        for i, a in f["athletes"].items():
            assert a["status"] in STATUSES, (f["frame"], i)
            assert 0 <= a["confidence"] <= 1
            if a["status"] == "lost":
                assert a["box"] is None and a["keypoints"] is None and a["confidence"] == 0
            else:
                assert a["box"] is not None
            if a["keypoints"] is not None:
                assert len(a["keypoints"]) == 17
                assert a["status"] != "lost"


def test_boxes_are_on_the_mat_not_the_crowd(case):
    _, _, d = case
    h = d["meta"]["crop_y"][1] - d["meta"]["crop_y"][0]
    for f in d["frames"]:
        for i, a in f["athletes"].items():
            if a["box"] is None:
                continue
            assert a["box"][3] >= 0.5 * h, f"frame {f['frame']} {i}: box bottom above the mat"
            if a["status"] in ("color_split", "color_recovered"):
                assert a["box"][1] >= 0.25 * h, f"frame {f['frame']} {i}: colour box reaches the crowd"


def test_poses_sit_on_their_own_athlete(case):
    _, _, d = case
    for f in d["frames"]:
        for i, a in f["athletes"].items():
            if a["keypoints"] is None:
                continue
            kp = np.array([[p["x"], p["y"], p["conf"]] for p in a["keypoints"]])
            conf = kp[:, 2] > 0.3
            assert conf.sum() >= 1
            x1, y1, x2, y2 = a["box"]
            w, h = x2 - x1, y2 - y1
            cx, cy = kp[conf, 0].mean(), kp[conf, 1].mean()
            assert x1 - 0.25 * w <= cx <= x2 + 0.25 * w and y1 - 0.25 * h <= cy <= y2 + 0.25 * h, \
                f"frame {f['frame']} {i}: skeleton is away from the athlete's box"


def test_report_flags_every_lost_run_longer_than_5(case):
    clip, out, d = case
    rep = Path("report.md") if out == "outputs" else Path(out) / "report.md"
    if not rep.exists():
        pytest.skip("report not generated yet")
    text = rep.read_text()
    for i in "AB":
        run = None
        for f in d["frames"] + [dict(frame=-1, athletes={"A": dict(status="x"), "B": dict(status="x")})]:
            lost = f["athletes"][i]["status"] == "lost"
            if lost and run is None:
                run = [f["frame"], f["frame"]]
            elif lost:
                run[1] = f["frame"]
            elif run is not None:
                if run[1] - run[0] + 1 > 5:
                    assert f"{run[0]}-{run[1]}" in text, (i, run)
                run = None
