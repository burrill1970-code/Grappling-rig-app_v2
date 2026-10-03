import json
import re
from pathlib import Path

import cv2
import pytest

from grappling.report import N_MAX_GAP, runs


def test_outputs_exist_and_video_decodes():
    c = cv2.VideoCapture("outputs/annotated.mp4")
    n = 0
    while c.read()[0]:
        n += 1
    assert n == 535
    assert Path("report.md").exists() and Path("outputs/tracks.json").exists()


def test_json_has_both_athletes_every_frame():
    d = json.load(open("outputs/tracks.json"))
    assert len(d["frames"]) == 535
    for f in d["frames"]:
        assert set(f["athletes"]) == {"A", "B"}
        for a in f["athletes"].values():
            assert a["status"] in ("detected", "color_split", "color_recovered", "lost")
            assert 0 <= a["confidence"] <= 1
            if a["status"] == "lost":
                assert a["box"] is None and a["keypoints"] is None
            if a["keypoints"] is not None:
                assert len(a["keypoints"]) == 17


def test_json_matches_fresh_tracking(tracked):
    res, _ = tracked
    d = json.load(open("outputs/tracks.json"))
    for fi in range(0, 535, 7):
        for i in "AB":
            assert d["frames"][fi]["athletes"][i]["status"] == res[fi][i]["status"]


def test_every_long_gap_is_reported(tracked):
    """No track missing for more than N frames without being flagged in report.md."""
    res, _ = tracked
    text = Path("report.md").read_text()
    for i in "AB":
        for a, b in runs([r[i]["status"] == "lost" for r in res]):
            if b - a + 1 > N_MAX_GAP:
                assert (f"{a}-{b}" in text), f"{i} lost {a}-{b} not in report"
    # and every lost run (any length) appears in the report's lost list
    for i in "AB":
        for a, b in runs([r[i]["status"] == "lost" for r in res]):
            assert (f"{a}-{b}" in text) or re.search(rf"(?<![\d-]){a}(?![\d-])", text), (i, a, b)


def test_report_states_failures():
    text = Path("report.md").read_text().lower()
    for needle in ("clear", "colour", "lost", "ground_truth.md", "failures"):
        assert needle in text


def test_ground_truth_if_present():
    gt = Path("ground_truth.md")
    if not gt.exists():
        pytest.skip("ground_truth.md not present: nothing to check (reported as skipped, not passed)")
    pytest.fail("ground_truth.md exists but its format is undefined; parser not written")
