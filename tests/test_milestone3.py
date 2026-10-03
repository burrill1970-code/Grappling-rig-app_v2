import numpy as np
from grappling.pose import attach_keypoints, SKELETON


def test_detected_athletes_have_17_keypoints(tracked, raw_dets):
    res = tracked[0]
    n = 0
    for r in res:
        for a in r.values():
            if a["status"] == "detected":
                kp = np.array(a["kp"])
                assert kp.shape == (17, 3)
                assert ((kp[:, 2] >= 0) & (kp[:, 2] <= 1)).all()
                n += 1
            else:
                assert a["kp"] is None or a["pose_source"] == "colour_assigned_shared_detection"
                if a["status"] == "lost":
                    assert a["kp"] is None  # no invented poses for lost athletes
    assert n > 250


def test_confident_keypoints_fall_inside_the_athlete_box(tracked, raw_dets):
    res = tracked[0]
    inside = tot = 0
    for r in res:
        for a in r.values():
            if a["status"] != "detected":
                continue
            x1, y1, x2, y2 = a["box"]
            for x, y, c in a["kp"]:
                if c > 0.5:
                    tot += 1
                    inside += (x1 - 15 <= x <= x2 + 15) and (y1 - 15 <= y <= y2 + 15)
    assert tot > 1000 and inside / tot > 0.95


def test_skeleton_indices_valid():
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON)
