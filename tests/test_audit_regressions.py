"""Regression tests for errors found by the vision audit (outputs/audit_results.json) and confirmed by two
independent verifiers. They pin specific failures so they cannot silently come back."""
import numpy as np
from grappling.athletes import referee_boxes
from grappling.tracker import containment

H = 608


def _boxes(res, kind=None):
    for fi, r in enumerate(res):
        for i, a in r.items():
            if a["box"] is not None and (kind is None or a["status"] in kind):
                yield fi, i, np.array(a["box"])


def test_no_athlete_box_is_in_the_crowd_or_scoreboard(tracked):
    """Audit: B colour-recovered box sat at y~210-248 near the scoreboard (frames 479, 481)."""
    res, _ = tracked
    for fi, i, b in _boxes(res):
        assert b[3] >= 0.5 * H, f"frame {fi} {i}: box bottom {b[3]:.0f} is above the mat"
    for fi, i, b in _boxes(res, ("color_split", "color_recovered")):
        assert b[1] >= 0.25 * H, f"frame {fi} {i}: colour box top {b[1]:.0f} reaches the crowd/scoreboard"


def test_no_athlete_label_on_the_referee(tracked, frames, raw_dets):
    """Audit: B colour-recovered box on the black-suited referee (479, 481)."""
    res, _ = tracked
    for fi, i, b in _boxes(res):
        for x1, y1, x2, y2 in referee_boxes(frames[fi], raw_dets[fi]):
            upper = np.array([x1, y1, x2, y1 + 0.6 * (y2 - y1)])  # shirt/tie; his legs may overlap a lying athlete
            assert containment(b, upper) < 0.5, f"frame {fi}: {i} box sits on the referee's upper body"


def test_poses_are_never_drawn_on_the_crowd(tracked):
    """Audit: stray red (A) skeleton on a spectator in the crowd, frames 456-525. A skeleton must sit on its
    own athlete: confident keypoints centred inside the athlete's box (grown 25%)."""
    res, _ = tracked
    for fi, r in enumerate(res):
        for i, a in r.items():
            if a["kp"] is None:
                continue
            kp = np.array(a["kp"])
            conf = kp[:, 2] > 0.3
            assert conf.sum() >= 1
            x1, y1, x2, y2 = a["box"]
            w, h = x2 - x1, y2 - y1
            cx, cy = kp[conf, 0].mean(), kp[conf, 1].mean()
            assert x1 - 0.25 * w <= cx <= x2 + 0.25 * w and y1 - 0.25 * h <= cy <= y2 + 0.25 * h, \
                f"frame {fi} {i}: skeleton centre ({cx:.0f},{cy:.0f}) is away from the athlete's box"


def test_audit_swap_frames_311_313(tracked):
    """Audit: B's solid box + skeleton were on the white athlete's legs at 311-313 (white athlete lies to the
    right of the blue one). A must now be the right-hand athlete and B must not take the white athlete's box."""
    res, _ = tracked
    for f in (311, 312, 313):
        a, b = res[f]["A"], res[f]["B"]
        assert a["box"] is not None and b["box"] is not None
        ca, cb = (a["box"][0] + a["box"][2]) / 2, (b["box"][0] + b["box"][2]) / 2
        assert ca > cb, f"frame {f}: A (white) should be right of B (blue)"


def test_audit_lost_but_visible_frames_recovered(tracked):
    """Audit: both athletes marked lost while clearly visible at 142-143 and 464; A lost at 309-313."""
    res, _ = tracked
    for f, who in ((142, "B"), (143, "B"), (143, "A"), (464, "A"), (464, "B"), (309, "A"), (310, "A"), (311, "A"),
                   (312, "A"), (313, "A")):
        assert res[f][who]["box"] is not None, f"frame {f}: {who} lost"
