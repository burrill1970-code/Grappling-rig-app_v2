"""Audit-confirmed failures on clip 2 (outputs/clip2/audit_results.json), pinned as regression tests.

History: these eight frames were pinned as STRICT xfail while the bug was open (the dark-suited referee was tracked
as an athlete). After the tracker learned a referee identity (grappling/tracker.py, REF_MARGIN) they started passing,
strict xfail flagged that, and the marker was removed. Frames were also checked by eye and re-audited.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from grappling.color import torso_hsv, color_fractions

OUT = Path("outputs/clip2")
pytestmark = pytest.mark.skipif(not (OUT / "tracks.json").exists(), reason="clip 2 outputs not present")

# (frame, athlete): the audit's two skeptics confirmed the athlete's box+skeleton are on the referee
REFEREE_AS_ATHLETE = [(81, "B"), (312, "B"), (347, "B"), (348, "B"), (349, "B"), (351, "B"), (156, "A"), (363, "A")]


def _iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


@pytest.fixture(scope="module")
def data():
    from grappling.video import read_frames
    frames = list(read_frames("test_clips/clip2.mp4"))
    raw = list(np.load(OUT / "raw_dets.npz", allow_pickle=True)["dets"])
    tracks = json.load(open(OUT / "tracks.json"))["frames"]
    return frames, raw, tracks


@pytest.mark.parametrize("frame,who", REFEREE_AS_ATHLETE)
def test_referee_is_not_tracked_as_an_athlete(data, frame, who):
    """The referee is a standing, narrow (aspect < 0.5), mostly dark-clothed person. An athlete's box must not
    be that detection."""
    frames, raw, tracks = data
    box = tracks[frame]["athletes"][who]["box"]
    for d in raw[frame]:
        if d[4] < 0.5:
            continue
        aspect = (d[2] - d[0]) / (d[3] - d[1])
        dark = color_fractions(torso_hsv(frames[frame], d))["dark"]
        if aspect < 0.5 and dark >= 0.5:
            assert _iou(box, d[:4]) < 0.8, f"frame {frame}: {who} box is the standing dark-suited person"


# --- open failure, pinned as strict xfail -----------------------------------------------------------------------
# B is lost while the blue athlete is clearly visible (audit-confirmed twice, targeted re-check included). In the second
# match he is behind the white athlete and the person segmenter returns only the white athlete's silhouette.
B_LOST_WHILE_VISIBLE = [137, 439, 440, 441, 442, 443, 450, 451, 452]


@pytest.mark.xfail(strict=True, reason="open bug: B lost while visible, hidden athlete not segmented (audit-confirmed)")
@pytest.mark.parametrize("frame", B_LOST_WHILE_VISIBLE)
def test_b_is_not_lost_while_visible(data, frame):
    _, _, tracks = data
    assert tracks[frame]["athletes"]["B"]["box"] is not None
