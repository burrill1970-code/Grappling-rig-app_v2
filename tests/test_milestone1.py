from grappling.video import detect_cuts, video_info
from grappling.athletes import candidates

MIN_TWO_ATHLETE_FRAC = 0.40  # measured 258/535 = 48%: regression floor, NOT a success claim


def test_video_decodes(frames):
    assert len(frames) == video_info("test_clips/clip.mp4")["n"] == 535
    assert frames[0].shape == (608, 1080, 3)


def test_one_shot_cut_at_frame_12(frames):
    assert detect_cuts(frames) == [12]


def test_detections_cover_every_frame(frames, raw_dets):
    assert len(raw_dets) == len(frames)


def test_referee_and_crowd_are_filtered(frames, raw_dets):
    # frame 400: referee (box x 120-257) must not be an athlete candidate
    c = candidates(frames[400], raw_dets[400])
    assert all(x["box"][0] > 300 for x in c)
    # frame 40: crowd false positives must be gone, 2 athletes remain
    assert len(candidates(frames[40], raw_dets[40])) == 2


def test_two_athlete_detection_rate_is_known_and_flagged(frames, raw_dets):
    n2 = [len(candidates(f, d)) >= 2 for f, d in zip(frames, raw_dets)]
    frac = sum(n2) / len(n2)
    # The detector merges tangled athletes into one box; those frames are listed, not hidden.
    assert frac >= MIN_TWO_ATHLETE_FRAC
    assert frac < 0.95, "detector unexpectedly perfect: re-check the filter"
