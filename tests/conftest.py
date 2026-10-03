import numpy as np
import pytest
from grappling.video import read_frames

CLIP = "test_clips/clip.mp4"
RAW = "outputs/raw_dets.npz"


@pytest.fixture(scope="session")
def frames():
    return list(read_frames(CLIP))


@pytest.fixture(scope="session")
def raw_dets():
    return list(np.load(RAW, allow_pickle=True)["dets"])


@pytest.fixture(scope="session")
def tracked(frames, raw_dets):
    from grappling.video import detect_cuts
    from grappling.tracker import track_video
    cuts = detect_cuts(frames)
    return track_video(frames, raw_dets, cuts), cuts
