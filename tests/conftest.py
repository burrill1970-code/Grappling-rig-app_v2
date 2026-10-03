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
def person_masks():
    from grappling.segment import run_segmenter
    return run_segmenter(None, cache="outputs/person_masks.npz")


@pytest.fixture(scope="session")
def tracked(frames, raw_dets, person_masks):
    from grappling.video import detect_cuts
    from grappling.tracker import track_video
    from grappling.pose import attach_keypoints
    cuts = detect_cuts(frames)
    return attach_keypoints(track_video(frames, raw_dets, cuts, person_masks), raw_dets, frames), cuts
