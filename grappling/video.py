"""Video loading, cropping to the actual picture, and scene-cut detection."""
import cv2
import numpy as np

# The clip is a phone screen recording of a YouTube player: the picture is the
# 16:9 band between the player chrome. Measured on frame 200 (see report.md).
CROP_Y0, CROP_Y1 = 346, 954


def read_frames(path, crop=True):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f[CROP_Y0:CROP_Y1] if crop else f
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS), w=int(cap.get(3)), h=int(cap.get(4)),
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def detect_cuts(frames, thresh=0.6):
    """Return frame indices that start a new shot (colour-histogram L1 jump)."""
    cuts, prev = [], None
    for i, f in enumerate(frames):
        h = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
        h /= h.sum()
        if prev is not None and np.abs(h - prev).sum() > thresh:
            cuts.append(i)
        prev = h
    return cuts
