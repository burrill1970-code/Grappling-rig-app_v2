"""Video loading, cropping to the actual picture, and scene-cut detection."""
import cv2
import numpy as np

# The clips are phone screen recordings of a YouTube player: the picture is the 16:9 band between the
# player chrome. For this recorder's layout it sits at rows 346-954 (measured by hand on clip 1).
CROP_Y0, CROP_Y1 = 346, 954
_CROPS = {}


def detect_crop(path, n=24, thresh=3):
    """Rows (y0, y1) of the picture: the longest run of rows whose median brightness over n sampled frames
    is above `thresh` (the player chrome is black). Snaps to the known layout if within 3 px of it."""
    key = str(path)
    if key in _CROPS:
        return _CROPS[key]
    cap = cv2.VideoCapture(key)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for k in np.linspace(5, max(total - 5, 6), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            rows.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float).mean(1))
    cap.release()
    mask = np.median(rows, 0) > thresh
    best, s0 = (0, 0, 0), None
    for i, v in enumerate(list(mask) + [False]):
        if v and s0 is None:
            s0 = i
        if not v and s0 is not None:
            best = max(best, (i - s0, s0, i))
            s0 = None
    y0, y1 = best[1], best[2]
    if abs(y0 - CROP_Y0) <= 3 and abs(y1 - CROP_Y1) <= 3:
        y0, y1 = CROP_Y0, CROP_Y1
    _CROPS[key] = (y0, y1)
    return y0, y1


def video_fps(path):
    """Average fps of the video stream (frames / duration); the recordings are variable-rate."""
    import json, subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_frames,duration", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True).stdout
        st = json.loads(out)["streams"][0]
        return float(st["nb_frames"]) / float(st["duration"])
    except Exception:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps


def read_frames(path, crop=True):
    y0, y1 = detect_crop(path) if crop else (0, None)
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f[y0:y1] if crop else f
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
