"""Labelled person crops for the marker study.

Labels come only from sources that are independent of the appearance features being studied:
  A / B : athlete boxes that were DIRECT detections (status 'detected') in frames where the vision audit judged that
          athlete 'correct'. (Audit verdicts exist for ~1/3 of frames per clip.)
  R     : any other confident (>= 0.5), on-mat, >= 40 px wide person that does not overlap (IoU > 0.3) the two verified
          athlete boxes in a frame where BOTH athletes are verified. Labelled by exclusion, using no colour, so colour
          markers are not tested against a colour-defined label. Checked by eye (outputs/marker_study/referee_check.jpg).
Segments = runs of consecutive labelled frames, used for leave-one-segment-out evaluation.
"""
import json
import numpy as np
import cv2
from pathlib import Path

from grappling.video import read_frames
from grappling.segment import run_segmenter
from grappling.segment_x import run_segmenter_x, combine
from grappling.color import torso_hsv, color_fractions

CLIPS = [
    dict(name="clip1", video="test_clips/clip.mp4", out="outputs", audit="outputs/audit3_results.json"),
    dict(name="clip2", video="test_clips/clip2.mp4", out="outputs/clip2", audit="outputs/clip2/audit_results.json"),
]
CROP_H = 192


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def crop(frame, mask, box):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    c, m = frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    s = CROP_H / c.shape[0]
    w = max(8, int(round(c.shape[1] * s)))
    return (cv2.resize(c, (w, CROP_H), interpolation=cv2.INTER_AREA),
            cv2.resize(m.astype(np.uint8), (w, CROP_H), interpolation=cv2.INTER_NEAREST).astype(bool))


insts = []
for cl in CLIPS:
    frames = list(read_frames(cl["video"]))
    H = frames[0].shape[0]
    raw = list(np.load(Path(cl["out"]) / "raw_dets.npz", allow_pickle=True)["dets"])
    pm = combine(run_segmenter(None, cache=Path(cl["out"]) / "person_masks.npz"),
                 run_segmenter_x(None, cache=Path(cl["out"]) / "person_masks_x.npz"))
    tracks = json.load(open(Path(cl["out"]) / "tracks.json"))["frames"]
    audit = json.load(open(cl["audit"]))
    verdict = {p["frame"]: p for g in audit for p in g["audit"]["per_frame"]}
    n_ab = n_r = 0
    for f in sorted(verdict):
        ath_boxes = []
        for who in "AB":
            a = tracks[f]["athletes"][who]
            if a["status"] == "detected" and verdict[f][who] == "correct":
                d = raw[f][[i for i, r in enumerate(raw[f]) if np.allclose(r[:4], a["box"], atol=1.0)][0]]
                insts.append(dict(clip=cl["name"], frame=f, label=who, box=np.array(a["box"]), kp=d[5:].reshape(17, 3),
                                  conf=float(d[4]), crop=crop(frames[f], pm[f], a["box"])))
                ath_boxes.append(np.array(a["box"]))
                n_ab += 1
        # referee / other official, labelled by EXCLUSION (no colour used): in a frame where BOTH athletes are verified
        # (audited-correct direct detections), any other confident on-mat person that does not overlap them is not an athlete.
        if len(ath_boxes) == 2:
            for d in raw[f]:
                x1, y1, x2, y2, c = [float(v) for v in d[:5]]
                if c < 0.5 or y2 < 0.5 * H or (x2 - x1) < 40:
                    continue
                if any(iou(np.array([x1, y1, x2, y2]), b) > 0.3 for b in ath_boxes):
                    continue
                insts.append(dict(clip=cl["name"], frame=f, label="R", box=np.array([x1, y1, x2, y2]), kp=d[5:].reshape(17, 3),
                                  conf=c, crop=crop(frames[f], pm[f], (x1, y1, x2, y2))))
                n_r += 1
    print(cl["name"], "A/B instances", n_ab, "referee instances", n_r)

# segments: consecutive labelled frames of the same clip (gap > 6 frames starts a new segment)
insts.sort(key=lambda d: (d["clip"], d["frame"]))
seg, last = 0, None
for d in insts:
    key = (d["clip"])
    if last is None or last[0] != key or d["frame"] - last[1] > 6:
        seg += 1
    d["segment"] = seg
    last = (key, d["frame"])
# contact flag: another labelled box in the same frame overlaps this one
by = {}
for d in insts:
    by.setdefault((d["clip"], d["frame"]), []).append(d)
for d in insts:
    d["contact"] = any(o is not d and iou(d["box"], o["box"]) > 0.1 for o in by[(d["clip"], d["frame"])])
import pickle
pickle.dump(insts, open("outputs/marker_study/dataset.pkl", "wb"))
import collections
print(collections.Counter((d["clip"], d["label"]) for d in insts), "segments", seg,
      "contact", sum(d["contact"] for d in insts))

# montage to eyeball the referee labels
rng = np.random.default_rng(0)
tiles = []
for clipn in ("clip1", "clip2"):
    rs = [d for d in insts if d["clip"] == clipn and d["label"] == "R"]
    for d in rng.choice(rs, size=min(8, len(rs)), replace=False) if rs else []:
        c = cv2.resize(d["crop"][0], (96, 192)); cv2.putText(c, f"{clipn[-1]}:{d['frame']}", (2, 14), 0, 0.5, (0, 255, 255), 1); tiles.append(c)
cv2.imwrite("outputs/marker_study/referee_check.jpg", cv2.hconcat(tiles))
