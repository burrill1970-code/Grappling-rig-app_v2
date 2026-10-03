"""Full pipeline: detect -> segment -> track -> pose -> outputs (annotated video, JSON, stats).

    python run_pipeline.py                                   # clip 1 -> outputs/
    python run_pipeline.py --clip test_clips/clip2.mp4 --out outputs/clip2
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from grappling.video import read_frames, detect_cuts, detect_crop, video_fps
from grappling.detect import run_detector
from grappling.segment import run_segmenter
from grappling.segment_x import run_segmenter_x, combine
from grappling.tracker import track_video
from grappling.pose import attach_keypoints, KP_NAMES
from grappling.render import write_video
from grappling.report import analyse

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="test_clips/clip.mp4")
ap.add_argument("--out", default="outputs")
args = ap.parse_args()
CLIP, OUT = args.clip, Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

frames = list(read_frames(CLIP))
fps = video_fps(CLIP)
y0, y1 = detect_crop(CLIP)
cuts = detect_cuts(frames)
raw = run_detector(frames, cache=OUT / "raw_dets.npz")
pm = combine(run_segmenter(frames, cache=OUT / "person_masks.npz"),
             run_segmenter_x(frames, cache=OUT / "person_masks_x.npz"))
res = attach_keypoints(track_video(frames, raw, cuts, pm), raw, frames, pm)

tmp = OUT / "_tmp_annotated.mp4"
write_video(tmp, frames, res, set(cuts), fps)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "23", str(OUT / "annotated.mp4")], check=True)
os.remove(tmp)

shot = np.searchsorted(cuts, np.arange(len(res)), side="right")
out = dict(
    meta=dict(video=CLIP, fps=fps, frames=len(res), crop_y=[y0, y1],
              note="coordinates are in the cropped picture; add crop_y[0] to y for full-frame",
              shot_cuts=cuts,
              identity=dict(A="white gi", B="blue gi"),
              status_legend=dict(detected="own detector box+pose",
                                 color_split="detector merged both athletes; box from gi-colour pixels (low conf)",
                                 color_recovered="detector missed athlete; box from gi-colour pixels near last position (low conf)",
                                 lost="no usable evidence; box and keypoints are null"),
              keypoint_names=KP_NAMES),
    frames=[])
for fi, rec in enumerate(res):
    ath = {}
    for i, r in rec.items():
        kp = None if r["kp"] is None else [dict(name=KP_NAMES[k], x=round(x, 1), y=round(y, 1), conf=round(c, 3))
                                           for k, (x, y, c) in enumerate(r["kp"])]
        ath[i] = dict(status=r["status"], box=None if r["box"] is None else [round(v, 1) for v in r["box"]],
                      confidence=round(r["conf"], 3), keypoints=kp, pose_conf=round(r["pose_conf"], 3),
                      pose_source=r["pose_source"], note=r.get("note"))
    out["frames"].append(dict(frame=fi, t=round(fi / fps, 3), shot=int(shot[fi]), athletes=ath))
json.dump(out, open(OUT / "tracks.json", "w"))
stats = analyse(res, frames, raw, cuts, pm)
json.dump(stats, open(OUT / "stats.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print({k: stats[k] for k in ("counts", "pose", "clear_n", "colour_conflict_n", "lost_long")})
print("cuts", cuts)
