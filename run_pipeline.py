"""Full pipeline: detect -> segment -> track -> pose -> outputs (annotated video, JSON, report data)."""
import json
import subprocess
import numpy as np
from grappling.video import read_frames, video_info, detect_cuts, CROP_Y0, CROP_Y1
from grappling.detect import run_detector
from grappling.segment import run_segmenter
from grappling.tracker import track_video
from grappling.pose import attach_keypoints, KP_NAMES
from grappling.render import write_video
from grappling.report import analyse

CLIP = "test_clips/clip.mp4"
frames = list(read_frames(CLIP))
info = video_info(CLIP)
fps = 535 / 16.923689  # stream is variable-rate; nb_frames / duration from ffprobe
cuts = detect_cuts(frames)
raw = run_detector(frames, cache="outputs/raw_dets.npz")
pm = run_segmenter(frames, cache="outputs/person_masks.npz")
res = attach_keypoints(track_video(frames, raw, cuts, pm), raw, frames, pm)

write_video("outputs/_tmp_annotated.mp4", frames, res, set(cuts), fps)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "outputs/_tmp_annotated.mp4", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "23", "outputs/annotated.mp4"], check=True)
import os; os.remove("outputs/_tmp_annotated.mp4")

shot = np.searchsorted(cuts, np.arange(len(res)), side="right")
out = dict(
    meta=dict(video=CLIP, fps=fps, frames=len(res), crop_y=[CROP_Y0, CROP_Y1],
              note="coordinates are in the cropped picture (1080x608); add crop_y[0] to y for full-frame",
              shot_cuts=cuts,
              identity=dict(A="white gi", B="blue gi"),
              status_legend=dict(detected="own detector box+pose", color_split="detector merged both athletes; box from gi-colour pixels (low conf)",
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
json.dump(out, open("outputs/tracks.json", "w"))
stats = analyse(res, frames, raw, cuts)
json.dump(stats, open("outputs/stats.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print({k: stats[k] for k in ("counts", "pose", "clear_n", "colour_conflict_n", "lost_long")})
