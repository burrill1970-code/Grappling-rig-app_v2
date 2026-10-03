from grappling.video import read_frames
from grappling.detect import run_detector
dets = run_detector(read_frames("test_clips/clip.mp4"), cache="outputs/raw_dets.npz")
print("frames", len(dets), "people/frame histogram",
      {n: sum(len(d) == n for d in dets) for n in range(8)})
