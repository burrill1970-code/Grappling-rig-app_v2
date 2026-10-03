from grappling.video import read_frames
from grappling.segment import run_segmenter
m = run_segmenter(read_frames("test_clips/clip.mp4"), cache="outputs/person_masks.npz")
print("masks", len(m), "done")
