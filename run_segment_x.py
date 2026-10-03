from grappling.video import read_frames
from grappling.segment_x import run_segmenter_x
m = run_segmenter_x(read_frames("test_clips/clip.mp4"), cache="outputs/person_masks_x.npz")
print("masks", len(m), "done")
