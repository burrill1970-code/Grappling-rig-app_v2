# Grappling tracking: report

## Status: BLOCKED at environment verification. No tracking was run.

No results exist. Nothing was mocked or faked.

### Blocker
`test_clips/` does not exist in the repo checkout. The repo has no commits, and
`git ls-remote origin` returns no branches, so the video is not on the remote either.
A filesystem search found no `*.mp4`, `*.mov`, `*.mkv` or `ground_truth.md` anywhere
on the machine.

The video is most likely uncommitted locally, or it was never pushed to
`burrill1970-code/Grappling-rig-app_v2`. Because the clip can't be decoded, I could
not inspect the first frames to identify the gi colors, so athletes A and B are not
yet defined.

### Environment checks
| Check | Result |
|---|---|
| Python | 3.11.15, 4 CPUs, 15 GB RAM, CPU only |
| ffmpeg / ffprobe | present (`/usr/bin/ffmpeg`) |
| Video decodes | **NOT VERIFIED**: no video file found |
| pip installs | OK (`pip download ultralytics` fetched 8.4.171) |
| Model weights | OK (`yolo11n-pose.pt` fetched from the GitHub release, HTTP 200, 6.26 MB) |
| `ground_truth.md` | not present, so ground-truth checks will be skipped |

### Planned approach (not started)
Ultralytics YOLO11-pose (person detection plus 17 keypoints, pretrained, no training)
with ByteTrack. Identity comes from a torso-region HSV histogram per gi color, which
overrides track IDs whenever tracks merge, split or swap. Frames will be downsampled
for CPU speed.

### Needed to proceed
Commit `test_clips/<video>` to the branch `ccr-6b606fea-rlzjik` (or tell me where it
is), and I'll continue from milestone 1.
