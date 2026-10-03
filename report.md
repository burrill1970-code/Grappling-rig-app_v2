# Grappling athlete tracking: report

Clip: `test_clips/clip.mp4`, 535 frames, ~31.6 fps, 16.9s. Identity labels: **A = white gi, B = blue gi**.

## Verdict (read this first)

* Two tracks, A and B, exist on **every one of the 535 frames**, and A is always white and B always blue.
  Only 77 frames (14%) are *clear* (both athletes directly detected and not overlapping):
  frames 0-11, 17-80, 145. **Identity is only independently verifiable by separation on those frames.**
* From about frame 80 to the end the athletes are in contact (standing grips, throw, ground). The detector returns one
  box for the pair on most of those frames, so for ~72% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. Those boxes are
  often too small or wrap around the other athlete. They are labelled `color_split` / `color_recovered`, confidence capped at 0.40.
* Identity during contact therefore rests on **gi colour alone**. The A/B label follows the gi colour, so a motion-style
  ID swap is unlikely, but a box can still sit on the wrong body or cover only part of one, and I have no ground truth to prove otherwise.
* Pose is partial: 371/535 frames have a skeleton for A and 424/535 for B. On colour-split frames only
  the keypoints whose own pixels match the athlete's gi colour are kept, so skeletons are fragments.
* Some failures I found by eye and could not fix (listed below). `ground_truth.md` does **not** exist in the repo, so no ground-truth checks were run (nothing is reported as passed).

## What the clip is (affects results)

* It is a **phone screen recording of a YouTube player**, not a clean match feed: player chrome above and below the picture
  (cropped to rows 346-953), and a pause/skip overlay drawn over the picture in the early frames.
* It is **judo**, not BJJ (judogi, referee in black suit, tatami, Budapest 2016 boards).
* There is a **shot cut at frame 12** (0.38s). Frames 0-11 show a different match (white gi standing, blue gi on the mat).
  A/B in those frames are the same *colour labels* but **different people** from frames 12 onward. Colour identity cannot tell that apart.
* A referee in a black suit is in many frames. He is excluded by colour (torso mostly dark, no white or blue gi pixels) and by position.
  The crowd and ad boards are excluded by position/size. This rule would fail for a referee in white or blue.

## Method

YOLO11m-pose (detection + 17 keypoints) and YOLO11m-seg (person masks), both pretrained, CPU, no training. Per frame, athlete candidates
are matched to A/B by Hungarian assignment: primary cost is torso-HSV-histogram distance to each identity's reference plus a
white/blue colour-class prior; secondary cost is distance to the constant-velocity predicted box. Motion state resets at the shot cut.
When one box covers both athletes, or an athlete is missed, that athlete's box is taken from its gi-colour pixels inside the person mask.
This is a purpose-built two-target tracker, not ByteTrack/DeepSORT: with exactly two athletes and a colour cue it was simpler to constrain it directly.

## Per-athlete status counts

| status | A (white) | B (blue) | meaning |
|---|---|---|---|
| detected | 147 | 135 | own detector box and pose |
| color_split | 371 | 374 | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | 14 | 26 | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | 3 | 0 | nothing usable; box and keypoints are null |

Pose: A 371 frames (224 from colour-filtered shared detections), B 424 frames (289).

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges: frames
12-16, 81-144, 146-534 (458 frames). On these frames A/B come from colour only. No swap was *seen* in the frames I inspected
(a few dozen frames sampled across the clip), but I did not inspect them all and I have no ground truth.

**Track lost (box null), listed in full.**
* A: 140, 142, 308 (3 frames)
* B: none (0 frames)
* Runs longer than N=5 frames: A none, B none. Every lost frame carries status `lost` in the JSON and in the video HUD.
* In frames 142-143, 309-313 (A) and 464 an athlete is clearly visible and still marked lost. At 142-143 and 464 *both* athletes are lost (motion blur / too few gi pixels). These are tracker failures, not real occlusions.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: 90-91, 95, 98, 102-118, 120-129, 131-139, 141, 143-144, 152-153, 156-160, 162-181, 185, 188, 191-307, 309, 311-344, 346-397, 399-412, 421-434, 439-443, 445-462, 465-472, 482, 484-488, 490-502, 504-534
* B: 90-91, 95, 98, 102-118, 120-129, 131-144, 152-153, 156-160, 162-181, 185, 188, 191-344, 346-397, 399-412, 415-416, 419-434, 439-443, 445-450, 453-462, 464-474, 476-482, 484-488, 490-502, 504-534

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: 84-85, 145, 147, 154-155, 184, 186-187, 189-190, 310, 413, 416-417, 436-438, 463-464, 473-474, 479, 481, 483, 503
* B: 1, 5, 9, 413-414, 417-418, 435-436, 444, 451, 463, 475, 483, 489, 503

**Colour disagreement on direct detections** (torso colour says the other gi; automatic check):
A 84-85, 89, 92-94, 96-97, B none.
I looked at 84, 90, 94, 97, 187, 472 and 490. In 84-97, 187 and 490 IDs are correct and the flag is contamination from overlapping boxes (e.g. the blue sleeve inside A's box). In 472 B's low-confidence own-detection skeleton extends across the white athlete's legs: a pose from a detection spanning both bodies can belong to either, and the status `detected` does not guard against that. Frames 89 and the rest of the flagged ranges were not individually checked.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only 147 A / 135 B frames are direct detections).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin, and wrap around the other athlete: in a sample of colour-derived boxes, about 13% contained more of the other athlete's colour than their own (checked by `tests/test_milestone4.py`).
3. A is often mostly hidden under B. Little white is visible there, so A's box can be small.
4. Poses on tangled bodies are poor. Keypoints from the detector are unreliable when bodies overlap, and the colour filter only drops keypoints it can attribute, it does not make the rest correct.
5. The colour classes assume a white gi vs a blue gi. Shadowed white falls back to a lower brightness threshold, which is only safe inside the person mask.
6. Only one shot cut was detected (histogram threshold 0.6). A softer cut would be missed and motion state would carry over it.
7. Colour does not identify *people* across the cut at frame 12.
8. Frame rate is variable in the source; the annotated video uses 31.61 fps (frames / duration).

## Tests

`python -m pytest tests`: see `tests/`. Notes on what they do and do not prove:
* Exactly 2 tracks on every frame and every missing track is flagged `lost`: passes.
* No swaps in clear segments: passes, but covers only the 77 clear frames above.
* Colour re-identification after occlusion: checks detections right after non-detected stretches match their gi colour (<=10% disagreement allowed).
* Two-athlete detection rate (48% of frames) and colour-box quality (13% bad, floor 25%) are **measured regression floors, not success claims**.
* Ground truth: ground_truth.md not found; test is skipped, not passed.

## Outputs

* `outputs/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/stats.json`: ranges used in this report. Regenerate with `python run_pipeline.py && python make_report.py`.
