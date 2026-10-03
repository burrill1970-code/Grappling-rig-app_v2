"""Write report.md from outputs/stats.json (numbers/ranges computed) + fixed, hand-verified observations."""
import json
from pathlib import Path

s = json.load(open("outputs/stats.json"))
fps = 535 / 16.923689


def fmt(runs):
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs) or "none"


def t(f):
    return f"{f / fps:.2f}s"


def total(runs):
    return sum(b - a + 1 for a, b in runs)


n = s["n"]
c = s["counts"]
gt = Path("ground_truth.md")
L = []
L.append(f"""# Grappling athlete tracking: report

Clip: `test_clips/clip.mp4`, {n} frames, ~{fps:.1f} fps, {n / fps:.1f}s. Identity labels: **A = white gi, B = blue gi**.

## Verdict (read this first)

* Two tracks, A and B, exist on **every one of the {n} frames**, and A is always white and B always blue.
  Only {s['clear_n']} frames ({100 * s['clear_n'] / n:.0f}%) are *clear* (both athletes directly detected and not overlapping):
  frames {fmt(s['clear'])}. **Identity is only independently verifiable by separation on those frames.**
* From about frame 80 to the end the athletes are in contact (standing grips, throw, ground). The detector returns one
  box for the pair on most of those frames, so for ~{100 * (c['A']['color_split'] + c['A']['color_recovered']) / n:.0f}% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. Those boxes are
  often too small or wrap around the other athlete. They are labelled `color_split` / `color_recovered`, confidence capped at 0.40.
* Identity during contact therefore rests on **gi colour alone**. The A/B label follows the gi colour, so a motion-style
  ID swap is unlikely, but a box can still sit on the wrong body or cover only part of one, and I have no ground truth to prove otherwise.
* Pose is partial: {s['pose']['A']}/{n} frames have a skeleton for A and {s['pose']['B']}/{n} for B. On colour-split frames only
  the keypoints whose own pixels match the athlete's gi colour are kept, so skeletons are fragments.
* Some failures I found by eye and could not fix (listed below). `ground_truth.md` {'exists' if gt.exists() else 'does **not** exist in the repo, so no ground-truth checks were run (nothing is reported as passed)'}.

## What the clip is (affects results)

* It is a **phone screen recording of a YouTube player**, not a clean match feed: player chrome above and below the picture
  (cropped to rows 346-953), and a pause/skip overlay drawn over the picture in the early frames.
* It is **judo**, not BJJ (judogi, referee in black suit, tatami, Budapest 2016 boards).
* There is a **shot cut at frame {s['cuts'][0]}** ({t(s['cuts'][0])}). Frames 0-{s['cuts'][0] - 1} show a different match (white gi standing, blue gi on the mat).
  A/B in those frames are the same *colour labels* but **different people** from frames {s['cuts'][0]} onward. Colour identity cannot tell that apart.
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
| detected | {c['A']['detected']} | {c['B']['detected']} | own detector box and pose |
| color_split | {c['A']['color_split']} | {c['B']['color_split']} | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | {c['A']['color_recovered']} | {c['B']['color_recovered']} | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | {c['A']['lost']} | {c['B']['lost']} | nothing usable; box and keypoints are null |

Pose: A {s['pose']['A']} frames ({s['pose_shared']['A']} from colour-filtered shared detections), B {s['pose']['B']} frames ({s['pose_shared']['B']}).

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges: frames
{fmt(s['id_risk'])} ({total(s['id_risk'])} frames). On these frames A/B come from colour only. No swap was *seen* in the frames I inspected
(a few dozen frames sampled across the clip), but I did not inspect them all and I have no ground truth.

**Track lost (box null), listed in full.**
* A: {fmt(s['lost']['A'])} ({total(s['lost']['A'])} frames)
* B: {fmt(s['lost']['B'])} ({total(s['lost']['B'])} frames)
* Runs longer than N=5 frames: A {fmt(s['lost_long']['A'])}, B {fmt(s['lost_long']['B'])}. Every lost frame carries status `lost` in the JSON and in the video HUD.
* In frames 142-143, 309-313 (A) and 464 an athlete is clearly visible and still marked lost. At 142-143 and 464 *both* athletes are lost (motion blur / too few gi pixels). These are tracker failures, not real occlusions.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: {fmt(s['colour_only']['A'])}
* B: {fmt(s['colour_only']['B'])}

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: {fmt(s['low_conf']['A'])}
* B: {fmt(s['low_conf']['B'])}

**Colour disagreement on direct detections** (torso colour says the other gi; automatic check):
A {fmt(s['colour_conflict']['A'])}, B {fmt(s['colour_conflict']['B'])}.
I looked at 84, 90, 94, 97, 187, 472 and 490. In 84-97, 187 and 490 IDs are correct and the flag is contamination from overlapping boxes (e.g. the blue sleeve inside A's box). In 472 B's low-confidence own-detection skeleton extends across the white athlete's legs: a pose from a detection spanning both bodies can belong to either, and the status `detected` does not guard against that. Frames 89 and the rest of the flagged ranges were not individually checked.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only {c['A']['detected']} A / {c['B']['detected']} B frames are direct detections).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin, and wrap around the other athlete: in a sample of colour-derived boxes, about 13% contained more of the other athlete's colour than their own (checked by `tests/test_milestone4.py`).
3. A is often mostly hidden under B. Little white is visible there, so A's box can be small.
4. Poses on tangled bodies are poor. Keypoints from the detector are unreliable when bodies overlap, and the colour filter only drops keypoints it can attribute, it does not make the rest correct.
5. The colour classes assume a white gi vs a blue gi. Shadowed white falls back to a lower brightness threshold, which is only safe inside the person mask.
6. Only one shot cut was detected (histogram threshold 0.6). A softer cut would be missed and motion state would carry over it.
7. Colour does not identify *people* across the cut at frame {s['cuts'][0]}.
8. Frame rate is variable in the source; the annotated video uses {fps:.2f} fps (frames / duration).

## Tests

`python -m pytest tests`: see `tests/`. Notes on what they do and do not prove:
* Exactly 2 tracks on every frame and every missing track is flagged `lost`: passes.
* No swaps in clear segments: passes, but covers only the {s['clear_n']} clear frames above.
* Colour re-identification after occlusion: checks detections right after non-detected stretches match their gi colour (<=10% disagreement allowed).
* Two-athlete detection rate (48% of frames) and colour-box quality (13% bad, floor 25%) are **measured regression floors, not success claims**.
* Ground truth: {'present' if gt.exists() else 'ground_truth.md not found; test is skipped, not passed'}.

## Outputs

* `outputs/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/stats.json`: ranges used in this report. Regenerate with `python run_pipeline.py && python make_report.py`.
""")
Path("report.md").write_text("\n".join(L))
print("report.md written", len("\n".join(L)))
