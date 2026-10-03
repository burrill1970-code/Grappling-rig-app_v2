# Grappling athlete tracking: report

Clip: `test_clips/clip2.mp4`, 453 frames, ~32.6 fps, 13.9s. Identity labels: **A = white gi, B = blue gi** (both clips checked by eye on the first frames; other colour pairs are not supported).

## Verdict (read this first)

* Two tracks, A (white gi) and B (blue gi), exist on **every one of the 453 frames**. That says nothing about whether each track sits on the right person; the audits below measure that.
  Only 27 frames (6%) are *clear* (both athletes directly detected and not overlapping):
  frames 80-83, 102-111, 118-125, 148-149, 156, 163, 166. **Only on those frames is identity independently verifiable by separation.**
* Outside the clear frames the athletes overlap or the detector merges them into one box, so for ~64% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. These boxes are
  low-confidence (capped at 0.40), often too small, and sometimes wrap around the other athlete.
* Identity during contact rests on **gi colour alone**. There is **no ground truth** (`ground_truth.md`
  is not in the repo), so none of the checks below is a ground-truth check.
* The evidence for quality is **vision audits**: model reviewers judged contact sheets of the annotated video and a second round of skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the reviewers are language models. Results are in the Audits section.
* Latest audit (1, 165 of 453 frames sampled): 10 wrong_body verdicts; A judged correct on 157 and B on 146 of 165 frames (95% / 88%); the remaining verdicts are partial boxes (5 A, 3 B) and lost-but-visible (0 A, 9 B). Sampling means this is an estimate, not a count over all frames.
* Pose is partial: 390/453 frames carry a skeleton for A and 358/453 for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## Clip notes (affect results)

* Shot cuts detected at frames: 437 (1 cut; colour-histogram jump > 0.6, so soft cuts and dissolves may be missed).
  Motion state resets at each cut. Gi colour cannot tell *people* apart across a cut: A/B keep the same colour labels but may be different athletes.
* Phone screen recording of a YouTube player: player chrome is cropped away (rows 346-954); a pause/skip overlay can appear over the picture.
* The referee is meant to be excluded by colour (torso mostly dark, no gi colour) and position, the crowd by position and size. This works for a black suit and **fails when the suit reads as blue** (a navy suit, see the audit); it would also fail for a referee in white.
* Source: a screen recording of "Judo Paris Grand Slam 2026 - Day One Highlights" (title overlay). It is **judo**, not BJJ.
* Frames 0-436 are one continuous shot of one match (white gi vs blue gi, scoreboard FRA/HUN). Frames 437-452 (16 frames) are a **different match** with large white "HARVEST" ad boards behind the athletes. The two matches are different people; A/B keep the same colour labels.
* The camera zooms in during the early ground phase (about frames 42-51) and the athletes are in contact almost all the time: only a handful of short stretches are clear.
* On-screen graphics: pause/skip button overlay in the early frames, JUDO TV logos, "DISCOUNT CODE" text, a scoreboard, white "adidas"/"PARIS" ad boards.
* **The referee wears a dark navy suit, white shirt and blue tie, and stands next to the athletes for long stretches.** On frame 80 his torso reads 33% blue / 62% dark, close to a blue gi in shadow. In clip 1 the referee wore black and this did not arise. The blue gi here is bright royal blue; in clip 1 it was dark navy.
* In the second match the blue athlete is mostly hidden behind the white athlete, back to the camera.

## Method

YOLO11m-pose (detection + 17 keypoints), YOLO11m-seg and a second slower YOLO11x-seg pass (person masks), all pretrained, CPU, no training.
Per frame, athlete candidates are matched to A/B by Hungarian assignment. Primary cost: torso-HSV-histogram distance to the identity's
reference plus a white/blue class prior (class taken from all gi pixels in the box). Secondary cost: distance to the constant-velocity predicted
box. Motion state resets at shot cuts. When one box covers both athletes, or an athlete is missed, that athlete's box is the bounding box of its
gi colour inside the person mask (referee's upper body removed; colour boxes must sit on the mat and be at least 40x30 px). If the segmenter
missed a person, the detector box stands in for the mask. The search widens in steps (near last box, other athlete's surroundings, whole mat).
This is a purpose-built two-target tracker, not ByteTrack/DeepSORT: with exactly two athletes and a colour cue it was simpler to constrain directly.

## Per-athlete status counts

| status | A (white) | B (blue) | meaning |
|---|---|---|---|
| detected | 164 | 165 | own detector box and pose |
| color_split | 283 | 274 | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | 6 | 5 | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | 0 | 9 | nothing usable; box and keypoints are null |

Pose: A 390 frames (226 from colour-filtered shared detections), B 358 frames (193).

## Audits (vision, model-based; not ground truth)

Each audit: reviewers over contact sheets (every third frame, at a different offset per audit, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|
| 1 | build as delivered by the first run on this clip (no clip-specific tuning); nothing was changed after the audit | 165 | 157 / 5 / 3 / 0 of 165 | 146 / 3 / 7 / 9 of 165 | 19 / 1 / 0 |

**Latest audit confirmed errors** (19): 81 B referee_or_crowd_labelled, 137 B lost_but_visible, 156 A referee_or_crowd_labelled, 216 A box_on_wrong_body, 312 B referee_or_crowd_labelled, 347 B referee_or_crowd_labelled, 347 A pose_on_wrong_athlete, 348 B referee_or_crowd_labelled, 349 B referee_or_crowd_labelled, 351 B referee_or_crowd_labelled, 363 A referee_or_crowd_labelled, 439 B lost_but_visible, 440 B lost_but_visible, 441 B lost_but_visible, 442 B lost_but_visible, 443 B lost_but_visible, 450 B lost_but_visible, 451 B lost_but_visible, 452 B lost_but_visible. It covered 165 of 453 frames; frames not sampled are unaudited.

This is the **first and only audit of clip 2**, run on the build exactly as delivered (the same code that was tuned on clip 1; nothing was changed for this clip and nothing was changed after the audit). It is a transfer test, and the result is clearly worse than on clip 1.

**Confirmed by both skeptics (19 of 20 serious claims):**
* **Referee tracked as an athlete (8 frames):** B's solid box and skeleton are on the referee at 81, 312, 347, 348, 349, 351; A's are on the referee at 156 and 363. At 81 the real blue athlete has no B overlay at all; at 349-351 he is outside or mostly outside B's box.
* **Wrong body, A:** 216 (A's colour box covers the blue athlete's back, including the white back panel of his gi); 347 (A's skeleton sits on the blue athlete).
* **B lost while visible (9 frames):** 137, 439-443, 450-452. In the second match the blue athlete is mostly hidden behind the white athlete: the only blue visible is a sliver under 40 px wide (rejected by the minimum-width rule) and the other blue blobs are ad boards.
* 1 claim was refuted (138, B).

**Cause of the referee failures (diagnosed, not fixed):** the athlete filter drops a person only if the torso is mostly dark with no gi colour. The navy suit reads 33% blue / 62% dark on frame 80, which passes as "blue gi", and that candidate earns the blue-class prior while the real blue athlete's box (overlapping the white athlete) is classed `mixed` and earns none. The assignment then hands B to the referee. Colour alone cannot separate a navy suit from a dark blue gi (clip 1's blue gi reads up to 70% dark); a fix needs context, for example that the two athletes are in contact while the referee stands apart. That would change shared assignment logic and needs clip 1 re-validated, so it was not done in this step. The eight referee frames are pinned as strict-xfail tests in `tests/test_clip2_known_failures.py`.

**Unaudited:** 288 of 453 frames were not sampled (the audit covered every third frame from frame 0, plus the flagged frames). The referee error may be more frequent than the 8 sampled frames show: the sheets sample about a third of the frames and the referee stands next to the athletes for long stretches.

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
0-79, 84-101, 112-117, 126-147, 150-155, 157-162, 164-165, 167-452 (426 frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: none (0 frames)
* B: 137, 439-443, 450-452 (9 frames)
* Runs longer than N=5 frames: A none, B none. Every lost frame carries status `lost` in the JSON and the video HUD.
* A frame listed as lost can still show the athlete (colour not found, e.g. blur or shadow): it is a tracker failure, not proof of occlusion.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: 23-60, 64-79, 84-87, 132-140, 177-249, 251-252, 282-293, 295-309, 314-339, 353-360, 364-394, 396-427, 429-436, 438-452
* B: 23-60, 64-79, 84-87, 132-136, 138-140, 177-215, 217-249, 251-252, 282-293, 295-309, 314-339, 353-360, 364-394, 396-427, 429-436, 438, 444-449

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: 344-346, 349
* B: 395, 428, 437

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A 347, B 363, 437.
Of these 3 frames, the latest audit sampled 3 and judged 2 wrong-body (347, 363). This check can fire on correct IDs when an overlapping box captures the other athlete's sleeve, so it flags contamination as well as errors.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only 164 A / 165 B frames are direct detections; two athlete-like detector boxes exist on 74% of frames, and some of those are duplicate boxes on the same pair).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin and can wrap around the other athlete: 27 of 112 sampled colour boxes (24%) contain more of the other athlete's colour than their own.
3. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct.
4. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. A blue gi's white patches (collar, back label) can look like A. White ad boards and logos are rejected only by the person mask.
5. Cuts are found by a colour-histogram jump (threshold 0.6); soft cuts are missed and motion state would carry over them. Colour does not identify *people* across cuts.
6. Frame rate is variable in the source; the annotated video uses 32.60 fps (frames / duration).
7. Audits (where run) sample a subset of frames and are done by language-model reviewers, who can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Clip 1: exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and audit-found errors pinned to specific frames: pass. Every clip also runs the generic checks in `tests/test_generic.py` (counts, `lost` flagging, boxes on the mat, poses inside their box), which cannot see a referee being tracked as an athlete. Known open failures are pinned as strict-xfail tests (clip 2: `tests/test_clip2_known_failures.py`).
* No swaps in clear segments: passes, but covers only the 27 clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete-box rate (74% of frames) and colour-box quality (27/112 bad) are **measured numbers; clip 1 has regression floors on them, not success claims**.
* Ground truth: ground_truth.md not found; the test is skipped, not passed.

## Outputs

* `outputs/clip2/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/clip2/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/clip2/stats.json` (and `audit_summary.json`, `audit*_results.json` where audits were run): the numbers and evidence used here.
* Regenerate: `python run_pipeline.py --clip test_clips/clip2.mp4 --out outputs/clip2 && python tools/summarise_audits.py --out outputs/clip2 && python make_report.py --clip test_clips/clip2.mp4 --out outputs/clip2 --report outputs/clip2/report.md`.
