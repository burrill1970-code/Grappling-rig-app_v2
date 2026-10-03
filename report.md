# Grappling athlete tracking: report

Clip: `test_clips/clip.mp4`, 535 frames, ~31.6 fps, 16.9s. Identity labels: **A = white gi, B = blue gi**.

## Verdict (read this first)

* Two tracks, A and B, exist on **every one of the 535 frames**, A always white and B always blue.
  Only 77 frames (14%) are *clear* (both athletes directly detected and not overlapping):
  frames 0-11, 17-80, 145. **Only on those frames is identity independently verifiable by separation.**
* From about frame 80 to the end the athletes are in contact (standing grips, throw, ground). The detector returns one
  box for the pair on most of those frames, so for ~72% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. These boxes are
  low-confidence (capped at 0.40), often too small, and sometimes wrap around the other athlete.
* Identity during contact rests on **gi colour alone**. There is **no ground truth** (`ground_truth.md`
  is not in the repo), so none of the checks below is a ground-truth check.
* The evidence for quality is **three vision audits**: model reviewers judged contact sheets of the annotated video
  and a second round of skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the
  reviewers are language models. Results are in the Audits section; the audits found real bugs which are fixed.
* Pose is partial: 371/535 frames carry a skeleton for A and 424/535 for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## What the clip is (affects results)

* A **phone screen recording of a YouTube player**, not a clean match feed: player chrome above and below the picture
  (cropped to rows 346-953) and a pause/skip overlay over the picture in the early frames.
* It is **judo**, not BJJ (judogi, referee in a black suit, tatami, Budapest 2016 boards).
* A **shot cut at frame 12** (0.38s). Frames 0-11 show a different match (white standing, blue on the mat).
  A/B there carry the same *colour labels* but are **different people** from frame 12 on. Colour identity cannot tell that apart.
* The referee (black suit, blue shirt) is excluded by colour and position, the crowd by position and size. This would fail for a referee in white or blue.

## Method

YOLO11m-pose (detection + 17 keypoints), YOLO11m-seg and a second slower YOLO11x-seg pass (person masks), all pretrained, CPU, no training.
Per frame, athlete candidates are matched to A/B by Hungarian assignment. Primary cost: torso-HSV-histogram distance to the identity's
reference plus a white/blue class prior (class taken from all gi pixels in the box). Secondary cost: distance to the constant-velocity predicted
box. Motion state resets at the shot cut. When one box covers both athletes, or an athlete is missed, that athlete's box is the bounding box of its
gi colour inside the person mask (referee's upper body removed; colour boxes must sit on the mat and be at least 40x30 px). If the segmenter
missed a person, the detector box stands in for the mask. The search widens in steps (near last box, other athlete's surroundings, whole mat).
This is a purpose-built two-target tracker, not ByteTrack/DeepSORT: with exactly two athletes and a colour cue it was simpler to constrain directly.

## Per-athlete status counts

| status | A (white) | B (blue) | meaning |
|---|---|---|---|
| detected | 147 | 135 | own detector box and pose |
| color_split | 371 | 374 | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | 14 | 26 | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | 3 | 0 | nothing usable; box and keypoints are null |

Pose: A 371 frames (224 from colour-filtered shared detections), B 424 frames (289).

## Audits (vision, model-based; not ground truth)

Each audit: 12 reviewers over contact sheets (every third frame at a different offset, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|
| 1 | first full pipeline, before audit fixes | 208 | 189 / 9 / 2 / 8 of 208 | 189 / 10 / 6 / 3 of 208 | 23 / 3 / 1 |
| 2 | after round-1 audit fixes (crowd/referee/311-313/lost), before 2nd segmenter + loose-white fallback | 209 | 190 / 12 / 0 / 7 of 209 | 202 / 7 / 0 / 0 of 209 | 7 / 0 / 0 |

Only audit 3 describes the build delivered here. Audits 1 and 2 are kept because they found the bugs that were fixed.

Bugs the audits found that I had missed by eye, now fixed and pinned by `tests/test_audit_regressions.py`: crowd members and the referee labelled as A/B (frames 456-525, 479, 481);
B's solid box and skeleton on the white athlete at 311-313 and 472; both athletes marked lost while clearly visible at 142-143 and 464, and A at 309-313.

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
12-16, 81-144, 146-534 (458 frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: 140, 142, 308 (3 frames)
* B: none (0 frames)
* Runs longer than N=5 frames: A none, B none. Every lost frame carries status `lost` in the JSON and the video HUD.
* These are tracker failures, not real occlusions, when the athlete is visible: the white athlete's gi is motion-blurred and desaturated in the throw (about 136-144), and audits 1 and 2 confirmed A visible on the lost frames they sampled (earlier builds).

**Identity from colour only (`color_split` / `color_recovered`).**
* A: 90-91, 95, 98, 102-118, 120-129, 131-139, 141, 143-144, 152-153, 156-160, 162-181, 185, 188, 191-307, 309, 311-344, 346-397, 399-412, 421-434, 439-443, 445-462, 465-472, 482, 484-488, 490-502, 504-534
* B: 90-91, 95, 98, 102-118, 120-129, 131-144, 152-153, 156-160, 162-181, 185, 188, 191-344, 346-397, 399-412, 415-416, 419-434, 439-443, 445-450, 453-462, 464-474, 476-482, 484-488, 490-502, 504-534

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: 84-85, 145, 147, 154-155, 184, 186-187, 189-190, 310, 413, 416-417, 436-438, 463-464, 473-474, 479, 481, 483, 503
* B: 1, 5, 9, 413-414, 417-418, 435-436, 444, 451, 463, 475, 483, 489, 503

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A 84-85, 89, 92-94, 96-97, B none.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only 147 A / 135 B frames are direct detections; two athlete candidates exist on 59% of frames).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin and can wrap around the other athlete: 14 of 155 sampled colour boxes (9%) contain more of the other athlete's colour than their own. The audits also rate a number of colour boxes `partial`.
3. A is often mostly hidden under B, so A's box can be small.
4. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct. Audits rate some poses garbage on tangled frames.
5. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. The blue gi has white patches (collar, back label) that can look like A.
6. Only one shot cut was detected (histogram threshold 0.6). A softer cut would be missed and motion state would carry over it.
7. Colour does not identify *people* across the cut at frame 12.
8. Frame rate is variable in the source; the annotated video uses 31.61 fps (frames / duration).
9. No audit covered every frame: audits 1, 2 and 3 each sampled one third of the frames (frame numbers 0, 1 and 2 mod 3) plus the frames this report flags. Reviewers are language models and can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and the audit-found errors pinned to specific frames: pass.
* No swaps in clear segments: passes, but covers only the 77 clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete detection rate (59% of frames) and colour-box quality (14/155 bad, floor 25%) are **measured regression floors, not success claims**.
* Ground truth: ground_truth.md not found; the test is skipped, not passed.

## Outputs

* `outputs/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/stats.json`, `outputs/audit_summary.json`, `outputs/audit*_results.json`: the numbers and audit evidence used here.
* Regenerate: `python run_pipeline.py && python tools/summarise_audits.py && python make_report.py`.
