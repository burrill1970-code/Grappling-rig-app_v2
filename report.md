# Grappling athlete tracking: report

Clip: `test_clips/clip.mp4`, 535 frames, ~31.6 fps, 16.9s. Identity labels: **A = white gi, B = blue gi** (both clips checked by eye on the first frames; other colour pairs are not supported).

## Verdict (read this first)

* Two tracks, A and B, exist on **every one of the 535 frames**, A always white and B always blue.
  Only 77 frames (14%) are *clear* (both athletes directly detected and not overlapping):
  frames 0-11, 17-80, 145. **Only on those frames is identity independently verifiable by separation.**
* Outside the clear frames the athletes overlap or the detector merges them into one box, so for ~72% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. These boxes are
  low-confidence (capped at 0.40), often too small, and sometimes wrap around the other athlete.
* Identity during contact rests on **gi colour alone**. There is **no ground truth** (`ground_truth.md`
  is not in the repo), so none of the checks below is a ground-truth check.
* The evidence for quality is **vision audits**: model reviewers judged contact sheets of the annotated video and a second round of skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the reviewers are language models. Results are in the Audits section.
* Latest audit (3, 210 of 535 frames sampled): 0 wrong_body verdicts; A judged correct on 199 and B on 199 of 210 frames (95% / 95%); the remaining verdicts are partial boxes (8 A, 11 B) and lost-but-visible (3 A, 0 B). Sampling means this is an estimate, not a count over all frames.
* Pose is partial: 371/535 frames carry a skeleton for A and 424/535 for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## Clip notes (affect results)

* Shot cuts detected at frames: 12 (1 cut; colour-histogram jump > 0.6, so soft cuts and dissolves may be missed).
  Motion state resets at each cut. Gi colour cannot tell *people* apart across a cut: A/B keep the same colour labels but may be different athletes.
* Phone screen recording of a YouTube player: player chrome is cropped away (rows 346-954); a pause/skip overlay can appear over the picture.
* The referee (dark suit) is excluded by colour and position, the crowd by position and size. This would fail for a referee in white or blue.
* It is **judo**, not BJJ (judogi, referee in a black suit and blue shirt, tatami, Budapest 2016 boards).
* The shot cut at frame 12 separates two different matches: frames 0-11 show a white athlete standing and a blue athlete on the mat; from frame 12 the Budapest match. A/B in frames 0-11 carry the same colour labels but are different people.
* From about frame 80 to the end the athletes are in contact (standing grips, a throw at about 136-144, then ground work).
* A is often mostly hidden under B on the ground, so A's colour box can be small.

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
| detected | 147 | 135 | own detector box and pose |
| color_split | 371 | 374 | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | 15 | 26 | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | 2 | 0 | nothing usable; box and keypoints are null |

Pose: A 371 frames (224 from colour-filtered shared detections), B 424 frames (289).

## Audits (vision, model-based; not ground truth)

Each audit: 12 reviewers over contact sheets (every third frame at a different offset, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|
| 1 | first full pipeline, before audit fixes | 208 | 189 / 9 / 2 / 8 of 208 | 189 / 10 / 6 / 3 of 208 | 23 / 3 / 1 |
| 2 | after round-1 audit fixes (crowd/referee/311-313/lost), before 2nd segmenter + loose-white fallback | 209 | 190 / 12 / 0 / 7 of 209 | 202 / 7 / 0 / 0 of 209 | 7 / 0 / 0 |
| 3 | delivered build, except A at frames 308-309 (changed after the audit: 308 lost -> small low-confidence box) | 210 | 199 / 8 / 0 / 3 of 210 | 199 / 11 / 0 / 0 of 210 | 3 / 0 / 0 |

**Latest audit confirmed errors** (3): 140 A lost_but_visible, 142 A lost_but_visible, 308 A lost_but_visible. It covered 210 of 535 frames; frames not sampled are unaudited.

Only audit 3 describes (almost exactly) the build delivered here: the delivered output differs from the audited one only in A at frames 308-309 (308 was lost in the audited output and now has a small low-confidence `color_recovered` box; not re-audited). Audits 1 and 2 are kept because they found the bugs that were fixed.

Bugs the audits found that I had missed by eye, now fixed and pinned by `tests/test_audit_regressions.py`: crowd members and the referee labelled as A/B (frames 456-525, 479, 481); B's solid box and skeleton on the white athlete at 311-313 and 472; both athletes marked lost while clearly visible at 142-143 and 464, and A at 308-313. All fixed except A at 140 and 142 (the motion-blurred throw, where the white gi is blurred and desaturated), which are still lost.

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
12-16, 81-144, 146-534 (458 frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: 140, 142 (2 frames)
* B: none (0 frames)
* Runs longer than N=5 frames: A none, B none. Every lost frame carries status `lost` in the JSON and the video HUD.
* A frame listed as lost can still show the athlete (colour not found, e.g. blur or shadow): it is a tracker failure, not proof of occlusion.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: 90-91, 95, 98, 102-118, 120-129, 131-139, 141, 143-144, 152-153, 156-160, 162-181, 185, 188, 191-309, 311-344, 346-397, 399-412, 421-434, 439-443, 445-462, 465-472, 482, 484-488, 490-502, 504-534
* B: 90-91, 95, 98, 102-118, 120-129, 131-144, 152-153, 156-160, 162-181, 185, 188, 191-344, 346-397, 399-412, 415-416, 419-434, 439-443, 445-450, 453-462, 464-474, 476-482, 484-488, 490-502, 504-534

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: 84-85, 145, 147, 154-155, 184, 186-187, 189-190, 310, 413, 416-417, 436-438, 463-464, 473-474, 479, 481, 483, 503
* B: 1, 5, 9, 413-414, 417-418, 435-436, 444, 451, 463, 475, 483, 489, 503

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A 84-85, 89, 92-94, 96-97, B none.
Of these 8 frames, the latest audit sampled 8 and judged 0 wrong-body (none). This check can fire on correct IDs when an overlapping box captures the other athlete's sleeve, so it flags contamination as well as errors.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only 147 A / 135 B frames are direct detections; two athlete-like detector boxes exist on 59% of frames, and some of those are duplicate boxes on the same pair).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin and can wrap around the other athlete: 14 of 155 sampled colour boxes (9%) contain more of the other athlete's colour than their own.
3. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct.
4. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. A blue gi's white patches (collar, back label) can look like A. White ad boards and logos are rejected only by the person mask.
5. Cuts are found by a colour-histogram jump (threshold 0.6); soft cuts are missed and motion state would carry over them. Colour does not identify *people* across cuts.
6. Frame rate is variable in the source; the annotated video uses 31.61 fps (frames / duration).
7. Audits (where run) sample a subset of frames and are done by language-model reviewers, who can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Clip 1: exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and audit-found errors pinned to specific frames: pass. Clip 2 runs the generic checks in `tests/test_generic.py`.
* No swaps in clear segments: passes, but covers only the 77 clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete-box rate (59% of frames) and colour-box quality (14/155 bad) are **measured numbers; clip 1 has regression floors on them, not success claims**.
* Ground truth: ground_truth.md not found; the test is skipped, not passed.

## Outputs

* `outputs/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/stats.json` (and `audit_summary.json`, `audit*_results.json` where audits were run): the numbers and evidence used here.
* Regenerate: `python run_pipeline.py --clip test_clips/clip.mp4 --out outputs && python tools/summarise_audits.py --out outputs && python make_report.py --clip test_clips/clip.mp4 --out outputs --report report.md`.
