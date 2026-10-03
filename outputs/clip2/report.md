# Grappling athlete tracking: report

Clip: `test_clips/clip2.mp4`, 453 frames, ~32.6 fps, 13.9s. Identity labels: **A = white gi, B = blue gi** (both clips checked by eye on the first frames; other colour pairs are not supported).

## Verdict (read this first)

* Two tracks, A (white gi) and B (blue gi), exist on **every one of the 453 frames**. That says nothing about whether each track sits on the right person; the audits below measure that.
  Only 22 frames (5%) are *clear* (both athletes directly detected and not overlapping):
  frames 102-111, 118-125, 148-149, 163, 166. **Only on those frames is identity independently verifiable by separation.**
* Outside the clear frames the athletes overlap or the detector merges them into one box, so for ~67% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. These boxes are
  low-confidence (capped at 0.40), often too small, and sometimes wrap around the other athlete.
* Identity during contact rests on **gi colour alone**. There is **no ground truth** (`ground_truth.md`
  is not in the repo), so none of the checks below is a ground-truth check.
* The evidence for quality is **vision audits**: model reviewers judged contact sheets of the annotated video and a second round of skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the reviewers are language models. Results are in the Audits section.
* Latest random-sample audit (2, 170 of 453 frames sampled; the table below says which build it measured, and later targeted re-checks are in the audit notes): 2 wrong_body verdicts; A judged correct on 164 and B on 156 of 170 frames (96% / 92%); the remaining verdicts are partial boxes (3 A, 5 B), lost-but-visible (0 A, 9 B) and unclear (1 A, 0 B). Sampling means this is an estimate, not a count over all frames.
* Pose is partial: 390/453 frames carry a skeleton for A and 347/453 for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## Clip notes (affect results)

* Shot cuts detected at frames: 437 (1 cut; colour-histogram jump > 0.6, so soft cuts and dissolves may be missed).
  Motion state resets at each cut. Gi colour cannot tell *people* apart across a cut: A/B keep the same colour labels but may be different athletes.
* Phone screen recording of a YouTube player: player chrome is cropped away (rows 346-954); a pause/skip overlay can appear over the picture.
* The referee is excluded by colour (torso mostly dark, no gi colour) and position, the crowd by position and size. A suit that reads as gi colour (clip 2's navy suit) is caught by a referee appearance gallery learned online from the unambiguous dark detections; that needs the referee to have been seen unambiguously first, and it would fail for a referee in white or for one never seen that way.
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
missed a person, the detector box stands in for the mask. The search widens in steps (near last box, other athlete's surroundings, whole mat); white blobs wholly enclosed by the blue gi (back patches) are not counted as A, and a last fallback unions all blobs of an athlete's colour in the region.
The referee is a third, hidden identity: his appearance is learned online from strict-dark detections and any candidate that matches that gallery clearly better than A or B is set aside before assignment.
Appearance markers beyond colour (gradients/HOG, texture, colour profile, part-based colour, pose shape, a pretrained re-ID embedding, motion/context) were evaluated and none beat the colour histogram; see `outputs/marker_study/REPORT.md`.
This is a purpose-built two-target tracker, not ByteTrack/DeepSORT: with exactly two athletes and a colour cue it was simpler to constrain directly.

## Per-athlete status counts

| status | A (white) | B (blue) | meaning |
|---|---|---|---|
| detected | 149 | 149 | own detector box and pose |
| color_split | 298 | 288 | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | 6 | 7 | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | 0 | 9 | nothing usable; box and keypoints are null |

Pose: A 390 frames (241 from colour-filtered shared detections), B 347 frames (198).

## Audits (vision, model-based; not ground truth)

Each audit: reviewers over contact sheets (every third frame, at a different offset per audit, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|
| 1 | first run on this clip (tuned on clip 1 only; no referee identity) | 165 | 157 / 5 / 3 / 0 of 165 | 146 / 3 / 7 / 9 of 165 | 19 / 1 / 0 |
| 2 | after the referee became a third learned identity; the delivered build differs from it in 16 athlete-frames (patch-aware A search), re-checked in a targeted audit (see notes) | 170 | 164 / 3 / 2 / 0 of 170 | 156 / 5 / 0 / 9 of 170 | 11 / 0 / 0 |

**Latest audit confirmed errors** (11): 49 A box_on_wrong_body, 137 B lost_but_visible, 216 A box_on_wrong_body, 439 B lost_but_visible, 440 B lost_but_visible, 441 B lost_but_visible, 442 B lost_but_visible, 443 B lost_but_visible, 450 B lost_but_visible, 451 B lost_but_visible, 452 B other. It covered 170 of 453 frames; frames not sampled are unaudited.

Audit 1 was a transfer test of the clip-1 tuned build (no clip-specific changes) and was clearly worse than on clip 1. Audit 2 was run after the tracker learned the referee as a third identity. Both are random samples (every third frame, at offsets 0 and 1, plus flagged frames).

**Audit 1 (19 of 20 serious claims confirmed):** the dark-suited referee tracked as an athlete at 81, 156, 312, 347-349, 351, 363; A's box on the blue athlete's back at 216; A's skeleton on the blue athlete at 347; B lost while visible at 137, 439-443, 450-452.

**Cause and fix of the referee failures:** the athlete filter dropped a person only if the torso was mostly dark with no gi colour. The navy suit read 33% blue / 62% dark on frame 80, passed as "blue gi", earned the blue-class prior, and the real blue athlete's box (overlapping the white athlete) was classed `mixed` and earned none, so B went to the referee. A marker study (`outputs/marker_study/REPORT.md`) showed no extra marker (gradients, texture, colour profile, parts, posture, deep re-ID) beats the plain colour histogram; the failure was structural. The tracker now learns the referee's appearance online from strict-dark detections and sets aside any candidate that matches that gallery clearly better than A or B. Audit 2 judged all eight previously wrong frames correct.

**Audit 2 (11 serious claims, all confirmed):** A's box on the blue athlete's white back patch at 49 and 216; B lost while visible at 137, 439-443, 450-452.

**Change after audit 2:** the A search now drops white blobs that sit wholly inside the blue silhouette (the blue gi's "HUN / CAF" back panel), and the blob union has a fallback tier. This changed A or B in 16 athlete-frames. **Targeted re-check (38 frames: the changed frames, previous error frames and flagged frames; not a random sample, no rates):** the eight referee frames are all correct; A at 49-51 improved from wrong-body to partial; **A at 216 is still on the blue athlete's back panel** (the white athlete is hidden); **B is still lost while visible at 137, 439-443, 450-452** (all 10 serious claims confirmed).

**Why B is still lost (diagnosed, not fixed):** in the second match the blue athlete is behind the white one and the person segmenter returns only the white athlete's silhouette, so the blue athlete's trousers and feet are outside the mask. Searching those slivers without a mask would also pick up the navy "HARVEST" lettering on the ad boards behind them.

**Unaudited:** frames outside the samples. Each random audit covered about a third of the frames (165 and 170 of 453) and audit 2 describes the build before the last change; the delivered build was re-checked only on 38 targeted frames.

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
0-101, 112-117, 126-147, 150-162, 164-165, 167-452 (431 frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: none (0 frames)
* B: 137, 439-443, 450-452 (9 frames)
* Runs longer than N=5 frames: A none, B none. Every lost frame carries status `lost` in the JSON and the video HUD.
* A frame listed as lost can still show the athlete (colour not found, e.g. blur or shadow): it is a tracker failure, not proof of occlusion.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: 23-60, 64-79, 84-87, 132-140, 177-249, 251-252, 282-293, 295-340, 344, 347-394, 396-427, 429-436, 438-452
* B: 23-60, 64-79, 84-87, 132-136, 138-140, 177-249, 251-252, 282-293, 295-340, 344, 347-394, 396-427, 429-436, 438, 444-449

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: 345-346
* B: 395, 428, 437

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A none, B 437.
Of these 1 frames, the latest audit sampled 1 and judged 0 wrong-body (none). This check can fire on correct IDs when an overlapping box captures the other athlete's sleeve, so it flags contamination as well as errors.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only 149 A / 149 B frames are direct detections; two athlete-like detector boxes exist on 74% of frames, and some of those are duplicate boxes on the same pair).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin and can wrap around the other athlete: 30 of 118 sampled colour boxes (25%) contain more of the other athlete's colour than their own.
3. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct.
4. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. A blue gi's white patches (collar, back label) can look like A. White ad boards and logos are rejected only by the person mask.
5. Cuts are found by a colour-histogram jump (threshold 0.6); soft cuts are missed and motion state would carry over them. Colour does not identify *people* across cuts.
6. Frame rate is variable in the source; the annotated video uses 32.60 fps (frames / duration).
7. Audits (where run) sample a subset of frames and are done by language-model reviewers, who can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Clip 1: exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and audit-found errors pinned to specific frames: pass. Every clip also runs the generic checks in `tests/test_generic.py` (counts, `lost` flagging, boxes on the mat, poses inside their box), which cannot see a referee being tracked as an athlete. Known open failures are pinned as strict-xfail tests (clip 2: `tests/test_clip2_known_failures.py`).
* No swaps in clear segments: passes, but covers only the 22 clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete-box rate (74% of frames) and colour-box quality (30/118 bad) are **measured numbers; clip 1 has regression floors on them, not success claims**.
* Ground truth: ground_truth.md not found; the test is skipped, not passed.

## Outputs

* `outputs/clip2/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/clip2/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/clip2/stats.json` (and `audit_summary.json`, `audit*_results.json` where audits were run): the numbers and evidence used here.
* Regenerate: `python run_pipeline.py --clip test_clips/clip2.mp4 --out outputs/clip2 && python tools/summarise_audits.py --out outputs/clip2 && python make_report.py --clip test_clips/clip2.mp4 --out outputs/clip2 --report outputs/clip2/report.md`.
