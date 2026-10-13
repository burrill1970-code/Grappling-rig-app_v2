# Grappling athlete tracking: report

Clip: `test_clips/clip.mp4`, 535 frames, ~31.6 fps, 16.9s. Identity labels: **A = white gi, B = blue gi** (both clips checked by eye on the first frames; other colour pairs are not supported).

## Verdict (read this first)

* Two tracks, A (white gi) and B (blue gi), exist on **every one of the 535 frames**. That says nothing about whether each track sits on the right person; the audits below measure that.
  Only 77 frames (14%) are *clear* (both athletes directly detected and not overlapping):
  frames 0-11, 17-80, 145. **Only on those frames is identity independently verifiable by separation.**
* Outside the clear frames the athletes overlap or the detector merges them into one box, so for ~73% of the clip each athlete's box is
  **not a detection**: it is built from that athlete's gi-coloured pixels inside a person mask, then grown over the rest of the
  body (head, hands, feet) by the person-mask pixels nearer that gi than the other one. These boxes are low-confidence (capped at 0.40);
  when the athletes are wrapped together they often cover both, and the growth can overreach onto the other athlete.
* Identity during contact rests on **gi colour alone**.
* **Ground truth** (hand-labelled boxes on 76 regularly sampled frames, every 7th): the track sits on the right person for A on 74/76 visible frames (97%, 95% CI 91-99%) and for B on 76/76 (100%, 95-100%); wrong-body A 0, B 0; box covering both A 2, B 0; identity swaps on 0 frames. Box quality (IoU >= 0.5 with the labelled box): A 83%, B 91%. Details in the Ground truth section.
* The evidence for quality is **vision audits**: model reviewers judged contact sheets of the annotated video and a second round of skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the reviewers are language models. Results are in the Audits section.
* Latest random-sample audit (3, 210 of 535 frames sampled; the table below says which build it measured, and later targeted re-checks are in the audit notes): 0 wrong_body verdicts; A judged correct on 199 and B on 199 of 210 frames (95% / 95%); the remaining verdicts are partial boxes (8 A, 11 B), lost-but-visible (3 A, 0 B) and unclear (0 A, 0 B). Sampling means this is an estimate, not a count over all frames.
* Pose is partial: 373/535 frames carry a skeleton for A and 424/535 for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## Clip notes (affect results)

* Shot cuts detected at frames: 12 (1 cut; colour-histogram jump > 0.6, so soft cuts and dissolves may be missed).
  Motion state resets at each cut. Gi colour cannot tell *people* apart across a cut: A/B keep the same colour labels but may be different athletes.
* Phone screen recording of a YouTube player: player chrome is cropped away (rows 346-954); a pause/skip overlay can appear over the picture.
* The referee is excluded by colour (torso mostly dark, no gi colour) and position, the crowd by position and size. A suit that reads as gi colour (clip 2's navy suit) is caught by a referee appearance gallery learned online from the unambiguous dark detections; that needs the referee to have been seen unambiguously first, and it would fail for a referee in white or for one never seen that way.
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
missed a person, the detector box stands in for the mask. The reported box is then grown from that gi box over the athlete's skin, hair and
limbs: person-mask pixels nearer this athlete's gi than the other's, within half the gi box's longer side (setting chosen on clip 1, checked once on
clip 2). Tracking itself still runs on the gi box, which `tracks.json` keeps as `gi_box`. The search widens in steps (near last box, other athlete's surroundings, whole mat); white blobs wholly enclosed by the blue gi (back patches) are not counted as A, and a last fallback unions all blobs of an athlete's colour in the region.
The referee is a third, hidden identity: his appearance is learned online from strict-dark detections and any candidate that matches that gallery clearly better than A or B is set aside before assignment.
Appearance markers beyond colour (gradients/HOG, texture, colour profile, part-based colour, pose shape, a pretrained re-ID embedding, motion/context) were evaluated and none beat the colour histogram; see `outputs/marker_study/REPORT.md`.
This is a purpose-built two-target tracker, not ByteTrack/DeepSORT: with exactly two athletes and a colour cue it was simpler to constrain directly.

## Per-athlete status counts

| status | A (white) | B (blue) | meaning |
|---|---|---|---|
| detected | 147 | 137 | own detector box and pose |
| color_split | 371 | 372 | detector merged the pair; box from gi-colour pixels, grown over the body (conf <= 0.40) |
| color_recovered | 17 | 26 | detector missed the athlete; box from gi-colour pixels, grown over the body (conf <= 0.40) |
| lost | 0 | 0 | nothing usable; box and keypoints are null |

Pose: A 373 frames (226 from colour-filtered shared detections), B 424 frames (287).

## Ground truth (hand labels)

A person drew a box around each athlete's whole visible body, or marked them hidden, on the raw frames (never seeing the tracker's output), using
`tools/gt_label.py`. `tools/eval_ground_truth.py` scores each track box by IoU with that athlete's box ("own") and the other athlete's ("other"):
correct = own IoU >= 0.5; partial = right person, poor box (0 < own IoU < 0.5, own >= other); wrong_body = other IoU > own IoU; off_target = on
neither athlete; lost_but_visible = no track box for a visible athlete; hidden_ok / box_on_hidden = athlete labelled hidden, without / with a track box.
When the athletes are tangled their labelled boxes overlap heavily and IoU cannot say which one a track box is on, so wherever the box touches both
and favours its own by less than 0.15 IoU, a person answered which athlete it is on (`tools/gt_identity.py`, which does show the track box):
right keeps the IoU verdict, wrong = wrong_body, both = covers_both; unanswered = undecided, left out of the rates.
Colour-derived track boxes cover gi pixels only, so they score lower IoU than a whole-body label even on the right person: *partial* is their expected verdict.

Regular sample: 76 frames (every 7th). One labeller, no second-labeller agreement check.

| verdict | A (white) | B (blue) |
|---|---|---|
| correct | 63 | 69 |
| partial | 11 | 7 |
| wrong_body | 0 | 0 |
| off_target | 0 | 0 |
| covers_both | 2 | 0 |
| undecided | 0 | 0 |
| lost_but_visible | 0 | 0 |
| hidden_ok | 0 | 0 |
| box_on_hidden | 0 | 0 |

Identity checked by a person: A 7, B 4 of the regular frames; undecided (left out of the rates): A 0, B 0.

| on visible, decided frames | A | B |
|---|---|---|
| right person (correct + partial) | 74/76 (97%, CI 91-99%) | 76/76 (100%, CI 95-100%) |
| correct box (IoU >= 0.5) | 83% (CI 73-90%) | 91% (CI 82-95%) |
| box covers both athletes | 3% | 0% |
| wrong body | 0% (CI 0-5%) | 0% (CI 0-5%) |
| mean IoU (all visible frames) | 0.65 | 0.67 |
| direct detections: right person / correct / mean IoU | 20/20 / 20/20 / 0.73 | 20/20 / 20/20 / 0.77 |
| colour-derived boxes: right person / correct / mean IoU | 54/56 / 43/56 / 0.63 | 56/56 / 49/56 / 0.63 |

**Errors on labelled frames:** A: 262 covers_both, 283 covers_both; B: none. **Identity swaps:** none.
**Targeted frames** (chosen because audits or tests found failures there; not a sample, no rate): 140 A partial / B correct, 142 A partial / B correct, 308 A partial / B correct.

## Audits (vision, model-based; not ground truth)

Each audit: reviewers over contact sheets (every third frame, at a different offset per audit, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|
| 1 | first full pipeline, before audit fixes | 208 | 189 / 9 / 2 / 8 of 208 | 189 / 10 / 6 / 3 of 208 | 23 / 3 / 1 |
| 2 | after round-1 audit fixes (crowd/referee/311-313/lost), before 2nd segmenter + loose-white fallback | 209 | 190 / 12 / 0 / 7 of 209 | 202 / 7 / 0 / 0 of 209 | 7 / 0 / 0 |
| 3 | delivered build as of that audit; since then A changed at frames 116, 118, 140, 142, 272, 273, 278, 308, 309 (patch-aware A search, union fallback, widened recovery), re-checked in a targeted audit (see notes) | 210 | 199 / 8 / 0 / 3 of 210 | 199 / 11 / 0 / 0 of 210 | 3 / 0 / 0 |

**Latest audit confirmed errors** (3): 140 A lost_but_visible, 142 A lost_but_visible, 308 A lost_but_visible. It covered 210 of 535 frames; frames not sampled are unaudited.

Only audit 3 describes (almost exactly) the build that was delivered at the time; since then the tracker changed in A at frames 116, 118, 140, 142, 272, 273, 278, 308 and 309 (see below). Audits 1 and 2 are kept because they found the bugs that were fixed.

Bugs the audits found that I had missed by eye, now fixed and pinned by `tests/test_audit_regressions.py`: crowd members and the referee labelled as A/B (frames 456-525, 479, 481); B's solid box and skeleton on the white athlete at 311-313 and 472; both athletes marked lost while clearly visible at 142-143 and 464, and A at 308-313.

**Targeted re-check after the last tracker changes (50 frames: the changed frames plus every flagged frame; not a random sample, so no rate can be quoted):** B judged correct on all 50; A correct on 47 and partial on 3 (140, 142, 308). No wrong-body verdicts and nothing lost while visible. A at 140 and 142, which audit 3 confirmed as lost-but-visible, now has a box that covers only part of the airborne white athlete; 308 has a small (about 30x30 px) low-confidence box. The skeletons at 489 and 503 were rated poor.

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
12-16, 81-144, 146-534 (458 frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: none (0 frames)
* B: none (0 frames)
* Runs longer than N=5 frames: A none, B none. Every lost frame carries status `lost` in the JSON and the video HUD.
* A frame listed as lost can still show the athlete (colour not found, e.g. blur or shadow): it is a tracker failure, not proof of occlusion.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: 90-91, 95, 98, 102-118, 120-129, 131-144, 152-153, 156-160, 162-181, 185, 188, 191-309, 311-344, 346-397, 399-412, 421-434, 439-443, 445-462, 465-472, 482, 484-488, 490-502, 504-534
* B: 90-91, 95, 98, 102-118, 120-129, 131-142, 152-153, 156-160, 162-181, 185, 188, 191-344, 346-397, 399-412, 415-416, 419-434, 439-443, 445-450, 453-462, 464-474, 476-482, 484-488, 490-502, 504-534

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: 84-85, 145, 147, 154-155, 184, 186-187, 189-190, 310, 413, 416-417, 436-438, 463-464, 473-474, 479, 481, 483, 503
* B: 1, 5, 9, 413-414, 417-418, 435-436, 444, 451, 463, 475, 483, 489, 503

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A 84-85, 89, 92-94, 96-97, B none.
Of these 8 frames, the latest audit sampled 8 and judged 0 wrong-body (none). This check can fire on correct IDs when an overlapping box captures the other athlete's sleeve, so it flags contamination as well as errors.

## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only 147 A / 137 B frames are direct detections; two athlete-like detector boxes exist on 59% of frames, and some of those are duplicate boxes on the same pair).
2. Colour-derived boxes start from gi pixels and are grown over the body by distance to each athlete's gi; where the athletes are wrapped together the grown box often covers both, and it can overreach onto the other athlete where a limb of one lies nearer the other's gi (see the ground-truth errors). The underlying gi box can wrap around the other athlete: 14 of 156 sampled gi boxes (9%) contain more of the other athlete's colour than their own.
3. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct.
4. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. A blue gi's white patches (collar, back label) can look like A. White ad boards and logos are rejected only by the person mask.
5. Cuts are found by a colour-histogram jump (threshold 0.6); soft cuts are missed and motion state would carry over them. Colour does not identify *people* across cuts.
6. Frame rate is variable in the source; the annotated video uses 31.61 fps (frames / duration).
7. Audits (where run) sample a subset of frames and are done by language-model reviewers, who can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Clip 1: exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and audit-found errors pinned to specific frames: pass. Every clip also runs the generic checks in `tests/test_generic.py` (counts, `lost` flagging, boxes on the mat, poses inside their box), which cannot see a referee being tracked as an athlete. Known open failures are pinned as strict-xfail tests (clip 2: `tests/test_clip2_known_failures.py`).
* No swaps in clear segments: passes, but covers only the 77 clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete-box rate (59% of frames) and colour-box quality (14/156 bad) are **measured numbers; clip 1 has regression floors on them, not success claims**.
* Ground truth: labels in `outputs/ground_truth.json` are checked for completeness and scored; the numbers are measurements, with no pass/fail floor yet.

## Outputs

* `outputs/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/stats.json` (and `audit_summary.json`, `audit*_results.json` where audits were run): the numbers and evidence used here.
* Regenerate: `python run_pipeline.py --clip test_clips/clip.mp4 --out outputs && python tools/summarise_audits.py --out outputs && python tools/eval_ground_truth.py --out outputs && python make_report.py --clip test_clips/clip.mp4 --out outputs --report report.md`.
