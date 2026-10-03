"""Write a report from <out>/stats.json (+ audit_summary.json if present).

    python make_report.py                                              # clip 1 -> report.md
    python make_report.py --clip test_clips/clip2.mp4 --out outputs/clip2 --report outputs/clip2/report.md

Every number and frame range is computed. Fixed prose is generic; clip-specific observations come from
<out>/clip_notes.md and <out>/audit_notes.md (hand-written after looking at the clip / audits), if they exist.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs")
ap.add_argument("--clip", default="test_clips/clip.mp4")
ap.add_argument("--report", default="report.md")
args = ap.parse_args()
OUT = Path(args.out)


def read(p):
    return Path(p).read_text().strip() if Path(p).exists() else ""


s = json.load(open(OUT / "stats.json"))
aud = json.load(open(OUT / "audit_summary.json")) if (OUT / "audit_summary.json").exists() else {}
meta = json.load(open(OUT / "tracks.json"))["meta"]
fps = meta["fps"]
clip_notes, audit_notes = read(OUT / "clip_notes.md"), read(OUT / "audit_notes.md")
gt = Path("ground_truth.md")


def fmt(runs):
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs) or "none"


def total(runs):
    return sum(b - a + 1 for a, b in runs)


n, c, cuts = s["n"], s["counts"], s["cuts"]
colour_pct = 100 * (c["A"]["color_split"] + c["A"]["color_recovered"]) / n
final = aud.get("audit3") or (aud[sorted(aud)[-1]] if aud else None)
final_key = "audit3" if "audit3" in aud else (sorted(aud)[-1] if aud else None)


def pf(a, who):
    d = a["per_frame"][who]
    return " / ".join(str(d.get(k, 0)) for k in ("correct", "partial", "wrong_body", "lost_but_visible")) + f" of {a['frames']}"


# ---- verdict bullets about audits ----
if final:
    v = final["per_frame"]
    wrong = v["A"].get("wrong_body", 0) + v["B"].get("wrong_body", 0)
    fr = final["frames"]
    audit_bullets = (
        "* The evidence for quality is **vision audits**: model reviewers judged contact sheets of the annotated video and a second round of "
        "skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the reviewers are language models. "
        "Results are in the Audits section.\n"
        f"* Latest audit ({final_key[-1]}, {fr} of {n} frames sampled): {wrong} wrong_body verdicts; "
        f"A judged correct on {v['A'].get('correct', 0)} and B on {v['B'].get('correct', 0)} of {fr} frames "
        f"({100 * v['A'].get('correct', 0) / fr:.0f}% / {100 * v['B'].get('correct', 0) / fr:.0f}%); the remaining verdicts are partial boxes "
        f"({v['A'].get('partial', 0)} A, {v['B'].get('partial', 0)} B) and lost-but-visible "
        f"({v['A'].get('lost_but_visible', 0)} A, {v['B'].get('lost_but_visible', 0)} B). Sampling means this is an estimate, not a count over all frames.\n")
else:
    audit_bullets = ("* **No vision audit has been run on this clip**: the numbers below are tracker statistics, not a quality check. "
                     "Treat every colour-derived frame as unverified.\n")

L = [f"""# Grappling athlete tracking: report

Clip: `{args.clip}`, {n} frames, ~{fps:.1f} fps, {n / fps:.1f}s. Identity labels: **A = white gi, B = blue gi** (both clips checked by eye on the first frames; other colour pairs are not supported).

## Verdict (read this first)

* Two tracks, A (white gi) and B (blue gi), exist on **every one of the {n} frames**. That says nothing about whether each track sits on the right person; the audits below measure that.
  Only {s['clear_n']} frames ({100 * s['clear_n'] / n:.0f}%) are *clear* (both athletes directly detected and not overlapping):
  frames {fmt(s['clear'])}. **Only on those frames is identity independently verifiable by separation.**
* Outside the clear frames the athletes overlap or the detector merges them into one box, so for ~{colour_pct:.0f}% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. These boxes are
  low-confidence (capped at 0.40), often too small, and sometimes wrap around the other athlete.
* Identity during contact rests on **gi colour alone**. There is **no ground truth** (`ground_truth.md`
  {'exists' if gt.exists() else 'is not in the repo'}), so none of the checks below is a ground-truth check.
{audit_bullets}* Pose is partial: {s['pose']['A']}/{n} frames carry a skeleton for A and {s['pose']['B']}/{n} for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## Clip notes (affect results)

* Shot cuts detected at frames: {fmt([(x, x) for x in cuts])} ({len(cuts)} cut{'s' if len(cuts) != 1 else ''}; colour-histogram jump > 0.6, so soft cuts and dissolves may be missed).
  Motion state resets at each cut. Gi colour cannot tell *people* apart across a cut: A/B keep the same colour labels but may be different athletes.
* Phone screen recording of a YouTube player: player chrome is cropped away (rows {meta['crop_y'][0]}-{meta['crop_y'][1]}); a pause/skip overlay can appear over the picture.
* The referee is meant to be excluded by colour (torso mostly dark, no gi colour) and position, the crowd by position and size. This works for a black suit and **fails when the suit reads as blue** (a navy suit, see the audit); it would also fail for a referee in white.
{clip_notes}

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
| detected | {c['A']['detected']} | {c['B']['detected']} | own detector box and pose |
| color_split | {c['A']['color_split']} | {c['B']['color_split']} | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | {c['A']['color_recovered']} | {c['B']['color_recovered']} | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | {c['A']['lost']} | {c['B']['lost']} | nothing usable; box and keypoints are null |

Pose: A {s['pose']['A']} frames ({s['pose_shared']['A']} from colour-filtered shared detections), B {s['pose']['B']} frames ({s['pose_shared']['B']}).
"""]

# ---- audits section (only if audits exist) ----
if aud:
    L.append("""## Audits (vision, model-based; not ground truth)

Each audit: reviewers over contact sheets (every third frame, at a different offset per audit, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|""")
    for k in sorted(aud):
        a = aud[k]
        L.append(f"| {k[-1]} | {a['build']} | {a['frames']} | {pf(a, 'A')} | {pf(a, 'B')} | "
                 f"{a['serious_confirmed']} / {a['serious_refuted']} / {a['serious_split']} |")
    conf = final["confirmed"]
    conf_txt = ", ".join("%d %s %s" % (x["frame"], x["athlete"], x["type"]) for x in conf) or "none"
    L.append(f"\n**Latest audit confirmed errors** ({len(conf)}): {conf_txt}. "
             f"It covered {final['frames']} of {n} frames; frames not sampled are unaudited.")
    if audit_notes:
        L.append("\n" + audit_notes)
    L.append("")

L.append(f"""## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
{fmt(s['id_risk'])} ({total(s['id_risk'])} frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: {fmt(s['lost']['A'])} ({total(s['lost']['A'])} frames)
* B: {fmt(s['lost']['B'])} ({total(s['lost']['B'])} frames)
* Runs longer than N=5 frames: A {fmt(s['lost_long']['A'])}, B {fmt(s['lost_long']['B'])}. Every lost frame carries status `lost` in the JSON and the video HUD.
* A frame listed as lost can still show the athlete (colour not found, e.g. blur or shadow): it is a tracker failure, not proof of occlusion.

**Identity from colour only (`color_split` / `color_recovered`).**
* A: {fmt(s['colour_only']['A'])}
* B: {fmt(s['colour_only']['B'])}

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: {fmt(s['low_conf']['A'])}
* B: {fmt(s['low_conf']['B'])}

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A {fmt(s['colour_conflict']['A'])}, B {fmt(s['colour_conflict']['B'])}.""")
if final:
    flagged = sorted({f for i in "AB" for a0, b0 in s["colour_conflict"][i] for f in range(a0, b0 + 1)})
    seen = {f: final["verdicts"][str(f)] for f in flagged if str(f) in final["verdicts"]}
    wrong_f = [f for f, v in seen.items() if "wrong_body" in (v["A"], v["B"])]
    unaud = len(flagged) - len(seen)
    tail = "" if unaud == 0 else f"; the other {unaud} are unaudited"
    L.append(f"Of these {len(flagged)} frames, the latest audit sampled {len(seen)} and judged {len(wrong_f)} wrong-body "
             f"({', '.join(map(str, wrong_f)) or 'none'}){tail}. This check can fire on correct IDs when an overlapping box captures the other athlete's sleeve, so it flags contamination as well as errors.")
else:
    L.append("Not audited: this check can fire on correct IDs (an overlapping box capturing the other athlete's sleeve), so treat it as a pointer, not a verdict.")

L.append(f"""
## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only {c['A']['detected']} A / {c['B']['detected']} B frames are direct detections; two athlete-like detector boxes exist on {100 * s['two_athlete_frac']:.0f}% of frames, and some of those are duplicate boxes on the same pair).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin and can wrap around the other athlete: {s.get('colour_box_bad', '?')} of {s.get('colour_box_total', '?')} sampled colour boxes ({100 * s.get('colour_box_bad', 0) / max(s.get('colour_box_total', 1), 1):.0f}%) contain more of the other athlete's colour than their own.
3. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct.
4. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. A blue gi's white patches (collar, back label) can look like A. White ad boards and logos are rejected only by the person mask.
5. Cuts are found by a colour-histogram jump (threshold 0.6); soft cuts are missed and motion state would carry over them. Colour does not identify *people* across cuts.
6. Frame rate is variable in the source; the annotated video uses {fps:.2f} fps (frames / duration).
7. Audits (where run) sample a subset of frames and are done by language-model reviewers, who can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Clip 1: exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and audit-found errors pinned to specific frames: pass. Every clip also runs the generic checks in `tests/test_generic.py` (counts, `lost` flagging, boxes on the mat, poses inside their box), which cannot see a referee being tracked as an athlete. Known open failures are pinned as strict-xfail tests (clip 2: `tests/test_clip2_known_failures.py`).
* No swaps in clear segments: passes, but covers only the {s['clear_n']} clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete-box rate ({100 * s['two_athlete_frac']:.0f}% of frames) and colour-box quality ({s.get('colour_box_bad', '?')}/{s.get('colour_box_total', '?')} bad) are **measured numbers; clip 1 has regression floors on them, not success claims**.
* Ground truth: {'present' if gt.exists() else 'ground_truth.md not found; the test is skipped, not passed'}.

## Outputs

* `{OUT}/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `{OUT}/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `{OUT}/stats.json` (and `audit_summary.json`, `audit*_results.json` where audits were run): the numbers and evidence used here.
* Regenerate: `python run_pipeline.py --clip {args.clip} --out {OUT} && python tools/summarise_audits.py --out {OUT} && python make_report.py --clip {args.clip} --out {OUT} --report {args.report}`.
""")
Path(args.report).write_text("\n".join(L))
print(args.report, "written", len("\n".join(L)))
