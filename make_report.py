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
gt = OUT / "ground_truth.json"
ge = json.load(open(OUT / "gt_eval.json"))["summary"] if gt.exists() and (OUT / "gt_eval.json").exists() else None


def pct(x):
    return f"{100 * x:.0f}%"


def ci(x):
    return f"{100 * x[0]:.0f}-{100 * x[1]:.0f}%"


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


# ---- ground truth (hand labels scored by tools/eval_ground_truth.py) ----
if ge:
    gA, gB = ge["athletes"]["A"], ge["athletes"]["B"]
    gt_bullet = (f"* **Ground truth** (hand-labelled boxes on {ge['regular_frames']} regularly sampled frames, every 7th): the track sits on the right person "
                 f"for A on {gA['right_person']}/{gA['decided']} visible frames ({pct(gA['right_person_rate'])}, 95% CI {ci(gA['right_person_ci95'])}) and "
                 f"for B on {gB['right_person']}/{gB['decided']} ({pct(gB['right_person_rate'])}, {ci(gB['right_person_ci95'])}); "
                 f"wrong-body A {gA['counts']['wrong_body']}, B {gB['counts']['wrong_body']}; box covering both A {gA['counts']['covers_both']}, B {gB['counts']['covers_both']}; identity swaps on {len(ge['swap_frames'])} frame{'' if len(ge['swap_frames']) == 1 else 's'}. "
                 f"Box quality (IoU >= 0.5 with the labelled box): A {pct(gA['correct_rate'])}, B {pct(gB['correct_rate'])}. Details in the Ground truth section.")
else:
    gt_bullet = (f"* There is **no ground truth** for this clip (`{gt}` is absent; label it with `tools/gt_label.py`), "
                 "so none of the checks below is a ground-truth check.")

# ---- verdict bullets about audits ----
if final:
    v = final["per_frame"]
    wrong = v["A"].get("wrong_body", 0) + v["B"].get("wrong_body", 0)
    fr = final["frames"]
    audit_bullets = (
        "* The evidence for quality is **vision audits**: model reviewers judged contact sheets of the annotated video and a second round of "
        "skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the reviewers are language models. "
        "Results are in the Audits section.\n"
        f"* Latest random-sample audit ({final_key[-1]}, {fr} of {n} frames sampled; the table below says which build it measured, and later targeted re-checks are in the audit notes): {wrong} wrong_body verdicts; "
        f"A judged correct on {v['A'].get('correct', 0)} and B on {v['B'].get('correct', 0)} of {fr} frames "
        f"({100 * v['A'].get('correct', 0) / fr:.0f}% / {100 * v['B'].get('correct', 0) / fr:.0f}%); the remaining verdicts are partial boxes "
        f"({v['A'].get('partial', 0)} A, {v['B'].get('partial', 0)} B), lost-but-visible "
        f"({v['A'].get('lost_but_visible', 0)} A, {v['B'].get('lost_but_visible', 0)} B) and unclear "
        f"({v['A'].get('unclear', 0)} A, {v['B'].get('unclear', 0)} B). Sampling means this is an estimate, not a count over all frames.\n")
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
  **not a detection**: it is built from that athlete's gi-coloured pixels inside a person mask, then grown over the rest of the
  body (head, hands, feet) by the person-mask pixels nearer that gi than the other one. These boxes are low-confidence (capped at 0.40);
  when the athletes are wrapped together they often cover both, and the growth can overreach onto the other athlete.
* Identity during contact rests on **gi colour alone**.
{gt_bullet}
{audit_bullets}* Pose is partial: {s['pose']['A']}/{n} frames carry a skeleton for A and {s['pose']['B']}/{n} for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## Clip notes (affect results)

* Shot cuts detected at frames: {fmt([(x, x) for x in cuts])} ({len(cuts)} cut{'s' if len(cuts) != 1 else ''}; colour-histogram jump > 0.6, so soft cuts and dissolves may be missed).
  Motion state resets at each cut. Gi colour cannot tell *people* apart across a cut: A/B keep the same colour labels but may be different athletes.
* Phone screen recording of a YouTube player: player chrome is cropped away (rows {meta['crop_y'][0]}-{meta['crop_y'][1]}); a pause/skip overlay can appear over the picture.
* The referee is excluded by colour (torso mostly dark, no gi colour) and position, the crowd by position and size. A suit that reads as gi colour (clip 2's navy suit) is caught by a referee appearance gallery learned online from the unambiguous dark detections; that needs the referee to have been seen unambiguously first, and it would fail for a referee in white or for one never seen that way.
{clip_notes}

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
| detected | {c['A']['detected']} | {c['B']['detected']} | own detector box and pose |
| color_split | {c['A']['color_split']} | {c['B']['color_split']} | detector merged the pair; box from gi-colour pixels, grown over the body (conf <= 0.40) |
| color_recovered | {c['A']['color_recovered']} | {c['B']['color_recovered']} | detector missed the athlete; box from gi-colour pixels, grown over the body (conf <= 0.40) |
| occluded | {c['A'].get('occluded', 0)} | {c['B'].get('occluded', 0)} | hidden behind the other athlete: no evidence, last box carried (up to 10 frames, conf 0.05) |
| lost | {c['A']['lost']} | {c['B']['lost']} | nothing usable; box and keypoints are null |

Pose: A {s['pose']['A']} frames ({s['pose_shared']['A']} from colour-filtered shared detections), B {s['pose']['B']} frames ({s['pose_shared']['B']}).
"""]

# ---- ground-truth section (only if labels have been scored) ----
if ge:
    from tools.eval_ground_truth import ID_MARGIN, VERDICTS as VERD
    L.append(f"""## Ground truth (hand labels)

A person drew a box around each athlete's whole visible body, or marked them hidden, on the raw frames (never seeing the tracker's output), using
`tools/gt_label.py`. `tools/eval_ground_truth.py` scores each track box by IoU with that athlete's box ("own") and the other athlete's ("other"):
correct = own IoU >= 0.5; partial = right person, poor box (0 < own IoU < 0.5, own >= other); wrong_body = other IoU > own IoU; off_target = on
neither athlete; lost_but_visible = no track box for a visible athlete; hidden_ok / box_on_hidden = athlete labelled hidden, without / with a track box.
When the athletes are tangled their labelled boxes overlap heavily and IoU cannot say which one a track box is on, so wherever the box touches both
and favours its own by less than {ID_MARGIN} IoU, a person answered which athlete it is on (`tools/gt_identity.py`, which does show the track box):
right keeps the IoU verdict, wrong = wrong_body, both = covers_both; unanswered = undecided, left out of the rates.
Colour-derived track boxes cover gi pixels only, so they score lower IoU than a whole-body label even on the right person: *partial* is their expected verdict.

Regular sample: {ge['regular_frames']} frames (every 7th){'; unusable frames skipped: ' + ', '.join(map(str, ge['unusable_frames'])) if ge['unusable_frames'] else ''}. One labeller, no second-labeller agreement check.

| verdict | A (white) | B (blue) |
|---|---|---|""")
    for v in VERD:
        L.append(f"| {v} | {gA['counts'][v]} | {gB['counts'][v]} |")
    L.append(f"""
Identity checked by a person: A {gA['id_checked']}, B {gB['id_checked']} of the regular frames; undecided (left out of the rates): A {gA['counts']['undecided']}, B {gB['counts']['undecided']}.

| on visible, decided frames | A | B |
|---|---|---|
| right person (correct + partial) | {gA['right_person']}/{gA['decided']} ({pct(gA['right_person_rate'])}, CI {ci(gA['right_person_ci95'])}) | {gB['right_person']}/{gB['decided']} ({pct(gB['right_person_rate'])}, CI {ci(gB['right_person_ci95'])}) |
| correct box (IoU >= 0.5) | {pct(gA['correct_rate'])} (CI {ci(gA['correct_ci95'])}) | {pct(gB['correct_rate'])} (CI {ci(gB['correct_ci95'])}) |
| box covers both athletes | {pct(gA['covers_both_rate'])} | {pct(gB['covers_both_rate'])} |
| wrong body | {pct(gA['wrong_body_rate'])} (CI {ci(gA['wrong_body_ci95'])}) | {pct(gB['wrong_body_rate'])} (CI {ci(gB['wrong_body_ci95'])}) |
| mean IoU (all visible frames) | {gA['mean_iou_visible']:.2f} | {gB['mean_iou_visible']:.2f} |""")
    for g, name in (("detected", "direct detections"), ("colour", "colour-derived boxes")):
        a, b = gA["by_status"][g], gB["by_status"][g]
        L.append(f"| {name}: right person / correct / mean IoU | {a['right_person']}/{a['n']} / {a['correct']}/{a['n']} / {'-' if a['mean_iou'] is None else format(a['mean_iou'], '.2f')} "
                 f"| {b['right_person']}/{b['n']} / {b['correct']}/{b['n']} / {'-' if b['mean_iou'] is None else format(b['mean_iou'], '.2f')} |")
    errs = "; ".join(f"{w}: " + (", ".join(f"{e['frame']} {e['verdict']}" + (" (targeted)" if e['kind'] == 'targeted' else "")
                                          for e in ge["errors"][w]) or "none") for w in "AB")
    L.append(f"\n**Errors on labelled frames:** {errs}. **Identity swaps:** {', '.join(map(str, ge['swap_frames'])) or 'none'}.")
    if ge["targeted"]:
        L.append("**Targeted frames** (chosen because audits or tests found failures there; not a sample, no rate): "
                 + ", ".join(f"{t['frame']} A {t['A']} / B {t['B']}" for t in ge["targeted"]) + ".")
    L.append("")

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

**Hidden behind the other athlete (`occluded`: box carried from the last sighting, not observed).** A: {fmt(s.get('occluded', {}).get('A', []))}; B: {fmt(s.get('occluded', {}).get('B', []))}.

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
2. Colour-derived boxes start from gi pixels and are grown over the body by distance to each athlete's gi; where the athletes are wrapped together the grown box often covers both, and it can overreach onto the other athlete where a limb of one lies nearer the other's gi (see the ground-truth errors). The underlying gi box can wrap around the other athlete: {s.get('colour_box_bad', '?')} of {s.get('colour_box_total', '?')} sampled gi boxes ({100 * s.get('colour_box_bad', 0) / max(s.get('colour_box_total', 1), 1):.0f}%) contain more of the other athlete's colour than their own.
3. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct.
4. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. A blue gi's white patches (collar, back label) can look like A. White ad boards and logos are rejected only by the person mask.
5. Cuts are found by a colour-histogram jump (threshold 0.6); soft cuts are missed and motion state would carry over them. Colour does not identify *people* across cuts.
6. Frame rate is variable in the source; the annotated video uses {fps:.2f} fps (frames / duration).
7. Audits (where run) sample a subset of frames and are done by language-model reviewers, who can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Clip 1: exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and audit-found errors pinned to specific frames: pass. Every clip also runs the generic checks in `tests/test_generic.py` (counts, `lost` flagging, boxes on the mat, poses inside their box), which cannot see a referee being tracked as an athlete. Failures found by audits are pinned in `tests/test_audit_regressions.py` and `tests/test_clip2_regressions.py` (clip 2's B-lost-while-visible frames now carry an `occluded` box).
* No swaps in clear segments: passes, but covers only the {s['clear_n']} clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete-box rate ({100 * s['two_athlete_frac']:.0f}% of frames) and colour-box quality ({s.get('colour_box_bad', '?')}/{s.get('colour_box_total', '?')} bad) are **measured numbers; clip 1 has regression floors on them, not success claims**.
* Ground truth: {'labels in `' + str(gt) + '` are checked for completeness and scored; the numbers are measurements, with no pass/fail floor yet' if ge else '`' + str(gt) + '` not found; the test is skipped, not passed'}.

## Outputs

* `{OUT}/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `{OUT}/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `{OUT}/stats.json` (and `audit_summary.json`, `audit*_results.json` where audits were run): the numbers and evidence used here.
* Regenerate: `python run_pipeline.py --clip {args.clip} --out {OUT} && python tools/summarise_audits.py --out {OUT} && python tools/eval_ground_truth.py --out {OUT} && python make_report.py --clip {args.clip} --out {OUT} --report {args.report}`.
""")
Path(args.report).write_text("\n".join(L))
print(args.report, "written", len("\n".join(L)))
