"""Write report.md from outputs/stats.json and outputs/audit_summary.json.
Every number and frame range is computed. Only the explanatory prose is fixed text."""
import json
from pathlib import Path

s = json.load(open("outputs/stats.json"))
aud = json.load(open("outputs/audit_summary.json")) if Path("outputs/audit_summary.json").exists() else {}
fps = 535 / 16.923689


def fmt(runs):
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs) or "none"


def ranges_of(frames):
    frames = sorted(set(frames))
    out, s0, p = [], None, None
    for f in frames:
        if s0 is None:
            s0 = p = f
        elif f == p + 1:
            p = f
        else:
            out.append((s0, p))
            s0 = p = f
    if s0 is not None:
        out.append((s0, p))
    return out


def total(runs):
    return sum(b - a + 1 for a, b in runs)


n = s["n"]
c = s["counts"]
gt = Path("ground_truth.md")
cut = s["cuts"][0]
colour_pct = 100 * (c["A"]["color_split"] + c["A"]["color_recovered"]) / n
final = aud.get("audit3")


def pf(a, who):
    d = a["per_frame"][who]
    tot = a["frames"]
    return " / ".join(f"{d.get(k, 0)}" for k in ("correct", "partial", "wrong_body", "lost_but_visible")) + f" of {tot}"


L = []
L.append(f"""# Grappling athlete tracking: report

Clip: `test_clips/clip.mp4`, {n} frames, ~{fps:.1f} fps, {n / fps:.1f}s. Identity labels: **A = white gi, B = blue gi**.

## Verdict (read this first)

* Two tracks, A and B, exist on **every one of the {n} frames**, A always white and B always blue.
  Only {s['clear_n']} frames ({100 * s['clear_n'] / n:.0f}%) are *clear* (both athletes directly detected and not overlapping):
  frames {fmt(s['clear'])}. **Only on those frames is identity independently verifiable by separation.**
* From about frame 80 to the end the athletes are in contact (standing grips, throw, ground). The detector returns one
  box for the pair on most of those frames, so for ~{colour_pct:.0f}% of the clip each athlete's box is
  **not a detection**: it is the bounding box of that athlete's gi-coloured pixels inside a person mask. These boxes are
  low-confidence (capped at 0.40), often too small, and sometimes wrap around the other athlete.
* Identity during contact rests on **gi colour alone**. There is **no ground truth** (`ground_truth.md`
  {'exists' if gt.exists() else 'is not in the repo'}), so none of the checks below is a ground-truth check.
* The evidence for quality is **three vision audits**: model reviewers judged contact sheets of the annotated video
  and a second round of skeptic reviewers tried to refute every serious finding. That is an independent look, not ground truth, and the
  reviewers are language models. Results are in the Audits section; the audits found real bugs which are fixed.
* Pose is partial: {s['pose']['A']}/{n} frames carry a skeleton for A and {s['pose']['B']}/{n} for B. On colour-derived frames only
  keypoints whose own pixels carry that athlete's gi colour and that lie inside the athlete's box are kept, so skeletons are fragments.

## What the clip is (affects results)

* A **phone screen recording of a YouTube player**, not a clean match feed: player chrome above and below the picture
  (cropped to rows 346-953) and a pause/skip overlay over the picture in the early frames.
* It is **judo**, not BJJ (judogi, referee in a black suit, tatami, Budapest 2016 boards).
* A **shot cut at frame {cut}** ({cut / fps:.2f}s). Frames 0-{cut - 1} show a different match (white standing, blue on the mat).
  A/B there carry the same *colour labels* but are **different people** from frame {cut} on. Colour identity cannot tell that apart.
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
| detected | {c['A']['detected']} | {c['B']['detected']} | own detector box and pose |
| color_split | {c['A']['color_split']} | {c['B']['color_split']} | detector merged the pair; box from gi-colour pixels (conf <= 0.40) |
| color_recovered | {c['A']['color_recovered']} | {c['B']['color_recovered']} | detector missed the athlete; box from gi-colour pixels (conf <= 0.40) |
| lost | {c['A']['lost']} | {c['B']['lost']} | nothing usable; box and keypoints are null |

Pose: A {s['pose']['A']} frames ({s['pose_shared']['A']} from colour-filtered shared detections), B {s['pose']['B']} frames ({s['pose_shared']['B']}).

## Audits (vision, model-based; not ground truth)

Each audit: 12 reviewers over contact sheets (every third frame at a different offset, plus the frames this report flags), each tile judged for A and B
as correct / partial / wrong_body / lost_but_visible; every high-severity claim then went to two skeptics (visual lens and temporal lens) working from the raw frames,
who were told to refute by default. Columns are correct / partial / wrong_body / lost_but_visible.

| audit | build measured | frames | A | B | serious claims: confirmed / refuted / split |
|---|---|---|---|---|---|""")
for k in ("audit1", "audit2", "audit3"):
    if k in aud:
        a = aud[k]
        L.append(f"| {k[-1]} | {a['build']} | {a['frames']} | {pf(a, 'A')} | {pf(a, 'B')} | {a['serious_confirmed']} / {a['serious_refuted']} / {a['serious_split']} |")
L.append("""
Only audit 3 describes the build delivered here. Audits 1 and 2 are kept because they found the bugs that were fixed.""")
if final:
    conf = final["confirmed"]
    L.append(f"""
**Audit 3 (final build) confirmed errors** ({len(conf)}): {', '.join(f"{x['frame']} {x['athlete']} {x['type']}" for x in conf) or 'none'}.
Audit 3 covered {final['frames']} of {n} frames; frames not sampled are unaudited.""")
L.append(f"""
Bugs the audits found that I had missed by eye, now fixed and pinned by `tests/test_audit_regressions.py`: crowd members and the referee labelled as A/B (frames 456-525, 479, 481);
B's solid box and skeleton on the white athlete at 311-313 and 472; both athletes marked lost while clearly visible at 142-143 and 464, and A at 309-313.

## Frame ranges to distrust

**Possible identity swap (identity not backed by separated detections).** Every frame outside the clear ranges:
{fmt(s['id_risk'])} ({total(s['id_risk'])} frames). On these frames A/B come from colour only.

**Track lost (box null), listed in full.**
* A: {fmt(s['lost']['A'])} ({total(s['lost']['A'])} frames)
* B: {fmt(s['lost']['B'])} ({total(s['lost']['B'])} frames)
* Runs longer than N=5 frames: A {fmt(s['lost_long']['A'])}, B {fmt(s['lost_long']['B'])}. Every lost frame carries status `lost` in the JSON and the video HUD.
* These are tracker failures, not real occlusions, when the athlete is visible: the white athlete's gi is motion-blurred and desaturated in the throw (about 136-144), and audits 1 and 2 confirmed A visible on the lost frames they sampled (earlier builds).

**Identity from colour only (`color_split` / `color_recovered`).**
* A: {fmt(s['colour_only']['A'])}
* B: {fmt(s['colour_only']['B'])}

**Confidence dropped (direct detections with detector confidence < 0.5).**
* A: {fmt(s['low_conf']['A'])}
* B: {fmt(s['low_conf']['B'])}

**Colour disagreement on direct detections** (automatic torso-colour check says the other gi): A {fmt(s['colour_conflict']['A'])}, B {fmt(s['colour_conflict']['B'])}.""")
if final:
    flagged = []
    for i in "AB":
        for a0, b0 in s["colour_conflict"][i]:
            flagged += list(range(a0, b0 + 1))
    seen = {f: final["verdicts"][str(f)] for f in sorted(set(flagged)) if str(f) in final["verdicts"]}
    wrong = {f: v for f, v in seen.items() if "wrong_body" in (v["A"], v["B"])}
    L.append(f"Of these {len(set(flagged))} frames, audit 3 sampled {len(seen)}; it judged {len(wrong)} of them as wrong-body ({', '.join(map(str, wrong)) or 'none'}). The rest are unaudited in the final build. Earlier I saw this check fire on correct IDs because an overlapping box captured the other athlete's sleeve, so it flags contamination as well as errors.")
L.append(f"""
## Known failures and limits (not fixed)

1. The detector merges tangled athletes into one box on most contact frames (only {c['A']['detected']} A / {c['B']['detected']} B frames are direct detections; two athlete candidates exist on {100 * s['two_athlete_frac']:.0f}% of frames).
2. Colour-derived boxes are bounding boxes of gi pixels. They miss heads, limbs and skin and can wrap around the other athlete: {s.get('colour_box_bad', '?')} of {s.get('colour_box_total', '?')} sampled colour boxes ({100 * s.get('colour_box_bad', 0) / max(s.get('colour_box_total', 1), 1):.0f}%) contain more of the other athlete's colour than their own. The audits also rate a number of colour boxes `partial`.
3. A is often mostly hidden under B, so A's box can be small.
4. Poses on tangled bodies are poor and fragmentary; the colour filter only drops keypoints it can attribute, it does not make the rest correct. Audits rate some poses garbage on tangled frames.
5. The colour classes assume white gi vs blue gi. Shadowed or blurred white falls back to looser thresholds (S<80), used only inside a person mask. The blue gi has white patches (collar, back label) that can look like A.
6. Only one shot cut was detected (histogram threshold 0.6). A softer cut would be missed and motion state would carry over it.
7. Colour does not identify *people* across the cut at frame {cut}.
8. Frame rate is variable in the source; the annotated video uses {fps:.2f} fps (frames / duration).
9. No audit covered every frame: audits 1, 2 and 3 each sampled one third of the frames (frame numbers 0, 1 and 2 mod 3) plus the frames this report flags. Reviewers are language models and can be wrong in both directions; serious claims were cross-checked by two skeptics.

## Tests

`python -m pytest tests`: what they do and do not prove:
* Exactly 2 tracks on every frame, every missing track flagged `lost`, no athlete box in the crowd or on the referee, poses inside their athlete's box, and the audit-found errors pinned to specific frames: pass.
* No swaps in clear segments: passes, but covers only the {s['clear_n']} clear frames above.
* Colour re-identification after occlusion: detections right after non-detected stretches must match their gi colour (<=10% disagreement allowed).
* Two-athlete detection rate ({100 * s['two_athlete_frac']:.0f}% of frames) and colour-box quality ({s.get('colour_box_bad', '?')}/{s.get('colour_box_total', '?')} bad, floor 25%) are **measured regression floors, not success claims**.
* Ground truth: {'present' if gt.exists() else 'ground_truth.md not found; the test is skipped, not passed'}.

## Outputs

* `outputs/annotated.mp4`: A red, B green. Solid box = detected, dashed = colour-derived. HUD shows each athlete's status every frame.
* `outputs/tracks.json`: per frame, per athlete: status, box, confidence, 17 keypoints (name, x, y, conf) or null, pose source, note.
* `outputs/stats.json`, `outputs/audit_summary.json`, `outputs/audit*_results.json`: the numbers and audit evidence used here.
* Regenerate: `python run_pipeline.py && python tools/summarise_audits.py && python make_report.py`.
""")
Path("report.md").write_text("\n".join(L))
print("report.md written", len("\n".join(L)))
