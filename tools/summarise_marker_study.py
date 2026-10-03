"""Write outputs/marker_study/REPORT.md from results.json (explorer + verifier output) and a fresh fusion run."""
import importlib
import json
import io
import contextlib
from tools.marker_fuse import fused_eval
from tools.marker_baseline import hsv_torso

R = json.load(open("outputs/marker_study/results.json"))

cp = importlib.import_module("tools.markers.colour_profile").FEATURES
pa = importlib.import_module("tools.markers.parts_markers").FEATURES
ps = importlib.import_module("tools.markers.pose_shape").FEATURES
base = ("hsv", hsv_torso, "chi2", "mean", 1.0)
lab = ("labprofile", cp["Lab_K6_med_gradboth"], "l2", "mean", 1.0)
legs = ("parts", pa["axis_quad_belt_legs_hsv"], "chi2", "nn", 1.0)
pose = ("pose", ps["pose_box"], "l2", "nn", 1.0)
combos = [("HSV torso histogram alone (baseline)", [base]), ("+ Lab colour profile", [base, lab]),
          ("+ legs/belt parts", [base, legs]), ("+ Lab profile + parts", [base, lab, legs]),
          ("+ pose/posture", [base, pose]), ("+ Lab + parts + pose", [base, lab, legs, pose])]
fus = []
for name, ms in combos:
    with contextlib.redirect_stdout(io.StringIO()):
        r = fused_eval(ms, name)
    fus.append((name, r["block"], r["start"]))

L = ["# Marker study: which cues identify a unique body?", "",
     "Question: can gradients and other markers tell athlete A (white gi), athlete B (blue gi) and the referee R apart, "
     "especially where gi colour alone is ambiguous (clip 2's referee wears a navy suit that reads as blue)?", "",
     "## Method", "",
     "* **Dataset** (`tools/build_marker_dataset.py`, 324 person crops): A/B = directly detected boxes in frames where the vision audit judged that athlete correct "
     "(clip 1: 78 A, 71 B; clip 2: 57 A, 54 B). R = any other confident on-mat person in frames where both athletes are verified, labelled by exclusion with **no colour used** "
     "(clip 1: 22, clip 2: 42). An earlier labelling of R by 'mostly dark' was discarded because it leaked into brightness features.",
     "* **Protocols** (`tools/marker_eval.py`): nearest-prototype classification into A/B/R per clip. *block* = leave-one-24-frame-block-out with a one-block buffer; "
     "*start* = gallery from the first 30% of the clip, test on the rest (how a tracker would use it).",
     "* Seven families were explored by independent agents under fixed rules: tune on clip 1 only, report clip 2 as held-out, at most 15 variants, list all variants. "
     "A second agent per family re-ran the best variant and audited it for leakage.", "",
     "## Results (balanced accuracy over A/B/R; chance = 0.33)", "",
     "| family | best variant (clip-1 tuned) | block | start | clip 1 | clip 2 | contact | A-vs-B | B-vs-R (clip 2) | ms/crop | verifier |",
     "|---|---|---|---|---|---|---|---|---|---|---|",
     "| **baseline: HSV torso histogram** | chi2, class means | 0.953 | 0.967 | 0.972 | 0.940 | 0.954 | 0.954 | 0.955 | <1 | (reference) |"]
for r in R:
    e, v, b = r["explore"], r["verify"], r["explore"]["best"]
    L.append(f"| {r['family']} | {b['name']} | {b['block_bal']:.3f} | {b['start_bal']:.3f} | {b['clip1_block']:.3f} | {b['clip2_block']:.3f} | "
             f"{b.get('contact_acc_block', float('nan')):.3f} | {b.get('A_vs_B_block', float('nan')):.3f} | {b.get('B_vs_R_clip2_block', float('nan')):.3f} | "
             f"{e.get('ms_per_instance', float('nan')):.1f} | {v['verdict'] if v else 'n/a'}{' (leakage flagged)' if v and v['leakage_found'] else ''} |")
L += ["", "No family beat the baseline. All seven verdicts were `do_not_use` for replacing colour; every best variant reproduced in the verifier's independent re-run.", "",
      "## What each family showed", ""]
for r in R:
    e, v = r["explore"], r["verify"]
    L.append(f"* **{r['family']}** — {e['adds_to_baseline']}")
    L.append(f"  * Failure modes: {e['failure_modes'][:600]}")
    if v:
        L.append(f"  * Verifier ({v['verdict']}): {v['reasoning'][:500]}")
L += ["", "## Fusion of the complementary markers (distance-level, `tools/marker_fuse.py`)", "",
      "| markers | block bal | block contact | block clip 2 | block A-vs-B | start bal |", "|---|---|---|---|---|---|"]
for name, b, s in fus:
    L.append(f"| {name} | {b['bal']:.3f} | {b['contact']:.3f} | {b['clip2']:.3f} | {b['A_vs_B']:.3f} | {s['bal']:.3f} |")
L += ["", "Differences of ±0.01 are 3 or 4 of 324 instances, i.e. noise. Fusion helps A-vs-B a little (0.954 to 0.981) and hurts contact accuracy (0.954 to about 0.92).", "",
      "## Conclusion", "",
      "1. **Gradients (HOG, edge/orientation/magnitude statistics) and texture (LBP/GLCM/Gabor) do not identify a unique body here.** The best gradient variant scores 0.66 and the best texture variant 0.76 (many variants sit near chance, 0.33); they mostly "
      "encode blur, exposure and scale, and in contact they pool both bodies' gradients. Fused with colour they are inert or harmful.",
      "2. **The colour histogram, matched against a per-person gallery that includes the referee, already separates the referee from B (95 to 98%).** The information was never missing. "
      "The tracker failed because it had no referee identity and decided by colour-class thresholds and priors.",
      "3. **Other markers add narrow things only:** legs/belt colour helps A-vs-B (0.973 vs 0.954); posture/body proportions separate the referee without any colour (0.92) but cannot tell A from B; "
      "a pretrained re-ID embedding reaches 0.905 at 254 ms/crop, below colour.",
      "4. **So the fix was structural, not a new marker:** the tracker now learns the referee's appearance online from unambiguous strict-dark detections and sets aside any candidate "
      "that matches that gallery clearly better than A or B (`grappling/tracker.py`, `REF_MARGIN`). Clip 1 output is unchanged (0 frames differ); the eight audited referee errors on clip 2 no longer occur.", "",
      "## Limits", "",
      "* 324 crops from two clips, one referee per clip. Per-person galleries are built within a clip; nothing here shows cross-clip re-identification.",
      "* A/B instances are direct detections in audited-correct frames, which is easier than the contested frames where the tracker fails.",
      "* Marker hyperparameters were tuned on clip 1 only; some selected variants generalise poorly to clip 2 (see the table).",
      "* Agents wrote the family code (`tools/markers/*.py`); it was re-run by a second agent but not hand-reviewed line by line."]
open("outputs/marker_study/REPORT.md", "w").write("\n".join(L))
print("written", len("\n".join(L)))
for name, b, s in fus:
    print(f"{name:42s} block {b['bal']:.3f} contact {b['contact']:.3f} start {s['bal']:.3f}")
