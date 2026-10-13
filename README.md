# Grappling athlete tracker

Tracks the two athletes in a gi-vs-gi grappling video from first frame to last and keeps their identities
stable: **A = white gi, B = blue gi**. It uses pretrained YOLO11 pose and segmentation models (no training,
CPU only), a purpose-built two-athlete tracker with colour as the main identity cue, and a learned referee
identity so the referee is never labelled as an athlete.

Results and their limits are in the reports. Read them before trusting any output:

* `report.md`: clip 1 (`test_clips/clip.mp4`, Budapest 2016, 535 frames)
* `outputs/clip2/report.md`: clip 2 (`test_clips/clip2.mp4`, Paris Grand Slam 2026 highlights, 453 frames)
* `outputs/marker_study/REPORT.md`: which visual cues identify a person (gradients, texture, pose, re-ID, …)

There is **no ground truth** for either clip. Quality figures in the reports come from vision audits by
language-model reviewers, not from measured accuracy.

## Setup

Python 3.11. A GPU is not needed.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
# optional, CPU-only machines: avoids the large CUDA wheels
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Model weights (`yolo11m-pose.pt`, `yolo11m-seg.pt`, `yolo11x-seg.pt`) are not in git. Ultralytics downloads them
from its GitHub releases the first time the pipeline runs, into the working directory (they are git-ignored).

## Check it works

```bash
python -m pytest tests        # about 1 minute
```

The tests run on the committed outputs and model caches, so no model is run. Expected result: everything passes
except one skipped test (`ground_truth.md` is absent) and 9 expected failures (`xfail`). The xfails pin a known,
unfixed bug: on clip 2, athlete B is lost on 9 frames while visible.

## Run the pipeline

```bash
python run_pipeline.py                                                   # clip 1 -> outputs/
python run_pipeline.py --clip test_clips/clip2.mp4 --out outputs/clip2   # any clip -> its own folder
```

Each run writes these files to the output folder:

* `annotated.mp4`: A in red, B in green. A solid box is a direct detection, a dashed box is derived from gi colour. The status of each athlete is shown in the top-left corner on every frame.
* `tracks.json`: per frame and per athlete: status, box, confidence, 17 COCO keypoints (or null) and pose source.
* `stats.json`: lost, low-confidence and colour-only frame ranges, plus counts.
* Caches: `raw_dets.npz`, `person_masks.npz` and `person_masks_x.npz`. When they exist, the models are not re-run. Delete them to recompute.

On a 4-CPU machine the first run of a clip takes about 35 minutes:

| stage | time |
|---|---|
| YOLO11m pose | about 4.5 min |
| YOLO11m seg | about 4 min |
| YOLO11x seg | about 25 min |

Later runs that reuse the caches take a few minutes, most of it re-encoding the annotated video.

The picture crop is detected automatically. It assumes a phone screen recording of a YouTube player, where the
picture sits in a band between black player chrome. The colour pair white vs blue is built into
`grappling/color.py`, `athletes.py`, `occlusion.py` and `tracker.py`. Other gi colours are not supported yet.

## Write a report

```bash
python tools/summarise_audits.py --out outputs/clip2      # only if audit results exist in that folder
python make_report.py --clip test_clips/clip2.mp4 --out outputs/clip2 --report outputs/clip2/report.md
```

`make_report.py` with no arguments regenerates clip 1's `report.md`. Every number in a report is computed from
`stats.json` and `audit_summary.json`. Observations that are specific to a clip come from hand-written files in
the output folder:

* `clip_notes.md`: what the clip contains.
* `audit_notes.md`: what the audits found.
* `audit_builds.json`: which build each audit measured.

### Adding a new clip

1. Put the clip in `test_clips/`.
2. Run `run_pipeline.py --clip ... --out outputs/<name>`.
3. Watch `annotated.mp4` and write `outputs/<name>/clip_notes.md`.
4. Optionally run an audit (see below).
5. Run `make_report.py`.
6. Add the new folder to `CASES` in `tests/test_generic.py` so the generic checks run on it.

## Vision audits (optional; needs Claude Code)

An audit has reviewer agents judge contact sheets of the annotated video. Every serious finding is then checked
by two skeptic agents who try to refute it.

1. Make the sheets:
   `python tools/make_audit_sheets.py SHEET_DIR OFFSET OUT_DIR [EXTRA_FRAMES]`
   * `OFFSET` is 0, 1 or 2 to sample every third frame at that offset, or -1 for targeted frames only.
   * Frames flagged in `stats.json` are always added.
2. In Claude Code, run the workflow `tools/workflows/audit_generic.js` with these arguments:
   `{scratch, repo, dir, out, clip, crop: [346, 954], regular, flagged, perGroup: 3, brief}`.
   * `regular` and `flagged` are the sheet counts printed in step 1.
   * `brief` describes the clip to the reviewers.
3. Save the array the workflow returns as `OUT_DIR/audit_results.json`, `audit2_results.json` or `audit3_results.json`.
4. Run `tools/summarise_audits.py --out OUT_DIR`.

The scripts in `tools/workflows/history/` are the clip-specific versions used for clip 1's audits. They are kept
for provenance only.

## Marker study (how identity cues were evaluated)

```bash
PYTHONPATH=. python tools/build_marker_dataset.py      # rebuilds outputs/marker_study/dataset.pkl (see note below)
PYTHONPATH=. python tools/marker_baseline.py           # colour-histogram baseline
PYTHONPATH=. python tools/summarise_marker_study.py    # regenerates outputs/marker_study/REPORT.md
```

* The committed `outputs/marker_study/dataset.pkl` is the one the study used. Rebuilding it uses today's `tracks.json`, which changed slightly after the study, so the instances can differ a little.
* `tools/marker_eval.py` and `tools/marker_fuse.py` hold the evaluation protocols.
* `tools/markers/*.py` holds the seven marker families.
* `tools/workflows/marker_study.js` is the Claude Code workflow that explored and verified them, with arguments `{repo, scratch}`.

## Layout

```
grappling/            the library
  video.py            crop detection, frame reading, fps, shot-cut detection
  detect.py           YOLO11m-pose: person boxes + 17 keypoints (cached)
  segment.py          YOLO11m-seg person masks (cached)
  segment_x.py        YOLO11x-seg second mask pass, combined with the first (cached)
  color.py            torso region and colour histograms
  athletes.py         athlete candidates (drops crowd and referee), colour class per box
  tracker.py          two-athlete tracker, merged-box handling, colour recovery, referee identity
  occlusion.py        gi-colour boxes when the detector merges or misses an athlete
  pose.py             keypoints per athlete (own detection, or colour-attributed fragments)
  render.py           annotated video
  report.py           failure-range analysis for stats.json
run_pipeline.py       end-to-end run for one clip
make_report.py        report generator
tools/                audit sheets, audit summary, marker study, Claude Code workflow scripts
tests/                milestone tests, generic per-clip checks, regressions pinned from audits
test_clips/           the two input clips
outputs/              clip 1 outputs (top level), outputs/clip2/, outputs/marker_study/
```

`run_detect.py`, `run_segment.py` and `run_segment_x.py` are the early per-stage scripts for clip 1.
`run_pipeline.py` replaces them.

## Known limitations

These are summarised here. The reports give frame ranges and evidence.

* **Contact frames.** When the athletes touch, which is most of both clips, the detector often returns one box for both. Each athlete's box is then derived from gi-colour pixels, so it is low-confidence and often partial.
* **Partial poses.** Skeletons on tangled bodies are fragments.
* **Hidden athlete.** An athlete hidden behind the other can be lost. Clip 2 frames 137, 439-443 and 450-452 are examples, because the segmenter does not see that athlete.
* **Fixed colours.** Gi colours are fixed to white vs blue. A referee dressed in white would not be excluded.
* **No ground truth.** The `ground_truth.md` check is skipped, not passed.
