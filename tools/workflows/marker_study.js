export const meta = {
  name: 'marker-study',
  description: 'Evaluate appearance/gradient/shape/context markers for telling A, B and the referee apart; verify each family for leakage',
  phases: [
    { title: 'Explore', detail: '7 marker families, each tuned on clip 1 and reported on held-out clip 2' },
    { title: 'Verify', detail: 'independent re-run plus leakage audit per family' },
  ],
}

// args: { repo: absolute repo path, scratch: a writable temp directory for the agents }
const REPO = args.repo
const SCRATCH = args.scratch

const FAMILIES = [
  { key: 'gradient_hog', brief: 'IMAGE-GRADIENT markers: HOG (try cell sizes / orientation bins / block norms) on the whole-body crop masked to person pixels and separately on torso and legs; gradient-magnitude and gradient-orientation histograms (Sobel/Scharr); edge density and fold patterns of a gi versus a smooth suit. Try cv2.HOGDescriptor and/or skimage if installed.' },
  { key: 'colour_profile', brief: 'VERTICAL COLOUR PROFILE / colour gradient down the body: per-horizontal-band statistics of colour over person-mask pixels (HSV, Lab, chromaticity), and band-to-band colour gradients. Prefer brightness-invariant choices (hue/saturation, Lab a*b*, chromaticity). Try different band counts.' },
  { key: 'texture', brief: 'TEXTURE markers: LBP histograms (uniform, several radii), Gabor-bank energies, local variance, gradient-energy per body part, co-occurrence statistics (skimage graycomatrix if installed). Gi weave/folds/grip creases versus smooth suit cloth.' },
  { key: 'pose_shape', brief: 'BODY-SHAPE BIOMETRICS from the 17 COCO keypoints (inst["kp"], x,y,conf) and the box: limb-length ratios normalised by torso length, shoulder/hip width ratios, head size, box aspect, posture (crouched vs upright). Use only keypoints with conf>0.3 and handle missing ones (return None if too few). Beware that pose is unreliable when bodies overlap.' },
  { key: 'parts_markers', brief: 'PART-BASED COLOUR and OTHER MARKERS, placed with the keypoints: head/hair/skin region (above the shoulders), belt line near the hips, legs versus jacket (separate histograms), collar / back-patch regions, shirt-and-tie region. Compare each part and combinations of parts; report which part carries the discriminative signal.' },
  { key: 'deep_embedding', brief: 'PRETRAINED OPEN-SOURCE EMBEDDING (no training, CPU): find any model whose weights can actually be downloaded in this sandbox. Candidates: ultralytics classification weights (yolo11n-cls.pt etc. via GitHub releases, which have worked here), torchvision weights, torchreid/OSNet, timm, open_clip. Use penultimate-layer features of the person crop (masked or not). If a family of weights cannot be fetched, report EXACTLY what failed (URL, error) and move on; do not fake results.' },
  { key: 'context_motion', brief: 'NON-APPEARANCE markers: stationarity/motion (displacement of that person across neighbouring frames, using the raw detections in outputs/raw_dets.npz for clip1 and outputs/clip2/raw_dets.npz for clip2, matched by IoU), box height relative to the frame, aspect, distance to / contact with the other labelled people. HARD RULE: do not encode absolute position or time in a way that lets the classifier memorise segments; use relative or invariant quantities and justify each one.' },
]

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    family: { type: 'string' },
    file: { type: 'string' },
    best: {
      type: 'object',
      properties: {
        name: { type: 'string' }, metric: { type: 'string' }, mode: { type: 'string' },
        block_bal: { type: 'number' }, start_bal: { type: 'number' }, contact_acc_block: { type: 'number' },
        A_vs_B_block: { type: 'number' }, B_vs_R_clip2_block: { type: 'number' },
        clip1_block: { type: 'number' }, clip2_block: { type: 'number' }, abstain: { type: 'integer' },
      },
      required: ['name', 'metric', 'mode', 'block_bal', 'start_bal', 'clip1_block', 'clip2_block'],
    },
    variants_tried: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, block_bal: { type: 'number' }, start_bal: { type: 'number' }, clip2_block: { type: 'number' } }, required: ['name', 'block_bal'] } },
    ms_per_instance: { type: 'number' },
    beats_baseline: { type: 'boolean' },
    adds_to_baseline: { type: 'string' },
    failure_modes: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['family', 'file', 'best', 'variants_tried', 'beats_baseline', 'failure_modes'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    reproduced: { type: 'boolean' },
    block_bal: { type: 'number' }, start_bal: { type: 'number' }, clip2_block: { type: 'number' },
    leakage_found: { type: 'boolean' },
    leakage_detail: { type: 'string' },
    verdict: { type: 'string', enum: ['use', 'use_with_caveats', 'do_not_use'] },
    reasoning: { type: 'string' },
  },
  required: ['reproduced', 'leakage_found', 'verdict', 'reasoning'],
}

const SETUP = `You are evaluating appearance markers for a judo two-athlete tracker. Work in ${REPO}.
GOAL: find image-based (or geometric) markers that identify a UNIQUE BODY, i.e. tell athlete A (white gi), athlete B (blue gi) and the referee R (dark suit) apart, robustly when bodies overlap. On a second clip the referee wears a navy suit that the current colour thresholds mistake for the blue gi.

Read first: tools/marker_eval.py (the harness and its protocols), tools/marker_baseline.py (the current baseline), tools/build_marker_dataset.py (how labels were made).
Dataset: outputs/marker_study/dataset.pkl via tools.marker_eval.load(): 324 instances (clip1: 78 A, 71 B, 22 R; clip2: 57 A, 54 B, 42 R) (A/B on directly detected boxes in frames the vision audit judged correct; R = any other confident on-mat person in frames where both athletes are verified, labelled WITHOUT colour). Each instance: clip, frame, label, box, kp (17x3), conf, crop=(BGR 192xW image, person mask), contact (bodies overlap), segment.
Run python from the repo root as: PYTHONPATH=. python -c "from tools.marker_eval import evaluate; ..."  (evaluate(feature_fn, name, metric='cos'|'l1'|'l2'|'chi2', mode='mean'|'nn')). A feature_fn(inst) returns a 1-D vector or None (abstain).
BASELINE to beat (HSV torso histogram, chi2, mean prototypes): block bal_acc 0.953, start 0.967, contact 0.954, A-vs-B 0.954, B-vs-R(clip2) 0.955; clip1 0.972, clip2 0.940. Brightness-free hue/sat histogram: block 0.974, start 0.953.
You may read frames with grappling.video.read_frames('test_clips/clip.mp4' or 'test_clips/clip2.mp4') if you need full-frame context; the crops are usually enough.

PROTOCOL RULES (to avoid fooling ourselves):
- Tune hyperparameters using clip1 numbers ONLY; clip2 is the HELD-OUT report. Never pick a variant because it scored better on clip2.
- Try at most 15 variants. List ALL of them in variants_tried, including the bad ones. Do not report only the winner.
- No training of models. Pretrained open-source models are allowed if their weights can really be downloaded here. CPU only.
- Do not use anything that identifies the segment/time (absolute box position, frame number, crop size/width, image position). Colour-threshold features that mirror how R was labelled are not an issue (R was labelled by exclusion, not colour) but say so if you use darkness.
- Write your feature code ONLY in tools/markers/<family>.py (create tools/markers/__init__.py if missing). Do not modify any other repo file. Temporary files go under ${SCRATCH}/.
- Report honestly. If your family does not beat the baseline, say so; that is a valid result. State what it ADDS (e.g. helps only A-vs-B under contact).
- Report ms per instance for your best variant.`

const results = await pipeline(
  FAMILIES,
  f => agent(
    `${SETUP}\n\nYOUR FAMILY: ${f.key}\n${f.brief}\n\nImplement the variants in tools/markers/${f.key}.py (a FEATURES dict name -> feature_fn), evaluate them with the harness, pick the best by CLIP1 block-protocol numbers, and return the structured result. Fill best.* from the harness output for that variant (clip1_block / clip2_block = the clip1_acc / clip2_acc under the block protocol).`,
    { label: `explore:${f.key}`, phase: 'Explore', schema: RESULT_SCHEMA }
  ),
  (r, f) => {
    if (!r) return { family: f.key, explore: null, verify: null }
    return agent(
      `${SETUP}\n\nYou are the VERIFIER for family "${f.key}". The explorer claims its best variant is "${r.best.name}" (metric ${r.best.metric}, mode ${r.best.mode}) in ${r.file} with block_bal ${r.best.block_bal}, start_bal ${r.best.start_bal}, clip1 ${r.best.clip1_block}, clip2 ${r.best.clip2_block}; beats_baseline=${r.beats_baseline}. Failure modes it reports: ${r.failure_modes}\n\n` +
      `1) Re-run that variant yourself from scratch with the harness and report the numbers you get (reproduced=true only if they agree within 0.02).\n` +
      `2) LEAKAGE AUDIT: read the code in ${r.file}. Does the feature encode anything that identifies the time segment or the clip rather than the person (absolute position, crop width, frame number, image size, mask area that correlates with segment)? Test it: e.g. shuffle labels within time blocks, or evaluate with the feature computed on a position-randomised crop, and see whether accuracy collapses as it should. Report leakage_found and details.\n` +
      `3) Check the explorer did not cherry-pick: it must have tuned on clip1 only. Look at variants_tried: ${JSON.stringify(r.variants_tried)}.\n` +
      `Return verdict use / use_with_caveats / do_not_use with reasoning. Do not modify repo files other than none (read-only except scratch).`,
      { label: `verify:${f.key}`, phase: 'Verify', schema: VERIFY_SCHEMA }
    ).then(v => ({ family: f.key, explore: r, verify: v }))
  }
)

const ok = results.filter(Boolean)
log(`families returned: ${ok.length}/${FAMILIES.length}`)
return ok
