export const meta = {
  name: 'audit-tracking',
  description: 'Visually audit annotated grappling tracking for ID swaps, wrong-body boxes, lost-but-visible, wrong poses; adversarially verify serious findings',
  phases: [
    { title: 'Audit', detail: '12 auditors over contact sheets of the annotated video' },
    { title: 'Verify', detail: 'two skeptic lenses per serious finding, using raw frames' },
  ],
}
// HISTORICAL: the clip-1-specific audit script used for clip 1's audit 1 (round1) and audits 2-3 (rounds2_3).
// Kept for provenance of outputs/audit*_results.json. Use tools/workflows/audit_generic.js for new audits.

const SCRATCH = args.scratch
const REPO = args.repo

const ISSUE_TYPES = ['id_swap', 'box_on_wrong_body', 'box_partial', 'lost_but_visible', 'pose_on_wrong_athlete', 'pose_garbage', 'referee_or_crowd_labelled', 'other']
const PER_FRAME = ['correct', 'wrong_body', 'partial', 'lost_but_visible', 'not_visible', 'unclear']

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    per_frame: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          frame: { type: 'integer' },
          A: { type: 'string', enum: PER_FRAME },
          B: { type: 'string', enum: PER_FRAME },
          note: { type: 'string' },
        },
        required: ['frame', 'A', 'B'],
      },
    },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          frame: { type: 'integer' },
          athlete: { type: 'string', enum: ['A', 'B', 'both'] },
          type: { type: 'string', enum: ISSUE_TYPES },
          severity: { type: 'string', enum: ['low', 'medium', 'high'] },
          description: { type: 'string' },
        },
        required: ['frame', 'athlete', 'type', 'severity', 'description'],
      },
    },
  },
  required: ['per_frame', 'issues'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    confirmed: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    evidence: { type: 'string' },
  },
  required: ['confirmed', 'confidence', 'evidence'],
}

const CONTEXT = `You are auditing the output of a two-athlete tracker on a judo clip (screen recording, picture cropped to 1080x608).
Athlete A = the athlete in the WHITE gi, drawn in RED (box, skeleton, label). Athlete B = the athlete in the BLUE gi, drawn in GREEN.
Solid box = direct detector box. Dashed box = box derived from gi-colour pixels (may legitimately be partial). Skeleton lines are pose keypoints; on tangled frames only fragments are expected.
A black-suited referee and the crowd must never be labelled A or B.
The top-left HUD of each tile says each athlete's status (detected / colour-split / colour-recovered / LOST). Each tile has its frame number printed at the bottom as #N.
Frames 0-11 are a DIFFERENT match (white standing, blue on mat); judge those on colour alone.
Be strict and independent. Do NOT give the benefit of the doubt. If you cannot tell, say 'unclear'. Do not invent problems either: a dashed box that covers part of the correct athlete is 'partial', not 'wrong_body'.`

const DEFS = `Per-frame verdicts for each athlete (A, B):
- correct: the label/box (and skeleton if drawn) sit on the right-coloured athlete.
- wrong_body: the A (red) overlay is mostly on the blue-gi athlete or the B (green) overlay is mostly on the white-gi athlete, or on background/referee. This includes ID swaps.
- partial: right athlete but the box covers clearly less than about half of that athlete's visible body.
- lost_but_visible: HUD says LOST (no box) but that athlete is clearly visible in the picture.
- not_visible: no box and the athlete is genuinely hidden/off-screen.
- unclear: you cannot judge.
Issue types: ${ISSUE_TYPES.join(', ')}. Severity high = identity wrong (id_swap / box_on_wrong_body / pose_on_wrong_athlete / referee labelled) or lost_but_visible; medium = box_partial or pose_garbage; low = cosmetic.
List an issue only for frames you actually looked at, once per frame per athlete.`

// 12 groups: 10 regular (3 sheets each), 2 flagged (3 + 2 sheets)
const pad = n => String(n).padStart(3, '0')
const groups = []
for (let g = 0; g < 10; g++) groups.push({ kind: 'regular', label: `regular-${g}`, sheets: [3 * g, 3 * g + 1, 3 * g + 2].map(n => `${SCRATCH}/audit/regular_${pad(n)}.jpg`) })
groups.push({ kind: 'flagged', label: 'flagged-0', sheets: [0, 1, 2].map(n => `${SCRATCH}/audit/flagged_${pad(n)}.jpg`) })
groups.push({ kind: 'flagged', label: 'flagged-1', sheets: [3, 4].map(n => `${SCRATCH}/audit/flagged_${pad(n)}.jpg`) })

const SERIOUS = new Set(['id_swap', 'box_on_wrong_body', 'lost_but_visible', 'pose_on_wrong_athlete', 'referee_or_crowd_labelled'])

const results = await pipeline(
  groups,
  g => agent(
    `${CONTEXT}\n\n${DEFS}\n\nRead each of these contact-sheet images with the Read tool (each shows 6 frames, 2 columns x 3 rows) and judge EVERY tile on every sheet:\n${g.sheets.map(s => '- ' + s).join('\n')}\n\nReturn per_frame for every tile you judged (frame number from the #N label) and the issues list. Do not modify any files.`,
    { label: `audit:${g.label}`, phase: 'Audit', schema: AUDIT_SCHEMA }
  ),
  (audit, g) => {
    if (!audit) return { group: g.label, audit: null, verified: [] }
    const serious = audit.issues.filter(i => SERIOUS.has(i.type) || i.severity === 'high')
    return parallel(serious.map(iss => async () => {
      const mk = lens => agent(
        `You are a skeptic. An auditor claimed this tracking error: frame ${iss.frame}, athlete ${iss.athlete}, type ${iss.type}, severity ${iss.severity}: "${iss.description}".\n` +
        `${CONTEXT}\n\nYour job: try to REFUTE the claim. Default to confirmed=false if the evidence is ambiguous or you cannot see the problem clearly.\n` +
        `Work in ${REPO} (read-only; write images only under ${SCRATCH}/verify/). Use python+cv2 to extract frames, e.g.:\n` +
        `  mkdir -p ${SCRATCH}/verify && python -c "import cv2;c=cv2.VideoCapture('outputs/annotated.mp4');c.set(1,F);ok,f=c.read();cv2.imwrite('${SCRATCH}/verify/ann_F.png',f)"\n` +
        `  RAW picture: same with test_clips/clip.mp4 then crop rows 346:954 (f[346:954]).\n` +
        `Then Read the PNGs. ${lens}\nReturn confirmed=true only if the error is really present at that frame.`,
        { label: `verify:${iss.frame}:${iss.type}:${lens.slice(0, 8)}`, phase: 'Verify', schema: VERDICT_SCHEMA }
      )
      const votes = await parallel([
        () => mk('LENS 1 (visual): view the annotated frame ' + iss.frame + ' at full size next to the raw frame; decide from the picture alone who is the white-gi and blue-gi athlete and whether the red/green overlays sit on them.'),
        () => mk('LENS 2 (temporal): view annotated frames ' + [iss.frame - 3, iss.frame, iss.frame + 3].filter(f => f >= 0 && f <= 534).join(', ') + ' and the raw frame ' + iss.frame + '; decide whether the overlay is consistently on the wrong athlete or only a momentary glitch, and whether the claim holds at exactly frame ' + iss.frame + '.'),
      ])
      const v = votes.filter(Boolean)
      return { ...iss, votes: v, confirmed: v.length === 2 && v.every(x => x.confirmed), split: v.length === 2 && v[0].confirmed !== v[1].confirmed }
    })).then(verified => ({ group: g.label, audit, verified: verified.filter(Boolean) }))
  }
)

const ok = results.filter(Boolean)
log(`audit groups returned: ${ok.length}/${groups.length}`)
return ok
