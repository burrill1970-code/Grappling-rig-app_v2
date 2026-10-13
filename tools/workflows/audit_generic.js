export const meta = {
  name: 'audit-tracking-generic',
  description: 'Visually audit annotated grappling tracking for any clip; adversarially verify serious findings',
  phases: [
    { title: 'Audit', detail: 'reviewers over contact sheets of the annotated video' },
    { title: 'Verify', detail: 'two skeptic lenses per serious finding, using raw frames' },
  ],
}

// args: { scratch, repo, dir, out, clip, crop:[y0,y1], regular, flagged, brief, perGroup }
const SCRATCH = args.scratch
const REPO = args.repo
const PER = args.perGroup || 3

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

const CONTEXT = `You are auditing the output of a two-athlete tracker on a judo clip (a phone screen recording of a YouTube player; the picture has been cropped to the video area).
Athlete A = the athlete in the WHITE gi, drawn in RED (box, skeleton, label). Athlete B = the athlete in the BLUE gi, drawn in GREEN.
Solid box = direct detector box. Dashed box = box derived from gi-colour pixels (may legitimately be partial). Skeleton lines are pose keypoints; on tangled frames only fragments are expected.
Referees (dark suits), coaches, officials and the crowd must never be labelled A or B. Only the two judoka in gi are athletes.
The top-left HUD of each tile says each athlete's status (detected / colour-split / colour-recovered / LOST). Each tile has its frame number printed at the bottom as #N.
Clip briefing: ${args.brief}
Be strict and independent. Do NOT give the benefit of the doubt. If you cannot tell, say 'unclear'. Do not invent problems either: a dashed box that covers part of the correct athlete is 'partial', not 'wrong_body'.`

const DEFS = `Per-frame verdicts for each athlete (A, B):
- correct: the label/box (and skeleton if drawn) sit on the right-coloured athlete.
- wrong_body: the A (red) overlay is mostly on the blue-gi athlete or the B (green) overlay is mostly on the white-gi athlete, or on background / a referee / a spectator. This includes ID swaps.
- partial: right athlete but the box covers clearly less than about half of that athlete's visible body.
- lost_but_visible: HUD says LOST (no box) or no overlay is drawn for that athlete, but the athlete is clearly visible in the picture.
- not_visible: no box and the athlete is genuinely hidden/off-screen.
- unclear: you cannot judge.
Issue types: ${ISSUE_TYPES.join(', ')}. Severity high = identity wrong (id_swap / box_on_wrong_body / pose_on_wrong_athlete / referee or crowd labelled) or lost_but_visible; medium = box_partial or pose_garbage; low = cosmetic.
List an issue only for frames you actually looked at, once per frame per athlete.`

const pad = n => String(n).padStart(3, '0')
const groups = []
for (let s = 0; s < args.regular; s += PER) {
  groups.push({ label: `regular-${s / PER}`, sheets: Array.from({ length: Math.min(PER, args.regular - s) }, (_, k) => `${SCRATCH}/${args.dir}/regular_${pad(s + k)}.jpg`) })
}
for (let s = 0; s < args.flagged; s += PER) {
  groups.push({ label: `flagged-${s / PER}`, sheets: Array.from({ length: Math.min(PER, args.flagged - s) }, (_, k) => `${SCRATCH}/${args.dir}/flagged_${pad(s + k)}.jpg`) })
}

const SERIOUS = new Set(['id_swap', 'box_on_wrong_body', 'lost_but_visible', 'pose_on_wrong_athlete', 'referee_or_crowd_labelled'])
const [Y0, Y1] = args.crop

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
        `  mkdir -p ${SCRATCH}/verify && python -c "import cv2;c=cv2.VideoCapture('${args.out}/annotated.mp4');c.set(1,F);ok,f=c.read();cv2.imwrite('${SCRATCH}/verify/c2ann_F.png',f)"\n` +
        `  RAW picture: same with ${args.clip} then crop rows ${Y0}:${Y1} (f[${Y0}:${Y1}]).\n` +
        `Then Read the PNGs. ${lens}\nReturn confirmed=true only if the error is really present at that frame.`,
        { label: `verify:${iss.frame}:${iss.type}:${lens.slice(0, 8)}`, phase: 'Verify', schema: VERDICT_SCHEMA }
      )
      const votes = await parallel([
        () => mk('LENS 1 (visual): view the annotated frame ' + iss.frame + ' at full size next to the raw frame; decide from the picture alone who is the white-gi and blue-gi athlete and whether the red/green overlays sit on them.'),
        () => mk('LENS 2 (temporal): view annotated frames ' + [iss.frame - 3, iss.frame, iss.frame + 3].filter(f => f >= 0).join(', ') + ' and the raw frame ' + iss.frame + '; decide whether the overlay is consistently on the wrong athlete or only a momentary glitch, and whether the claim holds at exactly frame ' + iss.frame + '.'),
      ])
      const v = votes.filter(Boolean)
      return { ...iss, votes: v, confirmed: v.length === 2 && v.every(x => x.confirmed), split: v.length === 2 && v[0].confirmed !== v[1].confirmed }
    })).then(verified => ({ group: g.label, audit, verified: verified.filter(Boolean) }))
  }
)

const ok = results.filter(Boolean)
log(`audit groups returned: ${ok.length}/${groups.length}`)
return ok
