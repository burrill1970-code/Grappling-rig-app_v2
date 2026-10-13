"""Identity check: a person says which athlete a track box is on, where box labels cannot decide.

    python tools/gt_identity.py         # both clips, then open http://localhost:8765

Asked only for the boxes tools/eval_ground_truth.py marks as needing it (the box touches both labelled athletes
and favours its own by less than ID_MARGIN IoU). Unlike tools/gt_label.py this page shows the tracker's box: the
question is a yes/no check of that box, not a label drawn from scratch. One box per question, so the other
athlete's box cannot sway the answer. Answers save to <out>/identity_check.json after every question, each with the box it is about: if the tracker
later changes that box, the answer no longer applies and the box is asked about again.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.eval_ground_truth import answer_for, evaluate, needs_id_check  # noqa: E402
from tools.gt_label import CLIPS, CONTEXT, read_jpgs  # noqa: E402

GI = {"A": "WHITE", "B": "BLUE"}


def load_clip(video, out):
    out = Path(out)
    tracks = json.load(open(out / "tracks.json"))
    rows = evaluate(out, write=False)["per_frame"]
    tasks = []
    for r in rows:
        if r["unusable"]:
            continue
        for who in "AB":
            x = r[who]
            box = tracks["frames"][r["frame"]]["athletes"][who]["box"]
            if box is not None and needs_id_check(x["iou"], x["iou_other"]):
                tasks.append(dict(frame=r["frame"], athlete=who, box=box))
    jpgs = read_jpgs(video, tracks["meta"]["crop_y"], sorted({t["frame"] for t in tasks}))
    return dict(video=video, out=out, tasks=tasks, jpgs=jpgs)


def answers_path(clip):
    return clip["out"] / "identity_check.json"


def read_answers(clip):
    p = answers_path(clip)
    return json.load(open(p)) if p.exists() else {
        "video": clip["video"], "question": "is this track box on the named athlete?",
        "answers": {"right": "on that athlete", "wrong": "on the other athlete", "both": "covers both about equally"},
        "frames": {}}


def make_handler(clips):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/tasks":
                tasks = []
                for ci, c in enumerate(clips):
                    ans = read_answers(c)["frames"]
                    for t in c["tasks"]:
                        ctx = [t["frame"] + d for d in CONTEXT if (t["frame"] + d) in c["jpgs"]]
                        tasks.append(dict(t, clip=ci, video=c["video"], gi=GI[t["athlete"]], context=ctx,
                                          answer=answer_for(ans, t["frame"], t["athlete"], t["box"])))
                return self.send(json.dumps(tasks).encode(), "application/json")
            if self.path.startswith("/img/"):
                try:
                    ci, f = self.path[5:].removesuffix(".jpg").split("/")
                    return self.send(clips[int(ci)]["jpgs"][int(f)], "image/jpeg")
                except (KeyError, ValueError, IndexError):
                    return self.send(b"not found", "text/plain", 404)
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/answer":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["answer"] not in ("right", "wrong", "both") or body["athlete"] not in ("A", "B"):
                return self.send(b"bad answer", "text/plain", 400)
            c = clips[int(body["clip"])]
            data = read_answers(c)
            task = next(t for t in c["tasks"] if t["frame"] == int(body["frame"]) and t["athlete"] == body["athlete"])
            data["frames"].setdefault(str(task["frame"]), {})[task["athlete"]] = dict(answer=body["answer"], box=task["box"])
            p = answers_path(c)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, p)
            self.send(b'{"ok":true}', "application/json")

    return H


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Identity check</title>
<style>
:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa0a6;--ok:#5ad17a}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
#top{display:flex;gap:14px;align-items:center;padding:8px 12px;flex-wrap:wrap}
#q{font-size:18px}#q b{padding:1px 6px;border-radius:3px}
#prog{flex:1;min-width:160px;height:6px;background:#333;border-radius:3px;overflow:hidden}#prog div{height:100%;background:var(--ok)}
canvas{width:calc(100% - 24px);margin:0 12px;display:block}
#ctx{display:flex;gap:6px;margin:6px 12px}#ctx figure{margin:0;flex:1;text-align:center;color:var(--mut);font-size:12px}#ctx img{width:100%}
button{background:#2a2d33;color:var(--fg);border:1px solid #444;border-radius:4px;padding:6px 12px;cursor:pointer;font-size:14px}
button.sel{border-color:var(--ok);color:var(--ok)}
#help{color:var(--mut);padding:4px 12px 16px}kbd{background:#333;border-radius:3px;padding:1px 5px}
</style></head><body>
<div id="top"><span id="q"></span>
<button data-a="right"><kbd>R</kbd> Yes, on that athlete</button>
<button data-a="wrong"><kbd>W</kbd> No, on the other athlete</button>
<button data-a="both"><kbd>B</kbd> Covers both about equally</button>
<button id="prev">◀ Back</button><div id="prog"><div></div></div><span id="count"></span></div>
<canvas id="cv"></canvas><div id="ctx"></div>
<div id="help">Judge which athlete's body the box mainly sits on, not how well it fits. Thumbnails below are the raw neighbouring frames.
&nbsp;<kbd>R</kbd>/<kbd>W</kbd>/<kbd>B</kbd> answer and go on &nbsp; <kbd>←</kbd> back &nbsp; <kbd>N</kbd> next unanswered</div>
<script>
let tasks=[],i=0,img=new Image();const cv=document.getElementById('cv'),cx=cv.getContext('2d'),$=id=>document.getElementById(id);
async function init(){tasks=await (await fetch('/api/tasks')).json();i=tasks.findIndex(t=>!t.answer);if(i<0)i=0;show()}
function show(){const t=tasks[i];img=new Image();img.onload=()=>{cv.width=img.width;cv.height=img.height;draw()};img.src=`/img/${t.clip}/${t.frame}.jpg`;
 const col=t.gi==='WHITE'?'background:#eee;color:#111':'background:#1f5fd6;color:#fff';
 $('q').innerHTML=`${t.video} frame ${t.frame}: is the box on the <b style="${col}">${t.gi} gi</b> athlete?`;
 $('ctx').innerHTML=t.context.map(f=>`<figure><img src="/img/${t.clip}/${f}.jpg"><figcaption>frame ${f} (${f>t.frame?'+':''}${f-t.frame})</figcaption></figure>`).join('');
 document.querySelectorAll('[data-a]').forEach(b=>b.classList.toggle('sel',b.dataset.a===t.answer));
 const n=tasks.filter(t=>t.answer).length;$('count').textContent=`${n} / ${tasks.length} answered · #${i+1}`;$('prog').firstChild.style.width=(100*n/tasks.length)+'%'}
function draw(){const b=tasks[i].box,lw=Math.max(3,cv.width/250);cx.drawImage(img,0,0);
 cx.lineWidth=lw+4;cx.strokeStyle='#000';cx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
 cx.lineWidth=lw;cx.strokeStyle='#ff2bd6';cx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}
async function answer(a){const t=tasks[i];t.answer=a;
 await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip:t.clip,frame:t.frame,athlete:t.athlete,answer:a})});
 // go to the next UNANSWERED question: stepping onto an answered one invites re-answering it by accident
 const j=tasks.findIndex((t,k)=>k>i&&!t.answer),k=j>=0?j:tasks.findIndex(t=>!t.answer);
 if(k>=0){i=k;show()}else{show();alert('All answered. You can close this page.')}}
document.querySelectorAll('[data-a]').forEach(b=>b.onclick=()=>answer(b.dataset.a));
$('prev').onclick=()=>{if(i>0){i--;show()}};
window.onkeydown=e=>{const k=e.key.toLowerCase();if(k==='r')answer('right');else if(k==='w')answer('wrong');else if(k==='b')answer('both');
 else if(k==='arrowleft'&&i>0){i--;show()}else if(k==='n'){const j=tasks.findIndex(t=>!t.answer);if(j>=0){i=j;show()}}};
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    clips = []
    for video, out, _ in CLIPS:
        c = load_clip(video, out)
        print(f"{video}: {len(c['tasks'])} boxes to check -> {answers_path(c)}")
        clips.append(c)
    print(f"open http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(clips)).serve_forever()


if __name__ == "__main__":
    main()
