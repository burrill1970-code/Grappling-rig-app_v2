"""Hand-labelling tool for ground truth: a local web page where a person draws a box around each athlete.

    python tools/gt_label.py            # both clips, then open http://localhost:8765

The page shows the RAW cropped frame (never the tracker's output, so the labels are not biased by it) with
small context thumbnails of nearby frames. Labels save to <out>/ground_truth.json after every frame, so the
session can be stopped and resumed. Box coordinates are in the cropped picture, the same frame of reference
as tracks.json.

Which frames: every 7th frame of each clip ("regular", an unbiased sample that rates can be quoted from) plus
frames where audits or tests found failures ("targeted", reported separately because they are not a sample).
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STEP = 7
CONTEXT = (-4, -2, 2, 4)
CLIPS = [
    # (video, output folder, targeted frames: audit-confirmed failures and the frames pinned by xfail tests)
    ("test_clips/clip.mp4", "outputs", [140, 142, 308]),
    ("test_clips/clip2.mp4", "outputs/clip2", [137, 439, 440, 441, 442, 443, 450, 451, 452]),
]


def task_frames(n, targeted):
    regular = list(range(3, n, STEP))
    extra = [f for f in targeted if f not in regular and 0 <= f < n]
    return [(f, "regular") for f in regular] + [(f, "targeted") for f in extra]


def read_jpgs(video, crop_y, frames):
    """JPEG bytes of the given frames (and their CONTEXT neighbours), cropped to the picture."""
    want = {f + d for f in frames for d in (0,) + CONTEXT}
    y0, y1 = crop_y
    jpgs = {}
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            jpgs[i] = cv2.imencode(".jpg", f[y0:y1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        i += 1
    cap.release()
    return jpgs


def load_clip(video, out, targeted):
    meta = json.load(open(Path(out) / "tracks.json"))["meta"]
    tasks = task_frames(meta["frames"], targeted)
    jpgs = read_jpgs(video, meta["crop_y"], [f for f, _ in tasks])
    return dict(video=video, out=out, n=meta["frames"], tasks=tasks, jpgs=jpgs)


def labels_path(clip):
    return Path(clip["out"]) / "ground_truth.json"


def read_labels(clip):
    p = labels_path(clip)
    return json.load(open(p)) if p.exists() else {"video": clip["video"], "coords": "cropped picture (tracks.json frame)",
                                                   "identity": {"A": "white gi", "B": "blue gi"}, "frames": {}}


def write_labels(clip, data):
    p = labels_path(clip)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, p)


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
                    lab = read_labels(c)["frames"]
                    for f, kind in c["tasks"]:
                        ctx = [f + d for d in CONTEXT if (f + d) in c["jpgs"]]
                        tasks.append(dict(clip=ci, video=c["video"], frame=f, kind=kind, context=ctx,
                                          label=lab.get(str(f))))
                return self.send(json.dumps(tasks).encode(), "application/json")
            if self.path.startswith("/img/"):
                try:
                    ci, f = self.path[5:].removesuffix(".jpg").split("/")
                    return self.send(clips[int(ci)]["jpgs"][int(f)], "image/jpeg")
                except (KeyError, ValueError, IndexError):
                    return self.send(b"not found", "text/plain", 404)
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/label":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            c = clips[int(body["clip"])]
            data = read_labels(c)
            data["frames"][str(int(body["frame"]))] = {k: body[k] for k in ("A", "B", "unusable", "kind")}
            write_labels(c, data)
            self.send(b'{"ok":true}', "application/json")

    return H


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Ground truth labelling</title>
<style>
:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa0a6;--a:#ff9f1c;--b:#2ec4ff;--ok:#5ad17a;--bad:#ff5d5d}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
#top{display:flex;gap:16px;align-items:center;padding:8px 12px;flex-wrap:wrap}
#prog{flex:1;min-width:200px;height:6px;background:#333;border-radius:3px;overflow:hidden}#prog div{height:100%;background:var(--ok)}
.who{padding:4px 10px;border-radius:4px;border:2px solid transparent;cursor:pointer}
.who.A{color:var(--a)}.who.B{color:var(--b)}.who.cur{border-color:currentColor}
#wrap{position:relative;margin:0 12px}canvas{width:100%;display:block;cursor:crosshair}
#ctx{display:flex;gap:6px;margin:6px 12px}#ctx figure{margin:0;flex:1;text-align:center;color:var(--mut);font-size:12px}#ctx img{width:100%}
#help{color:var(--mut);padding:4px 12px 16px;line-height:1.6}kbd{background:#333;border-radius:3px;padding:1px 5px}
button{background:#2a2d33;color:var(--fg);border:1px solid #444;border-radius:4px;padding:4px 10px;cursor:pointer}
.flag{color:var(--bad)}
</style></head><body>
<div id="top"><b id="title"></b><span id="kind"></span>
<span class="who A" data-w="A">A (white)<span id="sA"></span></span>
<span class="who B" data-w="B">B (blue)<span id="sB"></span></span>
<span id="un" class="flag"></span>
<button id="prev">◀ Back</button><button id="next">Save &amp; next ▶</button>
<div id="prog"><div></div></div><span id="count"></span></div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="ctx"></div>
<div id="help">Drag a box around the athlete's whole <b>visible</b> body (head and limbs too, only the part you can see).
After A it switches to B. If an athlete cannot be seen well enough to box, mark them hidden. Ignore the referee and the crowd.<br>
<kbd>1</kbd>/<kbd>2</kbd> choose A/B &nbsp; <kbd>H</kbd> hidden &nbsp; <kbd>U</kbd> unsure &nbsp; <kbd>Backspace</kbd> clear box &nbsp;
<kbd>X</kbd> whole frame unusable (overlay, blur) &nbsp; <kbd>Enter</kbd>/<kbd>→</kbd> save &amp; next &nbsp; <kbd>←</kbd> back &nbsp;
<kbd>N</kbd> jump to next unlabelled</div>
<script>
let tasks=[],i=0,cur='A',lab=null,img=new Image(),drag=null;
const cv=document.getElementById('cv'),cx=cv.getContext('2d'),$=id=>document.getElementById(id);
const blank=()=>({A:{box:null,hidden:false,unsure:false},B:{box:null,hidden:false,unsure:false},unusable:false});
const done=l=>l&&(l.unusable||['A','B'].every(w=>l[w].box||l[w].hidden));
async function init(){tasks=await (await fetch('/api/tasks')).json();i=tasks.findIndex(t=>!done(t.label));if(i<0)i=0;show()}
function show(){const t=tasks[i];lab=t.label?JSON.parse(JSON.stringify(t.label)):blank();cur='A';
 img=new Image();img.onload=()=>{cv.width=img.width;cv.height=img.height;draw()};img.src=`/img/${t.clip}/${t.frame}.jpg`;
 $('title').textContent=`${t.video}  frame ${t.frame}`;$('kind').textContent=t.kind==='targeted'?'(targeted)':'';
 $('ctx').innerHTML=t.context.map(f=>`<figure><img src="/img/${t.clip}/${f}.jpg"><figcaption>frame ${f} (${f>t.frame?'+':''}${f-t.frame})</figcaption></figure>`).join('');
 status()}
function status(){for(const w of['A','B']){const l=lab[w];$('s'+w).textContent=l.hidden?' — hidden':l.box?' ✓':' …';if(l.unsure)$('s'+w).textContent+=' (unsure)'}
 document.querySelectorAll('.who').forEach(e=>e.classList.toggle('cur',e.dataset.w===cur));
 $('un').textContent=lab.unusable?'FRAME UNUSABLE':'';const n=tasks.filter(t=>done(t.label)).length;
 $('count').textContent=`${n} / ${tasks.length} done · #${i+1}`;$('prog').firstChild.style.width=(100*n/tasks.length)+'%'}
function draw(){cx.drawImage(img,0,0);const lw=Math.max(2,cv.width/400);
 for(const w of['A','B']){const b=lab[w].box;if(!b)continue;const c=w==='A'?'#ff9f1c':'#2ec4ff';cx.strokeStyle=c;cx.lineWidth=lw;
  cx.setLineDash(lab[w].unsure?[10,6]:[]);cx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);cx.setLineDash([]);
  cx.fillStyle=c;cx.font=`bold ${lw*10}px system-ui`;cx.fillText(w,b[0]+lw*2,b[1]+lw*10)}
 if(drag&&drag.b){const b=drag.b;cx.strokeStyle=cur==='A'?'#ff9f1c':'#2ec4ff';cx.lineWidth=lw;cx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}}
function pt(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height]}
cv.onmousedown=e=>{drag={p:pt(e),b:null}};
window.onmousemove=e=>{if(!drag)return;const q=pt(e),p=drag.p;
 drag.b=[Math.max(0,Math.min(p[0],q[0])),Math.max(0,Math.min(p[1],q[1])),Math.min(cv.width,Math.max(p[0],q[0])),Math.min(cv.height,Math.max(p[1],q[1]))].map(Math.round);draw()};
window.onmouseup=()=>{if(!drag)return;const b=drag.b;drag=null;
 if(b&&b[2]-b[0]>4&&b[3]-b[1]>4){lab[cur].box=b;lab[cur].hidden=false;if(cur==='A'&&!lab.B.box&&!lab.B.hidden)cur='B'}draw();status()};
async function save(){const t=tasks[i];t.label=lab;
 await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip:t.clip,frame:t.frame,kind:t.kind,...lab})})}
async function next(){if(!done(lab)){alert('Box each athlete or mark them hidden (or mark the frame unusable) first.');return}
 await save();if(i<tasks.length-1){i++;show()}else{status();alert('All frames labelled. You can close this page.')}}
function back(){if(i>0){i--;show()}}
document.querySelectorAll('.who').forEach(e=>e.onclick=()=>{cur=e.dataset.w;status()});
$('next').onclick=next;$('prev').onclick=back;
window.onkeydown=e=>{const k=e.key;
 if(k==='1'){cur='A'}else if(k==='2'){cur='B'}
 else if(k==='h'||k==='H'){lab[cur].hidden=!lab[cur].hidden;if(lab[cur].hidden)lab[cur].box=null;if(cur==='A'&&lab.A.hidden&&!lab.B.box&&!lab.B.hidden)cur='B'}
 else if(k==='u'||k==='U'){lab[cur].unsure=!lab[cur].unsure}
 else if(k==='x'||k==='X'){lab.unusable=!lab.unusable}
 else if(k==='Backspace'){lab[cur].box=null;e.preventDefault()}
 else if(k==='Enter'||k==='ArrowRight'){next();return}
 else if(k==='ArrowLeft'){back();return}
 else if(k==='n'||k==='N'){const j=tasks.findIndex(t=>!done(t.label));if(j>=0){i=j;show()}return}
 else return;draw();status()};
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    clips = []
    for video, out, targeted in CLIPS:
        c = load_clip(video, out, targeted)
        print(f"{video}: {len(c['tasks'])} frames to label -> {labels_path(c)}")
        clips.append(c)
    print(f"open http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(clips)).serve_forever()


if __name__ == "__main__":
    main()
