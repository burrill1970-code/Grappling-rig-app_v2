"""Summarise the vision audits (<out>/audit_results.json, audit2/audit3_results.json) into <out>/audit_summary.json."""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs")
OUT = ap.parse_args().out

out = {}
BUILD = {"audit1": "first full pipeline, before audit fixes",
         "audit2": "after round-1 audit fixes (crowd/referee/311-313/lost), before 2nd segmenter + loose-white fallback",
         "audit3": "delivered build, except A at frames 308-309 (changed after the audit: 308 lost -> small low-confidence box)"}
import os
if os.path.exists(f"{OUT}/audit_builds.json"):
    BUILD.update(json.load(open(f"{OUT}/audit_builds.json")))  # per-clip labels override the clip-1 ones
for name, path in (("audit1", f"{OUT}/audit_results.json"), ("audit2", f"{OUT}/audit2_results.json"),
                   ("audit3", f"{OUT}/audit3_results.json")):
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        continue
    pf = [p for g in data for p in g["audit"]["per_frame"]]
    ver = [v for g in data for v in g["verified"]]
    out[name] = dict(
        build=BUILD[name],
        frames=len(pf),
        verdicts={str(p["frame"]): dict(A=p["A"], B=p["B"]) for p in pf},
        per_frame={a: dict(collections.Counter(p[a] for p in pf)) for a in "AB"},
        serious_claimed=len(ver),
        serious_confirmed=sum(v["confirmed"] for v in ver),
        serious_refuted=sum((not v["confirmed"]) and (not v["split"]) for v in ver),
        serious_split=sum(v["split"] for v in ver),
        confirmed=[dict(frame=v["frame"], athlete=v["athlete"], type=v["type"]) for v in sorted(ver, key=lambda v: v["frame"]) if v["confirmed"]],
    )
json.dump(out, open(f"{OUT}/audit_summary.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confirmed"} for k, v in out.items()}, indent=1))
