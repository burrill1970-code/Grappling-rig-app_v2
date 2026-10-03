"""Summarise the vision audits (outputs/audit_results.json, audit2_results.json) into outputs/audit_summary.json."""
import collections, json

out = {}
for name, path in (("audit1", "outputs/audit_results.json"), ("audit2", "outputs/audit2_results.json")):
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        continue
    pf = [p for g in data for p in g["audit"]["per_frame"]]
    ver = [v for g in data for v in g["verified"]]
    out[name] = dict(
        frames=len(pf),
        per_frame={a: dict(collections.Counter(p[a] for p in pf)) for a in "AB"},
        serious_claimed=len(ver),
        serious_confirmed=sum(v["confirmed"] for v in ver),
        serious_refuted=sum((not v["confirmed"]) and (not v["split"]) for v in ver),
        serious_split=sum(v["split"] for v in ver),
        confirmed=[dict(frame=v["frame"], athlete=v["athlete"], type=v["type"]) for v in sorted(ver, key=lambda v: v["frame"]) if v["confirmed"]],
    )
json.dump(out, open("outputs/audit_summary.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confirmed"} for k, v in out.items()}, indent=1))
