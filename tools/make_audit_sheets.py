"""Contact sheets of the annotated video for visual audit.
Usage: python tools/make_audit_sheets.py SHEETDIR [OFFSET_MOD_3] [TRACKING_OUT_DIR]"""
import json, sys
import cv2, numpy as np

out = sys.argv[1]
offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # sample frames offset, mod 3
src = sys.argv[3] if len(sys.argv) > 3 else "outputs"
import os; os.makedirs(out, exist_ok=True)
stats = json.load(open(f"{src}/stats.json"))
flag = set()
for key in ("lost", "colour_conflict", "low_conf"):
    for i in "AB":
        for a, b in stats[key][i]:
            flag.update(range(a, b + 1))
cap = cv2.VideoCapture(f"{src}/annotated.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
base = set(range(offset, len(frames), 3))
sets = {"regular": sorted(base), "flagged": sorted(f for f in flag - base if f < len(frames))}
index = {}
for name, idxs in sets.items():
    sheets = []
    for s in range(0, len(idxs), 6):
        chunk = idxs[s:s + 6]
        tiles = []
        for fi in chunk:
            t = cv2.resize(frames[fi][60:], (720, 391))
            cv2.putText(t, f"#{fi}", (300, 380), 0, 1.0, (0, 0, 0), 5)
            cv2.putText(t, f"#{fi}", (300, 380), 0, 1.0, (0, 255, 255), 2)
            tiles.append(t)
        while len(tiles) < 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = cv2.vconcat([cv2.hconcat(tiles[0:2]), cv2.hconcat(tiles[2:4]), cv2.hconcat(tiles[4:6])])
        p = f"{out}/{name}_{s // 6:03d}.jpg"
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheets.append(dict(path=p, frames=chunk))
    index[name] = sheets
json.dump(index, open(f"{out}/index.json", "w"))
print({k: len(v) for k, v in index.items()}, "flagged-only frames:", len(sets["flagged"]))
