"""Annotated video: A and B in fixed colours, labelled, skeleton overlay, status shown."""
import cv2
import numpy as np
from .pose import SKELETON

COLORS = {"A": (40, 40, 255), "B": (60, 200, 40)}  # BGR: A red, B green (fixed for the whole clip)
NAMES = {"A": "A (white gi)", "B": "B (blue gi)"}
SHORT = {"detected": "detected", "color_split": "colour-split", "color_recovered": "colour-recovered", "lost": "LOST"}


def _dashed_rect(img, b, col, th=2, dash=12):
    x1, y1, x2, y2 = b
    for (p, q) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(int(np.hypot(q[0] - p[0], q[1] - p[1]) // dash), 1)
        for k in range(0, n, 2):
            a = (int(p[0] + (q[0] - p[0]) * k / n), int(p[1] + (q[1] - p[1]) * k / n))
            e = (int(p[0] + (q[0] - p[0]) * min(k + 1, n) / n), int(p[1] + (q[1] - p[1]) * min(k + 1, n) / n))
            cv2.line(img, a, e, col, th)


def draw(frame, rec, fi, shot):
    img = frame.copy()
    for ident, r in rec.items():
        col = COLORS[ident]
        if r["kp"] is not None:
            kp = np.array(r["kp"])
            for a, b in SKELETON:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), col, 3)
            for x, y, c in kp:
                if c > 0.3:
                    cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1)
                    cv2.circle(img, (int(x), int(y)), 4, col, 1)
        if r["box"] is not None:
            b = [int(v) for v in r["box"]]
            if r["status"] == "detected":
                cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, 3)
            else:
                _dashed_rect(img, b, col, 2)
            label = f'{NAMES[ident]} {SHORT[r["status"]]} {r["conf"]:.2f}'
            (tw, th), _ = cv2.getTextSize(label, 0, 0.6, 2)
            ty = max(b[1], th + 6)
            cv2.rectangle(img, (b[0], ty - th - 6), (b[0] + tw + 6, ty), col, -1)
            cv2.putText(img, label, (b[0] + 3, ty - 4), 0, 0.6, (255, 255, 255), 2)
    # HUD: one line per athlete so a lost track is always visible
    for k, ident in enumerate(("A", "B")):
        r = rec[ident]
        txt = f'{NAMES[ident]}: {SHORT[r["status"]]}' + ("" if r["kp"] is not None else "  (no pose)")
        cv2.putText(img, txt, (10, 24 + 24 * k), 0, 0.65, (0, 0, 0), 4)
        cv2.putText(img, txt, (10, 24 + 24 * k), 0, 0.65, COLORS[ident], 2)
    cv2.putText(img, f"frame {fi}  shot {shot}", (img.shape[1] - 190, 24), 0, 0.6, (0, 0, 0), 4)
    cv2.putText(img, f"frame {fi}  shot {shot}", (img.shape[1] - 190, 24), 0, 0.6, (255, 255, 255), 1)
    return img


def write_video(path, frames, results, cuts, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    shot = 0
    for fi, (f, rec) in enumerate(zip(frames, results)):
        if fi in cuts:
            shot += 1
        vw.write(draw(f, rec, fi, shot))
    vw.release()
