"""Person-pixel masks from pretrained YOLO11-seg (no training). Used to keep gi-colour pixels that belong
to a person and drop same-coloured background (ad boards, scoreboard)."""
import cv2
import numpy as np
from pathlib import Path

WEIGHTS = "yolo11m-seg.pt"
IMGSZ = 1088
CONF = 0.20


def run_segmenter(frames, cache=None):
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        shape = tuple(z["shape"])
        return [np.unpackbits(m)[: shape[0] * shape[1]].reshape(shape).astype(bool) for m in z["masks"]]
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    masks, shape = [], None
    for i, f in enumerate(frames):
        shape = f.shape[:2]
        r = model.predict(f, imgsz=IMGSZ, conf=CONF, classes=[0], device="cpu", verbose=False)[0]
        m = np.zeros(shape, np.uint8)
        if r.masks is not None:
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        masks.append(m.astype(bool))
        if i % 50 == 0:
            print("seg frame", i, flush=True)
    if cache:
        arr = np.empty(len(masks), object)
        for i, m in enumerate(masks):
            arr[i] = np.packbits(m)
        np.savez_compressed(cache, masks=arr, shape=np.array(shape))
    return masks
