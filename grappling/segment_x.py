"""Second, slower person-mask pass (YOLO11x-seg, low confidence) to catch athletes the first pass misses
(motion blur, athlete hidden behind the other). Only athlete-like instances are kept: on the mat, wide enough,
confidence >= 0.15. Unioned with the first mask by `combine`."""
import cv2
import numpy as np
from pathlib import Path

WEIGHTS = "yolo11x-seg.pt"
IMGSZ = 1280
CONF = 0.05
KEEP_CONF = 0.15
MIN_BOTTOM = 0.5
MIN_WIDTH = 60


def run_segmenter_x(frames, cache=None):
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
            for poly, b, c in zip(r.masks.xy, r.boxes.xyxy.numpy(), r.boxes.conf.numpy()):
                if c >= KEEP_CONF and b[3] >= MIN_BOTTOM * shape[0] and (b[2] - b[0]) >= MIN_WIDTH and len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        masks.append(m.astype(bool))
        if i % 25 == 0:
            print("segx frame", i, flush=True)
    if cache:
        arr = np.empty(len(masks), object)
        for i, m in enumerate(masks):
            arr[i] = np.packbits(m)
        np.savez_compressed(cache, masks=arr, shape=np.array(shape))
    return masks


def combine(first, second):
    return [a | b for a, b in zip(first, second)]
