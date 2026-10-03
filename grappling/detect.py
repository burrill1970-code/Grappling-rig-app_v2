"""Person detection + 17-keypoint pose using pretrained YOLO11-pose (no training).

Raw per-frame results are cached to an .npz so later stages do not re-run the model.
"""
import numpy as np
from pathlib import Path

WEIGHTS = "yolo11m-pose.pt"
IMGSZ = 1088
CONF = 0.15  # low on purpose: tracker/colour stage decides; we do not hide weak detections


def run_detector(frames, cache=None):
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["dets"])
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    out = []
    for i, f in enumerate(frames):
        r = model.predict(f, imgsz=IMGSZ, conf=CONF, classes=[0], device="cpu", verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 5 + 17 * 3), np.float32))
        else:
            b = r.boxes.xyxy.numpy()
            c = r.boxes.conf.numpy()[:, None]
            k = r.keypoints.data.numpy().reshape(len(b), -1)  # x,y,conf * 17
            out.append(np.hstack([b, c, k]).astype(np.float32))
        if i % 50 == 0:
            print("detect frame", i, "people", len(out[-1]), flush=True)
    if cache:
        arr = np.empty(len(out), object)
        for i, d in enumerate(out):
            arr[i] = d
        np.savez_compressed(cache, dets=arr)
    return out
