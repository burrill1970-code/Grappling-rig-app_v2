"""Attach COCO-17 keypoints to athlete records."""
import numpy as np

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
            "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def keypoints_of(det):
    """det row -> 17x3 array (x, y, conf)."""
    return np.asarray(det[5:], np.float32).reshape(17, 3)


def attach_keypoints(results, raw_dets):
    """Add 'kp' (17x3 list or None) and 'pose_conf' to each athlete record.

    Only 'detected' records own their detection's pose. 'merged' records share one detection between two
    bodies, so which athlete the pose belongs to is not known here: kp stays None (resolved in milestone 4).
    """
    for fi, rec in enumerate(results):
        for r in rec.values():
            r["kp"], r["pose_conf"] = None, 0.0
            if r["status"] == "detected" and r["cand"] is not None:
                kp = keypoints_of(raw_dets[fi][r["cand"]])
                r["kp"] = kp.tolist()
                r["pose_conf"] = float(kp[:, 2].mean())
    return results
