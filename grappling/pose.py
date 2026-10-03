"""Attach COCO-17 keypoints to athlete records."""
import numpy as np

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
            "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def keypoints_of(det):
    """det row -> 17x3 array (x, y, conf)."""
    return np.asarray(det[5:], np.float32).reshape(17, 3)


def attach_keypoints(results, raw_dets, frames=None):
    """Add 'kp' (17x3 list or None), 'pose_conf' and 'pose_source' to each athlete record.

    'detected' records own their detection's pose. For colour-derived records (merged / recovered) a pose
    from an unclaimed detection is given to the athlete only if its limb/torso keypoints sit on that
    athlete's gi colour AND mostly inside that athlete's box. Otherwise kp stays None (never invented).
    """
    from .occlusion import pose_owner, grow
    for fi, rec in enumerate(results):
        claimed = set()
        for r in rec.values():
            r["kp"], r["pose_conf"], r["pose_source"] = None, 0.0, None
            if r["status"] == "detected" and r["cand"] is not None:
                kp = keypoints_of(raw_dets[fi][r["cand"]])
                r.update(kp=kp.tolist(), pose_conf=float(kp[:, 2].mean()), pose_source="own_detection")
                claimed.add(r["cand"])
        if frames is None:
            continue
        for ident, r in rec.items():
            if r["status"] not in ("color_split", "color_recovered") or r["box"] is None:
                continue
            best = None
            region = grow(r["box"], frames[fi].shape, 0.25, y_floor_frac=0.0)
            for j, d in enumerate(raw_dets[fi]):
                if j in claimed or d[4] < 0.2:
                    continue
                kp = keypoints_of(d)
                owner, share, n = pose_owner(frames[fi], kp)
                if owner != ident:
                    continue
                conf = kp[:, 2] > 0.3
                if conf.sum() < 4:
                    continue
                inside = ((kp[conf, 0] >= region[0]) & (kp[conf, 0] <= region[2]) &
                          (kp[conf, 1] >= region[1]) & (kp[conf, 1] <= region[3])).mean()
                if inside < 0.6:
                    continue
                score = float(kp[:, 2].mean())
                if best is None or score > best[0]:
                    best = (score, j, kp)
            if best is not None:
                r.update(kp=best[2].tolist(), pose_conf=best[0] * 0.5, pose_source="colour_assigned_shared_detection")
    return results
