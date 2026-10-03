"""Attach COCO-17 keypoints to athlete records."""
import numpy as np

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
            "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def keypoints_of(det):
    """det row -> 17x3 array (x, y, conf)."""
    return np.asarray(det[5:], np.float32).reshape(17, 3)


MIN_OWNED_KP = 4


def attach_keypoints(results, raw_dets, frames=None, person_masks=None):
    """Add 'kp' (17x3 list or None), 'pose_conf' and 'pose_source' to each athlete record.

    'detected' records own their detection's pose. For colour-derived records (merged / recovered) the
    pose comes from an unclaimed detection, but only the keypoints whose own pixels carry this athlete's gi
    colour are kept (others are zeroed), and the pose needs >= MIN_OWNED_KP of them. Partial skeletons are
    expected; a pose is never copied wholesale onto an athlete.
    """
    from .occlusion import keypoint_owners
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
        pm = None if person_masks is None else person_masks[fi]
        for ident, r in rec.items():
            if r["status"] not in ("color_split", "color_recovered") or r["box"] is None:
                continue
            best = None
            for j, d in enumerate(raw_dets[fi]):
                if j in claimed or d[4] < 0.2:
                    continue
                kp = keypoints_of(d)
                own = np.array([o == ident for o in keypoint_owners(frames[fi], kp, pm)])
                if own.sum() < MIN_OWNED_KP:
                    continue
                if best is None or own.sum() > best[0]:
                    best = (int(own.sum()), kp, own)
            if best is not None:
                kp = best[1].copy()
                kp[~best[2], 2] = 0.0
                r.update(kp=kp.tolist(), pose_conf=float(kp[best[2], 2].mean()) * 0.5,
                         pose_source="colour_assigned_shared_detection")
    return results
