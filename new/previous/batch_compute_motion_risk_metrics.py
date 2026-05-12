import os
import re
import csv
import math
import glob
import argparse
from pathlib import Path

import numpy as np


# =========================================================
# 0. 22-joint HumanML / MDM joint index
# =========================================================
IDX = {
    "pelvis": 0,
    "l_hip": 1,
    "r_hip": 2,
    "spine1": 3,
    "l_knee": 4,
    "r_knee": 5,
    "spine2": 6,
    "l_ankle": 7,
    "r_ankle": 8,
    "spine3": 9,
    "l_foot": 10,
    "r_foot": 11,
    "neck": 12,
    "l_collar": 13,
    "r_collar": 14,
    "head": 15,
    "l_shoulder": 16,
    "r_shoulder": 17,
    "l_elbow": 18,
    "r_elbow": 19,
    "l_wrist": 20,
    "r_wrist": 21,
}


# =========================================================
# 1. args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch compute robust motion risk metrics from official MDM results.npy"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more results.npy files, directories, or glob patterns"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results.npy",
        help="Filename pattern when scanning directories"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frame rate"
    )
    parser.add_argument(
        "--up_axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Vertical axis"
    )
    parser.add_argument(
        "--height_thresh",
        type=float,
        default=0.05,
        help="Contact height threshold"
    )
    parser.add_argument(
        "--slide_speed_thresh",
        type=float,
        default=0.08,
        help="Threshold for pivot-aware sliding ratio"
    )
    parser.add_argument(
        "--ground_quantile",
        type=float,
        default=0.02,
        help="Ground estimation quantile"
    )
    parser.add_argument(
        "--foot_width",
        type=float,
        default=0.10,
        help="Foot width used for support polygon"
    )
    parser.add_argument(
        "--root_low_speed_thresh",
        type=float,
        default=0.15,
        help="Pelvis planar speed threshold for quasi-static frames"
    )
    parser.add_argument(
        "--foot_stable_speed_thresh",
        type=float,
        default=0.10,
        help="Foot planar speed threshold for quasi-static frames"
    )
    parser.add_argument(
        "--smooth_win",
        type=int,
        default=5,
        help="Smoothing window for CoM / XCoM"
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Optional cap per results.npy"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many top suspicious cases to save"
    )
    return parser.parse_args()


# =========================================================
# 2. utils
# =========================================================
def get_axis_indices(up_axis="y"):
    axis_map = {"x": 0, "y": 1, "z": 2}
    up = axis_map[up_axis]
    horiz = [0, 1, 2]
    horiz.remove(up)
    return up, horiz


def safe_norm(v, axis=-1, keepdims=False, eps=1e-8):
    return np.sqrt(np.sum(v * v, axis=axis, keepdims=keepdims) + eps)


def moving_average_1d(x, w=5):
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(xp, kernel, mode="valid")


def moving_average_2d(x, w=5):
    y = np.zeros_like(x, dtype=float)
    for d in range(x.shape[1]):
        y[:, d] = moving_average_1d(x[:, d], w=w)
    return y


def safe_mean(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def safe_min(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.min(x))


def percentile_rank(values, reverse=False):
    """
    返回每个元素的 [0,1] percentile，值越大越“可疑”
    reverse=False: 数值越大越可疑
    reverse=True : 数值越小越可疑
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)

    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return out

    vals = arr[valid]
    if reverse:
        vals = -vals

    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=float)
    if len(vals) == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = np.arange(len(vals), dtype=float) / (len(vals) - 1)

    out[np.where(valid)[0]] = ranks
    return out


# =========================================================
# 3. discover + load official MDM results.npy
# =========================================================
def discover_result_files(inputs, pattern="results.npy", recursive=False):
    found = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            if p.name == pattern or p.suffix == ".npy":
                found.append(str(p.resolve()))
            continue

        if p.is_dir():
            matches = list(p.rglob(pattern)) if recursive else list(p.glob(pattern))
            found.extend(str(m.resolve()) for m in matches)
            continue

        # glob pattern
        matches = glob.glob(item, recursive=recursive)
        for m in matches:
            mp = Path(m)
            if mp.is_file():
                found.append(str(mp.resolve()))

    return sorted(set(found))


def load_results(results_path):
    obj = np.load(results_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported results.npy format: {results_path}")


def extract_motion(data, sample_idx=0):
    motion = np.asarray(data["motion"])
    if motion.ndim != 4:
        raise ValueError(f"Expected motion shape (N, J, 3, T), got {motion.shape}")

    arr = motion[sample_idx]                 # (J, 3, T)
    arr = np.transpose(arr, (2, 0, 1))      # -> (T, J, 3)

    lengths = data.get("lengths", None)
    if lengths is not None:
        length = int(np.asarray(lengths)[sample_idx])
        arr = arr[:length]
    else:
        length = arr.shape[0]

    prompt = ""
    text = data.get("text", None)
    if text is not None and sample_idx < len(text):
        prompt = str(text[sample_idx])

    return arr, prompt, length


# =========================================================
# 4. geometry
# =========================================================
def distance_point_to_segment_2d(p, a, b):
    ap = p - a
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-12:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def convex_hull(points):
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def point_in_polygon(point, poly):
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def point_to_polygon_distance(point, poly):
    """
    inside  -> negative
    outside -> positive
    """
    if len(poly) == 0:
        return np.nan
    if len(poly) == 1:
        return np.linalg.norm(point - poly[0])
    if len(poly) == 2:
        return distance_point_to_segment_2d(point, poly[0], poly[1])

    inside = point_in_polygon(point, poly)
    dmin = min(
        distance_point_to_segment_2d(point, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )
    return -dmin if inside else dmin


def foot_support_rectangle(ankle_2d, toe_2d, width=0.10):
    d = toe_2d - ankle_2d
    n = np.linalg.norm(d)
    if n < 1e-8:
        d = np.array([1.0, 0.0])
    else:
        d = d / n

    perp = np.array([-d[1], d[0]])
    hw = width / 2.0

    corners = np.stack([
        ankle_2d + hw * perp,
        ankle_2d - hw * perp,
        toe_2d   - hw * perp,
        toe_2d   + hw * perp,
    ], axis=0)
    return corners


# =========================================================
# 5. motion preprocessing + features
# =========================================================
def normalize_ground(motion, up_idx, ground_quantile=0.02):
    foot_ids = [IDX["l_ankle"], IDX["r_ankle"], IDX["l_foot"], IDX["r_foot"]]
    foot_heights = motion[:, foot_ids, up_idx].reshape(-1)
    ground = np.quantile(foot_heights, ground_quantile)
    motion2 = motion.copy()
    motion2[:, :, up_idx] -= ground
    return motion2, ground


def compute_foot_features(motion, up_idx, horiz_idx, fps=20.0):
    la = motion[:, IDX["l_ankle"]]
    ra = motion[:, IDX["r_ankle"]]
    lf = motion[:, IDX["l_foot"]]
    rf = motion[:, IDX["r_foot"]]

    l_height = np.minimum(la[:, up_idx], lf[:, up_idx])
    r_height = np.minimum(ra[:, up_idx], rf[:, up_idx])

    l_center = 0.5 * (la[:, horiz_idx] + lf[:, horiz_idx])
    r_center = 0.5 * (ra[:, horiz_idx] + rf[:, horiz_idx])

    l_center_speed = np.zeros(len(motion))
    r_center_speed = np.zeros(len(motion))
    l_center_speed[1:] = safe_norm(np.diff(l_center, axis=0), axis=1) * fps
    r_center_speed[1:] = safe_norm(np.diff(r_center, axis=0), axis=1) * fps

    l_toe_speed = np.zeros(len(motion))
    r_toe_speed = np.zeros(len(motion))
    l_toe_speed[1:] = safe_norm(np.diff(lf[:, horiz_idx], axis=0), axis=1) * fps
    r_toe_speed[1:] = safe_norm(np.diff(rf[:, horiz_idx], axis=0), axis=1) * fps

    l_vec = lf[:, horiz_idx] - la[:, horiz_idx]
    r_vec = rf[:, horiz_idx] - ra[:, horiz_idx]

    l_len = safe_norm(l_vec, axis=1)
    r_len = safe_norm(r_vec, axis=1)

    l_theta = np.unwrap(np.arctan2(l_vec[:, 1], l_vec[:, 0]))
    r_theta = np.unwrap(np.arctan2(r_vec[:, 1], r_vec[:, 0]))

    l_ang_speed = np.zeros(len(motion))
    r_ang_speed = np.zeros(len(motion))
    l_ang_speed[1:] = np.abs(np.diff(l_theta)) * fps
    r_ang_speed[1:] = np.abs(np.diff(r_theta)) * fps

    # pivot-aware translational residual:
    # 允许一部分支撑脚平面运动由足部旋转解释掉
    l_rot_equiv = 0.5 * np.median(l_len) * l_ang_speed
    r_rot_equiv = 0.5 * np.median(r_len) * r_ang_speed

    l_slide_residual = np.maximum(0.0, l_center_speed - l_rot_equiv)
    r_slide_residual = np.maximum(0.0, r_center_speed - r_rot_equiv)

    return {
        "l_height": l_height,
        "r_height": r_height,
        "l_center_speed": l_center_speed,
        "r_center_speed": r_center_speed,
        "l_toe_speed": l_toe_speed,
        "r_toe_speed": r_toe_speed,
        "l_vec": l_vec,
        "r_vec": r_vec,
        "l_len": l_len,
        "r_len": r_len,
        "l_ang_speed": l_ang_speed,
        "r_ang_speed": r_ang_speed,
        "l_slide_residual": l_slide_residual,
        "r_slide_residual": r_slide_residual,
        "l_center_2d": l_center,
        "r_center_2d": r_center,
        "l_toe_2d": lf[:, horiz_idx],
        "r_toe_2d": rf[:, horiz_idx],
        "l_ankle_2d": la[:, horiz_idx],
        "r_ankle_2d": ra[:, horiz_idx],
    }


def detect_contacts(features, height_thresh=0.05):
    l_contact = features["l_height"] < height_thresh
    r_contact = features["r_height"] < height_thresh
    return l_contact, r_contact


def compute_root_features(motion, horiz_idx, fps=20.0):
    pelvis2d = motion[:, IDX["pelvis"], horiz_idx]
    pelvis_vel = np.zeros(len(motion))
    pelvis_vel[1:] = safe_norm(np.diff(pelvis2d, axis=0), axis=1) * fps
    return {
        "pelvis2d": pelvis2d,
        "pelvis_vel": pelvis_vel,
    }


def estimate_com(motion):
    J = motion
    T = J.shape[0]

    segs = []
    weights = []

    trunk_center = np.mean(
        J[:, [IDX["pelvis"], IDX["spine1"], IDX["spine2"], IDX["spine3"], IDX["neck"]]],
        axis=1
    )
    segs.append(trunk_center); weights.append(0.50)

    head_center = 0.5 * (J[:, IDX["neck"]] + J[:, IDX["head"]])
    segs.append(head_center); weights.append(0.08)

    segs.append(0.5 * (J[:, IDX["l_shoulder"]] + J[:, IDX["l_elbow"]]))
    weights.append(0.03)
    segs.append(0.5 * (J[:, IDX["r_shoulder"]] + J[:, IDX["r_elbow"]]))
    weights.append(0.03)

    segs.append(0.5 * (J[:, IDX["l_elbow"]] + J[:, IDX["l_wrist"]]))
    weights.append(0.02)
    segs.append(0.5 * (J[:, IDX["r_elbow"]] + J[:, IDX["r_wrist"]]))
    weights.append(0.02)

    segs.append(0.5 * (J[:, IDX["l_hip"]] + J[:, IDX["l_knee"]]))
    weights.append(0.10)
    segs.append(0.5 * (J[:, IDX["r_hip"]] + J[:, IDX["r_knee"]]))
    weights.append(0.10)

    segs.append(0.5 * (J[:, IDX["l_knee"]] + J[:, IDX["l_ankle"]]))
    weights.append(0.0465)
    segs.append(0.5 * (J[:, IDX["r_knee"]] + J[:, IDX["r_ankle"]]))
    weights.append(0.0465)

    segs.append(0.5 * (J[:, IDX["l_ankle"]] + J[:, IDX["l_foot"]]))
    weights.append(0.0145)
    segs.append(0.5 * (J[:, IDX["r_ankle"]] + J[:, IDX["r_foot"]]))
    weights.append(0.0145)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    com = np.zeros((T, 3), dtype=float)
    for s, w in zip(segs, weights):
        com += w * s
    return com


def build_bos_polygons(motion, horiz_idx, l_contact, r_contact, foot_width=0.10):
    T = motion.shape[0]
    bos_polys = []
    for t in range(T):
        polys = []

        if l_contact[t]:
            polys.append(foot_support_rectangle(
                motion[t, IDX["l_ankle"], horiz_idx],
                motion[t, IDX["l_foot"], horiz_idx],
                width=foot_width
            ))
        if r_contact[t]:
            polys.append(foot_support_rectangle(
                motion[t, IDX["r_ankle"], horiz_idx],
                motion[t, IDX["r_foot"], horiz_idx],
                width=foot_width
            ))

        if len(polys) == 0:
            polys.append(foot_support_rectangle(
                motion[t, IDX["l_ankle"], horiz_idx],
                motion[t, IDX["l_foot"], horiz_idx],
                width=foot_width
            ))

        pts = np.concatenate(polys, axis=0)
        bos_polys.append(convex_hull(pts))
    return bos_polys


# =========================================================
# 6. mode inference
# =========================================================
def infer_analysis_mode(prompt, root_features, l_contact, r_contact):
    text = (prompt or "").lower()

    walk_keywords = ["walk", "walking", "turn", "step", "kick", "backward", "sideways", "pivot"]
    bend_keywords = ["bend", "stoop", "squat", "pick", "reach down", "sit", "stand up", "crouch", "kneel", "lift"]

    walk_hit = any(k in text for k in walk_keywords)
    bend_hit = any(k in text for k in bend_keywords)

    pelvis2d = root_features["pelvis2d"]
    total_disp = float(np.linalg.norm(pelvis2d[-1] - pelvis2d[0]))
    single_support_ratio = float(np.mean(np.logical_xor(l_contact, r_contact)))

    if walk_hit and not bend_hit:
        return "walk"
    if bend_hit and not walk_hit:
        return "bend"

    if total_disp > 0.8 and single_support_ratio > 0.15:
        return "walk"
    return "bend"


# =========================================================
# 7. metrics
# =========================================================
def compute_contact_metrics(features, l_contact, r_contact, slide_speed_thresh):
    l_slide_mean_raw = safe_mean(features["l_center_speed"][l_contact])
    r_slide_mean_raw = safe_mean(features["r_center_speed"][r_contact])

    l_slide_mean_pivot = safe_mean(features["l_slide_residual"][l_contact])
    r_slide_mean_pivot = safe_mean(features["r_slide_residual"][r_contact])

    l_slide_ratio_pivot = safe_mean((features["l_slide_residual"][l_contact] > slide_speed_thresh).astype(float))
    r_slide_ratio_pivot = safe_mean((features["r_slide_residual"][r_contact] > slide_speed_thresh).astype(float))

    l_penetration = np.maximum(0.0, -features["l_height"])
    r_penetration = np.maximum(0.0, -features["r_height"])

    slide_mean_pivot = safe_mean([
        l_slide_mean_pivot,
        r_slide_mean_pivot
    ])
    slide_ratio_pivot = safe_mean([
        l_slide_ratio_pivot,
        r_slide_ratio_pivot
    ])

    return {
        "left_slide_mean_raw": l_slide_mean_raw,
        "right_slide_mean_raw": r_slide_mean_raw,
        "left_slide_mean_pivot": l_slide_mean_pivot,
        "right_slide_mean_pivot": r_slide_mean_pivot,
        "left_slide_ratio_pivot": l_slide_ratio_pivot,
        "right_slide_ratio_pivot": r_slide_ratio_pivot,
        "slide_mean_pivot": slide_mean_pivot,
        "slide_ratio_pivot": slide_ratio_pivot,
        "penetration_mean": safe_mean(np.r_[l_penetration, r_penetration]),
        "penetration_max": safe_min(-np.r_[l_penetration, r_penetration]) * -1 if np.any(np.r_[l_penetration, r_penetration] > 0) else 0.0,
        "contact_height_mean": safe_mean(np.r_[features["l_height"][l_contact], features["r_height"][r_contact]]),
        "left_contact_ratio": float(np.mean(l_contact)),
        "right_contact_ratio": float(np.mean(r_contact)),
    }


def compute_root_jerk(motion, fps=20.0):
    pelvis = motion[:, IDX["pelvis"], :]
    if len(pelvis) < 4:
        return np.nan
    v = np.diff(pelvis, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    j = np.diff(a, axis=0) * fps
    return float(np.mean(safe_norm(j, axis=1)))


def compute_quasistatic_metrics(
    motion, com2d, bos_polys, com_margin,
    root_features, features, l_contact, r_contact,
    root_low_speed_thresh=0.15, foot_stable_speed_thresh=0.10
):
    low_speed = root_features["pelvis_vel"] < root_low_speed_thresh
    stable_contact = (
        low_speed
        & l_contact & r_contact
        & (features["l_center_speed"] < foot_stable_speed_thresh)
        & (features["r_center_speed"] < foot_stable_speed_thresh)
    )

    margin_sel = com_margin[stable_contact]

    return {
        "num_low_speed_frames": int(np.sum(low_speed)),
        "num_qs_frames": int(np.sum(stable_contact)),
        "qs_frame_ratio": float(np.mean(stable_contact)),
        "qs_margin_mean": safe_mean(margin_sel),
        "qs_margin_min": safe_min(margin_sel),
        "qs_negative_ratio": safe_mean((margin_sel < 0).astype(float)) if len(margin_sel) > 0 else np.nan,
    }


def compute_walk_metrics(
    com, com2d, bos_polys, up_idx, l_contact, r_contact,
    fps=20.0, smooth_win=5
):
    com2d_smooth = moving_average_2d(com2d, w=smooth_win)

    v2d = np.zeros_like(com2d_smooth)
    v2d[1:] = (com2d_smooth[1:] - com2d_smooth[:-1]) * fps

    l_pendulum = float(np.mean(com[:, up_idx]))
    l_pendulum = max(l_pendulum, 1e-3)
    omega0 = math.sqrt(9.81 / l_pendulum)

    xcom2d = com2d_smooth + v2d / omega0

    mos_signed = np.zeros(len(com2d), dtype=float)
    for t in range(len(com2d)):
        mos_signed[t] = point_to_polygon_distance(xcom2d[t], bos_polys[t])

    # inside positive, outside negative
    mos = -mos_signed
    stance_mask = l_contact | r_contact
    mos_stance = mos[stance_mask]

    return {
        "pendulum_length": l_pendulum,
        "omega0": omega0,
        "stance_ratio": float(np.mean(stance_mask)),
        "mos_mean": safe_mean(mos_stance),
        "mos_min": safe_min(mos_stance),
        "mos_negative_ratio": safe_mean((mos_stance < 0).astype(float)) if len(mos_stance) > 0 else np.nan,
    }


def compute_case_metrics(
    motion, prompt, results_path, sample_idx,
    fps=20.0, up_axis="y", height_thresh=0.05,
    slide_speed_thresh=0.08, ground_quantile=0.02,
    foot_width=0.10, root_low_speed_thresh=0.15,
    foot_stable_speed_thresh=0.10, smooth_win=5
):
    up_idx, horiz_idx = get_axis_indices(up_axis)

    motion, ground = normalize_ground(motion, up_idx=up_idx, ground_quantile=ground_quantile)
    features = compute_foot_features(motion, up_idx=up_idx, horiz_idx=horiz_idx, fps=fps)
    l_contact, r_contact = detect_contacts(features, height_thresh=height_thresh)
    root_features = compute_root_features(motion, horiz_idx=horiz_idx, fps=fps)

    mode = infer_analysis_mode(prompt, root_features, l_contact, r_contact)

    com = estimate_com(motion)
    com2d = com[:, horiz_idx]
    bos_polys = build_bos_polygons(motion, horiz_idx, l_contact, r_contact, foot_width=foot_width)

    signed_dist = np.array([point_to_polygon_distance(com2d[t], bos_polys[t]) for t in range(len(com2d))])
    com_margin = -signed_dist  # inside positive, outside negative

    row = {
        "results_path": results_path,
        "sample_idx": sample_idx,
        "prompt": prompt,
        "num_frames": motion.shape[0],
        "analysis_mode": mode,
        "estimated_ground_offset": ground,
        "root_jerk": compute_root_jerk(motion, fps=fps),
    }

    row.update(compute_contact_metrics(features, l_contact, r_contact, slide_speed_thresh))

    if mode == "walk":
        row.update(compute_walk_metrics(
            com=com,
            com2d=com2d,
            bos_polys=bos_polys,
            up_idx=up_idx,
            l_contact=l_contact,
            r_contact=r_contact,
            fps=fps,
            smooth_win=smooth_win
        ))
        row["qs_frame_ratio"] = np.nan
        row["qs_margin_mean"] = np.nan
        row["qs_margin_min"] = np.nan
        row["qs_negative_ratio"] = np.nan
        row["num_qs_frames"] = 0
        row["num_low_speed_frames"] = 0
    else:
        row.update(compute_quasistatic_metrics(
            motion=motion,
            com2d=com2d,
            bos_polys=bos_polys,
            com_margin=com_margin,
            root_features=root_features,
            features=features,
            l_contact=l_contact,
            r_contact=r_contact,
            root_low_speed_thresh=root_low_speed_thresh,
            foot_stable_speed_thresh=foot_stable_speed_thresh
        ))
        row["stance_ratio"] = float(np.mean(l_contact | r_contact))
        row["mos_mean"] = np.nan
        row["mos_min"] = np.nan
        row["mos_negative_ratio"] = np.nan
        row["pendulum_length"] = np.nan
        row["omega0"] = np.nan

    return row


# =========================================================
# 8. summaries + ranking
# =========================================================
def group_by_prompt(rows):
    groups = {}
    for r in rows:
        key = (r["prompt"], r["analysis_mode"])
        groups.setdefault(key, []).append(r)

    out = []
    metric_names = [
        "slide_mean_pivot",
        "slide_ratio_pivot",
        "penetration_max",
        "contact_height_mean",
        "root_jerk",
        "mos_mean",
        "mos_min",
        "mos_negative_ratio",
        "qs_margin_mean",
        "qs_margin_min",
        "qs_negative_ratio",
        "overall_contact_score",
        "overall_balance_score",
        "overall_suspicion_score",
    ]

    for (prompt, mode), items in groups.items():
        row = {
            "prompt": prompt,
            "analysis_mode": mode,
            "num_cases": len(items),
        }
        for m in metric_names:
            vals = [it.get(m, np.nan) for it in items]
            row[m + "_mean"] = safe_mean(vals)
        out.append(row)

    out.sort(key=lambda x: (x["analysis_mode"], -(x.get("overall_suspicion_score_mean", np.nan) if np.isfinite(x.get("overall_suspicion_score_mean", np.nan)) else -1)))
    return out


def add_percentile_scores(rows):
    # 定义哪些指标越大越可疑，哪些越小越可疑
    high_worse = [
        "slide_mean_pivot",
        "slide_ratio_pivot",
        "penetration_max",
        "contact_height_mean",
        "root_jerk",
        "mos_negative_ratio",
        "qs_negative_ratio",
    ]
    low_worse = [
        "mos_min",
        "qs_margin_min",
    ]

    for m in high_worse:
        vals = [r.get(m, np.nan) for r in rows]
        pct = percentile_rank(vals, reverse=False)
        for r, p in zip(rows, pct):
            r[m + "_pct"] = p

    for m in low_worse:
        vals = [r.get(m, np.nan) for r in rows]
        pct = percentile_rank(vals, reverse=True)
        for r, p in zip(rows, pct):
            r[m + "_pct"] = p

    # contact score
    for r in rows:
        contact_parts = [
            r.get("slide_mean_pivot_pct", np.nan),
            r.get("slide_ratio_pivot_pct", np.nan),
            r.get("penetration_max_pct", np.nan),
        ]
        r["overall_contact_score"] = safe_mean(contact_parts)

        if r["analysis_mode"] == "walk":
            balance_parts = [
                r.get("mos_negative_ratio_pct", np.nan),
                r.get("mos_min_pct", np.nan),
            ]
        else:
            balance_parts = [
                r.get("qs_negative_ratio_pct", np.nan),
                r.get("qs_margin_min_pct", np.nan),
            ]

        r["overall_balance_score"] = safe_mean(balance_parts)
        r["overall_suspicion_score"] = safe_mean([
            r["overall_contact_score"],
            r["overall_balance_score"],
        ])

    return rows


def write_csv(path, rows):
    if len(rows) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return

    fieldnames = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# =========================================================
# 9. main
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    result_files = discover_result_files(
        inputs=args.inputs,
        pattern=args.pattern,
        recursive=args.recursive
    )

    if len(result_files) == 0:
        print("No results.npy files found.")
        return

    print(f"Found {len(result_files)} results.npy files.")

    rows = []

    for results_path in result_files:
        print("=" * 80)
        print("Processing:", results_path)
        data = load_results(results_path)

        motion = np.asarray(data["motion"])
        num_cases = motion.shape[0]
        if args.max_samples_per_file is not None:
            num_cases = min(num_cases, args.max_samples_per_file)

        for sample_idx in range(num_cases):
            try:
                motion_tjc, prompt, _ = extract_motion(data, sample_idx=sample_idx)
                row = compute_case_metrics(
                    motion=motion_tjc,
                    prompt=prompt,
                    results_path=results_path,
                    sample_idx=sample_idx,
                    fps=args.fps,
                    up_axis=args.up_axis,
                    height_thresh=args.height_thresh,
                    slide_speed_thresh=args.slide_speed_thresh,
                    ground_quantile=args.ground_quantile,
                    foot_width=args.foot_width,
                    root_low_speed_thresh=args.root_low_speed_thresh,
                    foot_stable_speed_thresh=args.foot_stable_speed_thresh,
                    smooth_win=args.smooth_win
                )
                rows.append(row)
            except Exception as e:
                rows.append({
                    "results_path": results_path,
                    "sample_idx": sample_idx,
                    "prompt": "",
                    "analysis_mode": "error",
                    "error": str(e),
                })

    # 先给 percentile / suspicion score
    valid_rows = [r for r in rows if r.get("analysis_mode") != "error"]
    valid_rows = add_percentile_scores(valid_rows)

    error_rows = [r for r in rows if r.get("analysis_mode") == "error"]
    all_rows = valid_rows + error_rows

    # case-level csv
    case_csv = os.path.join(args.outdir, "case_metrics.csv")
    write_csv(case_csv, all_rows)

    # prompt-level summary
    prompt_rows = group_by_prompt(valid_rows)
    prompt_csv = os.path.join(args.outdir, "prompt_summary.csv")
    write_csv(prompt_csv, prompt_rows)

    # top suspect
    suspect_rows = [r for r in valid_rows if np.isfinite(r.get("overall_suspicion_score", np.nan))]
    suspect_rows.sort(key=lambda x: x["overall_suspicion_score"], reverse=True)
    topk_rows = suspect_rows[:args.topk]
    topk_csv = os.path.join(args.outdir, "top_suspect_cases.csv")
    write_csv(topk_csv, topk_rows)

    print("=" * 80)
    print("Done.")
    print("case-level csv   :", case_csv)
    print("prompt summary   :", prompt_csv)
    print("top suspect csv  :", topk_csv)


if __name__ == "__main__":
    main()