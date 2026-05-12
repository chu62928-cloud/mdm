import os
import csv
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# =========================================================
# 0. 参数
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze official MDM results.npy with foot contact + adaptive balance analysis"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to official MDM results.npy")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index inside results.npy")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Frame rate")
    parser.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"],
                        help="Vertical axis")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "walk", "bend"],
                        help="Balance analysis mode")
    parser.add_argument("--height_thresh", type=float, default=0.05,
                        help="Foot contact height threshold after ground normalization")
    parser.add_argument("--speed_thresh", type=float, default=0.08,
                        help="Foot skating speed threshold")
    parser.add_argument("--foot_width", type=float, default=0.10,
                        help="Foot support width for BoS rectangle")
    parser.add_argument("--ground_quantile", type=float, default=0.02,
                        help="Ground estimation quantile")
    parser.add_argument("--root_low_speed_thresh", type=float, default=0.15,
                        help="Pelvis planar speed threshold for bend-like low-speed frames")
    parser.add_argument("--foot_stable_speed_thresh", type=float, default=0.10,
                        help="Foot planar speed threshold for stable-contact bend frames")
    parser.add_argument("--smooth_win", type=int, default=5,
                        help="Smoothing window for CoM/XCoM")
    parser.add_argument("--title", type=str, default="MDM Case Analysis",
                        help="Figure title prefix")
    return parser.parse_args()


# =========================================================
# 1. 工具函数
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

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


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
    返回 signed distance:
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


def save_metrics_csv(out_csv, metrics_dict):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])


# =========================================================
# 2. 读取官方 MDM results.npy
# =========================================================

def load_mdm_results(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported npy format: {path}")


def extract_motion_from_results(data, sample_idx=0):
    if "motion" not in data:
        raise KeyError("'motion' key not found in results.npy")

    motion = np.asarray(data["motion"])
    if motion.ndim != 4:
        raise ValueError(f"Expected motion shape (N, J, 3, T), got {motion.shape}")

    arr = motion[sample_idx]                 # (J, 3, T)
    arr = np.transpose(arr, (2, 0, 1))      # -> (T, J, 3)

    lengths = data.get("lengths", None)
    text = data.get("text", None)

    length = arr.shape[0]
    if lengths is not None:
        length = int(np.asarray(lengths)[sample_idx])
        arr = arr[:length]

    prompt = ""
    if text is not None and sample_idx < len(text):
        prompt = str(text[sample_idx])

    return arr, prompt, length


# =========================================================
# 3. 22-joint 索引
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
# 4. 预处理 + foot features
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

    l_vel = np.zeros(len(motion))
    r_vel = np.zeros(len(motion))
    l_vel[1:] = safe_norm(np.diff(l_center, axis=0), axis=1) * fps
    r_vel[1:] = safe_norm(np.diff(r_center, axis=0), axis=1) * fps

    return {
        "l_height": l_height,
        "r_height": r_height,
        "l_vel": l_vel,
        "r_vel": r_vel,
        "l_traj": lf[:, horiz_idx],
        "r_traj": rf[:, horiz_idx],
        "l_center_2d": l_center,
        "r_center_2d": r_center,
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


# =========================================================
# 5. CoM + BoS
# =========================================================

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


def build_bos_polygon(motion, t, horiz_idx, l_contact, r_contact, foot_width=0.10):
    la = motion[t, IDX["l_ankle"], horiz_idx]
    ra = motion[t, IDX["r_ankle"], horiz_idx]
    lf = motion[t, IDX["l_foot"], horiz_idx]
    rf = motion[t, IDX["r_foot"], horiz_idx]

    polys = []

    if l_contact[t]:
        polys.append(foot_support_rectangle(la, lf, width=foot_width))
    if r_contact[t]:
        polys.append(foot_support_rectangle(ra, rf, width=foot_width))

    if len(polys) == 0:
        polys.append(foot_support_rectangle(la, lf, width=foot_width))

    pts = np.concatenate(polys, axis=0)
    hull = convex_hull(pts)
    return hull


def compute_bos_and_com_margin(motion, horiz_idx, l_contact, r_contact, foot_width=0.10):
    com = estimate_com(motion)
    com2d = com[:, horiz_idx]

    T = motion.shape[0]
    bos_polys = []
    signed_dist = np.zeros(T, dtype=float)

    for t in range(T):
        poly = build_bos_polygon(
            motion, t, horiz_idx,
            l_contact=l_contact,
            r_contact=r_contact,
            foot_width=foot_width
        )
        bos_polys.append(poly)
        signed_dist[t] = point_to_polygon_distance(com2d[t], poly)

    # 我们更喜欢 “margin > 0 inside, margin < 0 outside”
    com_margin = -signed_dist

    return {
        "com": com,
        "com2d": com2d,
        "bos_polys": bos_polys,
        "signed_dist": signed_dist,   # inside negative, outside positive
        "com_margin": com_margin,     # inside positive, outside negative
    }


# =========================================================
# 6. walk-like: XCoM / MoS
# =========================================================

def compute_xcom_mos(com, com2d, bos_polys, up_idx, fps=20.0, smooth_win=5):
    com2d_smooth = moving_average_2d(com2d, w=smooth_win)

    v2d = np.zeros_like(com2d_smooth)
    v2d[1:] = (com2d_smooth[1:] - com2d_smooth[:-1]) * fps

    l_pendulum = float(np.mean(com[:, up_idx]))
    l_pendulum = max(l_pendulum, 1e-3)
    omega0 = math.sqrt(9.81 / l_pendulum)

    xcom2d = com2d_smooth + v2d / omega0

    T = len(com2d)
    mos_signed = np.zeros(T, dtype=float)
    for t in range(T):
        mos_signed[t] = point_to_polygon_distance(xcom2d[t], bos_polys[t])

    # inside positive, outside negative
    mos = -mos_signed

    return {
        "com2d_smooth": com2d_smooth,
        "v2d": v2d,
        "xcom2d": xcom2d,
        "omega0": omega0,
        "pendulum_length": l_pendulum,
        "mos_signed": mos_signed,  # inside negative, outside positive
        "mos": mos,                # inside positive, outside negative
    }


# =========================================================
# 7. bend-like: 低速支撑帧 CoM margin
# =========================================================

def compute_low_speed_com_margin(
    com_margin,
    pelvis_vel,
    l_contact,
    r_contact,
    l_foot_vel,
    r_foot_vel,
    root_low_speed_thresh=0.15,
    foot_stable_speed_thresh=0.10
):
    low_speed = pelvis_vel < root_low_speed_thresh

    stable_contact = (
        low_speed
        & l_contact & r_contact
        & (l_foot_vel < foot_stable_speed_thresh)
        & (r_foot_vel < foot_stable_speed_thresh)
    )

    return {
        "low_speed": low_speed,
        "stable_contact": stable_contact,
        "com_margin_low_speed": com_margin[stable_contact] if np.any(stable_contact) else np.array([]),
    }


# =========================================================
# 8. 模式推断
# =========================================================

def infer_analysis_mode(prompt, root_features, l_contact, r_contact):
    text = (prompt or "").lower()

    walk_keywords = ["walk", "walking", "turn", "step", "kick", "backward", "sideways", "pivot"]
    bend_keywords = ["bend", "stoop", "squat", "pick", "reach down", "sit", "stand up", "crouch", "kneel", "lift"]

    walk_hit = any(k in text for k in walk_keywords)
    bend_hit = any(k in text for k in bend_keywords)

    pelvis2d = root_features["pelvis2d"]
    total_disp = float(np.linalg.norm(pelvis2d[-1] - pelvis2d[0]))

    single_support_ratio = np.mean(np.logical_xor(l_contact, r_contact))

    if walk_hit and not bend_hit:
        return "walk"
    if bend_hit and not walk_hit:
        return "bend"

    if total_disp > 0.8 and single_support_ratio > 0.15:
        return "walk"
    return "bend"


# =========================================================
# 9. Foot metrics
# =========================================================

def compute_foot_metrics(features, l_contact, r_contact, speed_thresh):
    l_skating_mean = np.mean(features["l_vel"][l_contact]) if np.any(l_contact) else np.nan
    r_skating_mean = np.mean(features["r_vel"][r_contact]) if np.any(r_contact) else np.nan

    l_skating_ratio = np.mean(features["l_vel"][l_contact] > speed_thresh) if np.any(l_contact) else np.nan
    r_skating_ratio = np.mean(features["r_vel"][r_contact] > speed_thresh) if np.any(r_contact) else np.nan

    l_penetration_mean = np.mean(np.maximum(0.0, -features["l_height"]))
    r_penetration_mean = np.mean(np.maximum(0.0, -features["r_height"]))

    l_contact_height_mean = np.mean(features["l_height"][l_contact]) if np.any(l_contact) else np.nan
    r_contact_height_mean = np.mean(features["r_height"][r_contact]) if np.any(r_contact) else np.nan

    return {
        "left_skating_mean": l_skating_mean,
        "right_skating_mean": r_skating_mean,
        "left_skating_ratio": l_skating_ratio,
        "right_skating_ratio": r_skating_ratio,
        "left_penetration_mean": l_penetration_mean,
        "right_penetration_mean": r_penetration_mean,
        "left_contact_height_mean": l_contact_height_mean,
        "right_contact_height_mean": r_contact_height_mean,
    }


def compute_root_jerk(motion, fps=20.0):
    pelvis = motion[:, IDX["pelvis"], :]
    if len(pelvis) < 4:
        return np.nan
    v = np.diff(pelvis, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    j = np.diff(a, axis=0) * fps
    return float(np.mean(safe_norm(j, axis=1)))


# =========================================================
# 10. 绘图：Foot Contact
# =========================================================

def plot_foot_contact_figure(
    motion, features, l_contact, r_contact, height_thresh, speed_thresh,
    outpath, title="Foot Contact Analysis"
):
    T = motion.shape[0]
    frames = np.arange(T)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.4], hspace=0.35)

    # (A) height
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(frames, features["l_height"], linewidth=2, label="Left foot height")
    ax1.plot(frames, features["r_height"], linewidth=2, label="Right foot height")
    ax1.axhline(height_thresh, linestyle="--", linewidth=1.5, label="Contact threshold")
    ax1.set_title("(A) Foot height over time")
    ax1.set_ylabel("Height")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", ncol=3)

    # (B) speed
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(frames, features["l_vel"], linewidth=2, label="Left foot planar speed")
    ax2.plot(frames, features["r_vel"], linewidth=2, label="Right foot planar speed")
    ax2.axhline(speed_thresh, linestyle="--", linewidth=1.5, label="Skating threshold")
    ax2.set_title("(B) Foot planar speed over time")
    ax2.set_ylabel("Speed")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", ncol=3)

    l_skate = l_contact & (features["l_vel"] > speed_thresh)
    r_skate = r_contact & (features["r_vel"] > speed_thresh)
    ax2.scatter(frames[l_skate], features["l_vel"][l_skate], s=28, marker="o", label="_nolegend_")
    ax2.scatter(frames[r_skate], features["r_vel"][r_skate], s=28, marker="s", label="_nolegend_")

    # (C) trajectory
    ax3 = fig.add_subplot(gs[2, 0])
    lxy = features["l_traj"]
    rxy = features["r_traj"]

    ax3.plot(lxy[:, 0], lxy[:, 1], linewidth=2, label="Left toe trajectory")
    ax3.plot(rxy[:, 0], rxy[:, 1], linewidth=2, label="Right toe trajectory")

    ax3.scatter(lxy[l_contact, 0], lxy[l_contact, 1], s=18, alpha=0.8, label="Left contact frames")
    ax3.scatter(rxy[r_contact, 0], rxy[r_contact, 1], s=18, alpha=0.8, label="Right contact frames")

    ax3.scatter(lxy[l_skate, 0], lxy[l_skate, 1], s=40, marker="x", linewidths=1.5, label="Left skating frames")
    ax3.scatter(rxy[r_skate, 0], rxy[r_skate, 1], s=40, marker="x", linewidths=1.5, label="Right skating frames")

    ax3.set_title("(C) Foot trajectories on the ground plane")
    ax3.set_xlabel("Horizontal axis 1")
    ax3.set_ylabel("Horizontal axis 2")
    ax3.axis("equal")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="best", ncol=2)

    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 11. 绘图：walk-like -> XCoM / MoS
# =========================================================

def plot_walk_balance_figure(
    motion, com2d, bos_polys, xcom2d, mos,
    l_contact, r_contact, outpath, title="Dynamic Balance Analysis (XCoM / MoS)"
):
    T = motion.shape[0]
    frames = np.arange(T)

    stance_mask = l_contact | r_contact
    mos_negative = mos < 0

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.35)

    # (A) CoM + XCoM + BoS
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(com2d[:, 0], com2d[:, 1], linewidth=2.0, alpha=0.7, label="Projected CoM trajectory")
    ax1.plot(xcom2d[:, 0], xcom2d[:, 1], linewidth=2.0, alpha=0.7, label="XCoM trajectory")

    key_frames = np.linspace(0, T - 1, min(6, T), dtype=int)
    for t in key_frames:
        poly = bos_polys[t]
        if len(poly) >= 3:
            patch = Polygon(poly, closed=True, fill=True, alpha=0.12, linewidth=1.3)
            ax1.add_patch(patch)
            ax1.text(xcom2d[t, 0], xcom2d[t, 1], f"{t}", fontsize=9)

    inside_idx = np.where(~mos_negative)[0]
    outside_idx = np.where(mos_negative)[0]
    if len(inside_idx) > 0:
        ax1.scatter(xcom2d[inside_idx, 0], xcom2d[inside_idx, 1], s=14, alpha=0.7, label="XCoM inside BoS")
    if len(outside_idx) > 0:
        ax1.scatter(xcom2d[outside_idx, 0], xcom2d[outside_idx, 1], s=16, alpha=0.85, label="XCoM outside BoS")

    ax1.set_title("(A) CoM / XCoM trajectories and Base of Support")
    ax1.set_xlabel("Horizontal axis 1")
    ax1.set_ylabel("Horizontal axis 2")
    ax1.axis("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", ncol=2)

    # (B) MoS
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(frames, mos, linewidth=2, label="Margin of Stability (MoS)")
    ax2.axhline(0.0, linestyle="--", linewidth=1.5)
    ax2.fill_between(frames, 0, mos, where=(mos < 0), alpha=0.25, interpolate=True,
                     label="Negative MoS (risk region)")

    ymin, ymax = ax2.get_ylim()
    band_h = (ymax - ymin) * 0.05 if ymax > ymin else 0.05
    ybase = ymin + band_h * 0.5
    for t in range(T):
        if l_contact[t]:
            ax2.plot([t, t], [ybase, ybase + band_h], linewidth=2.0)
        if r_contact[t]:
            ax2.plot([t, t], [ybase + 1.5 * band_h, ybase + 2.5 * band_h], linewidth=2.0)
    ax2.text(1, ybase + 0.55 * band_h, "L contact", fontsize=9, va="center")
    ax2.text(1, ybase + 2.05 * band_h, "R contact", fontsize=9, va="center")

    ax2.set_title("(B) Margin of Stability over time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("MoS")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 12. 绘图：bend-like -> low-speed CoM margin
# =========================================================

def plot_bend_balance_figure(
    motion, com2d, bos_polys, com_margin, stable_contact, pelvis_vel,
    outpath, title="Quasi-static Balance Analysis (Low-speed CoM Margin)"
):
    T = motion.shape[0]
    frames = np.arange(T)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.8, 1.0], hspace=0.32)

    # (A) CoM + BoS
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(com2d[:, 0], com2d[:, 1], linewidth=2.0, alpha=0.85, label="Projected CoM trajectory")

    key_frames = np.where(stable_contact)[0]
    if len(key_frames) > 0:
        # 最多抽 6 个关键低速帧
        idx = np.linspace(0, len(key_frames) - 1, min(6, len(key_frames)), dtype=int)
        key_frames = key_frames[idx]
    else:
        key_frames = np.linspace(0, T - 1, min(6, T), dtype=int)

    inside_idx = np.where(com_margin >= 0)[0]
    outside_idx = np.where(com_margin < 0)[0]
    if len(inside_idx) > 0:
        ax1.scatter(com2d[inside_idx, 0], com2d[inside_idx, 1], s=14, alpha=0.7, label="CoM inside BoS")
    if len(outside_idx) > 0:
        ax1.scatter(com2d[outside_idx, 0], com2d[outside_idx, 1], s=16, alpha=0.85, label="CoM outside BoS")

    for t in key_frames:
        poly = bos_polys[t]
        if len(poly) >= 3:
            patch = Polygon(poly, closed=True, fill=True, alpha=0.12, linewidth=1.3)
            ax1.add_patch(patch)
            ax1.text(com2d[t, 0], com2d[t, 1], f"{t}", fontsize=9)

    ax1.set_title("(A) Projected CoM trajectory and Base of Support")
    ax1.set_xlabel("Horizontal axis 1")
    ax1.set_ylabel("Horizontal axis 2")
    ax1.axis("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", ncol=2)

    # (B) root speed
    ax2 = fig.add_subplot(gs[1, 0], sharex=None)
    ax2.plot(frames, pelvis_vel, linewidth=2, label="Pelvis planar speed")
    ax2.scatter(frames[stable_contact], pelvis_vel[stable_contact], s=18, label="Low-speed support frames")
    ax2.set_title("(B) Root speed and selected quasi-static frames")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Speed")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    # (C) CoM margin
    ax3 = fig.add_subplot(gs[2, 0], sharex=None)
    ax3.plot(frames, com_margin, linewidth=2, label="CoM margin")
    ax3.axhline(0.0, linestyle="--", linewidth=1.5)
    ax3.fill_between(frames, 0, com_margin, where=(com_margin < 0), alpha=0.25, interpolate=True,
                     label="Negative margin (outside BoS)")

    ymin, ymax = ax3.get_ylim()
    band_h = (ymax - ymin) * 0.06 if ymax > ymin else 0.06
    ybase = ymin + 0.5 * band_h
    for t in np.where(stable_contact)[0]:
        ax3.plot([t, t], [ybase, ybase + band_h], linewidth=2.0)
    ax3.text(1, ybase + 0.5 * band_h, "Selected low-speed support frames", fontsize=9, va="center")

    ax3.set_title("(C) CoM margin over time")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("CoM margin")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper right")

    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 13. 主程序
# =========================================================

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = load_mdm_results(args.input)
    motion, prompt, length = extract_motion_from_results(data, sample_idx=args.sample_idx)

    if motion.shape[1] < 22:
        raise ValueError(
            f"This script assumes a 22-joint HumanML3D-style skeleton. "
            f"Current motion shape is {motion.shape}."
        )

    up_idx, horiz_idx = get_axis_indices(args.up_axis)

    # ground normalize
    motion, ground = normalize_ground(
        motion, up_idx=up_idx, ground_quantile=args.ground_quantile
    )

    # features
    features = compute_foot_features(
        motion, up_idx=up_idx, horiz_idx=horiz_idx, fps=args.fps
    )
    l_contact, r_contact = detect_contacts(
        features, height_thresh=args.height_thresh
    )
    root_features = compute_root_features(
        motion, horiz_idx=horiz_idx, fps=args.fps
    )

    # CoM / BoS / margin
    bos_res = compute_bos_and_com_margin(
        motion, horiz_idx=horiz_idx,
        l_contact=l_contact, r_contact=r_contact,
        foot_width=args.foot_width
    )
    com = bos_res["com"]
    com2d = bos_res["com2d"]
    bos_polys = bos_res["bos_polys"]
    signed_dist = bos_res["signed_dist"]
    com_margin = bos_res["com_margin"]

    # infer mode
    if args.mode == "auto":
        analysis_mode = infer_analysis_mode(prompt, root_features, l_contact, r_contact)
    else:
        analysis_mode = args.mode

    # foot metrics
    metrics = {
        "num_frames": motion.shape[0],
        "num_joints": motion.shape[1],
        "estimated_ground_offset": ground,
        "sample_idx": args.sample_idx,
        "analysis_mode": analysis_mode,
        "prompt": prompt,
    }
    metrics.update(compute_foot_metrics(features, l_contact, r_contact, args.speed_thresh))
    metrics["root_jerk"] = compute_root_jerk(motion, fps=args.fps)

    # plot foot analysis
    foot_fig_path = os.path.join(args.outdir, "foot_contact_analysis.png")
    plot_foot_contact_figure(
        motion=motion,
        features=features,
        l_contact=l_contact,
        r_contact=r_contact,
        height_thresh=args.height_thresh,
        speed_thresh=args.speed_thresh,
        outpath=foot_fig_path,
        title=f"{args.title} - Foot Contact"
    )

    # mode-specific analysis
    if analysis_mode == "walk":
        xcom_res = compute_xcom_mos(
            com=com,
            com2d=com2d,
            bos_polys=bos_polys,
            up_idx=up_idx,
            fps=args.fps,
            smooth_win=args.smooth_win
        )
        xcom2d = xcom_res["xcom2d"]
        mos = xcom_res["mos"]
        stance_mask = l_contact | r_contact

        metrics["pendulum_length"] = xcom_res["pendulum_length"]
        metrics["omega0"] = xcom_res["omega0"]
        metrics["mos_mean"] = float(np.mean(mos))
        metrics["mos_min"] = float(np.min(mos))
        metrics["mos_negative_ratio"] = float(np.mean(mos < 0))
        metrics["mos_mean_stance"] = float(np.mean(mos[stance_mask])) if np.any(stance_mask) else np.nan
        metrics["mos_negative_ratio_stance"] = float(np.mean(mos[stance_mask] < 0)) if np.any(stance_mask) else np.nan

        balance_fig_path = os.path.join(args.outdir, "balance_analysis_walk_xcom_mos.png")
        plot_walk_balance_figure(
            motion=motion,
            com2d=com2d,
            bos_polys=bos_polys,
            xcom2d=xcom2d,
            mos=mos,
            l_contact=l_contact,
            r_contact=r_contact,
            outpath=balance_fig_path,
            title=f"{args.title} - Dynamic Balance (XCoM / MoS)"
        )

        summary_text = (
            f"Analysis mode: WALK-LIKE\n"
            f"Prompt: {prompt}\n\n"
            f"Interpretation:\n"
            f"- Dynamic balance is assessed using XCoM / Margin of Stability (MoS).\n"
            f"- Positive MoS indicates the XCoM remains inside the support region.\n"
            f"- Negative MoS suggests dynamic balance risk.\n\n"
            f"Key metrics:\n"
            f"- mos_mean = {metrics['mos_mean']:.6f}\n"
            f"- mos_min = {metrics['mos_min']:.6f}\n"
            f"- mos_negative_ratio = {metrics['mos_negative_ratio']:.6f}\n"
            f"- mos_mean_stance = {metrics['mos_mean_stance']:.6f}\n"
            f"- mos_negative_ratio_stance = {metrics['mos_negative_ratio_stance']:.6f}\n"
        )

    else:
        bend_res = compute_low_speed_com_margin(
            com_margin=com_margin,
            pelvis_vel=root_features["pelvis_vel"],
            l_contact=l_contact,
            r_contact=r_contact,
            l_foot_vel=features["l_vel"],
            r_foot_vel=features["r_vel"],
            root_low_speed_thresh=args.root_low_speed_thresh,
            foot_stable_speed_thresh=args.foot_stable_speed_thresh
        )
        low_speed = bend_res["low_speed"]
        stable_contact = bend_res["stable_contact"]
        selected_margin = bend_res["com_margin_low_speed"]

        metrics["num_low_speed_frames"] = int(np.sum(low_speed))
        metrics["num_selected_quasistatic_frames"] = int(np.sum(stable_contact))
        metrics["com_margin_mean_all"] = float(np.mean(com_margin))
        metrics["com_margin_min_all"] = float(np.min(com_margin))
        metrics["com_margin_negative_ratio_all"] = float(np.mean(com_margin < 0))

        if len(selected_margin) > 0:
            metrics["com_margin_mean_low_speed"] = float(np.mean(selected_margin))
            metrics["com_margin_min_low_speed"] = float(np.min(selected_margin))
            metrics["com_margin_negative_ratio_low_speed"] = float(np.mean(selected_margin < 0))
        else:
            metrics["com_margin_mean_low_speed"] = np.nan
            metrics["com_margin_min_low_speed"] = np.nan
            metrics["com_margin_negative_ratio_low_speed"] = np.nan

        balance_fig_path = os.path.join(args.outdir, "balance_analysis_bend_low_speed_com_margin.png")
        plot_bend_balance_figure(
            motion=motion,
            com2d=com2d,
            bos_polys=bos_polys,
            com_margin=com_margin,
            stable_contact=stable_contact,
            pelvis_vel=root_features["pelvis_vel"],
            outpath=balance_fig_path,
            title=f"{args.title} - Quasi-static Balance (Low-speed CoM Margin)"
        )

        summary_text = (
            f"Analysis mode: BEND-LIKE / QUASI-STATIC\n"
            f"Prompt: {prompt}\n\n"
            f"Interpretation:\n"
            f"- Stability is assessed only on low-speed support frames.\n"
            f"- Positive CoM margin means the CoM projection remains inside the support polygon.\n"
            f"- Negative CoM margin indicates quasi-static balance risk.\n\n"
            f"Key metrics:\n"
            f"- num_low_speed_frames = {metrics['num_low_speed_frames']}\n"
            f"- num_selected_quasistatic_frames = {metrics['num_selected_quasistatic_frames']}\n"
            f"- com_margin_mean_low_speed = {metrics['com_margin_mean_low_speed']}\n"
            f"- com_margin_min_low_speed = {metrics['com_margin_min_low_speed']}\n"
            f"- com_margin_negative_ratio_low_speed = {metrics['com_margin_negative_ratio_low_speed']}\n"
        )

    # save metrics and summary
    metrics_csv = os.path.join(args.outdir, "metrics.csv")
    save_metrics_csv(metrics_csv, metrics)

    summary_txt = os.path.join(args.outdir, "analysis_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("=" * 72)
    print("Analysis finished.")
    print(f"Input results.npy: {args.input}")
    print(f"Sample idx:        {args.sample_idx}")
    print(f"Prompt:            {prompt}")
    print(f"Mode:              {analysis_mode}")
    print(f"Foot figure:       {foot_fig_path}")
    print(f"Metrics CSV:       {metrics_csv}")
    print(f"Summary TXT:       {summary_txt}")
    print("=" * 72)


if __name__ == "__main__":
    main()