import os
import csv
import math
import argparse
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon


# =========================================================
# 0. HumanML3D / MDM 22-joint 索引
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
# 1. 基础工具
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch compose panel videos: left mp4 + right dynamic foot/com analysis"
    )
    parser.add_argument("--manifest_csv", type=str, required=True,
                        help="CSV with columns: mp4_path, results_path, sample_idx, output_name")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--height_thresh", type=float, default=0.05)
    parser.add_argument("--speed_thresh", type=float, default=0.08)
    parser.add_argument("--foot_width", type=float, default=0.10)
    parser.add_argument("--ground_quantile", type=float, default=0.02)
    parser.add_argument("--hold_seconds", type=float, default=2.0,
                        help="How many seconds to hold the final static panel")
    parser.add_argument("--left_width", type=int, default=720)
    parser.add_argument("--right_width", type=int, default=1000)
    parser.add_argument("--canvas_height", type=int, default=900)
    parser.add_argument("--fps_override", type=float, default=None,
                        help="If set, use this fps instead of source mp4 fps")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def get_axis_indices(up_axis="y"):
    axis_map = {"x": 0, "y": 1, "z": 2}
    up = axis_map[up_axis]
    horiz = [0, 1, 2]
    horiz.remove(up)
    return up, horiz


def safe_norm(v, axis=-1, keepdims=False, eps=1e-8):
    return np.sqrt(np.sum(v * v, axis=axis, keepdims=keepdims) + eps)


def resize_with_pad(img, target_w, target_h, pad_value=255):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero size")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def load_video_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 20.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames, fps


def load_mdm_motion(results_path, sample_idx=0):
    obj = np.load(results_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        data = obj.item()
    elif isinstance(obj, dict):
        data = obj
    else:
        raise ValueError(f"Unsupported results format: {results_path}")

    motion = np.asarray(data["motion"])
    texts = data.get("text", None)
    lengths = data.get("lengths", None)

    if motion.ndim != 4:
        raise ValueError(f"Expected motion shape (N, J, 3, T), got {motion.shape}")

    arr = motion[sample_idx]  # (J, 3, T)
    arr = np.transpose(arr, (2, 0, 1))  # -> (T, J, 3)

    text = ""
    if texts is not None and sample_idx < len(texts):
        text = str(texts[sample_idx])

    if lengths is not None:
        length = int(np.asarray(lengths)[sample_idx])
        arr = arr[:length]
    else:
        length = arr.shape[0]

    return arr, text, length


def sanitize_text(text, max_len=90):
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# =========================================================
# 2. 几何 / 分析函数
# =========================================================
def normalize_ground(motion, up_idx, ground_quantile=0.02):
    foot_ids = [IDX["l_ankle"], IDX["r_ankle"], IDX["l_foot"], IDX["r_foot"]]
    foot_heights = motion[:, foot_ids, up_idx].reshape(-1)
    ground = np.quantile(foot_heights, ground_quantile)
    motion2 = motion.copy()
    motion2[:, :, up_idx] -= ground
    return motion2, ground


def compute_foot_features(motion, up_idx, horiz_idx):
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
    l_vel[1:] = safe_norm(np.diff(l_center, axis=0), axis=1)
    r_vel[1:] = safe_norm(np.diff(r_center, axis=0), axis=1)

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


# =========================================================
# 3. 预计算 analysis 数据
# =========================================================
def precompute_analysis(
    motion,
    up_axis="y",
    height_thresh=0.05,
    speed_thresh=0.08,
    foot_width=0.10,
    ground_quantile=0.02
):
    up_idx, horiz_idx = get_axis_indices(up_axis)
    motion, ground = normalize_ground(motion, up_idx=up_idx, ground_quantile=ground_quantile)
    features = compute_foot_features(motion, up_idx=up_idx, horiz_idx=horiz_idx)
    l_contact, r_contact = detect_contacts(features, height_thresh=height_thresh)
    com = estimate_com(motion)
    com2d = com[:, horiz_idx]

    T = motion.shape[0]
    bos_polys = []
    signed_dist = np.zeros(T, dtype=float)
    for t in range(T):
        poly = build_bos_polygon(
            motion, t, horiz_idx,
            l_contact=l_contact, r_contact=r_contact,
            foot_width=foot_width
        )
        bos_polys.append(poly)
        signed_dist[t] = point_to_polygon_distance(com2d[t], poly)

    return {
        "motion": motion,
        "up_idx": up_idx,
        "horiz_idx": horiz_idx,
        "features": features,
        "l_contact": l_contact,
        "r_contact": r_contact,
        "speed_thresh": speed_thresh,
        "height_thresh": height_thresh,
        "com": com,
        "com2d": com2d,
        "bos_polys": bos_polys,
        "signed_dist": signed_dist,
        "ground": ground,
    }


# =========================================================
# 4. 画右侧动态 analysis panel
# =========================================================
def fig_to_rgb_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_analysis_panel(pre, t, title="", final_static=False):
    features = pre["features"]
    l_contact = pre["l_contact"]
    r_contact = pre["r_contact"]
    speed_thresh = pre["speed_thresh"]
    height_thresh = pre["height_thresh"]
    com2d = pre["com2d"]
    bos_polys = pre["bos_polys"]
    signed_dist = pre["signed_dist"]

    T = len(features["l_height"])
    t = min(max(int(t), 0), T - 1)
    t_show = T - 1 if final_static else t
    frames = np.arange(T)

    l_skate = l_contact & (features["l_vel"] > speed_thresh)
    r_skate = r_contact & (features["r_vel"] > speed_thresh)
    violation = signed_dist > 0

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig = plt.figure(figsize=(8.0, 10.0), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.15, 1.0],
        left=0.08, right=0.98, bottom=0.05, top=0.94,
        hspace=0.22
    )

    # ============== Foot Analysis ==============
    gs_top = outer[0].subgridspec(3, 1, hspace=0.30)

    # (A1) foot height
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax1.plot(frames, features["l_height"], color="#1f77b4", alpha=0.20, linewidth=1.2)
    ax1.plot(frames, features["r_height"], color="#ff7f0e", alpha=0.20, linewidth=1.2)
    ax1.plot(frames[:t_show + 1], features["l_height"][:t_show + 1], color="#1f77b4", linewidth=2, label="Left")
    ax1.plot(frames[:t_show + 1], features["r_height"][:t_show + 1], color="#ff7f0e", linewidth=2, label="Right")
    ax1.axhline(height_thresh, color="gray", linestyle="--", linewidth=1.2, label="Contact threshold")
    if not final_static:
        ax1.axvline(t_show, color="red", linestyle="--", linewidth=1.3)
        ax1.scatter([t_show], [features["l_height"][t_show]], color="#1f77b4", s=26, zorder=5)
        ax1.scatter([t_show], [features["r_height"][t_show]], color="#ff7f0e", s=26, zorder=5)
    ax1.set_title("Foot Contact Analysis")
    ax1.set_ylabel("Height")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", ncol=3)

    # (A2) foot speed
    ax2 = fig.add_subplot(gs_top[1, 0], sharex=ax1)
    ax2.plot(frames, features["l_vel"], color="#1f77b4", alpha=0.20, linewidth=1.2)
    ax2.plot(frames, features["r_vel"], color="#ff7f0e", alpha=0.20, linewidth=1.2)
    ax2.plot(frames[:t_show + 1], features["l_vel"][:t_show + 1], color="#1f77b4", linewidth=2, label="Left")
    ax2.plot(frames[:t_show + 1], features["r_vel"][:t_show + 1], color="#ff7f0e", linewidth=2, label="Right")
    ax2.axhline(speed_thresh, color="gray", linestyle="--", linewidth=1.2, label="Skating threshold")
    if not final_static:
        ax2.axvline(t_show, color="red", linestyle="--", linewidth=1.3)
        ax2.scatter([t_show], [features["l_vel"][t_show]], color="#1f77b4", s=26, zorder=5)
        ax2.scatter([t_show], [features["r_vel"][t_show]], color="#ff7f0e", s=26, zorder=5)
    ax2.set_ylabel("Planar speed")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", ncol=3)

    # (A3) foot trajectory ground plane
    ax3 = fig.add_subplot(gs_top[2, 0])
    lxy = features["l_traj"]
    rxy = features["r_traj"]

    if final_static:
        show_idx = slice(0, T)
    else:
        show_idx = slice(0, t_show + 1)

    ax3.plot(lxy[:, 0], lxy[:, 1], color="#1f77b4", alpha=0.15, linewidth=1.0)
    ax3.plot(rxy[:, 0], rxy[:, 1], color="#ff7f0e", alpha=0.15, linewidth=1.0)
    ax3.plot(lxy[show_idx, 0], lxy[show_idx, 1], color="#1f77b4", linewidth=2.0, label="Left toe traj")
    ax3.plot(rxy[show_idx, 0], rxy[show_idx, 1], color="#ff7f0e", linewidth=2.0, label="Right toe traj")

    lc_show = np.where(l_contact[:t_show + 1])[0]
    rc_show = np.where(r_contact[:t_show + 1])[0]
    ls_show = np.where(l_skate[:t_show + 1])[0]
    rs_show = np.where(r_skate[:t_show + 1])[0]

    if len(lc_show) > 0:
        ax3.scatter(lxy[lc_show, 0], lxy[lc_show, 1], color="#1f77b4", s=10, alpha=0.7)
    if len(rc_show) > 0:
        ax3.scatter(rxy[rc_show, 0], rxy[rc_show, 1], color="#ff7f0e", s=10, alpha=0.7)
    if len(ls_show) > 0:
        ax3.scatter(lxy[ls_show, 0], lxy[ls_show, 1], color="red", s=18, marker="x", linewidths=1.2)
    if len(rs_show) > 0:
        ax3.scatter(rxy[rs_show, 0], rxy[rs_show, 1], color="red", s=18, marker="x", linewidths=1.2)

    if not final_static:
        ax3.scatter([lxy[t_show, 0]], [lxy[t_show, 1]], color="#1f77b4", s=40, edgecolor="black", zorder=6)
        ax3.scatter([rxy[t_show, 0]], [rxy[t_show, 1]], color="#ff7f0e", s=40, edgecolor="black", zorder=6)

    ax3.set_xlabel("Horizontal axis 1")
    ax3.set_ylabel("Horizontal axis 2")
    ax3.axis("equal")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best", ncol=2)

    # ============== CoM vs BoS ==============
    gs_bot = outer[1].subgridspec(2, 1, hspace=0.30)

    # (B1) CoM path + current BoS
    ax4 = fig.add_subplot(gs_bot[0, 0])
    ax4.plot(com2d[:, 0], com2d[:, 1], color="gray", alpha=0.15, linewidth=1.0)
    ax4.plot(com2d[:t_show + 1, 0], com2d[:t_show + 1, 1], color="#9467bd", linewidth=2.0, label="Projected CoM path")

    in_idx = np.where(~violation[:t_show + 1])[0]
    out_idx = np.where(violation[:t_show + 1])[0]
    if len(in_idx) > 0:
        ax4.scatter(com2d[in_idx, 0], com2d[in_idx, 1], color="#2ca02c", s=12, alpha=0.7, label="Inside BoS")
    if len(out_idx) > 0:
        ax4.scatter(com2d[out_idx, 0], com2d[out_idx, 1], color="#d62728", s=12, alpha=0.8, label="Outside BoS")

    current_poly = bos_polys[t_show]
    if len(current_poly) >= 3:
        patch = Polygon(current_poly, closed=True, facecolor="#8c564b", edgecolor="#8c564b", alpha=0.25, linewidth=1.5)
        ax4.add_patch(patch)
        ax4.plot(
            np.r_[current_poly[:, 0], current_poly[0, 0]],
            np.r_[current_poly[:, 1], current_poly[0, 1]],
            color="#8c564b", linewidth=1.5, label="Current BoS"
        )

    ax4.scatter([com2d[t_show, 0]], [com2d[t_show, 1]], color="#9467bd", s=45, edgecolor="black", zorder=6)
    ax4.set_title("CoM vs BoS Analysis")
    ax4.set_xlabel("Horizontal axis 1")
    ax4.set_ylabel("Horizontal axis 2")
    ax4.axis("equal")
    ax4.grid(alpha=0.25)
    ax4.legend(loc="best", ncol=2)

    # (B2) signed distance curve
    ax5 = fig.add_subplot(gs_bot[1, 0], sharex=ax1)
    ax5.plot(frames, signed_dist, color="gray", alpha=0.20, linewidth=1.0)
    ax5.plot(frames[:t_show + 1], signed_dist[:t_show + 1], color="#9467bd", linewidth=2.0, label="Signed distance")
    ax5.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    if not final_static:
        ax5.axvline(t_show, color="red", linestyle="--", linewidth=1.3)
        ax5.scatter([t_show], [signed_dist[t_show]], color="#9467bd", s=28, zorder=5)

    pos = signed_dist[:t_show + 1] > 0
    if np.any(pos):
        ax5.fill_between(frames[:t_show + 1], 0, signed_dist[:t_show + 1], where=pos,
                         color="#d62728", alpha=0.20, interpolate=True, label="Outside BoS")

    ax5.set_xlabel("Frame")
    ax5.set_ylabel("Signed dist.")
    ax5.grid(alpha=0.25)
    ax5.legend(loc="upper right")

    if title:
        fig.suptitle(title, fontsize=11, y=0.985)

    return fig_to_rgb_array(fig)


# =========================================================
# 5. 处理单个 case
# =========================================================
def compose_one_case(
    mp4_path,
    results_path,
    sample_idx,
    output_path,
    up_axis="y",
    height_thresh=0.05,
    speed_thresh=0.08,
    foot_width=0.10,
    ground_quantile=0.02,
    hold_seconds=2.0,
    left_width=720,
    right_width=1000,
    canvas_height=900,
    fps_override=None
):
    video_frames, src_fps = load_video_frames(mp4_path)
    fps = fps_override if fps_override is not None else src_fps

    motion, text, motion_len = load_mdm_motion(results_path, sample_idx=sample_idx)
    pre = precompute_analysis(
        motion,
        up_axis=up_axis,
        height_thresh=height_thresh,
        speed_thresh=speed_thresh,
        foot_width=foot_width,
        ground_quantile=ground_quantile,
    )

    T = min(len(video_frames), motion.shape[0])
    if T <= 0:
        raise RuntimeError(f"No valid frames to compose for {mp4_path}")

    title = f"sample_idx={sample_idx} | {sanitize_text(text)}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = left_width + right_width
    out_h = canvas_height
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    try:
        for t in range(T):
            left = resize_with_pad(video_frames[t], left_width, canvas_height, pad_value=255)
            right = render_analysis_panel(pre, t=t, title=title, final_static=False)
            right = resize_with_pad(right, right_width, canvas_height, pad_value=255)

            combo = np.concatenate([left, right], axis=1)
            combo_bgr = cv2.cvtColor(combo, cv2.COLOR_RGB2BGR)
            writer.write(combo_bgr)

        # 末尾 hold：右侧改成完整静态轨迹图
        hold_n = max(1, int(round(hold_seconds * fps)))
        left_last = resize_with_pad(video_frames[T - 1], left_width, canvas_height, pad_value=255)
        right_final = render_analysis_panel(pre, t=T - 1, title=title, final_static=True)
        right_final = resize_with_pad(right_final, right_width, canvas_height, pad_value=255)
        combo_final = np.concatenate([left_last, right_final], axis=1)
        combo_final_bgr = cv2.cvtColor(combo_final, cv2.COLOR_RGB2BGR)

        for _ in range(hold_n):
            writer.write(combo_final_bgr)

    finally:
        writer.release()


# =========================================================
# 6. 批处理入口
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        print("No rows found in manifest_csv.")
        return

    for i, row in enumerate(rows):
        mp4_path = row["mp4_path"]
        results_path = row["results_path"]
        sample_idx = int(row["sample_idx"])
        output_name = row.get("output_name", "").strip()

        if output_name == "":
            output_name = f"{Path(results_path).parent.name}_sample_{sample_idx:03d}"

        output_path = os.path.join(args.outdir, output_name + ".mp4")

        if args.skip_existing and os.path.exists(output_path):
            print(f"[SKIP] {output_path}")
            continue

        print("=" * 80)
        print(f"[{i+1}/{len(rows)}] composing")
        print("mp4_path    :", mp4_path)
        print("results_path:", results_path)
        print("sample_idx  :", sample_idx)
        print("output_path :", output_path)

        try:
            compose_one_case(
                mp4_path=mp4_path,
                results_path=results_path,
                sample_idx=sample_idx,
                output_path=output_path,
                up_axis=args.up_axis,
                height_thresh=args.height_thresh,
                speed_thresh=args.speed_thresh,
                foot_width=args.foot_width,
                ground_quantile=args.ground_quantile,
                hold_seconds=args.hold_seconds,
                left_width=args.left_width,
                right_width=args.right_width,
                canvas_height=args.canvas_height,
                fps_override=args.fps_override,
            )
            print(f"[OK] saved: {output_path}")
        except Exception as e:
            print(f"[FAIL] {output_path}")
            print("Error:", repr(e))


if __name__ == "__main__":
    main()