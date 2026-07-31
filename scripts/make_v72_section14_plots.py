"""Section 14 plots for the V7.2 three-band blind test (10 required figures).

Reads only already-frozen/collected artifacts:
  - output0727/v7_2_blind_test/pareto_scatter_data.json      (plots 1,2,3,5,7,10)
  - output0727/v7_2_blind_test/tau{05..25}/{method}/scorecard.json  (plot 4)
  - output0727/v7_2_blind_traces/tau{20,25}_{tag}_seed{n}/{v7-1,v7-2}/v7_trace.jsonl  (plots 6,9)
  - output0727/v7_2_blind_traces/.../{v7-1,v7-2}_run/**/comparison.npy            (plot 8)

Does not touch or recompute any frozen blind-test statistical result.
"""
import json
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- palette (validated: node scripts/validate_palette.js, light mode, --pairs all) ----
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

METHODS = [
    ("v2", "V2", "#2a78d6", "o"),
    ("v6", "V6", "#eb6834", "s"),
    ("v7-1", "V7.1", "#1baf7a", "^"),
    ("v7-2", "V7.2", "#4a3aa7", "D"),
]
TAUS = [5, 10, 15, 20, 25]

plt.rcParams.update({
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "legend.frameon": False,
    "font.size": 10,
})


def clean_axes(ax, x_grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.grid(axis="both" if x_grid else "y", color=GRID, linewidth=1.0)


def bootstrap_ci(values, seed, n_boot=2000, ci=95):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    return float(np.median(values)), float(np.percentile(values, 25)), float(np.percentile(values, 75))


# ---- pelvis-tilt geometry (mirrors scripts/diagnose_v7_results.py) ----

def get_joint_idx(name):
    mapping = {
        "pelvis": 0, "left_hip": 2, "right_hip": 1, "spine1": 3,
        "left_knee": 5, "right_knee": 4, "left_ankle": 8, "right_ankle": 7,
        "left_foot": 11, "right_foot": 10, "spine2": 6, "spine3": 9,
        "neck": 12, "head": 15, "left_collar": 13, "right_collar": 14,
        "left_shoulder": 17, "right_shoulder": 16, "left_elbow": 19,
        "right_elbow": 18, "left_wrist": 21, "right_wrist": 20,
    }
    return mapping.get(name, 0)


def pelvis_tilt_angle(q):
    pelvis = q[..., get_joint_idx("pelvis"), :]
    left_hip = q[..., get_joint_idx("left_hip"), :]
    right_hip = q[..., get_joint_idx("right_hip"), :]
    spine1 = q[..., get_joint_idx("spine1"), :]
    hip_center = (left_hip + right_hip) / 2.0
    pelvis_to_spine = spine1 - hip_center
    lr_axis = right_hip - left_hip
    lr_norm = np.linalg.norm(lr_axis, axis=-1, keepdims=True)
    lr_axis = lr_axis / np.maximum(lr_norm, 1e-12)
    lr_component = np.sum(pelvis_to_spine * lr_axis, axis=-1, keepdims=True) * lr_axis
    sagittal_vec = pelvis_to_spine - lr_component
    forward_proj = sagittal_vec[..., 2]
    upward_proj = sagittal_vec[..., 1]
    tilt = np.arctan2(forward_proj, np.maximum(upward_proj, 1e-12))
    return tilt


def compute_frame_angles(motion_xyz):
    if motion_xyz.ndim == 4:
        motion_xyz = motion_xyz[0]
    q = np.transpose(motion_xyz, (2, 0, 1))
    return np.degrees(pelvis_tilt_angle(q))


# ---- loaders ----

def load_trace(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_comparison(run_dir):
    matches = list(Path(run_dir).glob("**/comparison.npy"))
    if not matches:
        raise FileNotFoundError(f"no comparison.npy under {run_dir}")
    return np.load(matches[0], allow_pickle=True).item()


def load_scorecard(root, tau, method_key):
    p = root / f"tau{tau:02d}" / method_key / "scorecard.json"
    with open(p) as f:
        return json.load(f)


# ---- plots ----

def plot_01(pareto, out_dir):
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    for key, label, color, marker in METHODS:
        recs = pareto[key]
        meds, los, his = [], [], []
        for tau in TAUS:
            vals = [r["signed"] for r in recs if r["tau"] == tau]
            m, q1, q3 = median_iqr(vals)
            meds.append(m); los.append(q1); his.append(q3)
        ax.fill_between(TAUS, los, his, color=color, alpha=0.10, linewidth=0)
        ax.plot(TAUS, meds, color=color, linewidth=2, marker=marker, markersize=8,
                 markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.4,
                 label=label, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.text(TAUS[0], 0, " 0° (no bias)", color=MUTED, fontsize=8, va="bottom", ha="left")
    clean_axes(ax)
    ax.set_xticks(TAUS)
    ax.set_xlabel("target tilt τ (degrees)")
    ax.set_ylabel("signed error, guided − target (degrees, median ± IQR)")
    ax.set_title("1. Signed error vs. target", color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.savefig(out_dir / "plot01_signed_error_vs_target.png", bbox_inches="tight")
    plt.close(fig)


def plot_line_mean_ci(pareto, field, ylabel, title, fname, out_dir, pct=False):
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    for mi, (key, label, color, marker) in enumerate(METHODS):
        recs = pareto[key]
        means, los, his = [], [], []
        for ti, tau in enumerate(TAUS):
            vals = [r[field] for r in recs if r["tau"] == tau]
            m, lo, hi = bootstrap_ci(vals, seed=1000 * mi + ti)
            means.append(m); los.append(lo); his.append(hi)
        means = np.array(means); los = np.array(los); his = np.array(his)
        if pct:
            means, los, his = means * 100, los * 100, his * 100
        ax.fill_between(TAUS, los, his, color=color, alpha=0.10, linewidth=0)
        ax.plot(TAUS, means, color=color, linewidth=2, marker=marker, markersize=8,
                 markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.4,
                 label=label, zorder=3)
    clean_axes(ax)
    ax.set_xticks(TAUS)
    ax.set_xlabel("target tilt τ (degrees)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)


def plot_04(root, out_dir):
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    baseline_vals = []
    for key, label, color, marker in METHODS:
        meds, los, his = [], [], []
        for tau in TAUS:
            sc = load_scorecard(root, tau, key)
            fs = sc["control_metrics_aggregated"]["foot_skate_guided"]
            meds.append(fs["median"]); los.append(fs["ci_95_lo"]); his.append(fs["ci_95_hi"])
            if key == "v2":
                bsc = sc["control_metrics_aggregated"]["foot_skate_baseline"]
                baseline_vals.append(bsc["median"])
        ax.fill_between(TAUS, los, his, color=color, alpha=0.10, linewidth=0)
        ax.plot(TAUS, meds, color=color, linewidth=2, marker=marker, markersize=8,
                 markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.4,
                 label=label, zorder=3)
    ax.plot(TAUS, baseline_vals, color=MUTED, linewidth=2, linestyle=(0, (4, 2)),
             marker="o", markersize=6, markerfacecolor=MUTED, markeredgecolor=SURFACE,
             markeredgewidth=1.2, label="baseline (unguided)", zorder=2)
    clean_axes(ax)
    ax.set_xticks(TAUS)
    ax.set_xlabel("target tilt τ (degrees)")
    ax.set_ylabel("foot skate (median ± 95% CI)")
    ax.set_title("4. Foot-skate vs. target", color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.savefig(out_dir / "plot04_foot_skate_vs_target.png", bbox_inches="tight")
    plt.close(fig)


def plot_05(pareto, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    for key, label, color, marker in METHODS:
        recs = pareto[key]
        xs = np.array([r["abs"] for r in recs])
        ys = np.array([r["corr"] for r in recs])
        ax.scatter(xs, ys, s=22, color=color, alpha=0.30, edgecolor="none", zorder=2)
        mx, my = float(np.median(xs)), float(np.median(ys))
        ax.scatter([mx], [my], s=130, color=color, edgecolor=INK, linewidth=1.3,
                    marker=marker, zorder=4, label=f"{label} (median)")
    clean_axes(ax, x_grid=True)
    ax.set_xlabel("absolute error (degrees) — lower is better")
    ax.set_ylabel("temporal correlation — higher is better")
    ax.set_title("5. Accuracy – waveform preservation trade-off", color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(loc="lower left", fontsize=9, markerscale=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot05_pareto_scatter.png")
    plt.close(fig)


TRACE_REPS = [(20, 628, "hard_tail"), (20, 624, "good"), (25, 626, "hard_tail"), (25, 628, "good")]
TRACE_VARIANTS = [("v7-1", "#1baf7a", "^", "V7.1"), ("v7-2", "#4a3aa7", "D", "V7.2")]


def plot_06(traces_root, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=150, sharey=True)
    rng = np.random.default_rng(606)
    for ax, (tau, seed, tag) in zip(axes.flat, TRACE_REPS):
        for vkey, color, marker, label in TRACE_VARIANTS:
            trace_path = traces_root / f"tau{tau}_{tag}_seed{seed}" / vkey / "v7_trace.jsonl"
            recs = load_trace(trace_path)
            xs = np.array([r.get("distance_to_target_deg", np.nan) for r in recs], dtype=float)
            ys = np.array([r.get("proposal_count", np.nan) for r in recs], dtype=float)
            yj = ys + rng.uniform(-0.06, 0.06, size=len(ys))
            ax.scatter(xs, yj, color=color, s=30, marker=marker, alpha=0.45,
                        edgecolor="none", zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
        clean_axes(ax, x_grid=True)
        ax.set_title(f"τ={tau}°, seed {seed} ({tag.replace('_', ' ')})", fontsize=10, color=INK2, loc="left")
    for ax in axes[-1, :]:
        ax.set_xlabel("residual to target (degrees)")
    for ax in axes[:, 0]:
        ax.set_ylabel("proposals tried (this step)")
    handles = [Line2D([0], [0], marker=m, color="none", markerfacecolor=c, markeredgecolor="none",
                       markersize=8, alpha=0.8, label=l) for _, c, m, l in TRACE_VARIANTS]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(0.99, 0.985), frameon=False)
    fig.suptitle("6. Trust-region proposal count vs. remaining error", color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "plot06_proposal_count_vs_signed_error.png")
    plt.close(fig)


def plot_07(pareto, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=150, sharey=True)

    ax = axes[0]
    ax.axhspan(-1.5, 1.5, color="#1baf7a", alpha=0.14, linewidth=0, zorder=1)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    meds = []
    rng = np.random.default_rng(777)
    for tau in TAUS:
        vals = [r["signed"] for r in pareto["v7-1"] if r["tau"] == tau]
        jitter = rng.uniform(-0.6, 0.6, size=len(vals))
        ax.scatter(np.array([tau] * len(vals)) + jitter, vals, s=14, color="#1baf7a",
                    alpha=0.28, edgecolor="none", zorder=3)
        meds.append(float(np.median(vals)))
    ax.plot(TAUS, meds, color="#1baf7a", linewidth=2.4, marker="^", markersize=9,
             markerfacecolor="#1baf7a", markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    clean_axes(ax, x_grid=True)
    ax.set_xticks(TAUS)
    ax.set_title("V7.1 — single band (±1.5°)", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("target tilt τ (degrees)")
    ax.set_ylabel("signed error (degrees)")

    ax = axes[1]
    ax.axhspan(-3.0, 3.0, color="#4a3aa7", alpha=0.07, linewidth=0, zorder=1)
    ax.axhspan(-0.5, 0.5, color="#4a3aa7", alpha=0.16, linewidth=0, zorder=1)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    meds = []
    rng = np.random.default_rng(778)
    for tau in TAUS:
        vals = [r["signed"] for r in pareto["v7-2"] if r["tau"] == tau]
        jitter = rng.uniform(-0.6, 0.6, size=len(vals))
        ax.scatter(np.array([tau] * len(vals)) + jitter, vals, s=14, color="#4a3aa7",
                    alpha=0.28, edgecolor="none", zorder=3)
        meds.append(float(np.median(vals)))
    ax.plot(TAUS, meds, color="#4a3aa7", linewidth=2.4, marker="D", markersize=8.5,
             markerfacecolor="#4a3aa7", markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    clean_axes(ax, x_grid=True)
    ax.set_xticks(TAUS)
    ax.set_title("V7.2 — three-band (control ±0.5°, exit ±3.0°)", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("target tilt τ (degrees)")

    fig.suptitle("7. Realized bias vs. tolerance-band design", color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.01,
              "Hard-tail subgroup beyond ±2° at τ≥20°: n=12/40 (30%) at τ=20°, "
              "n=10/40 (25%) at τ=25° (Section 9.4).",
              color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_dir / "plot07_tolerance_band_vs_bias.png")
    plt.close(fig)


def plot_08(traces_root, out_dir):
    cases = [(628, "hard_tail", "hard-tail case"), (624, "good", "good case")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)
    tau = 20
    for ax, (seed, rep_tag, tag) in zip(axes, cases):
        base_dir = traces_root / f"tau{tau}_{rep_tag}_seed{seed}"
        d71 = load_comparison(base_dir / "v7-1_run")
        d72 = load_comparison(base_dir / "v7-2_run")
        baseline_deg = compute_frame_angles(d71["motion_xyz"])
        v71_deg = compute_frame_angles(d71["motion_xyz_guided"])
        v72_deg = compute_frame_angles(d72["motion_xyz_guided"])
        frames = np.arange(len(baseline_deg))
        ax.plot(frames, baseline_deg, color=MUTED, linewidth=2, linestyle=(0, (4, 2)),
                 label="baseline (unguided)", zorder=2)
        ax.plot(frames, v71_deg, color="#1baf7a", linewidth=2, label="V7.1", zorder=3)
        ax.plot(frames, v72_deg, color="#4a3aa7", linewidth=2, label="V7.2", zorder=4)
        ax.axhline(tau, color=INK2, linewidth=1.0, linestyle=(0, (1, 1.5)), zorder=1)
        ax.text(frames[-1], tau, f" target {tau}°", color=INK2, fontsize=8.5, va="bottom", ha="right",
                 zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
        clean_axes(ax)
        ax.set_xlabel("frame")
        ax.set_title(f"seed {seed} — {tag}", color=INK, fontsize=12, loc="left")
    axes[0].set_ylabel("pelvis tilt angle (degrees)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("8. Representative trajectories: baseline vs. V7.1 vs. V7.2", color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "plot08_trajectory_comparison.png")
    plt.close(fig)


def plot_09(traces_root, out_dir):
    n_rows = len(TRACE_REPS) * len(TRACE_VARIANTS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.05 * n_rows), dpi=150, sharex=True)
    row = 0
    for tau, seed, tag in TRACE_REPS:
        for vkey, color, marker, label in TRACE_VARIANTS:
            ax = axes[row]
            trace_path = traces_root / f"tau{tau}_{tag}_seed{seed}" / vkey / "v7_trace.jsonl"
            recs = load_trace(trace_path)
            steps = np.arange(len(recs))
            in_band = np.array([bool(r.get("in_control_band_after")) for r in recs])
            ax.fill_between(steps, 0, 1, where=in_band, color=color, alpha=0.18,
                              step="mid", linewidth=0, zorder=1)
            stop_steps = [i for i, r in enumerate(recs) if r.get("control_stop_triggered")]
            react_steps = [i for i, r in enumerate(recs) if r.get("reactivation_triggered")]
            if stop_steps:
                ax.scatter(stop_steps, [0.5] * len(stop_steps), marker="^", s=55,
                            color=GOOD, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            if react_steps:
                ax.scatter(react_steps, [0.5] * len(react_steps), marker="v", s=55,
                            color=CRITICAL, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlim(-0.5, len(recs) - 0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor(SURFACE)
            ax.text(-0.01, 0.5, f"τ{tau} {tag.replace('_', ' ')} — {label}",
                     transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=INK2)
            row += 1
    axes[-1].set_xlabel("diffusion sampling step (chronological)")
    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GOOD, markeredgecolor=SURFACE,
               markersize=9, label="control stop (entered band)"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=CRITICAL, markeredgecolor=SURFACE,
               markersize=9, label="reactivation (exited band)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(0.99, 0.975), frameon=False)
    fig.suptitle("9. Control stop / reactivation timeline", color=INK, fontsize=13, x=0.02, ha="left")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.93, bottom=0.06, hspace=0.15)
    fig.savefig(out_dir / "plot09_stop_reactivate_timeline.png")
    plt.close(fig)


def plot_10(pareto, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    rng = np.random.default_rng(2026)
    for i, tau in enumerate(TAUS):
        v71 = {r["seed"]: r["abs"] for r in pareto["v7-1"] if r["tau"] == tau}
        v72 = {r["seed"]: r["abs"] for r in pareto["v7-2"] if r["tau"] == tau}
        seeds = sorted(set(v71) & set(v72))
        diffs = np.array([v72[s] - v71[s] for s in seeds])
        jitter = rng.uniform(-0.28, 0.28, size=len(diffs))
        ax.scatter(diffs, i + jitter, s=10, color=MUTED, alpha=0.35, edgecolor="none", zorder=2)
        boots = rng.choice(diffs, size=(2000, len(diffs)), replace=True).mean(axis=1)
        m = float(diffs.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax.plot([lo, hi], [i, i], color=INK2, linewidth=1.6, zorder=3)
        ax.scatter([m], [i], marker="D", s=90, color="#4a3aa7", edgecolor=INK, linewidth=1.0, zorder=4)
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=1)
    ax.set_yticks(range(len(TAUS)))
    ax.set_yticklabels([f"τ={t}°" for t in TAUS])
    ax.invert_yaxis()
    clean_axes(ax, x_grid=True)
    ax.set_xlabel("paired difference in abs. error, V7.2 − V7.1 (degrees)  ← V7.2 better")
    ax.set_title("10. Per-target paired difference: V7.2 vs. V7.1 (abs. error)", color=INK, fontsize=13, loc="left", pad=12)
    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#4a3aa7", markeredgecolor=INK,
               markersize=9, label="mean diff (95% bootstrap CI)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="none",
               markersize=6, label="per-seed diff (n=40)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "plot10_paired_diff_forest.png")
    plt.close(fig)


def safe_call(name, fn, *args):
    try:
        print(f"[plot] {name} ...", flush=True)
        fn(*args)
        print(f"[plot] {name} OK", flush=True)
    except Exception as e:
        print(f"[plot] {name} FAILED: {e}", flush=True)
        traceback.print_exc()


def main():
    root = Path("/root/autodl-tmp/motion-diffusion-model/output0727/v7_2_blind_test")
    traces_root = Path("/root/autodl-tmp/motion-diffusion-model/output0727/v7_2_blind_traces")
    out_dir = root / "section14_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(root / "pareto_scatter_data.json") as f:
        pareto = json.load(f)

    safe_call("01 signed_error_vs_target", plot_01, pareto, out_dir)
    safe_call("02 frame_hit_vs_target", plot_line_mean_ci, pareto, "hit",
              "frame hit-band rate (mean ± 95% CI, %)", "2. Frame hit-band rate vs. target",
              "plot02_frame_hit_vs_target.png", out_dir, True)
    safe_call("03 temporal_corr_vs_target", plot_line_mean_ci, pareto, "corr",
              "temporal correlation (mean ± 95% CI)", "3. Temporal correlation vs. target",
              "plot03_temporal_corr_vs_target.png", out_dir, False)
    safe_call("04 foot_skate_vs_target", plot_04, root, out_dir)
    safe_call("05 pareto_scatter", plot_05, pareto, out_dir)
    safe_call("06 proposal_count_vs_signed_error", plot_06, traces_root, out_dir)
    safe_call("07 tolerance_band_vs_bias", plot_07, pareto, out_dir)
    safe_call("08 trajectory_comparison", plot_08, traces_root, out_dir)
    safe_call("09 stop_reactivate_timeline", plot_09, traces_root, out_dir)
    safe_call("10 paired_diff_forest", plot_10, pareto, out_dir)

    print("ALL DONE ->", out_dir)


if __name__ == "__main__":
    main()
