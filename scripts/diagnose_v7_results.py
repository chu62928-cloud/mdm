"""Phase 0.1 — Per-Seed Diagnostics CSV

Reads existing comparison.npy files and computes detailed per-seed metrics.

Per R2 (frozen metric definitions):
  - Frame-level band occupancy ≠ summary hit ≠ final-frame hit
  - Positive overshoot ≠ negative undershoot ≠ absolute MAE
  - Each reported independently.
"""

import csv, json, math, sys, os
from pathlib import Path
import numpy as np
from collections import defaultdict

# ---- Config ----
RESULT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output0727/v7_autocal")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output0727/v7_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {5: "tau05", 10: "tau10", 15: "tau15", 20: "tau20", 25: "tau25"}
VARIANTS = ["v7-auto-dps", "v2-dps", "v6-closed-loop"]
VARIANT_LABEL = {"v7-auto-dps": "V7", "v2-dps": "V2", "v6-closed-loop": "V6"}

# ---- Pelvis tilt computation (mirrors posture_guidance/angle_ops.py) ----

def get_joint_idx(name):
    """Mirror joint_indices.py get_joint_idx."""
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
    """Compute signed pelvis tilt angle. q: (N, J, 3) → (N,) radians."""
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


# ---- Frame-level computation ----

def compute_frame_angles(motion_xyz):
    """Compute per-frame pelvis tilt angles from motion_xyz.

    Args:
        motion_xyz: (1, J, 3, N) or (J, 3, N) — xyz coords.
    Returns:
        angles_deg: (N,) per-frame angles in degrees.
    """
    if motion_xyz.ndim == 4:
        motion_xyz = motion_xyz[0]
    # motion_xyz: (J, 3, N) → need (N, J, 3)
    q = np.transpose(motion_xyz, (2, 0, 1))  # (N, J, 3)
    angles_rad = pelvis_tilt_angle(q)          # (N,)
    return np.degrees(angles_rad)


def compute_per_seed_metrics(
    baseline_xyz, guided_xyz, target_deg, tolerance_deg=2.0
):
    """Compute all R2 metrics for one seed.

    Args:
        baseline_xyz: (1, J, 3, N) baseline joint coords.
        guided_xyz: (1, J, 3, N) guided joint coords.
        target_deg: target angle in degrees.
        tolerance_deg: band tolerance in degrees.

    Returns:
        dict of per-seed metrics.
    """
    bas_deg = compute_frame_angles(baseline_xyz)   # (N,)
    gud_deg = compute_frame_angles(guided_xyz)     # (N,)

    N = len(bas_deg)

    # ---- Summary (mean over all frames) ----
    baseline_mean = float(np.mean(bas_deg))
    guided_mean = float(np.mean(gud_deg))

    # ---- Per R2 definitions ----
    signed_error = guided_mean - target_deg
    abs_error = abs(signed_error)
    requested_change = target_deg - baseline_mean
    achieved_change = guided_mean - baseline_mean  # same as "delta"

    # ---- Per-frame statistics ----
    baseline_std = float(np.std(bas_deg))
    guided_std = float(np.std(gud_deg))

    # ---- Frame-level band occupancy ----
    in_band = np.abs(gud_deg - target_deg) <= tolerance_deg
    frame_hit_band = float(np.mean(in_band))

    # ---- Final summary in band (last 25% of frames) ----
    tail_N = max(N // 4, 1)
    tail_in_band = np.abs(gud_deg[-tail_N:] - target_deg) <= tolerance_deg
    final_summary_in_band = float(np.mean(tail_in_band))

    # ---- Loose hit (within 2x tolerance) ----
    loose_hit = float(np.mean(np.abs(gud_deg - target_deg) <= 2 * tolerance_deg))

    # ---- Positive overshoot (frames ABOVE target + tolerance) ----
    overshoot_frames = gud_deg - (target_deg + tolerance_deg)
    overshoot_frames = np.maximum(overshoot_frames, 0)
    positive_overshoot = float(np.mean(overshoot_frames[overshoot_frames > 0])) if np.any(overshoot_frames > 0) else 0.0

    # ---- Negative undershoot (frames BELOW target - tolerance) ----
    undershoot_frames = (target_deg - tolerance_deg) - gud_deg
    undershoot_frames = np.maximum(undershoot_frames, 0)
    negative_undershoot = float(np.mean(undershoot_frames[undershoot_frames > 0])) if np.any(undershoot_frames > 0) else 0.0

    # ---- Overshoot P90 (90th percentile of all positive deviations) ----
    deviations = gud_deg - target_deg
    overshoot_p90 = float(np.percentile(np.maximum(deviations, 0), 90))

    # ---- Temporal correlation (Pearson r between baseline and guided) ----
    bas_centered = bas_deg - np.mean(bas_deg)
    gud_centered = gud_deg - np.mean(gud_deg)
    corr_num = np.sum(bas_centered * gud_centered)
    corr_den = np.sqrt(np.sum(bas_centered**2) * np.sum(gud_centered**2))
    temporal_corr = float(corr_num / max(corr_den, 1e-12))

    # ---- Phase valid fraction (proxy: non-NaN frames / total) ----
    valid_fraction = 1.0  # All frames valid for "always" phase

    return {
        "baseline_mean_deg": baseline_mean,
        "guided_mean_deg": guided_mean,
        "baseline_std_deg": baseline_std,
        "guided_std_deg": guided_std,
        "requested_change_deg": requested_change,
        "achieved_change_deg": achieved_change,
        "signed_error_deg": signed_error,
        "abs_error_deg": abs_error,
        "frame_hit_band": frame_hit_band,
        "final_summary_in_band": final_summary_in_band,
        "loose_hit": loose_hit,
        "positive_overshoot_deg": positive_overshoot,
        "negative_undershoot_deg": negative_undershoot,
        "overshoot_p90_deg": overshoot_p90,
        "temporal_corr": temporal_corr,
        "phase_valid_fraction": valid_fraction,
        "active_frame_count": N,
    }


# ---- Main ----

def main():
    rows = []

    for target_deg, tau_name in sorted(TARGETS.items()):
        for vname in VARIANTS:
            run_dir = RESULT_ROOT / tau_name / vname
            if not run_dir.exists():
                print(f"  SKIP: {run_dir} not found")
                continue

            # Find all comparison.npy files
            npy_files = sorted(run_dir.glob("**/comparison.npy"))
            print(f"  {tau_name}/{vname}: {len(npy_files)} seeds")

            for npy_path in npy_files:
                try:
                    data = np.load(npy_path, allow_pickle=True).item()
                except Exception as e:
                    print(f"    ERROR loading {npy_path}: {e}")
                    continue

                seed = data.get("seed", "?")
                baseline_xyz = data.get("motion_xyz")
                guided_xyz = data.get("motion_xyz_guided")

                if baseline_xyz is None or guided_xyz is None:
                    print(f"    SKIP seed={seed}: missing xyz data")
                    continue

                metrics = compute_per_seed_metrics(
                    baseline_xyz, guided_xyz, target_deg
                )

                row = {
                    "seed": seed,
                    "method": VARIANT_LABEL.get(vname, vname),
                    "variant": vname,
                    "target_deg": target_deg,
                    **metrics,
                }
                rows.append(row)

    # ---- Write CSV ----
    csv_path = OUT_DIR / "per_seed_diagnostics.csv"
    if rows:
        fieldnames = [
            "seed", "method", "variant", "target_deg",
            "baseline_mean_deg", "guided_mean_deg",
            "baseline_std_deg", "guided_std_deg",
            "requested_change_deg", "achieved_change_deg",
            "signed_error_deg", "abs_error_deg",
            "frame_hit_band", "final_summary_in_band", "loose_hit",
            "positive_overshoot_deg", "negative_undershoot_deg",
            "overshoot_p90_deg", "temporal_corr",
            "phase_valid_fraction", "active_frame_count",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {csv_path}")
    else:
        print("No rows generated!")

    # ---- Per-method per-target summary ----
    summary_path = OUT_DIR / "per_target_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target_deg", "method", "n_seeds",
            "signed_error_median", "signed_error_mean",
            "abs_error_median", "abs_error_mean",
            "frame_hit_band_median", "frame_hit_band_mean",
            "final_summary_hit_median",
            "temporal_corr_median", "temporal_corr_mean",
            "positive_overshoot_median",
            "negative_undershoot_median",
            "overshoot_p90_median",
            "achieved_change_median",
            "baseline_mean_median",
            "guided_std_median",
        ])
        for target_deg in sorted(TARGETS):
            for vname in VARIANTS:
                vlabel = VARIANT_LABEL.get(vname, vname)
                subset = [r for r in rows if r["target_deg"] == target_deg and r["variant"] == vname]
                if not subset:
                    continue
                writer.writerow([
                    target_deg, vlabel, len(subset),
                    round(np.median([r["signed_error_deg"] for r in subset]), 2),
                    round(np.mean([r["signed_error_deg"] for r in subset]), 2),
                    round(np.median([r["abs_error_deg"] for r in subset]), 2),
                    round(np.mean([r["abs_error_deg"] for r in subset]), 2),
                    round(np.median([r["frame_hit_band"] for r in subset]), 3),
                    round(np.mean([r["frame_hit_band"] for r in subset]), 3),
                    round(np.median([r["final_summary_in_band"] for r in subset]), 3),
                    round(np.median([r["temporal_corr"] for r in subset]), 4),
                    round(np.mean([r["temporal_corr"] for r in subset]), 4),
                    round(np.median([r["positive_overshoot_deg"] for r in subset]), 2),
                    round(np.median([r["negative_undershoot_deg"] for r in subset]), 2),
                    round(np.median([r["overshoot_p90_deg"] for r in subset]), 2),
                    round(np.median([r["achieved_change_deg"] for r in subset]), 1),
                    round(np.median([r["baseline_mean_deg"] for r in subset]), 1),
                    round(np.median([r["guided_std_deg"] for r in subset]), 2),
                ])
    print(f"Saved summary to {summary_path}")

    # ---- Print key diagnostics tables ----
    print("\n=== SIGNED ERROR (negative = under-push) ===")
    print(f"{'Target':<8} {'V7 median':>10} {'V7 mean':>10} {'V2 median':>10} {'V2 mean':>10} {'V6 median':>10}")
    for target_deg in sorted(TARGETS):
        vals = {}
        for vname in VARIANTS:
            errs = [r["signed_error_deg"] for r in rows if r["target_deg"] == target_deg and r["variant"] == vname]
            vals[vname] = (np.median(errs), np.mean(errs)) if errs else (0, 0)
        print(f"{target_deg:<8} {vals['v7-auto-dps'][0]:+10.2f} {vals['v7-auto-dps'][1]:+10.2f} {vals['v2-dps'][0]:+10.2f} {vals['v2-dps'][1]:+10.2f} {vals['v6-closed-loop'][0]:+10.2f}")

    print("\n=== FRAME HIT-BAND vs SUMMARY HIT (V7) ===")
    print(f"{'Target':<8} {'frame_hit':>10} {'final_hit':>10} {'loose_hit':>10} {'abs_error':>10}")
    for target_deg in sorted(TARGETS):
        subset = [r for r in rows if r["target_deg"] == target_deg and r["variant"] == "v7-auto-dps"]
        if subset:
            print(f"{target_deg:<8} {np.median([r['frame_hit_band'] for r in subset]):10.3f} {np.median([r['final_summary_in_band'] for r in subset]):10.3f} {np.median([r['loose_hit'] for r in subset]):10.3f} {np.median([r['abs_error_deg'] for r in subset]):10.2f}")

    print("\n=== OVER/UNDER DECOMPOSITION (V7) ===")
    print(f"{'Target':<8} {'pos_overshoot':>14} {'neg_undershoot':>14} {'overshoot_p90':>14}")
    for target_deg in sorted(TARGETS):
        subset = [r for r in rows if r["target_deg"] == target_deg and r["variant"] == "v7-auto-dps"]
        if subset:
            print(f"{target_deg:<8} {np.median([r['positive_overshoot_deg'] for r in subset]):14.2f} {np.median([r['negative_undershoot_deg'] for r in subset]):14.2f} {np.median([r['overshoot_p90_deg'] for r in subset]):14.2f}")

    print(f"\n=== DONE: {csv_path} ===")


if __name__ == "__main__":
    main()
