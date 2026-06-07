#!/usr/bin/env python3
"""
三模式评估脚本 — evaluate_all_modes.py

对 joint / muscle / both 三种 guidance 模式的输出进行统一评估。

用法:
    python evaluate_all_modes.py <output_root>
    例如: python evaluate_all_modes.py output/posture_pipeline

输出:
    - evaluation_summary.csv  (汇总表)
    - evaluation_report.md    (可读报告)
"""
import sys, os, json, math
import numpy as np
from pathlib import Path

# ---- Path setup ----
_PROJ = "/root/autodl-tmp/motion-diffusion-model"
sys.path.insert(0, _PROJ)

from posture_guidance.angle_ops import pelvis_tilt_angle, signed_knee_angle, trunk_forward_lean
from posture_guidance.joint_indices import JOINT_IDX

# ---- Configuration ----
ANGLE_METRICS = {
    "骨盆前倾": {
        "fn": lambda q: (pelvis_tilt_angle(q) * 180.0 / math.pi).numpy(),
        "target": 20.0,
        "tolerance": 2.0,
        "direction": "greater_than",
        "unit": "deg",
    },
}

def load_comparison(npy_path):
    """Load comparison.npy and extract baseline/guided xyz data."""
    data = np.load(npy_path, allow_pickle=True).item()
    xyz_base = data["motion_xyz"]         # (B, J, 3, T)
    xyz_guided = data["motion_xyz_guided"]  # (B, J, 3, T)
    guidance_mode = data.get("guidance_mode", "joint")
    posture_instructions = data.get("posture_instructions", [])
    return xyz_base, xyz_guided, guidance_mode, posture_instructions


def compute_joint_metrics(xyz_base, xyz_guided, posture_name, fps=20):
    """Compute joint-angle-based metrics for a single sample."""
    import torch
    from scipy.stats import pearsonr

    q_base = torch.from_numpy(xyz_base).permute(0, 3, 1, 2).float()      # (1, T, J, 3)
    q_guided = torch.from_numpy(xyz_guided).permute(0, 3, 1, 2).float()

    spec = ANGLE_METRICS.get(posture_name)
    if spec is None:
        return None

    angle_base = spec["fn"](q_base)[0]        # (T,)
    angle_guided = spec["fn"](q_guided)[0]    # (T,)

    # Hit rate (band)
    target, tol = spec["target"], spec["tolerance"]
    hit_band = np.mean((angle_guided >= target - tol) & (angle_guided <= target + tol))

    # Hit rate (loose: one-sided)
    if spec["direction"] == "greater_than":
        hit_loose = np.mean(angle_guided >= target - tol)
    else:
        hit_loose = np.mean(angle_guided <= target + tol)

    # Target distance
    mean_angle = np.mean(angle_guided)
    target_dist = abs(mean_angle - target)

    # Temporal correlation
    if np.std(angle_base) > 1e-6 and np.std(angle_guided) > 1e-6:
        corr, _ = pearsonr(angle_base, angle_guided)
    else:
        corr = 0.0

    # RMSE between guided and baseline
    diff = xyz_guided - xyz_base
    rmse = np.sqrt(np.mean(diff ** 2))

    # Jitter (frame-to-frame variation)
    jitter = np.mean(np.linalg.norm(np.diff(xyz_guided[0], axis=-1), axis=1))

    # Foot skate (approximate: ankle velocity below threshold)
    ankle_idx = [7, 8, 10, 11]  # left/right ankle/foot
    ankle_vel = np.linalg.norm(np.diff(xyz_guided[0, ankle_idx], axis=-1), axis=1)
    foot_skate = np.mean(ankle_vel < 0.005)  # fraction of frames with near-zero ankle vel

    # Shape check
    shape_ok = (hit_loose > 0.05 and corr > 0 and jitter < 5.0 and target_dist < 5.0)

    return {
        "mean_baseline": float(np.mean(angle_base)),
        "mean_guided": float(mean_angle),
        "delta": float(mean_angle - np.mean(angle_base)),
        "hit_band": float(hit_band),
        "hit_loose": float(hit_loose),
        "target_distance": float(target_dist),
        "corr": float(corr),
        "rmse": float(rmse),
        "jitter": float(jitter),
        "foot_skate": float(foot_skate),
        "shape_ok": shape_ok,
    }


def compute_muscle_metrics(xyz_base, xyz_guided, npy_path):
    """
    Compute muscle-specific metrics.
    Since muscle activations are not directly in comparison.npy,
    we compute proxy metrics from the guided motion quality.
    """
    # The key muscle evaluation comes from posture_loss_torch
    # which is computed during generation. Here we check:
    # 1. Whether the muscle guidance succeeded in altering posture
    # 2. Motion quality metrics
    import torch

    q_base = torch.from_numpy(xyz_base).permute(0, 3, 1, 2).float()
    q_guided = torch.from_numpy(xyz_guided).permute(0, 3, 1, 2).float()

    # Pelvis tilt as a proxy for muscle guidance effectiveness
    apt_base = (pelvis_tilt_angle(q_base) * 180.0 / math.pi).numpy()[0]
    apt_guided = (pelvis_tilt_angle(q_guided) * 180.0 / math.pi).numpy()[0]

    # Motion smoothness
    diff = xyz_guided - xyz_base
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    jitter = float(np.mean(np.linalg.norm(np.diff(xyz_guided[0], axis=-1), axis=1)))

    return {
        "apt_baseline_mean": float(np.mean(apt_base)),
        "apt_guided_mean": float(np.mean(apt_guided)),
        "apt_delta": float(np.mean(apt_guided) - np.mean(apt_base)),
        "rmse_vs_baseline": rmse,
        "jitter": jitter,
    }


def evaluate_directory(result_dir):
    """Evaluate a single result directory."""
    npy_path = os.path.join(result_dir, "comparison.npy")
    if not os.path.exists(npy_path):
        return None

    try:
        xyz_base, xyz_guided, mode, postures = load_comparison(npy_path)
    except Exception as e:
        print(f"  ERROR loading {npy_path}: {e}")
        return None

    posture_name = postures[0] if postures else "骨盆前倾"
    result = {"mode": mode, "posture": posture_name, "dir": str(result_dir)}

    # Joint metrics (always available)
    joint_m = compute_joint_metrics(xyz_base, xyz_guided, posture_name)
    if joint_m:
        result.update(joint_m)

    # Muscle metrics (for muscle/both modes)
    if mode in ("muscle", "both"):
        muscle_m = compute_muscle_metrics(xyz_base, xyz_guided, npy_path)
        if muscle_m:
            result.update(muscle_m)

    return result


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "output/posture_pipeline"

    print(f"Scanning: {root}")
    print("=" * 80)

    results = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        npy = os.path.join(full, "comparison.npy")
        if not os.path.exists(npy):
            continue

        # Parse mode from directory name
        mode = entry.split("_")[0] if "_" in entry else "unknown"
        print(f"\n--- {entry} (mode={mode}) ---")

        r = evaluate_directory(full)
        if r:
            results.append(r)
            if r.get("hit_band") is not None:
                print(f"  Pelvis Tilt: {r.get('mean_baseline',0):.1f}deg -> {r.get('mean_guided',0):.1f}deg "
                      f"(delta={r.get('delta',0):+.1f}deg)")
                print(f"  Hit Band: {r['hit_band']*100:.1f}%  Hit Loose: {r['hit_loose']*100:.1f}%")
                print(f"  Corr: {r.get('corr',0):+.3f}  RMSE: {r.get('rmse',0):.4f}m  "
                      f"Jitter: {r.get('jitter',0):.3f}  FootSkate: {r.get('foot_skate',0)*100:.1f}%")
                print(f"  Shape OK: {r.get('shape_ok', False)}")
            if r.get("apt_delta") is not None:
                print(f"  Muscle APT delta: {r['apt_delta']:+.1f}deg")
        else:
            print(f"  SKIPPED (no valid data)")

    if not results:
        print("\nNo results found.")
        return

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Mode':<10}{'APT Base':>10}{'APT Guided':>12}{'Delta':>8}{'Hit(Band)':>10}{'Hit(Loose)':>11}{'Corr':>8}{'RMSE(m)':>9}{'Jitter':>8}{'Skate%':>8}{'Shape':>8}"
    print(header)
    print("-" * 100)
    for r in results:
        if r.get("hit_band") is not None:
            print(f"{r['mode']:<10}{r.get('mean_baseline',0):>9.1f}deg{r.get('mean_guided',0):>10.1f}deg"
                  f"{r.get('delta',0):>+7.1f}deg"
                  f"{r['hit_band']*100:>8.1f}%{r['hit_loose']*100:>9.1f}%"
                  f"{r.get('corr',0):>+7.3f}{r.get('rmse',0):>8.4f}{r.get('jitter',0):>7.3f}"
                  f"{r.get('foot_skate',0)*100:>7.1f}%"
                  f"{'OK' if r.get('shape_ok') else 'FAIL':>8}")
    print("=" * 100)

    # ---- Save ----
    out_csv = os.path.join(root, "evaluation_summary.csv")
    import csv
    if results:
        keys = results[0].keys()
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV: {out_csv}")

    # ---- Markdown report ----
    out_md = os.path.join(root, "evaluation_report.md")
    with open(out_md, "w") as f:
        f.write("# Posture Guidance Evaluation Report\n\n")
        f.write(f"Evaluated {len(results)} experiment(s)\n\n")
        f.write("## Results\n\n")
        f.write("| Mode | APT Baseline | APT Guided | Delta | Hit(Band) | Hit(Loose) | Corr | RMSE | Shape |\n")
        f.write("|------|-------------|-----------|-------|-----------|------------|------|------|-------|\n")
        for r in results:
            if r.get("hit_band") is not None:
                f.write(f"| {r['mode']} | {r.get('mean_baseline',0):.1f}deg | {r.get('mean_guided',0):.1f}deg | "
                        f"{r.get('delta',0):+.1f}deg | {r['hit_band']*100:.1f}% | {r['hit_loose']*100:.1f}% | "
                        f"{r.get('corr',0):+.3f} | {r.get('rmse',0):.4f} | "
                        f"{'OK' if r.get('shape_ok') else 'FAIL'} |\n")
        f.write("\n## Notes\n\n")
        f.write("- **Hit(Band)**: Fraction of frames within [target-tol, target+tol]\n")
        f.write("- **Hit(Loose)**: Fraction of frames on the correct side of target\n")
        f.write("- **Corr**: Temporal correlation between baseline and guided angle curves\n")
        f.write("- **Shape OK**: hit_loose>5%, corr>0, jitter<5, target_distance<5deg\n")
    print(f"Saved Report: {out_md}")


if __name__ == "__main__":
    main()
