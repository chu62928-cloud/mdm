"""
new/evaluate_ablation.py

Evaluate all ablation variants and produce a unified comparison table.

Reads each variant's comparison.npy, computes:
    - Δ angle (mean target angle shift)
    - hit_rate (frames in target band)
    - Pearson correlation (vs baseline trajectory)
    - RMSE (joint position deviation)
    - jitter (frame-to-frame angle change std)
    - foot_skate (sliding foot frames ratio)
    - inference_time (from log)

Then ranks variants by a composite score.

Usage:
    python -m new.evaluate_ablation ./output/ablation_walking_apt_seed42
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance import angle_ops


# Map posture name -> (angle_fn, target_deg, tolerance_deg, direction)
ANGLE_TARGETS = {
    "骨盆前倾":   ("pelvis_tilt_angle",      20.0, 2.0, "greater_than"),
    "骨盆前倾_深蹲": ("pelvis_tilt_angle",    25.0, 3.0, "greater_than"),
    "膝超伸":     ("signed_knee_angle_both", 185.0, 3.0, "greater_than"),
    "膝超伸_左":   ("signed_knee_angle_left", 185.0, 3.0, "greater_than"),
    "膝超伸_右":   ("signed_knee_angle_right", 185.0, 3.0, "greater_than"),
    "驼背":       ("spine_posterior_bulge",  0.08, 0.02, "greater_than"),
}


def get_angle_fn(name: str):
    """Resolve angle function by name, with special handlers for knee."""
    if name == "signed_knee_angle_both":
        def _both(q):
            l = angle_ops.signed_knee_angle(q, side="left")
            r = angle_ops.signed_knee_angle(q, side="right")
            return (l + r) / 2.0
        return _both
    if name == "signed_knee_angle_left":
        return lambda q: angle_ops.signed_knee_angle(q, side="left")
    if name == "signed_knee_angle_right":
        return lambda q: angle_ops.signed_knee_angle(q, side="right")
    return getattr(angle_ops, name)


def compute_metrics(npy_path: Path, posture: str) -> dict:
    """Compute all metrics for one variant's comparison.npy."""
    data = np.load(npy_path, allow_pickle=True).item()

    if posture not in ANGLE_TARGETS:
        raise ValueError(f"Unknown posture: {posture}")
    fn_name, target_deg, tol_deg, direction = ANGLE_TARGETS[posture]
    angle_fn = get_angle_fn(fn_name)

    q_b = torch.from_numpy(data["motion_xyz"][0]).float().permute(2, 0, 1)
    q_g = torch.from_numpy(data["motion_xyz_guided"][0]).float().permute(2, 0, 1)

    use_radians = "spine_posterior" not in fn_name

    a_b = angle_fn(q_b)
    a_g = angle_fn(q_g)

    if use_radians:
        a_b_deg = (a_b * 180.0 / math.pi).numpy()
        a_g_deg = (a_g * 180.0 / math.pi).numpy()
        target = target_deg
        tol = tol_deg
    else:
        a_b_deg = a_b.numpy()
        a_g_deg = a_g.numpy()
        target = target_deg
        tol = tol_deg

    delta = float(a_g_deg.mean() - a_b_deg.mean())

    if direction == "greater_than":
        hit_mask = a_g_deg >= (target - tol)
    elif direction == "less_than":
        hit_mask = a_g_deg <= (target + tol)
    else:
        hit_mask = np.abs(a_g_deg - target) <= tol
    hit_rate = float(hit_mask.mean())

    if a_b_deg.std() > 1e-6 and a_g_deg.std() > 1e-6:
        corr = float(pearsonr(a_b_deg, a_g_deg)[0])
    else:
        corr = 0.0

    diff = q_g.numpy() - q_b.numpy()
    rmse = float(np.sqrt((diff ** 2).mean()))

    jitter = float(np.diff(a_g_deg).std())

    foot_skate = compute_foot_skate(q_g.numpy())

    return {
        "baseline_mean": float(a_b_deg.mean()),
        "guided_mean":   float(a_g_deg.mean()),
        "delta":         delta,
        "hit_rate":      hit_rate,
        "corr":          corr,
        "rmse":          rmse,
        "jitter":        jitter,
        "foot_skate":    foot_skate,
    }


def compute_foot_skate(q: np.ndarray, ground_thresh: float = 0.05,
                      slide_thresh: float = 0.025) -> float:
    """Fraction of frames where any foot is on ground but moving sideways."""
    if q.shape[1] < 22:
        return 0.0
    L_FOOT, R_FOOT = 10, 11
    feet = q[:, [L_FOOT, R_FOOT], :]                # (T, 2, 3)
    on_ground = feet[:, :, 1] < ground_thresh       # (T, 2)
    motion = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1)
    sliding = motion > slide_thresh
    skate = on_ground[1:] & sliding
    return float(skate.any(axis=1).mean())


def parse_inference_time(log_path: Path) -> float:
    """Pull total sampling time out of pipeline.log if present."""
    if not log_path.exists():
        return float("nan")
    txt = log_path.read_text(errors="ignore")
    m = re.search(r"sampling.*?(\d+\.?\d*)\s*(?:s|sec|seconds)", txt, re.I)
    if m:
        return float(m.group(1))
    return float("nan")


def composite_score(m: dict) -> float:
    """
    Combined score: rewards meeting target AND preserving structure.
    Designed to penalize both 'too weak' and 'too strong' failures.

        score = hit_rate * corr / (1 + 5 * rmse + 0.1 * jitter + foot_skate)
    """
    if m["hit_rate"] < 0.05:
        return 0.0
    return (m["hit_rate"] * max(m["corr"], 0.0) /
            (1.0 + 5.0 * m["rmse"] + 0.1 * m["jitter"] + m["foot_skate"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_dir", type=str,
                        help="e.g. ./output/ablation_walking_apt_seed42")
    parser.add_argument("--posture", type=str, default=None,
                        help="override posture (else read from npy)")
    args = parser.parse_args()

    base = Path(args.ablation_dir)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        sys.exit(1)

    rows = []
    for variant_dir in sorted(base.iterdir()):
        if not variant_dir.is_dir():
            continue
        npy = variant_dir / "comparison.npy"
        if not npy.exists():
            print(f"  skip {variant_dir.name}: no comparison.npy")
            continue

        data = np.load(npy, allow_pickle=True).item()
        posture = args.posture or data.get("posture_instructions", "骨盆前倾")
        if isinstance(posture, list):
            posture = posture[0]

        try:
            m = compute_metrics(npy, posture)
            m["variant"] = variant_dir.name
            m["time_s"] = parse_inference_time(variant_dir / "pipeline.log")
            m["score"] = composite_score(m)
            rows.append(m)
        except Exception as e:
            print(f"  ERROR on {variant_dir.name}: {e}")
            continue

    if not rows:
        print("No variants evaluated.")
        return

    rows.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 110)
    print(f"  Ablation results — {base.name}")
    print(f"  Posture: {posture}")
    print("=" * 110)

    header = f"{'rank':<5}{'variant':<25}{'Δ':>8}{'hit':>8}{'corr':>8}{'RMSE':>8}{'jitter':>8}{'skate':>8}{'time(s)':>10}{'score':>10}"
    print(header)
    print("-" * 110)

    for i, r in enumerate(rows, start=1):
        print(f"{i:<5}{r['variant']:<25}"
              f"{r['delta']:+7.2f} "
              f"{r['hit_rate']*100:6.1f}%  "
              f"{r['corr']:+6.3f}  "
              f"{r['rmse']:6.3f}  "
              f"{r['jitter']:6.2f}  "
              f"{r['foot_skate']*100:5.1f}%  "
              f"{r['time_s']:8.1f}  "
              f"{r['score']:8.4f}")

    print("=" * 110)
    print(f"\n🏆 Best variant: {rows[0]['variant']} (score={rows[0]['score']:.4f})")
    print("\nMetric meanings:")
    print("  Δ      : mean angle shift (target - baseline). Larger = more pathology.")
    print("  hit    : % of frames inside target band [target±tol]. 40-80% is healthy.")
    print("  corr   : Pearson corr with baseline trajectory. >0.3 preserves motion.")
    print("  RMSE   : joint xyz deviation in meters. <0.15m keeps motion natural.")
    print("  jitter : frame-to-frame angle std. <2.0° is smooth.")
    print("  skate  : foot-skate frames (artifact). <5% is acceptable.")
    print("  score  : hit_rate * corr / (1 + 5*RMSE + 0.1*jitter + skate)")

    out_json = base / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {out_json}")

    out_md = base / "ablation_results.md"
    write_markdown_report(rows, posture, out_md)
    print(f"💾 Saved: {out_md}")


def write_markdown_report(rows, posture, path):
    """Write a markdown report ready to paste into the project notebook."""
    lines = []
    lines.append(f"# Ablation Results — {posture}\n")
    lines.append("| Rank | Variant | Δ (°) | Hit Rate | Corr | RMSE (m) | Jitter (°) | Foot Skate | Time (s) | Score |")
    lines.append("|------|---------|-------|----------|------|----------|------------|------------|----------|-------|")
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['delta']:+.2f} | {r['hit_rate']*100:.1f}% | "
            f"{r['corr']:+.3f} | {r['rmse']:.3f} | {r['jitter']:.2f} | "
            f"{r['foot_skate']*100:.1f}% | {r['time_s']:.1f} | **{r['score']:.4f}** |"
        )
    lines.append("\n---\n")
    lines.append(f"**Best**: `{rows[0]['variant']}` (score = {rows[0]['score']:.4f})\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()