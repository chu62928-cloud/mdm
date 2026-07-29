# eval/scorecard.py -- Unified evaluation scorecard v1.0
#
# One command turns a folder of saved motions into a complete,
# publishable scorecard with robust statistics and sagittal render.
#
# Usage:
#   python -m eval.scorecard --run_dir output0727/<name> \
#       --posture anterior_pelvic_tilt --target 20.0 --tolerance 2.0 \
#       --direction greater_than --judge_op pelvis_tilt

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from eval.control_metrics import (
    JUDGE_ANGLE_OPS, posture_target_metrics, foot_skate_ratio,
    assert_judge_not_guidance)
from eval.distribution_metrics import DistributionMetrics


def load_seed_data(run_dir):
    """Load all comparison.npy files from subdirectories.

    Returns:
        list of dicts, each with keys: seed, baseline_motion, guided_motion,
        baseline_xyz, guided_xyz, text, m_len, guidance_mode, guidance_config
    """
    run_dir = Path(run_dir)
    seeds = []

    # Also check if run_dir ITSELF contains comparison.npy (single-run case)
    direct_npy = run_dir / "comparison.npy"
    if direct_npy.exists():
        data = np.load(direct_npy, allow_pickle=True).item()
        seeds.append({
            "seed": data.get("seed", 0),
            "dir_name": run_dir.name,
            "baseline": data["motion_hml_tj"][0].astype(np.float32),
            "guided": data["motion_hml_tj_guided"][0].astype(np.float32),
            "baseline_xyz": data["motion_xyz"][0].transpose(2, 0, 1),
            "guided_xyz": data["motion_xyz_guided"][0].transpose(2, 0, 1),
            "text": data.get("text_prompt", ""),
            "m_len": data["motion_hml_tj"].shape[1],
            "guidance_mode": data.get("guidance_mode", "unknown"),
            "guidance_config": data.get("guidance_config", {}),
        })

    # Search subdirectories for comparison.npy
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        npy = d / "comparison.npy"
        if not npy.exists():
            continue
        data = np.load(npy, allow_pickle=True).item()
        seeds.append({
            "seed": data.get("seed", d.name),
            "dir_name": d.name,
            "baseline": data["motion_hml_tj"][0].astype(np.float32),
            "guided": data["motion_hml_tj_guided"][0].astype(np.float32),
            "baseline_xyz": data["motion_xyz"][0].transpose(2, 0, 1),
            "guided_xyz": data["motion_xyz_guided"][0].transpose(2, 0, 1),
            "text": data.get("text_prompt", ""),
            "m_len": data["motion_hml_tj"].shape[1],
            "guidance_mode": data.get("guidance_mode", "unknown"),
            "guidance_config": data.get("guidance_config", {}),
        })
    return seeds


def bootstrap_ci(values, n_bootstrap=2000, ci=95):
    """Bootstrap confidence interval for the mean."""
    values = np.asarray(values)
    n = len(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        means.append(values[idx].mean())
    means = np.sort(means)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def aggregate_metrics(metrics_list):
    """Aggregate per-seed metric dicts into robust summary statistics.

    Returns dict with median, q1, q3, iqr, bootstrap_ci_lo, bootstrap_ci_hi
    for each numeric metric.
    """
    if not metrics_list:
        return {}

    keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        vals_arr = np.array(vals)
        ci_lo, ci_hi = bootstrap_ci(vals_arr) if len(vals) >= 3 else (float("nan"), float("nan"))
        agg[k] = {
            "median": float(np.median(vals_arr)),
            "q1": float(np.percentile(vals_arr, 25)),
            "q3": float(np.percentile(vals_arr, 75)),
            "iqr": float(np.percentile(vals_arr, 75) - np.percentile(vals_arr, 25)),
            "mean": float(vals_arr.mean()),
            "std": float(vals_arr.std()),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
        }
    return agg


def compute_scorecard(run_dir, posture, target, tolerance, direction,
                      judge_op_name="pelvis_tilt",
                      guidance_op_name="pelvis_tilt_angle",
                      device="cuda",
                      compute_dist_metrics=True):
    """Compute a full scorecard for a run directory.

    Args:
        run_dir: path to directory with seed subdirectories
        posture: posture name string
        target: target angle value (degrees or meters)
        tolerance: band half-width
        direction: "greater_than" | "less_than" | "equal"
        judge_op_name: key into JUDGE_ANGLE_OPS
        guidance_op_name: name of the guidance angle op (for firewall check)
        device: "cuda" or "cpu"
        compute_dist_metrics: whether to compute FID/Diversity (slow, needs
            N>=100 for reliability)

    Returns:
        dict with full scorecard
    """
    run_dir = Path(run_dir)
    seeds = load_seed_data(run_dir)
    N = len(seeds)
    if N == 0:
        raise ValueError(f"No comparison.npy files found in {run_dir}")

    judge_op = JUDGE_ANGLE_OPS.get(judge_op_name)
    if judge_op is None:
        raise ValueError(f"Unknown judge op '{judge_op_name}'. "
                         f"Available: {list(JUDGE_ANGLE_OPS)}")

    # Firewall: judge must not be guidance
    try:
        assert_judge_not_guidance(judge_op, guidance_op_name)
    except AssertionError as e:
        warnings.warn(str(e))

    print(f"[scorecard] Processing {N} seeds from {run_dir}")

    # Collect metrics
    metric_keys = {"target_hit_band", "delta", "overshoot", "temporal_corr",
                   "phase_selectivity", "valid_fraction",
                   "foot_skate_baseline", "foot_skate_guided"}
    control_metrics = []
    for s in seeds:
        m = posture_target_metrics(
            s["guided_xyz"], s["baseline_xyz"],
            judge_op, target, tolerance, direction,
            judge_op_name=judge_op_name, unit="deg")
        m["foot_skate_baseline"] = foot_skate_ratio(s["baseline_xyz"])
        m["foot_skate_guided"] = foot_skate_ratio(s["guided_xyz"])
        m["seed"] = s["seed"]
        control_metrics.append(m)

    # Distribution metrics (optional, slow)
    dist_metrics = {}
    if compute_dist_metrics and N >= 10:
        print("[scorecard] Computing distribution metrics ...")
        dm = DistributionMetrics(device=device)
        bl_motions = [s["baseline"] for s in seeds]
        gd_motions = [s["guided"] for s in seeds]
        bl_lens = [s["m_len"] for s in seeds]
        gd_lens = [s["m_len"] for s in seeds]
        texts = [s["text"] for s in seeds]

        if N >= 100:
            dist_metrics["fid_baseline"] = dm.compute_fid(bl_motions, bl_lens)
            dist_metrics["fid_guided"] = dm.compute_fid(gd_motions, gd_lens)
        else:
            dist_metrics["fid_warning"] = f"N={N} < 100, FID skipped (unreliable)"

        if N >= 10:
            dist_metrics["diversity_baseline"] = dm.compute_diversity(
                bl_motions, bl_lens, times=min(300, N))
            dist_metrics["diversity_guided"] = dm.compute_diversity(
                gd_motions, gd_lens, times=min(300, N))

    # Aggregate
    agg = aggregate_metrics(control_metrics)

    # Build scorecard
    scorecard = {
        "eval_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "posture": posture,
        "target": target,
        "tolerance": tolerance,
        "direction": direction,
        "judge_op": judge_op_name,
        "guidance_op": guidance_op_name,
        "N_seeds": N,
        "guidance_mode": seeds[0]["guidance_mode"] if seeds else "unknown",
        "guidance_config": seeds[0]["guidance_config"] if seeds else {},
        "text_prompt": seeds[0]["text"] if seeds else "",
        "control_metrics_aggregated": agg,
        "per_seed_control": control_metrics,
        "distribution_metrics": dist_metrics,
    }
    return scorecard


def write_scorecard(scorecard, output_dir):
    """Write scorecard.json and scorecard.md to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "scorecard.json"
    with open(json_path, "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    print(f"[scorecard] Saved {json_path}")

    # Markdown
    md_path = output_dir / "scorecard.md"
    lines = []
    sc = scorecard
    agg = sc["control_metrics_aggregated"]
    dist = sc.get("distribution_metrics", {})

    lines.append(f"# Scorecard — {sc['posture']}")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Eval version | {sc['eval_version']} |")
    lines.append(f"| Timestamp | {sc['timestamp'][:19]} |")
    lines.append(f"| N seeds | {sc['N_seeds']} |")
    lines.append(f"| Target | {sc['target']} +- {sc['tolerance']} ({sc['direction']}) |")
    lines.append(f"| Judge op | {sc['judge_op']} |")
    lines.append(f"| Guidance mode | {sc['guidance_mode']} |")
    lines.append(f"| Text prompt | {sc['text_prompt']} |")
    lines.append("")

    lines.append("## Control Metrics (aggregated)")
    lines.append("")
    lines.append("| Metric | Median | Q1-Q3 | IQR | Mean +- Std | 95% CI |")
    lines.append("|--------|--------|-------|-----|-------------|--------|")
    for k, v in agg.items():
        ci = f"[{v['ci_95_lo']:.3f}, {v['ci_95_hi']:.3f}]"
        lines.append(f"| {k} | {v['median']:.4f} | "
                     f"[{v['q1']:.4f}, {v['q3']:.4f}] | {v['iqr']:.4f} | "
                     f"{v['mean']:.4f} +- {v['std']:.4f} | {ci} |")
    lines.append("")

    if dist:
        lines.append("## Distribution Metrics")
        lines.append("")
        for k, v in dist.items():
            lines.append(f"- **{k}**: {v}")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[scorecard] Saved {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--posture", default="anterior_pelvic_tilt")
    ap.add_argument("--target", type=float, default=20.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--direction", default="greater_than")
    ap.add_argument("--judge_op", default="pelvis_tilt")
    ap.add_argument("--guidance_op", default="pelvis_tilt_angle")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_dist_metrics", action="store_true")
    args = ap.parse_args()

    output_dir = args.output_dir or args.run_dir
    sc = compute_scorecard(
        args.run_dir, args.posture, args.target, args.tolerance,
        args.direction, args.judge_op, args.guidance_op,
        device=args.device,
        compute_dist_metrics=not args.no_dist_metrics,
    )
    write_scorecard(sc, output_dir)


if __name__ == "__main__":
    main()