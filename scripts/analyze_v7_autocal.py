"""Post-process V7 auto-calibration results.

Generates:
  - master_table.csv (per-variant, per-target metrics)
  - requested_vs_achieved.png
  - per_target_tables
  - paired_bootstrap.json
"""

import json, csv, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---- Config ----
RESULT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output0727/v7_autocal")
TARGETS = {
    5:  "tau05",
    10: "tau10",
    15: "tau15",
    20: "tau20",
    25: "tau25",
}
VARIANTS = ["v7-auto-dps", "v2-dps", "v6-closed-loop"]
VARIANT_LABELS = {
    "v7-auto-dps": "V7 (ours)",
    "v2-dps": "V2 (s=40)",
    "v6-closed-loop": "V6 (PID)",
}


def load_scorecard(path):
    """Load scorecard from a run directory."""
    sc_path = Path(path) / "scorecard.json"
    if sc_path.exists():
        with open(sc_path) as f:
            return json.load(f)
    return None


def collect_all_results():
    """Scan result directories and collect per-seed metrics."""
    results = []  # list of dicts

    for target_deg, tau_name in TARGETS.items():
        for vname in VARIANTS:
            run_dir = RESULT_ROOT / tau_name / vname
            sc = load_scorecard(run_dir)
            if sc is None:
                print(f"  WARNING: No scorecard for {tau_name}/{vname}")
                continue

            for seed_data in sc.get("per_seed_control", []):
                results.append({
                    "target_deg": target_deg,
                    "tau_name": tau_name,
                    "variant": vname,
                    "variant_label": VARIANT_LABELS.get(vname, vname),
                    "seed": seed_data.get("seed"),
                    "hit_band": seed_data.get("target_hit_band"),
                    "delta": seed_data.get("delta"),
                    "overshoot": seed_data.get("overshoot"),
                    "temporal_corr": seed_data.get("temporal_corr"),
                    "foot_skate_base": seed_data.get("foot_skate_baseline"),
                    "foot_skate_guided": seed_data.get("foot_skate_guided"),
                    "valid_fraction": seed_data.get("valid_fraction"),
                })

    return results


def compute_summary(results):
    """Compute per-variant per-target summary statistics."""
    summary = defaultdict(lambda: defaultdict(dict))

    for target_deg in TARGETS:
        for vname in VARIANTS:
            subset = [r for r in results
                      if r["target_deg"] == target_deg and r["variant"] == vname]

            if not subset:
                continue

            for metric in ["hit_band", "delta", "overshoot", "temporal_corr"]:
                values = [r[metric] for r in subset if r[metric] is not None]
                if values:
                    arr = np.array(values)
                    summary[target_deg][vname][metric] = {
                        "median": float(np.median(arr)),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "q1": float(np.percentile(arr, 25)),
                        "q3": float(np.percentile(arr, 75)),
                        "n": len(values),
                    }

    return summary


def paired_bootstrap(results_a, results_b, metric, n_bootstrap=10000):
    """Paired bootstrap comparison of two variants."""
    seeds_a = {r["seed"]: r[metric] for r in results_a if r[metric] is not None}
    seeds_b = {r["seed"]: r[metric] for r in results_b if r[metric] is not None}
    common = sorted(set(seeds_a) & set(seeds_b))

    if len(common) < 5:
        return {"error": "too few common seeds", "n": len(common)}

    diffs = np.array([seeds_a[s] - seeds_b[s] for s in common])
    obs_diff = np.mean(diffs)
    n = len(diffs)

    boot_diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_diffs.append(np.mean(diffs[idx]))

    boot_diffs = np.array(boot_diffs)
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    p_value = float(np.mean(np.abs(boot_diffs) >= np.abs(obs_diff)))

    return {
        "observed_diff": float(obs_diff),
        "ci_95": [ci_lo, ci_hi],
        "p_value": p_value,
        "n_pairs": n,
        "significant_at_05": p_value < 0.05,
    }


def main():
    print(f"Analyzing results from: {RESULT_ROOT}")
    print()

    # Generate scorecards first
    print("=== Generating scorecards ===")
    for target_deg, tau_name in TARGETS.items():
        for vname in VARIANTS:
            run_dir = RESULT_ROOT / tau_name / vname
            if not run_dir.exists():
                continue
            sc_path = run_dir / "scorecard.json"
            if not sc_path.exists():
                # Generate scorecard
                posture_name = "anterior_pelvic_tilt" if target_deg == 20 else f"anterior_pelvic_tilt_tau{target_deg:02d}"
                cmd = (
                    f"cd /root/autodl-tmp/motion-diffusion-model && "
                    f"source /root/miniconda3/bin/activate mdm5090 && "
                    f"python -m eval.scorecard --run_dir {run_dir} "
                    f"--posture {posture_name} --target {target_deg} --tolerance 2.0 "
                    f"--no_dist_metrics 2>/dev/null"
                )
                os.system(cmd)
                print(f"  Generated: {tau_name}/{vname}/scorecard.json")

    # Collect results
    print("\n=== Collecting results ===")
    results = collect_all_results()
    print(f"  Total per-seed records: {len(results)}")

    # Per-variant per-target summary
    summary = compute_summary(results)

    # Master table
    master_path = RESULT_ROOT / "master_table.csv"
    with open(master_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_deg", "variant", "n_seeds",
                         "hit_band_median", "hit_band_mean",
                         "delta_median", "overshoot_median",
                         "temporal_corr_median", "temporal_corr_mean"])
        for target_deg in sorted(TARGETS):
            for vname in VARIANTS:
                s = summary.get(target_deg, {}).get(vname, {})
                if not s:
                    continue
                writer.writerow([
                    target_deg, VARIANT_LABELS.get(vname, vname),
                    s.get("hit_band", {}).get("n", 0),
                    round(s.get("hit_band", {}).get("median", 0), 3),
                    round(s.get("hit_band", {}).get("mean", 0), 3),
                    round(s.get("delta", {}).get("median", 0), 1),
                    round(s.get("overshoot", {}).get("median", 0), 2),
                    round(s.get("temporal_corr", {}).get("median", 0), 4),
                    round(s.get("temporal_corr", {}).get("mean", 0), 4),
                ])
    print(f"  Master table: {master_path}")

    # Paired bootstrap: V7 vs V2, V7 vs V6
    print("\n=== Paired Bootstrap ===")
    bootstrap_results = {}
    for target_deg in TARGETS:
        v7_results = [r for r in results if r["target_deg"] == target_deg and r["variant"] == "v7-auto-dps"]
        v2_results = [r for r in results if r["target_deg"] == target_deg and r["variant"] == "v2-dps"]
        v6_results = [r for r in results if r["target_deg"] == target_deg and r["variant"] == "v6-closed-loop"]

        key = f"tau{target_deg:02d}"
        bootstrap_results[key] = {}

        for metric in ["hit_band", "delta", "overshoot", "temporal_corr"]:
            v7v2 = paired_bootstrap(v7_results, v2_results, metric)
            v7v6 = paired_bootstrap(v7_results, v6_results, metric)
            bootstrap_results[key][f"{metric}_v7_vs_v2"] = v7v2
            bootstrap_results[key][f"{metric}_v7_vs_v6"] = v7v6

            sig_v2 = "SIG" if v7v2.get("significant_at_05") else "ns"
            sig_v6 = "SIG" if v7v6.get("significant_at_05") else "ns"
            print(f"  tau={target_deg:02d} {metric:15s}: V7-V2={v7v2['observed_diff']:+.4f} [{sig_v2}]  V7-V6={v7v6['observed_diff']:+.4f} [{sig_v6}]")

    bp_path = RESULT_ROOT / "paired_bootstrap.json"
    with open(bp_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"  Bootstrap: {bp_path}")

    # Print key comparison table
    print("\n=== KEY COMPARISON ===")
    print(f"{'Target':<8} {'Metric':<15} {'V7':<10} {'V2':<10} {'V6':<10}")
    print("-" * 55)
    for target_deg in sorted(TARGETS):
        for metric, fmt in [("hit_band", ".3f"), ("delta", ".1f"), ("overshoot", ".2f"), ("temporal_corr", ".4f")]:
            vals = {}
            for vname in VARIANTS:
                s = summary.get(target_deg, {}).get(vname, {}).get(metric, {})
                vals[vname] = s.get("median", float("nan"))
            print(f"{target_deg:<8} {metric:<15} {vals['v7-auto-dps']:{fmt}}  {vals['v2-dps']:{fmt}}  {vals['v6-closed-loop']:{fmt}}")
        print()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
