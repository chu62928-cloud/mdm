"""
new/evaluate_sweep.py

Evaluate parameter sweep results (schedule and strength).
Same metrics as evaluate_ablation, but groups results by sweep dimension
and produces curves to find optimal hyperparameters.

Usage:
    python -m new.evaluate_sweep ./output/sweep_v2_dps_walking_seed42
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from new.evaluate_ablation import compute_metrics, composite_score, parse_inference_time

import numpy as np


def parse_run_name(name: str):
    """
    Parse 'sched_final', 'base_weight_30.0', 's_1.0', 'lr_0.05'  -> (dim, value).
    """
    if name.startswith("sched_"):
        return ("schedule", name[len("sched_"):])
    if name.startswith("base_weight_"):
        return ("base_weight", float(name[len("base_weight_"):]))
    if name.startswith("s_"):
        return ("s", float(name[len("s_"):]))
    if name.startswith("lr_"):
        return ("lr", float(name[len("lr_"):]))
    return ("unknown", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=str)
    parser.add_argument("--posture", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.sweep_dir)
    rows = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        npy = run_dir / "comparison.npy"
        if not npy.exists():
            continue
        try:
            data = np.load(npy, allow_pickle=True).item()
            posture = args.posture or data.get("posture_instructions", "骨盆前倾")
            if isinstance(posture, list):
                posture = posture[0]

            m = compute_metrics(npy, posture)
            dim, val = parse_run_name(run_dir.name)
            m["sweep_dim"] = dim
            m["sweep_val"] = val
            m["run_name"] = run_dir.name
            m["time_s"] = parse_inference_time(run_dir / "pipeline.log")
            m["score"] = composite_score(m)
            rows.append(m)
        except Exception as e:
            print(f"  skip {run_dir.name}: {e}")

    if not rows:
        print("No runs to evaluate.")
        return

    by_dim = {}
    for r in rows:
        by_dim.setdefault(r["sweep_dim"], []).append(r)

    for dim, items in by_dim.items():
        print(f"\n{'=' * 90}")
        print(f"  Sweep: {dim}")
        print("=" * 90)
        print(f"{'value':<15}{'Δ':>8}{'hit':>8}{'corr':>8}{'RMSE':>8}{'jitter':>8}{'skate':>8}{'score':>10}")
        print("-" * 90)

        if dim == "schedule":
            order = ["always", "decay", "second_half", "last_quarter", "final"]
            items_sorted = sorted(items, key=lambda r: order.index(str(r["sweep_val"]))
                                  if str(r["sweep_val"]) in order else 99)
        else:
            items_sorted = sorted(items, key=lambda r: r["sweep_val"])

        for r in items_sorted:
            print(f"{str(r['sweep_val']):<15}"
                  f"{r['delta']:+7.2f} "
                  f"{r['hit_rate']*100:6.1f}%  "
                  f"{r['corr']:+6.3f}  "
                  f"{r['rmse']:6.3f}  "
                  f"{r['jitter']:6.2f}  "
                  f"{r['foot_skate']*100:5.1f}%  "
                  f"{r['score']:8.4f}")

        best = max(items, key=lambda r: r["score"])
        print(f"\n  🏆 Best {dim}: {best['sweep_val']} (score={best['score']:.4f})")

    out_json = base / "sweep_results.json"
    json.dump(rows, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {out_json}")


if __name__ == "__main__":
    main()