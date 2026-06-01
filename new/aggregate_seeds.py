"""
new/aggregate_seeds.py

聚合多 seed 实验结果，计算每个 variant 的指标均值 ± 标准差。
判定 variant 是否"稳健"——好的方法应该跨 seed 表现一致。

使用：
    python -m new.aggregate_seeds ./output/seedtest_v2_dps_best ./output/seedtest_v2b_x0_edit_best
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from new.evaluate_ablation_v3 import compute_metrics, classify_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seedtest_dirs", nargs="+",
                        help="seed test 目录列表，例如 seedtest_v2_dps_best/")
    parser.add_argument("--posture", type=str, default=None,
                        help="评估目标体态。None 时根据第一个目录名自动推断。")
    args = parser.parse_args()

    # 自动推断 posture（避免漏传 --posture 把数据按错误目标评估）
    if args.posture is None:
        first = __import__('pathlib').Path(args.seedtest_dirs[0]).name.lower()
        if "trunk" in first or "躯干前倾" in first:
            args.posture = "躯干前倾"
        elif "flex_b" in first or "膝弯曲_b" in first:
            args.posture = "膝弯曲_B"
        elif "flex_a" in first or "膝弯曲_a" in first:
            args.posture = "膝弯曲_A"
        elif "flex" in first or "膝弯曲" in first:  # generic flex after specific A/B
            args.posture = "膝弯曲"
        elif "knee" in first or "膝" in first:
            args.posture = "膝超伸"
        elif "apt" in first or "pelvis" in first or "骨盆" in first:
            args.posture = "骨盆前倾"
        elif "kyphosis" in first or "驼背" in first:
            args.posture = "驼背"
        elif "fhp" in first or "头前伸" in first:
            args.posture = "头前伸"
        else:
            args.posture = "骨盆前倾"
            print(f"⚠ 无法从目录名 '{first}' 推断 posture，使用默认 '骨盆前倾'。"
                  f"如需评估其它体态，请显式传 --posture <name>。")
        print(f"[auto-detect] posture = {args.posture}")

    all_results = {}  # variant_name -> list of metric dicts

    for dir_path in args.seedtest_dirs:
        d = Path(dir_path)
        if not d.exists():
            print(f"skip: {d} 不存在")
            continue

        variant_name = d.name.replace("seedtest_", "")
        all_results[variant_name] = []

        for seed_dir in sorted(d.iterdir()):
            if not seed_dir.is_dir():
                continue
            npy = seed_dir / "comparison.npy"
            if not npy.exists():
                continue

            try:
                m = compute_metrics(npy, args.posture)
                m["seed_dir"] = seed_dir.name
                all_results[variant_name].append(m)
            except Exception as e:
                print(f"  ERR {seed_dir.name}: {e}")

    print("\n" + "=" * 110)
    print(f"  多 seed 聚合统计 — {args.posture}")
    print("=" * 110)
    print(f"{'variant':<25}{'N':>4}{'Δ(mean±std)':>18}{'hit_band':>16}{'corr':>16}"
          f"{'rmse':>14}{'shape (主导)':<28}")
    print("-" * 110)

    summary = []
    for variant, metrics in all_results.items():
        if not metrics:
            continue

        n = len(metrics)
        delta = [m["delta"] for m in metrics]
        hit   = [m["hit_rate"] for m in metrics]
        corr  = [m["corr"] for m in metrics]
        rmse  = [m["rmse"] for m in metrics]

        # 主导 shape
        shapes = [classify_shape(m) for m in metrics]
        from collections import Counter
        dominant_shape, dominant_count = Counter(shapes).most_common(1)[0]
        shape_str = f"{dominant_shape} ({dominant_count}/{n})"

        print(f"{variant:<25}{n:>4}"
              f"  {np.mean(delta):+5.2f}±{np.std(delta):4.2f}°  "
              f"  {np.mean(hit)*100:5.1f}±{np.std(hit)*100:4.1f}%  "
              f"  {np.mean(corr):+5.3f}±{np.std(corr):.3f}  "
              f"  {np.mean(rmse):.3f}±{np.std(rmse):.3f}  "
              f"  {shape_str}")

        summary.append({
            "variant": variant,
            "n": n,
            "delta_mean": np.mean(delta), "delta_std": np.std(delta),
            "hit_mean": np.mean(hit),     "hit_std": np.std(hit),
            "corr_mean": np.mean(corr),   "corr_std": np.std(corr),
            "rmse_mean": np.mean(rmse),   "rmse_std": np.std(rmse),
            "shape": shape_str,
        })

    print("=" * 110)

    print("\n稳健性判定：")
    for s in summary:
        cv_corr = (s["corr_std"] / max(abs(s["corr_mean"]), 1e-3)) * 100
        cv_delta = (s["delta_std"] / max(abs(s["delta_mean"]), 1e-3)) * 100

        verdict = "✅ 稳健" if (cv_corr < 30 and cv_delta < 20) else "⚠ 不稳定"
        print(f"  {s['variant']:<25} CV(corr)={cv_corr:5.1f}% CV(Δ)={cv_delta:5.1f}%  {verdict}")

    print("\nCV < 30% 视为稳健（跨 seed 一致），> 30% 说明结果依赖 seed")


if __name__ == "__main__":
    main()