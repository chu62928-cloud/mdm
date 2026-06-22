"""
scripts/aggregate_seeds.py

聚合多 seed 实验结果。输出两部分：
  1. 均值 ± 标准差表格（向后兼容）
  2. 稳健统计：median [Q1, Q3] + 95% bootstrap CI（mean）

用法：
    python -m scripts.aggregate_seeds ./output/seedtest_v2_dps_best ./output/seedtest_v6_best
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_ablation import compute_metrics, classify_shape


def bootstrap_ci(data, n_bootstrap=2000, ci=0.95):
    """95% bootstrap CI on the mean via percentile method."""
    data = np.asarray(data, dtype=float)
    if len(data) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        rng.choice(data, len(data), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return tuple(np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seedtest_dirs", nargs="+",
                        help="seed test 目录列表，例如 seedtest_v2_dps_best/")
    parser.add_argument("--posture", type=str, default=None,
                        help="评估目标体态。None 时根据第一个目录名自动推断。")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="跳过 bootstrap CI（加速，仅看均值/std）")
    args = parser.parse_args()

    # 自动推断 posture
    if args.posture is None:
        first = Path(args.seedtest_dirs[0]).name.lower()
        if "trunk" in first or "躯干前倾" in first:
            args.posture = "躯干前倾"
        elif "pelvic_tilt_lat" in first or "骨盆侧倾" in first:
            args.posture = "骨盆侧倾"
        elif "flex_b" in first or "膝弯曲_b" in first:
            args.posture = "膝弯曲_B"
        elif "flex_a" in first or "膝弯曲_a" in first:
            args.posture = "膝弯曲_A"
        elif "flex" in first or "膝弯曲" in first:
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

    # ── 主表格（均值 ± 标准差，向后兼容）──────────────────────────────
    W = 110
    print("\n" + "=" * W)
    print(f"  多 seed 聚合统计 — {args.posture}")
    print("=" * W)
    print(f"{'variant':<25}{'N':>4}{'Δ(mean±std)':>18}{'hit_band':>16}{'corr':>16}"
          f"{'rmse':>14}{'shape (主导)':<28}")
    print("-" * W)

    summary = []
    for variant, metrics in all_results.items():
        if not metrics:
            continue

        n = len(metrics)
        delta = [m["delta"] for m in metrics]
        hit   = [m["hit_rate"] for m in metrics]
        corr  = [m["corr"] for m in metrics]
        rmse  = [m["rmse"] for m in metrics]

        dominant_shape, dominant_count = Counter(
            [classify_shape(m) for m in metrics]
        ).most_common(1)[0]

        print(f"{variant:<25}{n:>4}"
              f"  {np.mean(delta):+5.2f}±{np.std(delta):4.2f}°  "
              f"  {np.mean(hit)*100:5.1f}±{np.std(hit)*100:4.1f}%  "
              f"  {np.mean(corr):+5.3f}±{np.std(corr):.3f}  "
              f"  {np.mean(rmse):.3f}±{np.std(rmse):.3f}  "
              f"  {dominant_shape} ({dominant_count}/{n})")

        summary.append({
            "variant": variant, "n": n,
            "delta": delta, "hit": hit, "corr": corr, "rmse": rmse,
            "delta_mean": np.mean(delta), "delta_std": np.std(delta),
            "hit_mean":   np.mean(hit),   "hit_std":   np.std(hit),
            "corr_mean":  np.mean(corr),  "corr_std":  np.std(corr),
            "rmse_mean":  np.mean(rmse),  "rmse_std":  np.std(rmse),
            "shape": f"{dominant_shape} ({dominant_count}/{n})",
        })

    print("=" * W)

    # ── 稳健性判定（CV，向后兼容）──────────────────────────────────────
    print("\n稳健性判定：")
    for s in summary:
        cv_corr  = (s["corr_std"]  / max(abs(s["corr_mean"]),  1e-3)) * 100
        cv_delta = (s["delta_std"] / max(abs(s["delta_mean"]), 1e-3)) * 100
        verdict = "✅ 稳健" if (cv_corr < 30 and cv_delta < 20) else "⚠ 不稳定"
        print(f"  {s['variant']:<25} CV(corr)={cv_corr:5.1f}% CV(Δ)={cv_delta:5.1f}%  {verdict}")
    print("\nCV < 30% 视为稳健（跨 seed 一致），> 30% 说明结果依赖 seed")

    if args.no_bootstrap:
        return

    # ── 稳健统计：median [IQR] + 95% bootstrap CI ────────────────────
    print("\n" + "=" * W)
    print("  稳健统计：median [Q1, Q3] | corr 95% bootstrap CI (mean)")
    print("=" * W)
    print(f"{'variant':<25}{'N':>4}  {'Δ median[IQR]':>18}  {'corr median[IQR]':>20}  {'corr 95%-CI':>22}")
    print("-" * W)

    for s in summary:
        delta_arr = np.array(s["delta"])
        corr_arr  = np.array(s["corr"])

        d_med = np.median(delta_arr)
        d_q1, d_q3 = np.percentile(delta_arr, [25, 75])
        c_med = np.median(corr_arr)
        c_q1, c_q3 = np.percentile(corr_arr, [25, 75])
        ci_lo, ci_hi = bootstrap_ci(corr_arr)

        print(f"{s['variant']:<25}{s['n']:>4}"
              f"  {d_med:+5.2f}° [{d_q1:+.2f},{d_q3:+.2f}]"
              f"  {c_med:+.3f} [{c_q1:+.3f},{c_q3:+.3f}]"
              f"  [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    print("=" * W)
    print("说明：median/IQR 对异常 seed 更稳健；bootstrap CI 检验均值估计精度。")
    print("      若 CI 区间不重叠 → 两变体均值有统计显著差异（α=0.05，无多重校正）。")


if __name__ == "__main__":
    main()
