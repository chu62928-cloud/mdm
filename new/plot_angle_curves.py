"""
new/plot_angle_curves.py

把所有 variant 的角度时间曲线画在一张图上，肉眼判断哪个 variant
在保留运动结构的同时达到目标角度。

设计哲学：
    - baseline 曲线只画一次（所有 variant 共享，从 v1_baseline 提取）
    - 每个 variant 一种颜色 + 实线
    - 目标角度画水平虚线
    - 把每个 variant 的统计指标（Δ, hit_rate, corr）打印在图例里
    - 同时生成单 variant 大图（每张图一个 variant，更清楚）

使用：
    python -m new.plot_angle_curves ./output/calibrated_walking_apt_seed42
    # 输出 angle_curves_all.png + angle_curves_<variant>.png
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance import angle_ops


# 目标角度配置（与 evaluate_ablation.py 保持一致）
POSTURE_TARGETS = {
    "骨盆前倾":  ("pelvis_tilt_angle",       20.0, 2.0),
    "膝超伸":    ("signed_knee_angle_both",  185.0, 3.0),
    "膝超伸_左": ("signed_knee_angle_left",  185.0, 3.0),
    "膝超伸_右": ("signed_knee_angle_right", 185.0, 3.0),
    "驼背":      ("spine_posterior_bulge",   0.08, 0.02),
}


def get_angle_fn(name):
    if name == "signed_knee_angle_both":
        return lambda q: (angle_ops.signed_knee_angle(q, side="left") +
                          angle_ops.signed_knee_angle(q, side="right")) / 2.0
    if name == "signed_knee_angle_left":
        return lambda q: angle_ops.signed_knee_angle(q, side="left")
    if name == "signed_knee_angle_right":
        return lambda q: angle_ops.signed_knee_angle(q, side="right")
    return getattr(angle_ops, name)


def load_angles(npy_path, posture):
    """加载一份 comparison.npy，返回 baseline + guided 的角度序列（度）。"""
    data = np.load(npy_path, allow_pickle=True).item()

    fn_name, target_deg, tol_deg = POSTURE_TARGETS[posture]
    angle_fn = get_angle_fn(fn_name)
    use_radians = "spine_posterior" not in fn_name

    q_b = torch.from_numpy(data["motion_xyz"][0]).float().permute(2, 0, 1)
    q_g = torch.from_numpy(data["motion_xyz_guided"][0]).float().permute(2, 0, 1)

    a_b = angle_fn(q_b)
    a_g = angle_fn(q_g)

    if use_radians:
        a_b_deg = (a_b * 180.0 / math.pi).numpy()
        a_g_deg = (a_g * 180.0 / math.pi).numpy()
    else:
        a_b_deg = a_b.numpy()
        a_g_deg = a_g.numpy()

    return a_b_deg, a_g_deg, target_deg, tol_deg


def compute_summary(a_b, a_g, target, tol):
    """返回 (Δ, hit_rate, corr) 三个汇总指标。"""
    delta = float(a_g.mean() - a_b.mean())
    hit = float((a_g >= target - tol).mean())
    if a_b.std() > 1e-6 and a_g.std() > 1e-6:
        corr = float(pearsonr(a_b, a_g)[0])
    else:
        corr = 0.0
    return delta, hit, corr


# ----------------------------------------------------------------
# 绘图主流程
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_dir", type=str)
    parser.add_argument("--posture", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.ablation_dir)
    if not base.exists():
        print(f"ERROR: {base} 不存在")
        sys.exit(1)

    # 收集所有 variant 数据
    runs = []  # list of (name, a_b, a_g, target, tol)
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        npy = run_dir / "comparison.npy"
        if not npy.exists():
            continue

        data = np.load(npy, allow_pickle=True).item()
        posture = args.posture or data.get("posture_instructions", "骨盆前倾")
        if isinstance(posture, list):
            posture = posture[0]

        if posture not in POSTURE_TARGETS:
            print(f"  skip {run_dir.name}: unknown posture {posture}")
            continue

        a_b, a_g, target, tol = load_angles(npy, posture)
        runs.append((run_dir.name, a_b, a_g, target, tol))

    if not runs:
        print("No runs to plot.")
        return

    posture = args.posture or "骨盆前倾"
    target = runs[0][3]
    tol = runs[0][4]

    print(f"\nLoaded {len(runs)} runs for posture '{posture}'")
    print(f"Target: {target}° ± {tol}°")

    # ============================================================
    # 大图：所有 variant 叠在一起
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 7))

    # baseline 用粗黑虚线（所有 variant 的 baseline 应该一致，画第一个就好）
    a_b_ref = runs[0][1]
    ax.plot(a_b_ref, color="black", linestyle="--", linewidth=2.5,
            label=f"baseline (mean={a_b_ref.mean():.1f}°)", zorder=10)

    # 目标线
    ax.axhline(y=target, color="red", linestyle=":", linewidth=2,
               label=f"target = {target}°", zorder=9)
    ax.axhspan(target - tol, target + tol, color="red", alpha=0.08, zorder=0)

    # 每个 variant 用不同颜色
    cmap = plt.cm.tab20
    for i, (name, a_b, a_g, _, _) in enumerate(runs):
        delta, hit, corr = compute_summary(a_b, a_g, target, tol)
        color = cmap(i / max(len(runs) - 1, 1))
        label = (f"{name:<22} Δ={delta:+5.1f}° "
                 f"hit={hit*100:4.1f}% corr={corr:+.2f}")
        ax.plot(a_g, color=color, linewidth=1.4, alpha=0.85, label=label)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel(f"{posture} angle (°)", fontsize=12)
    ax.set_title(f"Angle curves comparison — {posture} (target={target}°)",
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    # 图例放在外面（variant 多了挤不下）
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_all = base / "angle_curves_all.png"
    plt.savefig(out_all, dpi=140, bbox_inches="tight")
    print(f"\n💾 Saved: {out_all}")
    plt.close()

    # ============================================================
    # 每个 variant 一张图（更清楚地看个体行为）
    # ============================================================
    # 按 variant 前缀分组（v1_*, v2_dps_*, v2b_*, v3_*, v4_*, v5_*）
    groups = {}
    for run in runs:
        name = run[0]
        if name.startswith("v1"):
            key = "V1 — SGD on mu_t"
        elif name.startswith("v2_dps"):
            key = "V2 — DPS (gradient through MDM)"
        elif name.startswith("v2b") or name.startswith("v2_x0"):
            key = "V2b — direct edit x0_hat"
        elif name.startswith("v3"):
            key = "V3 — x0_direct (short loop)"
        elif name.startswith("v4"):
            key = "V4 — OmniControl (dynamic K)"
        elif name.startswith("v5"):
            key = "V5 — LGD (MC-smoothed DPS)"
        else:
            key = "other"
        groups.setdefault(key, []).append(run)

    n_groups = len(groups)
    n_cols = 2
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, (group_name, group_runs) in enumerate(groups.items()):
        ax = axes_flat[ax_idx]

        # baseline
        a_b_ref = group_runs[0][1]
        ax.plot(a_b_ref, color="black", linestyle="--", linewidth=2,
                label=f"baseline ({a_b_ref.mean():.1f}°)", zorder=10)

        # target
        ax.axhline(y=target, color="red", linestyle=":", linewidth=1.5,
                   label=f"target {target}°", zorder=9)
        ax.axhspan(target - tol, target + tol, color="red", alpha=0.08)

        # 该 group 的所有 variant
        sub_cmap = plt.cm.viridis
        for j, (name, _, a_g, _, _) in enumerate(group_runs):
            delta, hit, corr = compute_summary(a_b_ref, a_g, target, tol)
            color = sub_cmap(j / max(len(group_runs) - 1, 1))

            # 提取参数标签（去掉 variant 前缀）
            short_label = name.replace("v1_", "").replace("v2_dps_", "")\
                             .replace("v2b_x0_edit_", "").replace("v3_x0_direct_", "")\
                             .replace("v4_omni_", "").replace("v5_lgd_", "")
            label = f"{short_label} | Δ={delta:+.1f}° hit={hit*100:.0f}% r={corr:+.2f}"

            ax.plot(a_g, color=color, linewidth=1.5, alpha=0.85, label=label)

        ax.set_xlabel("Frame")
        ax.set_ylabel(f"{posture} (°)")
        ax.set_title(group_name, fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # 隐藏多余 subplot
    for k in range(len(groups), len(axes_flat)):
        axes_flat[k].axis("off")

    plt.tight_layout()
    out_grp = base / "angle_curves_per_variant.png"
    plt.savefig(out_grp, dpi=140, bbox_inches="tight")
    print(f"💾 Saved: {out_grp}")
    plt.close()

    # ============================================================
    # 打印汇总表（终端）
    # ============================================================
    print("\n" + "=" * 90)
    print(f"  Summary — {posture} (target={target}°)")
    print("=" * 90)
    print(f"{'variant':<28}{'Δ (°)':>10}{'hit %':>10}{'corr':>10}{'shape':>30}")
    print("-" * 90)

    for name, a_b, a_g, _, _ in runs:
        delta, hit, corr = compute_summary(a_b, a_g, target, tol)

        # 形状判断（启发式）
        if hit > 0.9 and abs(corr) < 0.15:
            shape = "❌ 塌平（静态姿势）"
        elif hit > 0.4 and corr > 0.3:
            shape = "✅ 平移+保结构"
        elif hit < 0.05:
            shape = "⚠ 推力不足"
        elif corr < 0:
            shape = "⚠ 时间结构破坏"
        else:
            shape = "中间状态"

        print(f"{name:<28}{delta:+8.2f}  {hit*100:7.1f}  {corr:+8.3f}  {shape}")

    print("=" * 90)
    print("\n判读规则：")
    print("  ✅ 平移+保结构  : hit_rate > 40% 且 corr > 0.3，是最佳目标")
    print("  ❌ 塌平         : hit_rate > 90% 但 corr 接近 0，所有帧被强行拉到目标")
    print("  ⚠ 推力不足      : hit_rate < 5%，guidance 没起作用")
    print("  ⚠ 时间结构破坏   : corr < 0，guided 的角度时间模式被反向")
    print()
    print(f"建议：选择 hit_rate ∈ [40%, 80%] 且 corr 最高的 variant 作为最佳。")


if __name__ == "__main__":
    main()