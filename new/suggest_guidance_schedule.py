"""
new/suggest_guidance_schedule.py

根据 baseline 角度分布自动推荐 guidance schedule。

原理（来自 N=15 实验总结）：
    - always    : 全程 guidance。高噪声步施压破坏时序结构 → corr↓, CV↑。
                  仅适合"需要极弱推力"的任务（目标 ≈ baseline）。
    - second_half: 后半程 guidance。平衡选择，但对强任务仍有时序风险。
    - last_quarter: 最后 25% 去噪步才施压。骨盆前倾/躯干前倾的 SOTA 选择。
                  噪声低时 x0_hat 可信，corr 损失最小。

推荐逻辑（基于 signal ratio = Δ_needed / σ_baseline）：
    Δ_needed = |target - baseline_mean|（需要移动多少度）
    σ_baseline = 跨 seed 的基线角度标准差（分布宽度）

    Δ/σ < 1.0  → 目标在基线分布内，几乎不需要推 → "always" 全程微调即可
    Δ/σ 1~3    → 需要中等推力 → "second_half" 平衡
    Δ/σ > 3    → 需要强推力  → "last_quarter" 保护时序结构（推力集中在低噪声步）

    注：Δ/σ > 5 伴随 OOD 风险（check_ood_risk.py 进一步验证）

用法：
    # 从已有 comparison.npy 自动估算基线
    python -m new.suggest_guidance_schedule --posture 骨盆前倾 ./output/n15_v2_base/

    # 手动输入基线 mean 和 std
    python -m new.suggest_guidance_schedule --posture 骨盆前倾 --baseline-mean 6.0 --baseline-std 3.0
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.registry import POSTURE_REGISTRY


def _load_baseline_from_npys(npy_files, angle_fn, angle_fn_kwargs, unit):
    """从 comparison.npy 列表提取 baseline 均值角度（度）。"""
    means = []
    for npy in npy_files:
        try:
            data = np.load(npy, allow_pickle=True).item()
            xyz = data["motion_xyz"][0]
            q = torch.from_numpy(xyz).float().permute(2, 0, 1)
            with torch.no_grad():
                angle = angle_fn(q, **angle_fn_kwargs)
            mean_rad = float(angle.mean().item())
            mean_deg = mean_rad * 180.0 / math.pi if unit == "deg" else mean_rad
            means.append(mean_deg)
        except Exception as e:
            print(f"  ⚠ 跳过 {npy}: {e}")
    return means


def recommend(target_deg: float, baseline_mean: float, baseline_std: float) -> dict:
    delta = abs(target_deg - baseline_mean)
    ratio = delta / max(baseline_std, 1e-3)

    if ratio < 1.0:
        schedule = "always"
        s_v2 = 10.0
        reason = (f"Δ={delta:.1f}° < σ={baseline_std:.1f}°，目标在基线分布内。"
                  f"全程弱推力即可，无需保护时序结构。")
    elif ratio < 3.0:
        schedule = "second_half"
        s_v2 = 25.0
        reason = (f"Δ={delta:.1f}°，Δ/σ={ratio:.1f}，中等推力。"
                  f"后半程 guidance 平衡 hit 和 corr。")
    else:
        schedule = "last_quarter"
        s_v2 = 40.0
        reason = (f"Δ={delta:.1f}°，Δ/σ={ratio:.1f}，强推力任务。"
                  f"仅在低噪声步施压，保护时序结构。")

    return {
        "schedule": schedule,
        "s_v2_suggested": s_v2,
        "delta": delta,
        "ratio": ratio,
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="根据 baseline 分布自动推荐 guidance schedule。"
    )
    parser.add_argument("--posture", required=True)
    parser.add_argument("dirs", nargs="*", help="含 comparison.npy 的目录")
    parser.add_argument("--baseline-mean", type=float, default=None,
                        help="手动指定 baseline 均值（度）")
    parser.add_argument("--baseline-std", type=float, default=None,
                        help="手动指定 baseline std（度）")
    args = parser.parse_args()

    if args.posture not in POSTURE_REGISTRY:
        print(f"❌ 未知体态 '{args.posture}'，可用：{list(POSTURE_REGISTRY.keys())}")
        sys.exit(1)

    spec = POSTURE_REGISTRY[args.posture]
    target_deg = spec.target_deg
    unit = spec.unit or "deg"

    # 决定 baseline_mean/std
    if args.baseline_mean is not None and args.baseline_std is not None:
        baseline_mean = args.baseline_mean
        baseline_std  = args.baseline_std
        source = "手动输入"
    elif args.dirs:
        npy_files = []
        for d in args.dirs:
            p = Path(d)
            npy_files.extend(sorted(p.rglob("comparison.npy")))
        if not npy_files:
            print("❌ 未找到 comparison.npy，请检查路径或用 --baseline-mean/std 手动输入")
            sys.exit(1)
        means = _load_baseline_from_npys(npy_files, spec.angle_fn, spec.angle_fn_kwargs, unit)
        if len(means) < 2:
            print("❌ 有效文件不足，无法估算 baseline 分布")
            sys.exit(1)
        baseline_mean = float(np.mean(means))
        baseline_std  = float(np.std(means))
        source = f"从 {len(means)} 个 npy 文件估算"
    else:
        print("❌ 请提供 comparison.npy 目录或 --baseline-mean/--baseline-std")
        sys.exit(1)

    rec = recommend(target_deg, baseline_mean, baseline_std)

    print(f"\n体态：{args.posture}  目标：{target_deg:.1f}°")
    print(f"基线：μ={baseline_mean:+.1f}°  σ={baseline_std:.1f}°  ({source})")
    print(f"Δ/σ = {rec['ratio']:.1f}")
    print()
    print(f"推荐 schedule：{rec['schedule'].upper()}")
    print(f"推荐 V2 步长：s ≈ {rec['s_v2_suggested']:.0f}")
    print(f"理由：{rec['reason']}")
    print()
    print("快速启动命令：")
    print(f"  V2: GUIDANCE_KWARGS_JSON='{{\"s\":{rec['s_v2_suggested']:.0f}.0,"
          f"\"schedule\":\"{rec['schedule']}\",\"base_weight\":1.0}}'")
    print(f"  V6: GUIDANCE_KWARGS_JSON='{{...\"spec_schedule_override\":\"{rec['schedule']}\"}}'")


if __name__ == "__main__":
    main()
