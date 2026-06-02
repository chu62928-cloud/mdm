"""
new/check_ood_risk.py

推理前 OOD 风险预检：判断目标体态是否落在 MDM 训练分布内。

原理：
    从已有的无引导 baseline 动作（comparison.npy 里的 motion_xyz）计算
    目标角度函数的基线分布均值 μ_b 和标准差 σ_b，再计算：

        OOD_score = |target_deg - μ_b| / σ_b   （单位：标准差）

    OOD_score < 1.5  → 分布内，guidance 可有效工作
    OOD_score 1.5-3  → 边界，guidance 可能推力不足
    OOD_score > 3    → 分布外，很可能失败（类似膝超伸 190° 的情况）

    同时输出基线角度的均值 ± std，帮助理解当前分布位置。

用法：
    # 直接从已有 comparison.npy 提取 baseline（推荐：用无引导运行的结果）
    python -m new.check_ood_risk --posture 骨盆前倾 ./output/baseline_runs/*/

    # 或者只给一个目录（脚本会递归找所有 comparison.npy）
    python -m new.check_ood_risk --posture 躯干前倾 ./output/n15_v2_base/

注意：
    若 comparison.npy 来自引导运行，则 motion_xyz（非 motion_xyz_guided）
    就是无引导基线，仍然可用。
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.registry import POSTURE_REGISTRY, resolve_instruction
from posture_guidance import angle_ops
from posture_guidance.mdm_integration import make_fk_fn


def _load_angle_fn(posture: str):
    """从注册表拿 angle_fn 和 target_deg。"""
    if posture not in POSTURE_REGISTRY:
        raise KeyError(f"未知体态 '{posture}'。可用：{list(POSTURE_REGISTRY.keys())}")
    spec = POSTURE_REGISTRY[posture]
    return spec.angle_fn, spec.angle_fn_kwargs, spec.target_deg, spec.unit


def _baseline_angles_from_npy(npy_path: Path, angle_fn, angle_fn_kwargs: dict) -> np.ndarray:
    """从单个 comparison.npy 提取 baseline（无引导）角度序列的帧均值（度）。"""
    data = np.load(npy_path, allow_pickle=True).item()
    xyz = data["motion_xyz"][0]          # (J, 3, T)
    q = torch.from_numpy(xyz).float().permute(2, 0, 1)  # (T, J, 3)

    with torch.no_grad():
        angle = angle_fn(q, **angle_fn_kwargs)  # (T,) in rad (usually)

    angle_np = angle.numpy()
    return angle_np  # per-frame values in rad


def main():
    parser = argparse.ArgumentParser(
        description="推理前 OOD 风险预检：计算目标角度与基线分布的距离。"
    )
    parser.add_argument("--posture", required=True, help="体态名称，如 '骨盆前倾'")
    parser.add_argument("dirs", nargs="+", help="含 comparison.npy 的目录（支持 glob）")
    parser.add_argument("--unit", default=None, choices=["deg", "rad"],
                        help="角度单位（None 时从注册表读）")
    args = parser.parse_args()

    try:
        angle_fn, angle_fn_kwargs, target_deg, unit = _load_angle_fn(args.posture)
    except KeyError as e:
        print(e)
        sys.exit(1)

    unit = args.unit or unit or "deg"

    # 收集所有 comparison.npy
    npy_files = []
    for d in args.dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".npy":
            npy_files.append(p)
        elif p.is_dir():
            npy_files.extend(sorted(p.rglob("comparison.npy")))

    if not npy_files:
        print(f"❌ 未找到任何 comparison.npy，检查路径：{args.dirs}")
        sys.exit(1)

    print(f"\n体态：{args.posture}   目标：{target_deg:.1f}°   找到 {len(npy_files)} 个 npy 文件")
    print("─" * 60)

    all_mean_angles = []  # per-file mean angle (in degrees)
    for npy in npy_files:
        try:
            angles_rad = _baseline_angles_from_npy(npy, angle_fn, angle_fn_kwargs)
            mean_rad = float(np.mean(angles_rad))
            mean_deg = mean_rad * 180.0 / math.pi if unit == "deg" else mean_rad
            all_mean_angles.append(mean_deg)
        except Exception as e:
            print(f"  ⚠ 跳过 {npy}: {e}")

    if len(all_mean_angles) < 2:
        print("❌ 有效文件不足，无法计算分布统计。")
        sys.exit(1)

    arr = np.array(all_mean_angles)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    ood_score = abs(target_deg - mu) / max(sigma, 1e-3)

    print(f"基线角度分布：μ = {mu:+.2f}°  σ = {sigma:.2f}°  (N={len(arr)} files)")
    print(f"目标角度：{target_deg:.1f}°")
    print(f"OOD score = |{target_deg:.1f} - ({mu:+.2f})| / {sigma:.2f} = {ood_score:.2f} σ")
    print()

    if ood_score < 1.5:
        verdict = "✅ 分布内  — guidance 可有效工作"
    elif ood_score < 3.0:
        verdict = "⚠ 边界区域 — guidance 可能推力不足，建议加大 s 或 Kp"
    else:
        verdict = "❌ 分布外  — 高 OOD 风险，guidance 极可能失败（类似膝超伸 190°）"

    print(f"判定：{verdict}")
    print()
    print("参考：OOD < 1.5σ = 分布内 | 1.5-3σ = 边界 | > 3σ = 分布外")
    print("      骨盆前倾(20°→5°基线, σ≈3°): score≈5 但在分布边界→仍可工作（快走包含前倾）")
    print("      膝超伸(190°→175°基线, σ≈5°): score≈3 且零密度→失败")


if __name__ == "__main__":
    main()
