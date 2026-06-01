"""
new/analyze_trunk_segments.py

躯干前倾"机制诊断"：把躯干前倾分解为下段（腰椎，hip→spine2）和上段（胸椎，spine2→肩）
两个分段角，度量 guidance 实现前倾的方式是"刚体整体前倾"还是"上背胸椎补偿"。

胸椎补偿指数 (Thoracic Compensation Index, TCI):
    TCI = Δ_upper / (Δ_lower + Δ_upper)
    - TCI ≈ 0.5  → 上下段同等参与 → 接近刚体整体前倾（自然、协调）
    - TCI → 1.0  → 几乎全靠上背圆弓 → 胸椎补偿捷径（不自然）

用途：检验 V6 (manifold_project=on) 是否比 V2 更倾向"协调多关节"的刚体前倾，
即流形投影是否把动作约束在自然行走流形上、避开胸椎补偿捷径。

用法：
    python -m new.analyze_trunk_segments ./output/cross_trunk_lean_*
聚合每个 variant 目录下所有 seed 的 comparison.npy。
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance import angle_ops


def _deg(a):
    return (a * 180.0 / math.pi).numpy()


def segment_metrics(npy_path):
    """返回单份 comparison.npy 的分段诊断指标。"""
    data = np.load(npy_path, allow_pickle=True).item()
    q_b = torch.from_numpy(data["motion_xyz"][0]).float().permute(2, 0, 1)
    q_g = torch.from_numpy(data["motion_xyz_guided"][0]).float().permute(2, 0, 1)

    # 三个角度序列（度）：总前倾、下段、上段
    tot_b = _deg(angle_ops.trunk_forward_lean(q_b))
    tot_g = _deg(angle_ops.trunk_forward_lean(q_g))
    low_b = _deg(angle_ops.trunk_lean_lower(q_b))
    low_g = _deg(angle_ops.trunk_lean_lower(q_g))
    upp_b = _deg(angle_ops.trunk_lean_upper(q_b))
    upp_g = _deg(angle_ops.trunk_lean_upper(q_g))

    d_tot = float(tot_g.mean() - tot_b.mean())
    d_low = float(low_g.mean() - low_b.mean())
    d_upp = float(upp_g.mean() - upp_b.mean())

    # 胸椎补偿指数：上段变化占上下段总变化的比例
    denom = abs(d_low) + abs(d_upp)
    tci = abs(d_upp) / denom if denom > 1e-6 else float("nan")

    return {"d_tot": d_tot, "d_low": d_low, "d_upp": d_upp, "tci": tci}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seedtest_dirs", nargs="+")
    args = parser.parse_args()

    print("\n" + "=" * 92)
    print("  躯干前倾机制诊断 — 上下段分解 (Δ in degrees)")
    print("=" * 92)
    print(f"{'variant':<42}{'N':>3}{'Δtot':>9}{'Δlower':>9}{'Δupper':>9}{'TCI':>8}")
    print("-" * 92)

    for dir_path in args.seedtest_dirs:
        d = Path(dir_path)
        if not d.exists():
            print(f"skip: {d} 不存在")
            continue

        rows = []
        for seed_dir in sorted(d.iterdir()):
            if not seed_dir.is_dir():
                continue
            npy = seed_dir / "comparison.npy"
            if not npy.exists():
                continue
            try:
                rows.append(segment_metrics(npy))
            except Exception as e:
                print(f"  ERR {seed_dir.name}: {e}")

        if not rows:
            continue

        n = len(rows)
        d_tot = np.mean([r["d_tot"] for r in rows])
        d_low = np.mean([r["d_low"] for r in rows])
        d_upp = np.mean([r["d_upp"] for r in rows])
        tci = np.nanmean([r["tci"] for r in rows])

        name = d.name
        print(f"{name:<42}{n:>3}{d_tot:>+9.2f}{d_low:>+9.2f}{d_upp:>+9.2f}{tci:>8.2f}")

    print("=" * 92)
    print("解读：")
    print("  TCI → 0.5  上下段同等参与 → 刚体整体前倾（协调、自然）")
    print("  TCI → 1.0  几乎全靠上背 → 胸椎补偿捷径（不自然）")
    print("  若 V6(manifold=on) 的 TCI 明显低于 V2 → 流形投影确实抑制了胸椎补偿捷径")
    print("=" * 92)


if __name__ == "__main__":
    main()
