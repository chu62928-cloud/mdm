#!/usr/bin/env python3
"""
new/decompose_pelvis_vs_trunk.py

确认 pelvis_tilt_angle 到底测的是"骨盆自身前旋"还是"下段躯干前倾"。

核心假设：pelvis_tilt_angle(hip_center→spine1) 与 trunk_lean 家族同构，
引导推它 → 实际产生的是躯干前弯，而非真骨盆前旋。

做什么（纯几何，不用 proxy，秒级）：
对 APT 段(+15°) 和 PPT 段(-15°) 的 baseline / guided，逐帧算下列角并取均值：
  1. pelvis_tilt_angle      —— 你优化的目标（hip_center→spine1）
  2. trunk_forward_lean     —— 整躯干前倾（hip_center→肩中点）
  3. trunk_lean_lower       —— 下段躯干（hip_center→spine2）
  4. trunk_lean_upper       —— 上段躯干（spine2→肩中点）
  5. pelvis_segment_tilt    —— 真·骨盆段前旋（pelvis,Lhip,Rhip 三点平面法向矢状角）
  6. hip_flexion            —— 髋屈角（大腿 vs 躯干轴），躯干在髋处前弯则增大

判读：
  若 guided 的 Δpelvis_tilt ≈ Δtrunk_lean（强相关），而 Δpelvis_segment_tilt ≈ 0
    → 实锤：你生成的是"躯干前倾"，不是"骨盆前旋"。proxy 无辜。
  若 Δpelvis_segment_tilt 与 Δpelvis_tilt 同步变大、而 trunk_lean 没动
    → 才是真骨盆前旋，proxy 反向(H_A)才需要认真对待。

用法：
    python new/decompose_pelvis_vs_trunk.py \
        --apt ./output/.../apt_seed42/comparison.npy \
        --ppt ./output/.../ppt_seed42/comparison.npy
    # 或传目录（自动找 comparison.npy）
    python new/decompose_pelvis_vs_trunk.py --apt ./output/apt_dir --ppt ./output/ppt_dir
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from posture_guidance.angle_ops import (              # noqa: E402
    pelvis_tilt_angle, trunk_forward_lean,
    trunk_lean_lower, trunk_lean_upper,
)
from posture_guidance.joint_indices import get_joint_idx  # noqa: E402

EPS = 1e-7
R2D = 180.0 / np.pi


# ---------------------------------------------------------------------------
# 两个补充角（angle_ops 里没有，本地实现）
# ---------------------------------------------------------------------------
def pelvis_segment_tilt(q: torch.Tensor) -> torch.Tensor:
    """
    真·骨盆段前旋角（弧度）。
    用 pelvis(根) / left_hip / right_hip 三点构成骨盆平面，取其法向在矢状面的倾角。
    与脊柱朝向解耦 —— 这才是临床意义上的骨盆前后倾。
    正负仅表方向，诊断只看 Δ 大小。
    """
    pelvis    = q[..., get_joint_idx("pelvis"),    :]
    left_hip  = q[..., get_joint_idx("left_hip"),  :]
    right_hip = q[..., get_joint_idx("right_hip"), :]

    v1 = left_hip  - pelvis
    v2 = right_hip - pelvis
    n = torch.cross(v1, v2, dim=-1)              # 骨盆平面法向
    n = F.normalize(n, dim=-1, eps=EPS)

    # 矢状面内法向的倾角：atan2(前后 z, 上下 y)。y 不 clamp，保留前后倾方向。
    return torch.atan2(n[..., 2], n[..., 1])


def hip_flexion(q: torch.Tensor) -> torch.Tensor:
    """
    髋屈角（左右平均，弧度）。大腿向量(hip→knee) 与躯干轴(hip_center→spine2) 的夹角。
    躯干在髋关节处前弯 → 髋屈角增大。直立行走 baseline 提供参考。
    """
    def one_side(side):
        hip   = q[..., get_joint_idx(f"{side}_hip"),  :]
        knee  = q[..., get_joint_idx(f"{side}_knee"), :]
        lh    = q[..., get_joint_idx("left_hip"),  :]
        rh    = q[..., get_joint_idx("right_hip"), :]
        spine2 = q[..., get_joint_idx("spine2"), :]
        hip_center = (lh + rh) / 2.0

        thigh = F.normalize(knee - hip, dim=-1, eps=EPS)
        trunk = F.normalize(spine2 - hip_center, dim=-1, eps=EPS)
        cos = (thigh * trunk).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.arccos(cos)

    return 0.5 * (one_side("left") + one_side("right"))


ANGLE_FNS = {
    "pelvis_tilt(目标)":   pelvis_tilt_angle,
    "trunk_forward_lean":  trunk_forward_lean,
    "trunk_lean_lower":    trunk_lean_lower,
    "trunk_lean_upper":    trunk_lean_upper,
    "pelvis_segment(真)":  pelvis_segment_tilt,
    "hip_flexion":         hip_flexion,
}


# ---------------------------------------------------------------------------
def load_xyz(path, key):
    if os.path.isdir(path):
        path = os.path.join(path, "comparison.npy")
    data = np.load(path, allow_pickle=True).item()
    if key not in data:
        raise KeyError(f"{path}: 缺少 key '{key}'. 现有: {list(data.keys())}")
    arr = data[key]                        # (B, J, 3, T)
    q = torch.from_numpy(arr[0]).float().permute(2, 0, 1)   # (T, J, 3)
    return q


def mean_deg(fn, q):
    with torch.no_grad():
        a = fn(q)                          # (T,)
    return float(a.mean()) * R2D


def analyze(label, npy_path):
    q_base   = load_xyz(npy_path, "motion_xyz")
    q_guided = load_xyz(npy_path, "motion_xyz_guided")

    print(f"\n{'='*78}\n{label}  ({npy_path})\n{'='*78}")
    print(f"{'angle':<22}{'baseline':>12}{'guided':>12}{'Δ(guided-base)':>18}")
    print("-" * 78)
    rows = {}
    for name, fn in ANGLE_FNS.items():
        b = mean_deg(fn, q_base)
        g = mean_deg(fn, q_guided)
        d = g - b
        rows[name] = (b, g, d)
        print(f"{name:<22}{b:>12.2f}{g:>12.2f}{d:>+18.2f}")
    return rows


def verdict(apt_rows, ppt_rows):
    print(f"\n{'='*78}\n判读\n{'='*78}")

    d_pelvis_tilt_apt = apt_rows["pelvis_tilt(目标)"][2]
    d_trunk_apt       = apt_rows["trunk_forward_lean"][2]
    d_trunk_low_apt   = apt_rows["trunk_lean_lower"][2]
    d_pelvis_seg_apt  = apt_rows["pelvis_segment(真)"][2]

    print(f"APT 段：")
    print(f"  目标指标 Δpelvis_tilt   = {d_pelvis_tilt_apt:+.2f}°")
    print(f"  躯干前倾 Δtrunk_lean    = {d_trunk_apt:+.2f}°  (下段 {d_trunk_low_apt:+.2f}°)")
    print(f"  真骨盆段 Δpelvis_segment= {d_pelvis_seg_apt:+.2f}°")

    # 相关判据：目标变化里有多少由躯干前倾解释
    if abs(d_pelvis_tilt_apt) > 1e-3:
        trunk_share = abs(d_trunk_low_apt) / (abs(d_pelvis_tilt_apt) + EPS)
    else:
        trunk_share = 0.0
    pelvis_share = abs(d_pelvis_seg_apt) / (abs(d_pelvis_tilt_apt) + EPS)

    print(f"\n  → 下段躯干前倾解释了目标变化的 ~{trunk_share*100:.0f}%")
    print(f"  → 真骨盆段转角仅占目标变化的 ~{pelvis_share*100:.0f}%")

    if trunk_share > 0.6 and pelvis_share < 0.4:
        print("\n  结论：✅ 实锤——pelvis_tilt 指标主要由【躯干前倾】驱动，")
        print("        真骨盆段几乎没转。你生成的是躯干前倾，不是骨盆前旋。")
        print("        → proxy 报出前弯躯干的伸肌负荷是【正确】的，没学反。")
    elif pelvis_share > 0.6:
        print("\n  结论：⚠️ 真骨盆段确实在转，躯干前倾不是主因。")
        print("        proxy 反向(H_A) 需要认真对待。")
    else:
        print("\n  结论：混合——躯干前倾与骨盆前旋都有贡献，需进一步看可视化。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apt", required=True, help="APT(+15°) 的 comparison.npy 或其目录")
    ap.add_argument("--ppt", required=True, help="PPT(-15°) 的 comparison.npy 或其目录")
    args = ap.parse_args()

    apt_rows = analyze("APT 段 (+15°)", args.apt)
    ppt_rows = analyze("PPT 段 (-15°)", args.ppt)
    verdict(apt_rows, ppt_rows)

    print(f"\n{'='*78}")
    print("提示：若 APT/PPT 的 pelvis_tilt 与 trunk_lean 几乎逐项同号同幅，")
    print("     即可在报告里直接说明指标混淆，并改用 pelvis_segment(真) 重新定义目标。")


if __name__ == "__main__":
    main()
