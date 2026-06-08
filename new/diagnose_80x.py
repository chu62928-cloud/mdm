#!/usr/bin/env python3
"""
new/diagnose_80x.py  —  H_A vs H_B 单脚本诊断

目标：区分
  H_A: proxy 学到了反向映射（APT 肌肉 ↔ PPT 几何）
  H_B: 80x 是拮抗比值分母被压向 0 的数值伪影

四个检查，全部离线（复用已有 comparison.npy，不重新生成）：

  CHECK 1 — 拮抗分母崩塌
    直接打印 muscle-mode guided 运动里每个拮抗对的 mean_under。
    mean_under < 1e-3 → 比值项 r = over/under → ∞，80x 是伪影（H_B 成立）。

  CHECK 2 — 模板交叉评估（核心）
    对 joint-mode（几何 APT, +15°）和 muscle-mode（几何 PPT, -11°）的运动，
    分别用 APT 和 PPT 模板算 clinical loss：
      H_A 预测：joint 运动对 PPT 模板得分更高（proxy 认为 APT 几何 = PPT 肌肉）
      H_B 预测：muscle 运动对 APT 模板得分高（分母被压，APT 比值炸），对 PPT 模板不一定高

  CHECK 3 — 各分量贡献分解
    看 muscle-mode 的 80x 主要来自哪个分量（antagonist/chain/synergy/stabilizer）。
    若 antagonist 独占 → H_B。若各分量均等 → 更倾向 H_A。

  CHECK 4 — 无 relu 的原始比值
    绕过 relu 阈值，直接算 r_current = mean_over / mean_under，
    与正常范围（1.0–2.0）对比，看是否异常。

用法：
    python new/diagnose_80x.py <output_root> \
        --muscle_ckpt motion2muscle/checkpoints/.../net_best_loss.pth \
        --joint_dir   joint   \   # output_root 下 joint 模式的子目录名
        --muscle_dir  muscle      # output_root 下 muscle 模式的子目录名
"""
import argparse
import os
import sys
import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_ASSETS = os.path.join(_ROOT, "motion2muscle")
for p in (_ROOT, _ASSETS):
    if p not in sys.path:
        sys.path.insert(0, p)

from muscle_rollup import get_indices
from posture_loss import POSTURE_PRIORS
from posture_loss_torch import build_group_index, compute_posture_loss_torch

EPS = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_hml(npy_path, key):
    data = np.load(npy_path, allow_pickle=True).item()
    if key not in data:
        raise KeyError(f"{npy_path}: missing key '{key}'. Keys: {list(data.keys())}")
    arr = data[key]   # (B, T, 263) or (N, T, 263)
    return arr


def get_activations(mg, hml_np, device):
    x = torch.from_numpy(hml_np).float().to(device)
    with torch.no_grad():
        a = mg._activations(x)        # (B, T, 402)
    return a


def group_temporal_mean(a, idx):
    """a: (B,T,402) tensor; idx: list[int] -> scalar float"""
    if not idx:
        return None
    return float(a[..., idx].mean())


# ---------------------------------------------------------------------------
# Check 1: antagonist denominator collapse
# ---------------------------------------------------------------------------
def check1_denominator(a_guided, mg, posture_name):
    print("\n" + "=" * 70)
    print("CHECK 1 — 拮抗分母崩塌（H_B 核心证据）")
    print(f"{'pair (over / under)':<48} {'mean_under':>12}  {'mean_over':>12}  {'r_current':>12}")
    print("-" * 70)
    cfg = POSTURE_PRIORS[posture_name]
    bil = cfg.get("bilateral", True)
    sides = ["_R", "_L"] if bil else [""]
    any_collapsed = False
    for over, under, delta in cfg.get("antagonist_imbalances", []):
        for s in sides:
            name_o, name_u = over + s, under + s
            idx_o = get_indices(name_o, mg.mint_cols)
            idx_u = get_indices(name_u, mg.mint_cols)
            if not idx_o or not idx_u:
                continue
            mo = group_temporal_mean(a_guided, idx_o)
            mu = group_temporal_mean(a_guided, idx_u)
            r  = mo / (mu + EPS)
            collapsed = mu < 1e-3
            tag = " ← COLLAPSED!" if collapsed else ""
            print(f"  {name_o:20s} / {name_u:20s}   {mu:12.4e}  {mo:12.4e}  {r:12.2f}{tag}")
            if collapsed:
                any_collapsed = True
    if any_collapsed:
        print("\n  결론: mean_under < 1e-3 → 分母崩塌，H_B 成立（80x 是数值伪影）")
    else:
        print("\n  결론: 分母未崩塌，需要 CHECK 2 进一步区分")


# ---------------------------------------------------------------------------
# Check 2: cross-template evaluation
# ---------------------------------------------------------------------------
def check2_cross_template(mg, a_joint, a_muscle, a_base, joint_ref_acts, muscle_ref_acts, device):
    """
    对 joint-mode 激活 和 muscle-mode 激活，分别用 APT / PPT 模板算 clinical loss。
    参考都用 baseline 激活（build_reference 已在外部完成两次，分别存进 joint/muscle ref_acts）。
    这里统一用 joint_ref_acts 作为参考基准（因为 baseline 两次是同一份）。
    """
    gi_apt = build_group_index(mg.mint_cols, "anterior_pelvic_tilt",  device=device)
    gi_ppt = build_group_index(mg.mint_cols, "posterior_pelvic_tilt", device=device)

    def score(a, gi, posture, ref):
        total, comp = compute_posture_loss_torch(
            a, gi, posture, ref, return_components=True)
        return float(total), comp

    loss_joint_apt, c_ja = score(a_joint,  gi_apt, "anterior_pelvic_tilt",  joint_ref_acts)
    loss_joint_ppt, c_jp = score(a_joint,  gi_ppt, "posterior_pelvic_tilt", joint_ref_acts)
    loss_musc_apt,  c_ma = score(a_muscle, gi_apt, "anterior_pelvic_tilt",  muscle_ref_acts)
    loss_musc_ppt,  c_mp = score(a_muscle, gi_ppt, "posterior_pelvic_tilt", muscle_ref_acts)

    print("\n" + "=" * 70)
    print("CHECK 2 — 模板交叉评估（H_A vs H_B 判决）")
    print(f"{'运动类型':<20} {'APT 模板 loss':>16} {'PPT 模板 loss':>16}  {'结论'}")
    print("-" * 70)

    # joint mode: geometric APT
    pref_j = "APT" if loss_joint_apt > loss_joint_ppt else "PPT"
    tag_j  = "✓ 正常（H_A=FALSE）" if pref_j == "APT" else "！H_A 候选"
    print(f"  joint-mode (+15°, 几何APT)    {loss_joint_apt:>16.4e} {loss_joint_ppt:>16.4e}  {pref_j}更高 → {tag_j}")

    # muscle mode: geometric PPT
    pref_m = "APT" if loss_musc_apt > loss_musc_ppt else "PPT"
    if pref_m == "APT":
        tag_m = "H_B 成立（分母伪影使 APT 虚高）"
    else:
        tag_m = "H_A 支持（proxy 确实更偏 PPT）"
    print(f"  muscle-mode (−11°, 几何PPT)   {loss_musc_apt:>16.4e} {loss_musc_ppt:>16.4e}  {pref_m}更高 → {tag_m}")

    print("\n  分量明细（muscle-mode, APT 模板）:")
    for k in ("antagonist", "chain", "synergy", "stabilizer"):
        frac = c_ma[k] / (float(sum(c_ma[k] for k in c_ma if isinstance(c_ma[k], float))) + EPS)
        print(f"    {k:<14s} {c_ma[k]:.4e}  ({frac*100:.1f}%)")


# ---------------------------------------------------------------------------
# Check 3: component breakdown (already in evaluate_muscle_space.py, re-print)
# ---------------------------------------------------------------------------
def check3_component_share(comp_guided_apt):
    print("\n" + "=" * 70)
    print("CHECK 3 — 分量占比（antagonist 独占 → H_B；均匀分布 → H_A）")
    total = sum(v for k, v in comp_guided_apt.items() if isinstance(v, float))
    for k in ("antagonist", "chain", "synergy", "stabilizer"):
        v = comp_guided_apt.get(k, 0.0)
        bar = "█" * int(40 * v / (total + EPS))
        print(f"  {k:<14s} {v:.4e}  {bar}")
    dom = max(("antagonist","chain","synergy","stabilizer"), key=lambda k: comp_guided_apt.get(k,0))
    if dom == "antagonist" and comp_guided_apt["antagonist"] / (total + EPS) > 0.7:
        print("  결론: antagonist 占 >70% → H_B 成立")
    else:
        print(f"  결론: {dom} 主导但未超70% → 需结合 CHECK 1/2")


# ---------------------------------------------------------------------------
# Check 4: raw ratio (no relu)
# ---------------------------------------------------------------------------
def check4_raw_ratio(a_base, a_guided, mg, posture_name):
    print("\n" + "=" * 70)
    print("CHECK 4 — 原始比值（无 relu 阈值，感受拮抗失衡实际幅度）")
    print(f"{'pair':<48} {'r_base':>10}  {'r_guided':>10}  {'倍增':>8}")
    print("-" * 70)
    cfg = POSTURE_PRIORS[posture_name]
    bil = cfg.get("bilateral", True)
    sides = ["_R", "_L"] if bil else [""]
    for over, under, _ in cfg.get("antagonist_imbalances", []):
        for s in sides:
            name_o, name_u = over + s, under + s
            idx_o = get_indices(name_o, mg.mint_cols)
            idx_u = get_indices(name_u, mg.mint_cols)
            if not idx_o or not idx_u:
                continue
            mo_b = group_temporal_mean(a_base,   idx_o)
            mu_b = group_temporal_mean(a_base,   idx_u)
            mo_g = group_temporal_mean(a_guided, idx_o)
            mu_g = group_temporal_mean(a_guided, idx_u)
            r_b  = mo_b / (mu_b + EPS)
            r_g  = mo_g / (mu_g + EPS)
            print(f"  {name_o:20s}/{name_u:20s}  {r_b:10.3f}  {r_g:10.3f}  {r_g/r_b+EPS:8.1f}x")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("--muscle_ckpt",     required=True)
    ap.add_argument("--joint_dir",       default="joint")
    ap.add_argument("--muscle_dir",      default="muscle")
    ap.add_argument("--posture",         default="anterior_pelvic_tilt")
    ap.add_argument("--muscle_assets",   default=_ASSETS)
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--same_norm",       action="store_true", default=True)
    ap.add_argument("--diff_norm",       dest="same_norm", action="store_false")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[diagnose_80x] device={device}, posture={args.posture}")

    # ---- 加载 MuscleGuidance ----
    from muscle_guidance_mdm import build_muscle_guidance
    mg = build_muscle_guidance(
        ckpt_path=args.muscle_ckpt,
        posture_name=args.posture,
        assets_dir=args.muscle_assets,
        same_normalization=args.same_norm,
        device=device,
    )

    # ---- 加载两个模式的 hml 数据 ----
    def load_npy(subdir, key):
        path = os.path.join(args.output_root, subdir, "comparison.npy")
        if not os.path.exists(path):
            # 兼容：直接在 output_root/<subdir>.npy
            path2 = os.path.join(args.output_root, subdir + ".npy")
            if os.path.exists(path2):
                path = path2
            else:
                raise FileNotFoundError(f"找不到 {path} 或 {path2}")
        return load_hml(path, key)

    hml_base_j   = load_npy(args.joint_dir,  "motion_hml_tj")           # (B,T,263)
    hml_guided_j = load_npy(args.joint_dir,  "motion_hml_tj_guided")
    hml_base_m   = load_npy(args.muscle_dir, "motion_hml_tj")
    hml_guided_m = load_npy(args.muscle_dir, "motion_hml_tj_guided")

    print(f"[data] joint  baseline {hml_base_j.shape}  guided {hml_guided_j.shape}")
    print(f"[data] muscle baseline {hml_base_m.shape}  guided {hml_guided_m.shape}")

    # ---- 激活计算 ----
    a_base_j   = get_activations(mg, hml_base_j,   device)
    a_guided_j = get_activations(mg, hml_guided_j, device)
    a_base_m   = get_activations(mg, hml_base_m,   device)
    a_guided_m = get_activations(mg, hml_guided_m, device)

    # ---- 建立各自参考（用 baseline 激活）----
    from posture_loss import build_reference_from_activations
    ref_j = build_reference_from_activations(a_base_j.cpu().numpy(), mg.mint_cols)
    ref_m = build_reference_from_activations(a_base_m.cpu().numpy(), mg.mint_cols)

    # ---- 四项检查 ----
    check1_denominator(a_guided_m, mg, args.posture)

    check2_cross_template(mg, a_guided_j, a_guided_m, a_base_j, ref_j, ref_m, device)

    # 拿 muscle-mode 的 APT 分量明细
    from posture_loss_torch import build_group_index
    gi_apt = build_group_index(mg.mint_cols, args.posture, device=device)
    _, comp_m_apt = compute_posture_loss_torch(
        a_guided_m, gi_apt, args.posture, ref_m, return_components=True)
    check3_component_share(comp_m_apt)

    check4_raw_ratio(a_base_m, a_guided_m, mg, args.posture)

    print("\n" + "=" * 70)
    print("DONE. 根据 CHECK 1-4 对照上方 H_A / H_B 标注判断。")


if __name__ == "__main__":
    main()
