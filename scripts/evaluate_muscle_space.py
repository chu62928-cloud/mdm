#!/usr/bin/env python3
"""
scripts/evaluate_muscle_space.py

肌肉空间评估（第二轮配套）。补上 evaluate_all_modes.py 缺的那一环：
**真正把生成的运动喂进冻结代理，算肌肉激活层面的指标**，而不是只看骨盆角。

为什么需要：joint 角度 hit_rate 衡量的是 Module 1 的目标；muscle 模块的交付物是
"临床上合理的肌肉激活模式"，必须在肌肉空间评（midterm Table 3/4）。否则 both≈joint
只是因为两者都用同一把（错的）尺子。

做什么（离线，无需重生成）：
对每个 comparison.npy 里**已保存的** motion_hml_tj(baseline) / motion_hml_tj_guided：
  1. 以 baseline 激活为参考（build_reference），算 guided 的 clinical 四分量 posture loss
     （total + 各分量）—— 复刻 midterm Table 3。
  2. 各功能肌群 baseline→guided 平均激活 + 期望方向 ✓/✗ —— 复刻 midterm Table 4。

用法：
    python scripts/evaluate_muscle_space.py <output_root> \
        --muscle_ckpt motion2muscle/checkpoints/.../net_best_loss.pth \
        [--muscle_posture anterior_pelvic_tilt] [--device cuda]
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

from muscle_rollup import get_indices, ROLLUP_GROUPS                 # noqa: E402
from posture_loss import POSTURE_PRIORS                             # noqa: E402
from posture_loss_torch import build_group_index, compute_posture_loss_torch  # noqa: E402


def expected_group_directions(posture_name):
    """从 POSTURE_PRIORS 推每个功能肌群的期望方向：'up' / 'down' / 'neutral'(冲突)。"""
    cfg = POSTURE_PRIORS[posture_name]
    bil = cfg.get("bilateral", True)
    sides = ["_R", "_L"] if bil else [""]
    votes = {}  # group -> set{'up','down'}

    def vote(name, d):
        for s in sides:
            votes.setdefault(name + s, set()).add(d)

    for over, under, _ in cfg.get("antagonist_imbalances", []):
        vote(over, "up"); vote(under, "down")
    for primary, comp, _ in cfg.get("compensation_chains", []):
        vote(primary, "down"); vote(comp, "up")
    for syn in cfg.get("synergy_imbalances", []):
        for n in syn["dominant"]:
            vote(n, "down")
        for n in syn["compensator"]:
            vote(n, "up")
    for name, _ in cfg.get("inhibited_stabilizers", []):
        vote(name, "down")

    out = {}
    for g, vs in votes.items():
        out[g] = next(iter(vs)) if len(vs) == 1 else "neutral"
    return out


def group_mean(acts, mint_cols, group):
    """acts (T,402) numpy -> 该肌群的标量平均激活；缺失返回 None。"""
    idx = get_indices(group, mint_cols)
    if not idx:
        return None
    return float(acts[..., idx].mean())


def evaluate_one(mg, hml_base, hml_guided, posture_name, device):
    """hml_*: (B,T,263) numpy（MDM 归一化空间）。返回指标 dict。"""
    xb = torch.from_numpy(hml_base).float().to(device)
    xg = torch.from_numpy(hml_guided).float().to(device)

    # 参考 = baseline（正常）激活；冻结
    mg.build_reference(xb)

    with torch.no_grad():
        ab = mg._activations(xb)                      # (B,T,402)
        ag = mg._activations(xg)

    # clinical 四分量 posture loss（midterm Table 3）：自洽参考≈0，guided 越大越病态
    total_b, comp_b = compute_posture_loss_torch(
        ab, mg.group_index, posture_name, mg.reference_acts, return_components=True)
    total_g, comp_g = compute_posture_loss_torch(
        ag, mg.group_index, posture_name, mg.reference_acts, return_components=True)

    # 各肌群方向性（midterm Table 4）
    dirs = expected_group_directions(posture_name)
    ab_np = ab.detach().cpu().numpy()
    ag_np = ag.detach().cpu().numpy()
    groups = []
    n_correct = n_scored = 0
    for g in sorted(dirs):
        mb = group_mean(ab_np, mg.mint_cols, g)
        mgd = group_mean(ag_np, mg.mint_cols, g)
        if mb is None or mgd is None:
            continue
        exp = dirs[g]
        delta = mgd - mb
        if exp == "up":
            ok = delta > 1e-6
        elif exp == "down":
            ok = delta < -1e-6
        else:
            ok = abs(delta) <= max(1e-6, 0.02 * abs(mb))  # neutral: 基本不变
        if exp != "neutral":
            n_scored += 1
            n_correct += int(ok)
        groups.append((g, exp, mb, mgd, delta, ok))

    return {
        "loss_baseline": float(total_b),
        "loss_guided": float(total_g),
        "loss_ratio": float(total_g) / (float(total_b) + 1e-12),
        "comp_baseline": comp_b,
        "comp_guided": comp_g,
        "groups": groups,
        "dir_correct": n_correct,
        "dir_scored": n_scored,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("--muscle_ckpt", required=True)
    ap.add_argument("--muscle_posture", default="anterior_pelvic_tilt")
    ap.add_argument("--muscle_assets_dir", default=_ASSETS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--same_norm", action="store_true", default=True)
    ap.add_argument("--diff_norm", dest="same_norm", action="store_false")
    ap.add_argument("--proxy_mean_path", default=None)
    ap.add_argument("--proxy_std_path", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from muscle_guidance_mdm import build_muscle_guidance
    mg = build_muscle_guidance(
        ckpt_path=args.muscle_ckpt,
        posture_name=args.muscle_posture,
        assets_dir=args.muscle_assets_dir,
        same_normalization=args.same_norm,
        proxy_mean_path=args.proxy_mean_path,
        proxy_std_path=args.proxy_std_path,
        device=device,
    )

    rows = []
    for entry in sorted(os.listdir(args.output_root)):
        d = os.path.join(args.output_root, entry)
        npy = os.path.join(d, "comparison.npy")
        if not os.path.isfile(npy):
            continue
        data = np.load(npy, allow_pickle=True).item()
        if "motion_hml_tj" not in data or "motion_hml_tj_guided" not in data:
            print(f"[skip] {entry}: comparison.npy 缺少 motion_hml_tj(_guided)")
            continue
        mode = data.get("guidance_mode", entry.split("_")[0])
        r = evaluate_one(mg, data["motion_hml_tj"], data["motion_hml_tj_guided"],
                         args.muscle_posture, device)
        r["mode"] = mode
        r["dir"] = entry
        rows.append(r)

        print(f"\n=== {entry} (mode={mode}) ===")
        print(f"  clinical posture loss: baseline={r['loss_baseline']:.4e} "
              f"guided={r['loss_guided']:.4e}  ratio={r['loss_ratio']:.3f}x "
              f"({'更病态↑' if r['loss_guided']>r['loss_baseline'] else '未变病态'})")
        print(f"  分量(guided): " + ", ".join(
            f"{k}={r['comp_guided'][k]:.3e}" for k in ("antagonist","chain","synergy","stabilizer")))
        print(f"  方向性: {r['dir_correct']}/{r['dir_scored']} 肌群朝期望方向")
        for g, exp, mb, mgd, delta, ok in r["groups"]:
            if exp == "neutral":
                continue
            arrow = "↑" if exp == "up" else "↓"
            print(f"    {('✓' if ok else '✗')} {g:28s} expect{arrow} "
                  f"{mb:.4e} -> {mgd:.4e} (Δ={delta:+.2e})")

    if not rows:
        print("没有可评估的 comparison.npy。")
        return

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':<8}{'loss_base':>12}{'loss_guided':>13}{'ratio':>8}{'dir✓/scored':>14}")
    print("-" * 78)
    for r in rows:
        dir_str = f"{r['dir_correct']}/{r['dir_scored']}"
        print(f"{r['mode']:<8}{r['loss_baseline']:>12.3e}{r['loss_guided']:>13.3e}"
              f"{r['loss_ratio']:>8.3f}{dir_str:>14}")
    print("=" * 78)

    out_md = os.path.join(args.output_root, "muscle_space_report.md")
    with open(out_md, "w") as f:
        f.write("# 肌肉空间评估 (muscle-space)\n\n")
        f.write(f"posture = {args.muscle_posture}, same_norm = {args.same_norm}\n\n")
        f.write("| mode | clinical loss base | guided | ratio | dir ✓/scored |\n")
        f.write("|------|-----|-----|-----|-----|\n")
        for r in rows:
            f.write(f"| {r['mode']} | {r['loss_baseline']:.3e} | {r['loss_guided']:.3e} | "
                    f"{r['loss_ratio']:.3f} | {r['dir_correct']}/{r['dir_scored']} |\n")
        f.write("\n> loss ratio > 1 表示 guided 在肌肉空间比 baseline 更朝该病态；"
                "dir ✓ 越多表示越多功能肌群朝临床期望方向变化（midterm Table 4）。\n")
    print(f"\nSaved: {out_md}")


if __name__ == "__main__":
    main()
