#!/usr/bin/env python3
"""
new/probe_proxy_inversion.py

坐实「冻结 motion→muscle proxy 对持续矢状骨盆倾角系统性逆映射」的成因，并产出论文证据图。

背景
----
诊断已确认 H_A：joint-guided 的真实 APT 几何（+15~20°）喂进 proxy，得到的绝对髋屈/伸比 < 1
（= PPT 肌肉形态）。换 reference 救不了，因为倒置在 proxy 的输入→激活 Jacobian 里。
但「为什么倒置」尚未实证区分三种成因：
  (a) OOD 外推 —— 分布内正确、仅在 APT/OOD 区间倒置；
  (b) 内在弱/被混淆自由度 —— proxy 从没真正编码骨盆倾角；
  (c) 量纲错配 —— proxy 输出瞬时激活 ≠ 临床张力性期望（本脚本无法直接证伪，仅在裁决中提示）。

本脚本三个子分析：
  A. 受控扫描（证据图）：沿 baseline↔joint-guided 方向插值，画 proxy flex/ext vs 真实骨盆角，
     预期单调递减；现有手点(baseline≈6.0, joint-APT≈0.73)应落在曲线上（回归锚点）。
  B. 分布内检验（决定性区分 a vs b/c）：真实 HumanML3D 步行片在自然角度范围内，
     看 proxy flex/ext↔骨盆角斜率符号。正确(正斜率)=纯 OOD；反/平=内在缺陷。
  C. 单自由度探针（隔离 Jacobian，实验性）：冻住四肢、只在矢状面旋转上半身改变骨盆倾角，
     re-encode→proxy，量纯倾角→激活响应，排除步态混淆。

用法
----
    python new/probe_proxy_inversion.py \
        --joint_npy output_0608/apt_joint_seed42/comparison.npy \
        --muscle_ckpt motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
        --data_dir dataset/HumanML3D \
        --out_dir output_0608/proxy_inversion \
        --max_clips 200 --probe --device cuda

  - 不传 --data_dir 则跳过 B；不传 --probe 则跳过 C。A 始终运行（只需 --joint_npy）。
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_ASSETS = os.path.join(_ROOT, "motion2muscle")
for p in (_ROOT, _ASSETS, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loaders.humanml.data.dataset import HumanML3D
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from muscle_guidance_mdm.build import build_muscle_guidance
from muscle_rollup import get_indices
from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_guidance.joint_indices import get_joint_idx

# 复用 analyze_pelvis_tilt 的分布内加载与直立过滤（同目录）
from analyze_pelvis_tilt import upright_mask, filter_walking_clips  # noqa: E402

# 关键肌群：髋屈肌(APT 应高) vs 髋伸肌(APT 应低)
FLEXORS = ["iliopsoas", "rectus_femoris"]
EXTENSOR = "gluteus_maximus"
REPORT_GROUPS = ["iliopsoas", "rectus_femoris", "gluteus_maximus",
                 "erector_spinae", "gluteus_medius"]


# ──────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────
def make_fk_fn(t2m, device):
    """(B,263,1,T) MDM 归一化特征 -> q (B,T,22,3) 全局关节。"""
    mean = torch.tensor(t2m.mean, dtype=torch.float32, device=device)
    std = torch.tensor(t2m.std, dtype=torch.float32, device=device)

    def fk_fn(mu):
        mu_inv = mu.permute(0, 3, 2, 1) * std + mean   # (B,T,1,263) 反归一化
        q = recover_from_ric(mu_inv, 22).squeeze(2)     # (B,T,22,3)
        return q
    return fk_fn


def group_frame_act(acts, mint_cols, group):
    """acts: (T,402) 或 (B,T,402) -> 该肌群(左右合并)逐帧均值 (T,) / (B,T)；缺失 None。"""
    idx = []
    for side in ("_R", "_L"):
        idx += get_indices(group + side, mint_cols)
    if not idx:
        return None
    return np.asarray(acts[..., idx]).mean(axis=-1)


def flex_ext_per_frame(acts, mint_cols, eps=1e-12):
    """逐帧屈/伸比 (T,)。屈=髂腰+股直均值，伸=臀大。"""
    flex = np.mean([group_frame_act(acts, mint_cols, g) for g in FLEXORS], axis=0)
    ext = group_frame_act(acts, mint_cols, EXTENSOR)
    return flex / (ext + eps)


def normalize_mdm(raw_263, t2m):
    """raw (T,263) -> MDM 归一化 (T,263)。"""
    return (raw_263 - t2m.mean) / t2m.std


# ──────────────────────────────────────────────────────────
# A. 受控扫描（证据图）
# ──────────────────────────────────────────────────────────
def analysis_A(mg, fk_fn, joint_npy, device, alphas):
    print("\n" + "=" * 70)
    print("A. 受控扫描：proxy flex/ext vs 真实骨盆角（沿 baseline↔joint-guided 插值）")
    print("=" * 70)
    data = np.load(joint_npy, allow_pickle=True).item()
    xb = torch.from_numpy(np.asarray(data["motion_hml_tj"], np.float32)).to(device)        # (1,T,263)
    xg = torch.from_numpy(np.asarray(data["motion_hml_tj_guided"], np.float32)).to(device)

    angles, ratios, groups = [], [], {g: [] for g in REPORT_GROUPS}
    print(f"\n  {'alpha':>6} {'pelvis°':>9} {'flex/ext':>10}")
    print("  " + "-" * 28)
    for a in alphas:
        x = xb + a * (xg - xb)                                  # (1,T,263) 归一化空间
        with torch.no_grad():
            q = fk_fn(x.permute(0, 2, 1).unsqueeze(2))          # (1,T,22,3)
            ang = float(torch.rad2deg(pelvis_tilt_angle(q).mean()))
            acts = mg._activations(x).cpu().numpy()[0]          # (T,402)
        r = float(np.mean(flex_ext_per_frame(acts, mg.mint_cols)))
        angles.append(ang); ratios.append(r)
        for g in REPORT_GROUPS:
            groups[g].append(float(group_frame_act(acts, mg.mint_cols, g).mean()))
        tag = " <-baseline" if abs(a) < 1e-6 else (" <-joint-APT" if abs(a - 1.0) < 1e-6 else "")
        print(f"  {a:>6.2f} {ang:>+8.1f}° {r:>10.3f}{tag}")

    angles, ratios = np.array(angles), np.array(ratios)
    # 单调性：骨盆角升序后 flex/ext 是否单调递减
    order = np.argsort(angles)
    mono = np.all(np.diff(ratios[order]) <= 1e-6)
    slope = float(np.polyfit(angles, ratios, 1)[0])
    print(f"\n  线性斜率 d(flex/ext)/d(pelvis°) = {slope:+.4f}"
          f"   {'(单调递减 ✓ = 逆映射)' if slope < 0 else '(非递减)'}")
    print(f"  严格单调递减: {mono}")
    return dict(angles=angles, ratios=ratios, groups=groups, slope=slope, mono=mono)


# ──────────────────────────────────────────────────────────
# B. 分布内检验（决定性区分 OOD vs 内在）
# ──────────────────────────────────────────────────────────
def analysis_B(mg, data_dir, t2m, device, max_clips, walking_only, min_height):
    print("\n" + "=" * 70)
    print("B. 分布内检验：真实步行片自然角度范围内 flex/ext vs 骨盆角")
    print("=" * 70)
    data_dir = Path(data_dir)
    vec_dir = data_dir / "new_joint_vecs"
    if not vec_dir.exists():
        print(f"  [跳过] 找不到 {vec_dir}")
        return None
    ids = [p.stem for p in sorted(vec_dir.glob("*.npy"))]
    if walking_only:
        ids = filter_walking_clips(data_dir, ids)
        print(f"  walking-prompt 过滤后: {len(ids)} clips")
    if max_clips > 0:
        ids = ids[:max_clips]
    print(f"  处理 {len(ids)} clips ...")

    A_all, R_all = [], []
    for cid in ids:
        try:
            raw = np.load(vec_dir / f"{cid}.npy").astype(np.float32)   # (T,263) 未归一化
            if raw.ndim != 2 or raw.shape[1] != t2m.mean.shape[0]:
                continue
            with torch.no_grad():
                q = recover_from_ric(torch.from_numpy(raw)[None], 22).squeeze(0)  # (T,22,3)
                m = upright_mask(q, min_height=min_height).numpy()                # (T,)
                if m.sum() < 1:
                    continue
                ang = pelvis_tilt_angle(q).numpy() * (180.0 / math.pi)           # (T,)
                xn = torch.from_numpy(normalize_mdm(raw, t2m))[None].to(device)  # (1,T,263)
                acts = mg._activations(xn).cpu().numpy()[0]                      # (T,402)
            r = flex_ext_per_frame(acts, mg.mint_cols)                          # (T,)
            A_all.append(ang[m]); R_all.append(r[m])
        except Exception as e:
            print(f"    [warn] {cid}: {e}")
            continue
    if not A_all:
        print("  [跳过] 无有效直立帧")
        return None
    A_all = np.concatenate(A_all); R_all = np.concatenate(R_all)
    slope = float(np.polyfit(A_all, R_all, 1)[0])
    r_pear = float(np.corrcoef(A_all, R_all)[0, 1])
    print(f"\n  直立帧数: {len(A_all):,}   骨盆角范围: "
          f"{A_all.min():+.1f}°..{A_all.max():+.1f}° (P5={np.percentile(A_all,5):+.1f}, "
          f"P95={np.percentile(A_all,95):+.1f})")
    print(f"  斜率 d(flex/ext)/d(pelvis°) = {slope:+.4f}   Pearson r = {r_pear:+.3f}")
    if slope > 0:
        print("  → 分布内斜率为正（前倾↑→flex/ext↑，方向正确）⇒ 成因 (a) OOD 外推")
    else:
        print("  → 分布内斜率≤0（方向已反/无关）⇒ 成因 (b)/(c) 内在弱自由度/量纲错配")
    return dict(angles=A_all, ratios=R_all, slope=slope, pearson=r_pear)


# ──────────────────────────────────────────────────────────
# C. 单自由度骨盆旋转探针（实验性）
# ──────────────────────────────────────────────────────────
def _rodrigues(axis, theta):
    """axis (3,) 单位向量, theta 标量(rad) -> R (3,3)。"""
    a = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def analysis_C(mg, t2m, joint_npy, device, deltas):
    """冻住四肢、绕髋中点 medio-lateral 轴旋转上半身，改变骨盆倾角，量纯激活响应。

    注意：通过对 recover_from_ric 得到的关节做几何旋转后，重新用 process_file 编码回 263。
    process_file 会重算 root 规范化/速度/足触，几何 tilt 应当保留。实验性，失败则跳过。
    """
    print("\n" + "=" * 70)
    print("C. 单自由度探针（实验性）：只旋转上半身改变骨盆倾角")
    print("=" * 70)
    try:
        from data_loaders.humanml.scripts.motion_process import process_file
    except Exception as e:
        print(f"  [跳过] 无法导入 process_file: {e}")
        return None

    UPPER = ["spine1", "spine2", "spine3", "neck", "head",
             "left_collar", "right_collar", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    upper_idx = [get_joint_idx(n) for n in UPPER]

    data = np.load(joint_npy, allow_pickle=True).item()
    xb = torch.from_numpy(np.asarray(data["motion_hml_tj"], np.float32))   # (1,T,263)
    with torch.no_grad():
        mu_inv = xb.permute(0, 2, 1).unsqueeze(2) * torch.tensor(t2m.std) + torch.tensor(t2m.mean)
        q0 = recover_from_ric(mu_inv, 22).squeeze(2).squeeze(0).numpy()    # (T,22,3)

    angles, ratios = [], []
    print(f"\n  {'Δapply°':>8} {'pelvis°':>9} {'flex/ext':>10}")
    print("  " + "-" * 30)
    for d in deltas:
        try:
            q = q0.copy()
            T = q.shape[0]
            for t in range(T):
                lh, rh = q[t, get_joint_idx("left_hip")], q[t, get_joint_idx("right_hip")]
                center = (lh + rh) / 2.0
                axis = rh - lh                      # medio-lateral 轴
                R = _rodrigues(axis, math.radians(d))
                for j in upper_idx:
                    q[t, j] = center + R @ (q[t, j] - center)
            feats = process_file(q, 0.002)[0]       # -> (T-?,263)
            feats = np.asarray(feats, np.float32)
            if feats.ndim != 2 or feats.shape[1] != t2m.mean.shape[0]:
                print(f"  {d:>8.1f}  process_file 输出形状异常 {feats.shape}，跳过该点")
                continue
            with torch.no_grad():
                qf = recover_from_ric(torch.from_numpy(feats)[None], 22).squeeze(0)
                ang = float(pelvis_tilt_angle(qf).mean() * 180.0 / math.pi)
                xn = torch.from_numpy(normalize_mdm(feats, t2m))[None].to(device)
                acts = mg._activations(xn).cpu().numpy()[0]
            r = float(np.mean(flex_ext_per_frame(acts, mg.mint_cols)))
            angles.append(ang); ratios.append(r)
            print(f"  {d:>8.1f} {ang:>+8.1f}° {r:>10.3f}")
        except Exception as e:
            print(f"  {d:>8.1f}  [warn] {e}")
            continue
    if len(angles) < 2:
        print("  [跳过] 有效点不足")
        return None
    slope = float(np.polyfit(angles, ratios, 1)[0])
    print(f"\n  探针斜率 d(flex/ext)/d(pelvis°) = {slope:+.4f}   "
          f"{'(负=纯倾角→激活也倒置)' if slope < 0 else '(非负)'}")
    return dict(angles=np.array(angles), ratios=np.array(ratios), slope=slope)


# ──────────────────────────────────────────────────────────
# 出图 + 裁决
# ──────────────────────────────────────────────────────────
def make_figure(rA, rB, rC, out_dir):
    n = 1 + (rB is not None) + (rC is not None)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    i = 0
    ax = axes[i]; i += 1
    o = np.argsort(rA["angles"])
    ax.plot(rA["angles"][o], rA["ratios"][o], "o-", color="C3")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_title(f"A. Controlled sweep (slope={rA['slope']:+.3f})")
    ax.set_xlabel("true pelvic tilt (deg, +=anterior)")
    ax.set_ylabel("proxy flex/ext ratio")
    ax.annotate("clinical APT expects flex/ext>1 here →",
                xy=(0.02, 0.92), xycoords="axes fraction", fontsize=8, color="gray")
    if rB is not None:
        ax = axes[i]; i += 1
        ax.scatter(rB["angles"], rB["ratios"], s=3, alpha=0.25, color="C0")
        xs = np.linspace(rB["angles"].min(), rB["angles"].max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(rB["angles"], rB["ratios"], 1), xs), "C1", lw=2)
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_title(f"B. In-distribution (slope={rB['slope']:+.3f}, r={rB['pearson']:+.2f})")
        ax.set_xlabel("true pelvic tilt (deg)"); ax.set_ylabel("proxy flex/ext")
    if rC is not None:
        ax = axes[i]; i += 1
        o = np.argsort(rC["angles"])
        ax.plot(rC["angles"][o], rC["ratios"][o], "s-", color="C2")
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_title(f"C. Isolated tilt probe (slope={rC['slope']:+.3f})")
        ax.set_xlabel("imposed pelvic tilt (deg)"); ax.set_ylabel("proxy flex/ext")
    fig.tight_layout()
    out = Path(out_dir) / "proxy_inversion.png"
    fig.savefig(out, dpi=150)
    print(f"\n  图已保存: {out}")


def save_csv(rA, out_dir):
    out = Path(out_dir) / "sweep_A.csv"
    cols = ["pelvis_deg", "flex_ext"] + REPORT_GROUPS
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(len(rA["angles"])):
            row = [f"{rA['angles'][k]:.3f}", f"{rA['ratios'][k]:.5f}"] + \
                  [f"{rA['groups'][g][k]:.6f}" for g in REPORT_GROUPS]
            f.write(",".join(row) + "\n")
    print(f"  CSV 已保存: {out}")


def verdict(rA, rB, rC):
    print("\n" + "=" * 70)
    print("裁决")
    print("=" * 70)
    print(f"  A 受控扫描斜率: {rA['slope']:+.4f}  "
          f"({'逆映射证据成立 ✓' if rA['slope'] < 0 else '未见逆映射 ✗'})")
    if rB is None:
        print("  B 分布内: 未运行（未传 --data_dir）—— 无法区分 OOD vs 内在")
        cause = "未定（需 B）"
    elif rB["slope"] > 0:
        cause = "(a) OOD 外推主导：proxy 分布内方向正确，仅在 APT/OOD 区间倒置"
    else:
        cause = "(b)/(c) 内在：proxy 在分布内也未正确编码骨盆倾角（弱自由度/量纲错配）"
    print(f"  B 分布内斜率: {('%.4f' % rB['slope']) if rB else 'N/A'}")
    if rC is not None:
        print(f"  C 探针斜率:   {rC['slope']:+.4f}  "
              f"({'纯倾角响应也倒置→支持内在' if rC['slope'] < 0 else '纯倾角响应正常→支持 OOD'})")
    print(f"\n  → 成因结论: {cause}")
    print("  注: (c) 量纲错配（瞬时激活≠临床张力性期望）本脚本不能单独证伪，")
    print("      若 B/C 指向内在，需结合真实 EMG/MinT 标签进一步分辨 (b) vs (c)。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_npy", required=True, help="joint 模式 comparison.npy（guided=真实 APT 几何）")
    ap.add_argument("--muscle_ckpt", required=True)
    ap.add_argument("--muscle_posture", default="anterior_pelvic_tilt")
    ap.add_argument("--data_dir", default=None, help="HumanML3D 根目录（含 new_joint_vecs/，传则跑 B）")
    ap.add_argument("--probe", action="store_true", help="跑 C 单自由度探针（实验性）")
    ap.add_argument("--out_dir", default="output_0608/proxy_inversion")
    ap.add_argument("--max_clips", type=int, default=200)
    ap.add_argument("--walking_only", action="store_true", default=True)
    ap.add_argument("--all_clips", dest="walking_only", action="store_false")
    ap.add_argument("--min_height", type=float, default=0.5)
    ap.add_argument("--dataset_opt", default="./dataset/humanml_opt.txt")
    ap.add_argument("--abs_path", default=".")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    print("加载 t2m dataset (FK/归一化统计) ...")
    t2m = HumanML3D(mode="eval", datapath=args.dataset_opt, device="cpu", abs_path=args.abs_path).t2m_dataset
    fk_fn = make_fk_fn(t2m, device)

    mg = build_muscle_guidance(
        ckpt_path=args.muscle_ckpt, posture_name=args.muscle_posture,
        assets_dir=_ASSETS, same_normalization=True, device=device)
    assert mg.mint_cols is not None

    alphas = np.linspace(-0.3, 1.3, 17)
    rA = analysis_A(mg, fk_fn, args.joint_npy, device, alphas)
    rB = analysis_B(mg, args.data_dir, t2m, device, args.max_clips,
                    args.walking_only, args.min_height) if args.data_dir else None
    deltas = np.linspace(-15, 25, 9)
    rC = analysis_C(mg, t2m, args.joint_npy, device, deltas) if args.probe else None

    save_csv(rA, args.out_dir)
    make_figure(rA, rB, rC, args.out_dir)
    verdict(rA, rB, rC)


if __name__ == "__main__":
    main()
