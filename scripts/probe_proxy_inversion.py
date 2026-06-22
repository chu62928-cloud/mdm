#!/usr/bin/env python3
"""
scripts/probe_proxy_inversion.py

坐实「冻结 motion→muscle proxy 对持续矢状骨盆倾角系统性逆映射」的成因，并产出论文证据图。

背景
----
诊断已确认 H_A：joint-guided 的真实 APT 几何（+15~20°）喂进 proxy，得到的绝对髋屈/伸比 < 1
（= PPT 肌肉形态）。换 reference 救不了，因为倒置在 proxy 的输入→激活 Jacobian 里。
本脚本实证区分三种成因：
  (a) OOD 外推 —— 分布内正确、仅在 APT/OOD 区间倒置；
  (b) 内在弱/未编码自由度 —— proxy 从没在数据里编码骨盆倾角→肌肉关系；
  (c) 读出/模板错配 —— proxy 编码了倾角，但不在临床模板用的 flex/ext 轴上（可换轴修复）。

子分析
------
  A. 受控扫描（证据图）：沿 baseline↔joint-guided 方向插值，画 proxy flex/ext vs 真实骨盆角，
     预期单调递减；现有手点(baseline≈6.0, joint-APT≈0.73)应落在曲线上（回归锚点）。
  B（分布内，真实 walking clips）三联：
     B1 跨片均值：每片一个点（均值倾角 vs 均值 flex/ext），隔离步态相位噪声 → 决定性区分 (a) vs (b)。
     B2 按角分箱：分布内帧按倾角分箱、画每箱均值 flex/ext → 去噪看分布内曲线形状。
     B3 全功能群扫描：逐 ROLLUP_GROUPS 算「该群激活 vs 倾角」的相关并排序 →
        无群编码=(b) 可表征性缺失；有群(正号)编码=(c) 读出错配、可换轴修复。
  （原 C 单自由度探针已弃用：只转上半身=解剖不可能姿势=又引回 OOD 混淆，且 process_file 重编码脆弱。）

用法
----
    python scripts/probe_proxy_inversion.py \
        --joint_npy output_0608/apt_joint_seed42/comparison.npy \
        --muscle_ckpt motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
        --data_dir dataset/HumanML3D \
        --out_dir output_0608/proxy_inversion \
        --max_clips 200 --device cuda

  - 不传 --data_dir 则只跑 A。
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
from muscle_rollup import ROLLUP_GROUPS, get_indices
from posture_guidance.angle_ops import pelvis_tilt_angle

# 复用 analyze_pelvis_tilt 的直立过滤与步行筛选（同目录）
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
    """acts: (T,402)/(B,T,402) -> 该肌群(左右合并)逐帧均值；缺失 None。"""
    idx = []
    for side in ("_R", "_L"):
        idx += get_indices(group + side, mint_cols)
    if not idx:
        return None
    return np.asarray(acts[..., idx]).mean(axis=-1)


def flex_ext_per_frame(acts, mint_cols, eps=1e-12):
    """逐帧屈/伸比。屈=髂腰+股直均值，伸=臀大。"""
    flex = np.mean([group_frame_act(acts, mint_cols, g) for g in FLEXORS], axis=0)
    ext = group_frame_act(acts, mint_cols, EXTENSOR)
    return flex / (ext + eps)


def normalize_mdm(raw_263, t2m):
    return (raw_263 - t2m.mean) / t2m.std


def base_group_names():
    """从 ROLLUP_GROUPS 的 _R/_L 键里取唯一基名。"""
    bases = set()
    for k in ROLLUP_GROUPS:
        if k.endswith("_R") or k.endswith("_L"):
            bases.add(k[:-2])
        else:
            bases.add(k)
    return sorted(bases)


# ──────────────────────────────────────────────────────────
# A. 受控扫描（证据图）
# ──────────────────────────────────────────────────────────
def analysis_A(mg, fk_fn, joint_npy, device, alphas):
    print("\n" + "=" * 70)
    print("A. 受控扫描：proxy flex/ext vs 真实骨盆角（沿 baseline↔joint-guided 插值）")
    print("=" * 70)
    data = np.load(joint_npy, allow_pickle=True).item()
    xb = torch.from_numpy(np.asarray(data["motion_hml_tj"], np.float32)).to(device)
    xg = torch.from_numpy(np.asarray(data["motion_hml_tj_guided"], np.float32)).to(device)

    angles, ratios, groups = [], [], {g: [] for g in REPORT_GROUPS}
    print(f"\n  {'alpha':>6} {'pelvis°':>9} {'flex/ext':>10}")
    print("  " + "-" * 28)
    for a in alphas:
        x = xb + a * (xg - xb)
        with torch.no_grad():
            q = fk_fn(x.permute(0, 2, 1).unsqueeze(2))
            ang = float(torch.rad2deg(pelvis_tilt_angle(q).mean()))
            acts = mg._activations(x).cpu().numpy()[0]
        r = float(np.mean(flex_ext_per_frame(acts, mg.mint_cols)))
        angles.append(ang); ratios.append(r)
        for g in REPORT_GROUPS:
            groups[g].append(float(group_frame_act(acts, mg.mint_cols, g).mean()))
        tag = " <-baseline" if abs(a) < 1e-6 else (" <-joint-APT" if abs(a - 1.0) < 1e-6 else "")
        print(f"  {a:>6.2f} {ang:>+8.1f}° {r:>10.3f}{tag}")

    angles, ratios = np.array(angles), np.array(ratios)
    slope = float(np.polyfit(angles, ratios, 1)[0])
    order = np.argsort(angles)
    mono = bool(np.all(np.diff(ratios[order]) <= 1e-6))
    print(f"\n  线性斜率 d(flex/ext)/d(pelvis°) = {slope:+.4f}"
          f"   {'(单调递减 ✓ = 逆映射)' if slope < 0 else '(非递减)'}   严格单调递减: {mono}")
    return dict(angles=angles, ratios=ratios, groups=groups, slope=slope, mono=mono)


# ──────────────────────────────────────────────────────────
# B. 分布内（真实 walking clips）：一次加载，缓存逐帧 + 跨片 + 全群
# ──────────────────────────────────────────────────────────
def collect_indist(mg, data_dir, t2m, device, max_clips, walking_only, min_height):
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

    # 预备各功能群的列下标（左右合并）
    bases = base_group_names()
    gidx = {}
    for b in bases:
        idx = get_indices(b + "_R", mg.mint_cols) + get_indices(b + "_L", mg.mint_cols)
        if idx:
            gidx[b] = idx

    ang_chunks, fe_chunks = [], []
    gchunks = {b: [] for b in gidx}
    clip_means = []
    n_dim = t2m.mean.shape[0]
    for cid in ids:
        try:
            raw = np.load(vec_dir / f"{cid}.npy").astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] != n_dim:
                continue
            with torch.no_grad():
                q = recover_from_ric(torch.from_numpy(raw)[None], 22).squeeze(0)   # (T,22,3)
                m = upright_mask(q, min_height=min_height).numpy()
                if m.sum() < 1:
                    continue
                ang = pelvis_tilt_angle(q).numpy() * (180.0 / math.pi)            # (T,)
                xn = torch.from_numpy(normalize_mdm(raw, t2m))[None].to(device)
                acts = mg._activations(xn).cpu().numpy()[0]                       # (T,402)
            a_m = ang[m]
            fe = flex_ext_per_frame(acts, mg.mint_cols)[m]
            acts_m = acts[m]
            ang_chunks.append(a_m); fe_chunks.append(fe)
            for b, idx in gidx.items():
                gchunks[b].append(acts_m[:, idx].mean(axis=-1))
            clip_means.append((float(a_m.mean()), float(fe.mean())))
        except Exception as e:
            print(f"    [warn] {cid}: {e}")
            continue
    if not ang_chunks:
        print("  [跳过] 无有效直立帧")
        return None
    return dict(
        angles=np.concatenate(ang_chunks),
        flexext=np.concatenate(fe_chunks),
        group_acts={b: np.concatenate(v) for b, v in gchunks.items()},
        clip_means=np.array(clip_means),
    )


def analysis_B(cache):
    print("\n" + "=" * 70)
    print("B. 分布内检验")
    print("=" * 70)
    ang, fe, cm = cache["angles"], cache["flexext"], cache["clip_means"]
    print(f"  直立帧 {len(ang):,}   骨盆角 {ang.min():+.1f}°..{ang.max():+.1f}° "
          f"(P5={np.percentile(ang,5):+.1f}, P95={np.percentile(ang,95):+.1f})   "
          f"clips={len(cm)}")

    # --- B0 逐帧（参照，含相位噪声）---
    s0 = float(np.polyfit(ang, fe, 1)[0]); r0 = float(np.corrcoef(ang, fe)[0, 1])
    print(f"\n  B0 逐帧:  slope={s0:+.4f}  r={r0:+.3f}  (相位噪声主导，仅参照)")

    # --- B1 跨片均值（去相位噪声，决定性）---
    s1 = float(np.polyfit(cm[:, 0], cm[:, 1], 1)[0]); r1 = float(np.corrcoef(cm[:, 0], cm[:, 1])[0, 1])
    print(f"  B1 跨片:  slope={s1:+.4f}  r={r1:+.3f}", end="  ")
    if r1 > 0.2:
        print("→ 分布内姿势耦合为正(方向正确) ⇒ 倾向 (a) OOD 外推")
    elif r1 < -0.2:
        print("→ 分布内姿势耦合为负(已倒置) ⇒ 倾向 (b)/(c) 内在")
    else:
        print("→ 分布内姿势与 flex/ext 解耦(≈0) ⇒ 倾向 (b) 未编码/弱自由度")

    # --- B2 按角分箱均值 ---
    print("\n  B2 分箱(均值 flex/ext):")
    lo, hi = np.percentile(ang, 2), np.percentile(ang, 98)
    edges = np.linspace(lo, hi, 9)
    centers, bin_means = [], []
    for i in range(len(edges) - 1):
        sel = (ang >= edges[i]) & (ang < edges[i + 1])
        if sel.sum() < 20:
            continue
        c = float((edges[i] + edges[i + 1]) / 2)
        v = float(fe[sel].mean())
        centers.append(c); bin_means.append(v)
        print(f"    [{edges[i]:+5.1f},{edges[i+1]:+5.1f})°  n={int(sel.sum()):6d}  flex/ext={v:.3f}")

    return dict(s0=s0, r0=r0, s1=s1, r1=r1, centers=np.array(centers), bin_means=np.array(bin_means))


def analysis_B3(cache, top=8):
    print("\n" + "=" * 70)
    print("B3. 全功能群扫描：各群激活 vs 骨盆倾角的分布内相关（逐帧）")
    print("=" * 70)
    ang = cache["angles"]
    rows = []
    for b, a in cache["group_acts"].items():
        if np.std(a) < 1e-9:
            continue
        r = float(np.corrcoef(ang, a)[0, 1])
        rows.append((b, r))
    rows.sort(key=lambda x: x[1])   # 升序：最负在前
    print(f"\n  最强正相关(倾角↑→激活↑)  Top{top}：")
    for b, r in rows[::-1][:top]:
        print(f"    {b:<26} r={r:+.3f}")
    print(f"\n  最强负相关(倾角↑→激活↓)  Top{top}：")
    for b, r in rows[:top]:
        print(f"    {b:<26} r={r:+.3f}")
    max_pos = max((r for _, r in rows), default=0.0)
    # 关注临床 APT 应升的髋屈/腰伸群是否有正相关
    clin_up = ["iliopsoas", "rectus_femoris", "erector_spinae", "psoas"]
    clin = {b: r for b, r in rows if b in clin_up}
    print("\n  临床 APT 期望↑的群的实际相关：")
    for b in clin_up:
        if b in clin:
            print(f"    {b:<26} r={clin[b]:+.3f}  {'✓正' if clin[b] > 0.2 else ('✗负' if clin[b] < -0.2 else '≈0')}")
    return dict(rows=rows, max_pos=max_pos, clin=clin)


# ──────────────────────────────────────────────────────────
# 出图 + 裁决
# ──────────────────────────────────────────────────────────
def make_figure(rA, rB, cache, out_dir):
    has_B = rB is not None
    n = 1 + (2 if has_B else 0)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    k = 0
    ax = axes[k]; k += 1
    o = np.argsort(rA["angles"])
    ax.plot(rA["angles"][o], rA["ratios"][o], "o-", color="C3")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_title(f"A. Controlled sweep (slope={rA['slope']:+.3f})")
    ax.set_xlabel("true pelvic tilt (deg, +=anterior)")
    ax.set_ylabel("proxy flex/ext ratio")
    if has_B:
        cm = cache["clip_means"]
        ax = axes[k]; k += 1
        ax.scatter(cm[:, 0], cm[:, 1], s=10, alpha=0.5, color="C0")
        xs = np.linspace(cm[:, 0].min(), cm[:, 0].max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(cm[:, 0], cm[:, 1], 1), xs), "C1", lw=2)
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_title(f"B1. Across-clip means (slope={rB['s1']:+.3f}, r={rB['r1']:+.2f})")
        ax.set_xlabel("clip-mean pelvic tilt (deg)"); ax.set_ylabel("clip-mean flex/ext")

        ax = axes[k]; k += 1
        if len(rB["centers"]):
            ax.plot(rB["centers"], rB["bin_means"], "s-", color="C2")
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_title("B2. In-distribution binned means")
        ax.set_xlabel("pelvic tilt bin (deg)"); ax.set_ylabel("mean flex/ext")
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


def verdict(rA, rB, rB3):
    print("\n" + "=" * 70)
    print("裁决")
    print("=" * 70)
    print(f"  A  受控扫描斜率 {rA['slope']:+.4f}  "
          f"({'逆映射成立 ✓' if rA['slope'] < 0 else '未见逆映射 ✗'})")
    if rB is None:
        print("  B/B3 未运行（未传 --data_dir）。")
        return
    print(f"  B1 跨片  slope={rB['s1']:+.4f}  r={rB['r1']:+.3f}")
    print(f"  B3 全群最强正相关 r={rB3['max_pos']:+.3f}")

    if rB["r1"] > 0.2:
        cause = "(a) OOD 外推：分布内姿势耦合为正，proxy 仅在 APT/OOD 区间倒置"
    elif rB3["max_pos"] > 0.3:
        top = max(rB3["rows"], key=lambda x: x[1])
        cause = (f"(c) 读出/模板错配：proxy 编码了倾角（如 {top[0]} r={top[1]:+.2f}），"
                 f"但不在临床用的 flex/ext 轴上 ⇒ 可换读出轴修复")
    else:
        cause = "(b) 可表征性缺失：分布内倾角与所有肌群均弱相关，proxy 未编码骨盆倾角"
    print(f"\n  → 成因结论: {cause}")
    if rB["r1"] <= 0.2 and rB3["max_pos"] <= 0.3:
        print("  注: (b) 与残余 (c) 的彻底分辨需真实 EMG/MinT 标签；本数据支持「未编码/弱」主导。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_npy", required=True, help="joint 模式 comparison.npy（guided=真实 APT 几何）")
    ap.add_argument("--muscle_ckpt", required=True)
    ap.add_argument("--muscle_posture", default="anterior_pelvic_tilt")
    ap.add_argument("--data_dir", default=None, help="HumanML3D 根目录（含 new_joint_vecs/，传则跑 B/B3）")
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

    rB = rB3 = cache = None
    if args.data_dir:
        cache = collect_indist(mg, args.data_dir, t2m, device,
                               args.max_clips, args.walking_only, args.min_height)
        if cache is not None:
            rB = analysis_B(cache)
            rB3 = analysis_B3(cache)

    save_csv(rA, args.out_dir)
    make_figure(rA, rB, cache, args.out_dir)
    verdict(rA, rB, rB3)


if __name__ == "__main__":
    main()
