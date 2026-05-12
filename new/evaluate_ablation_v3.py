"""
new/evaluate_ablation.py  (v3, with two-sided hit_rate)

核心变更：
    hit_rate 从单边 (>= target-tol) 改成区间 (target-tol, target+tol)。
    
    旧版问题：v2b_bw15 Δ=+36° 的曲线全在 30+ 区间，但 hit_rate 仍然 100%
    （因为 "≥ 18°" 永远满足）。这让"过度推力"的惩罚不能触发。
    
    新版：只有真正落在 [target-tol, target+tol] 区间内的帧才算 hit。
    这下 v2b_bw15 的 hit_rate 会暴跌到 ~10%，问题立刻暴露。
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance import angle_ops


ANGLE_TARGETS = {
    "骨盆前倾":   ("pelvis_tilt_angle",       20.0, 2.0),
    "骨盆前倾_深蹲": ("pelvis_tilt_angle",    25.0, 3.0),
    "膝超伸":     ("signed_knee_angle_both",  185.0, 3.0),
    "膝超伸_左":   ("signed_knee_angle_left", 185.0, 3.0),
    "膝超伸_右":   ("signed_knee_angle_right", 185.0, 3.0),
    "驼背":       ("spine_posterior_bulge",  0.08, 0.02),
}


def get_angle_fn(name):
    if name == "signed_knee_angle_both":
        def _both(q):
            l = angle_ops.signed_knee_angle(q, side="left")
            r = angle_ops.signed_knee_angle(q, side="right")
            return (l + r) / 2.0
        return _both
    if name == "signed_knee_angle_left":
        return lambda q: angle_ops.signed_knee_angle(q, side="left")
    if name == "signed_knee_angle_right":
        return lambda q: angle_ops.signed_knee_angle(q, side="right")
    return getattr(angle_ops, name)


def compute_metrics(npy_path, posture):
    data = np.load(npy_path, allow_pickle=True).item()

    if posture not in ANGLE_TARGETS:
        raise ValueError(f"Unknown posture: {posture}")
    fn_name, target_deg, tol_deg = ANGLE_TARGETS[posture]
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

    delta = float(a_g_deg.mean() - a_b_deg.mean())

    # ★ 关键改动：双边区间 hit_rate
    hit_mask = (a_g_deg >= target_deg - tol_deg) & (a_g_deg <= target_deg + tol_deg)
    hit_rate = float(hit_mask.mean())

    # 也算"宽松版"hit（仅 >= target-tol）做对比，写入但不参与 score
    hit_rate_loose = float((a_g_deg >= target_deg - tol_deg).mean())

    # 平均角度偏离目标的距离
    target_distance = float(abs(a_g_deg.mean() - target_deg))

    if a_b_deg.std() > 1e-6 and a_g_deg.std() > 1e-6:
        corr = float(pearsonr(a_b_deg, a_g_deg)[0])
    else:
        corr = 0.0

    diff = q_g.numpy() - q_b.numpy()
    rmse = float(np.sqrt((diff ** 2).mean()))

    jitter = float(np.diff(a_g_deg).std())

    foot_skate = compute_foot_skate(q_g.numpy())

    return {
        "baseline_mean":   float(a_b_deg.mean()),
        "guided_mean":     float(a_g_deg.mean()),
        "delta":           delta,
        "hit_rate":        hit_rate,           # 区间内
        "hit_rate_loose":  hit_rate_loose,     # 仅下限
        "target_distance": target_distance,    # |guided_mean - target|
        "corr":            corr,
        "rmse":            rmse,
        "jitter":          jitter,
        "foot_skate":      foot_skate,
    }


def compute_foot_skate(q, ground_thresh=0.05, slide_thresh=0.025):
    if q.shape[1] < 22:
        return 0.0
    L_FOOT, R_FOOT = 10, 11
    feet = q[:, [L_FOOT, R_FOOT], :]
    on_ground = feet[:, :, 1] < ground_thresh
    motion = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1)
    sliding = motion > slide_thresh
    skate = on_ground[1:] & sliding
    return float(skate.any(axis=1).mean())


def parse_inference_time(log_path):
    if not log_path.exists():
        return float("nan")
    txt = log_path.read_text(errors="ignore")
    m = re.search(r"sampling.*?(\d+\.?\d*)\s*(?:s|sec|seconds)", txt, re.I)
    if m:
        return float(m.group(1))
    return float("nan")


def classify_shape(m):
    if m["hit_rate_loose"] < 0.05:
        return "⚠ 推力不足"
    if m["corr"] < 0:
        return "❌ 时间结构反向"
    if m["hit_rate_loose"] > 0.9 and m["corr"] < 0.15:
        return "❌ 塌平（静态姿势）"
    if m["jitter"] > 5.0:
        return "❌ 抖动严重"
    # 新增：双边检测过度推力
    if m["target_distance"] > 5.0:                    # mean 距目标超过 5°
        return "❌ 过度推力"
    if m["hit_rate"] > 0.4 and m["corr"] > 0.3:
        return "✅ 平移+保结构"
    return "○ 中间状态"


def composite_score(m):
    """
    v3 评分函数：
        硬约束（任一触发 → 0 分）：
            - hit_rate_loose < 0.05
            - corr < 0
            - hit_rate_loose > 0.9 & corr < 0.15
            - jitter > 5.0
            - target_distance > 5.0       ← 新增：平均距目标超过 5° 算过度推力
        
        通过硬约束后：
            score = hit_rate · corr / (1 + 5·RMSE + 0.2·jitter + skate)
        
        注意 hit_rate 现在是双边区间（更严格）。
    """
    if m["hit_rate_loose"] < 0.05:                            return 0.0
    if m["corr"] < 0:                                         return 0.0
    if m["hit_rate_loose"] > 0.9 and m["corr"] < 0.15:        return 0.0
    if m["jitter"] > 5.0:                                     return 0.0
    if m["target_distance"] > 5.0:                            return 0.0

    return (m["hit_rate"] * m["corr"] /
            (1.0 + 5.0 * m["rmse"] + 0.2 * m["jitter"] + m["foot_skate"]))


def write_markdown_report(rows, posture, path):
    lines = []
    lines.append(f"# Ablation Results — {posture}\n")
    lines.append("| Rank | Variant | Δ (°) | Hit (band) | Hit (loose) | TgtDist | Corr | RMSE | Jit | Skate | Shape | Score |")
    lines.append("|------|---------|-------|-----------|-------------|---------|------|------|-----|-------|-------|-------|")
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['delta']:+.2f} | "
            f"{r['hit_rate']*100:.1f}% | {r['hit_rate_loose']*100:.1f}% | "
            f"{r['target_distance']:.1f}° | "
            f"{r['corr']:+.3f} | {r['rmse']:.3f} | {r['jitter']:.2f} | "
            f"{r['foot_skate']*100:.1f}% | {r['shape']} | **{r['score']:.4f}** |"
        )
    if rows:
        lines.append("\n---\n")
        lines.append(f"**Best**: `{rows[0]['variant']}` (score = {rows[0]['score']:.4f})\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_dir", type=str)
    parser.add_argument("--posture", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.ablation_dir)
    if not base.exists():
        print(f"ERROR: {base} 不存在")
        sys.exit(1)

    rows = []
    for variant_dir in sorted(base.iterdir()):
        if not variant_dir.is_dir():
            continue
        npy = variant_dir / "comparison.npy"
        if not npy.exists():
            continue

        data = np.load(npy, allow_pickle=True).item()
        posture = args.posture or data.get("posture_instructions", "骨盆前倾")
        if isinstance(posture, list):
            posture = posture[0]

        try:
            m = compute_metrics(npy, posture)
            m["variant"] = variant_dir.name
            m["time_s"] = parse_inference_time(variant_dir / "pipeline.log")
            m["shape"] = classify_shape(m)
            m["score"] = composite_score(m)
            rows.append(m)
        except Exception as e:
            print(f"  ERROR on {variant_dir.name}: {e}")
            continue

    if not rows:
        print("No variants evaluated.")
        return

    rows.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 140)
    print(f"  Ablation v3 (区间 hit_rate + 过度推力惩罚) — {base.name}")
    print(f"  Posture: {posture}")
    print("=" * 140)

    header = (f"{'rank':<5}{'variant':<28}"
              f"{'Δ':>8}{'hit(band)':>11}{'hit(loose)':>12}{'tgt_dist':>10}"
              f"{'corr':>9}{'RMSE':>8}{'jit':>7}{'skate':>8}"
              f"{'shape':<26}{'score':>9}")
    print(header)
    print("-" * 140)

    for i, r in enumerate(rows, start=1):
        print(f"{i:<5}{r['variant']:<28}"
              f"{r['delta']:+7.2f} "
              f"{r['hit_rate']*100:8.1f}%   "
              f"{r['hit_rate_loose']*100:8.1f}%   "
              f"{r['target_distance']:7.1f}°  "
              f"{r['corr']:+6.3f}  "
              f"{r['rmse']:6.3f}  "
              f"{r['jitter']:5.2f}  "
              f"{r['foot_skate']*100:5.1f}%  "
              f"{r['shape']:<26}"
              f"{r['score']:7.4f}")

    print("=" * 140)

    valid_rows = [r for r in rows if r["score"] > 0]
    if valid_rows:
        print(f"\n🏆 Best: {valid_rows[0]['variant']} (score={valid_rows[0]['score']:.4f})")
        print(f"    Δ={valid_rows[0]['delta']:+.2f}°  落带={valid_rows[0]['hit_rate']*100:.1f}%  "
              f"corr={valid_rows[0]['corr']:+.3f}  距目标={valid_rows[0]['target_distance']:.1f}°")
    else:
        print("\n❌ 没有 variant 通过硬约束。")

    print("\n硬约束惩罚（任一触发 → score=0）：")
    print("  ⚠ hit_loose < 5%      : guidance 没起作用")
    print("  ❌ corr < 0            : 时间结构被反向破坏")
    print("  ❌ hit>90% & corr<.15  : 塌成静态姿势")
    print("  ❌ jitter > 5°         : 剧烈抖动")
    print("  ❌ target_distance > 5°: 过度推力（平均角度距目标 >5°）")

    out_json = base / "ablation_results_v3.json"
    json.dump(rows, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {out_json}")

    out_md = base / "ablation_results_v3.md"
    write_markdown_report(rows, posture, out_md)
    print(f"💾 Saved: {out_md}")


if __name__ == "__main__":
    main()