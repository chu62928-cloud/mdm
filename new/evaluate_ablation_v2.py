"""
new/evaluate_ablation.py  (v2, with collapse penalty)

更新内容（基于第一次 ablation 的观察）：
    1. 塌平惩罚：hit_rate > 0.9 且 corr < 0.15 → score=0
       （第一次 V1 被这个规则正确判为塌平）
    2. 时间结构破坏惩罚：corr < 0 → score=0
       （V1 corr=-0.110 被这个规则判为失败）
    3. 抖动权重加大：jitter > 3° 严重扣分
       （V4 jitter=6.6° 是塌平+震荡的典型 signature）
    4. RMSE 权重不变（≥ 0.20m 严重扣分）
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
    "骨盆前倾":   ("pelvis_tilt_angle",      20.0, 2.0, "greater_than"),
    "骨盆前倾_深蹲": ("pelvis_tilt_angle",    25.0, 3.0, "greater_than"),
    "膝超伸":     ("signed_knee_angle_both", 185.0, 3.0, "greater_than"),
    "膝超伸_左":   ("signed_knee_angle_left", 185.0, 3.0, "greater_than"),
    "膝超伸_右":   ("signed_knee_angle_right", 185.0, 3.0, "greater_than"),
    "驼背":       ("spine_posterior_bulge",  0.08, 0.02, "greater_than"),
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
    fn_name, target_deg, tol_deg, direction = ANGLE_TARGETS[posture]
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

    if direction == "greater_than":
        hit_mask = a_g_deg >= (target_deg - tol_deg)
    elif direction == "less_than":
        hit_mask = a_g_deg <= (target_deg + tol_deg)
    else:
        hit_mask = np.abs(a_g_deg - target_deg) <= tol_deg
    hit_rate = float(hit_mask.mean())

    if a_b_deg.std() > 1e-6 and a_g_deg.std() > 1e-6:
        corr = float(pearsonr(a_b_deg, a_g_deg)[0])
    else:
        corr = 0.0

    diff = q_g.numpy() - q_b.numpy()
    rmse = float(np.sqrt((diff ** 2).mean()))

    jitter = float(np.diff(a_g_deg).std())

    foot_skate = compute_foot_skate(q_g.numpy())

    return {
        "baseline_mean": float(a_b_deg.mean()),
        "guided_mean":   float(a_g_deg.mean()),
        "delta":         delta,
        "hit_rate":      hit_rate,
        "corr":          corr,
        "rmse":          rmse,
        "jitter":        jitter,
        "foot_skate":    foot_skate,
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
    """根据指标判定 variant 的"形状"。"""
    if m["hit_rate"] < 0.05:
        return "⚠ 推力不足"
    if m["corr"] < 0:
        return "❌ 时间结构反向"
    if m["hit_rate"] > 0.9 and m["corr"] < 0.15:
        return "❌ 塌平（静态姿势）"
    if m["jitter"] > 5.0:
        return "❌ 抖动严重"
    if m["delta"] > 35.0:
        return "❌ 过度推力"
    if m["hit_rate"] > 0.4 and m["corr"] > 0.3:
        return "✅ 平移+保结构"
    return "○ 中间状态"


def composite_score(m):
    """
    v2 评分函数：加入塌平/反向/抖动的硬约束惩罚。

    硬约束（任一触发 → 0 分）：
        - hit_rate < 0.05    (推力不足)
        - corr < 0           (时间结构反向)
        - hit_rate>0.9 & corr<0.15  (塌平)
        - jitter > 5.0       (剧烈抖动)
        - delta > 35.0       (过度推力)

    通过硬约束后：
        score = hit_rate · corr / (1 + 5·RMSE + 0.2·jitter + skate)
    """
    if m["hit_rate"] < 0.05:               return 0.0
    if m["corr"] < 0:                      return 0.0
    if m["hit_rate"] > 0.9 and m["corr"] < 0.15:  return 0.0
    if m["jitter"] > 5.0:                  return 0.0
    if m["delta"] > 35.0:                  return 0.0

    return (m["hit_rate"] * m["corr"] /
            (1.0 + 5.0 * m["rmse"] + 0.2 * m["jitter"] + m["foot_skate"]))


def write_markdown_report(rows, posture, path):
    lines = []
    lines.append(f"# Ablation Results — {posture}\n")
    lines.append("| Rank | Variant | Δ (°) | Hit Rate | Corr | RMSE (m) | Jitter (°) | Foot Skate | Shape | Score |")
    lines.append("|------|---------|-------|----------|------|----------|------------|------------|-------|-------|")
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['delta']:+.2f} | {r['hit_rate']*100:.1f}% | "
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

    print("\n" + "=" * 130)
    print(f"  Ablation results (v2 评分，含塌平/反向惩罚) — {base.name}")
    print(f"  Posture: {posture}")
    print("=" * 130)

    header = (f"{'rank':<5}{'variant':<28}"
              f"{'Δ':>8}{'hit':>9}{'corr':>9}{'RMSE':>8}{'jitter':>8}{'skate':>8}"
              f"{'shape':<26}{'score':>9}")
    print(header)
    print("-" * 130)

    for i, r in enumerate(rows, start=1):
        print(f"{i:<5}{r['variant']:<28}"
              f"{r['delta']:+7.2f} "
              f"{r['hit_rate']*100:6.1f}%  "
              f"{r['corr']:+6.3f}  "
              f"{r['rmse']:6.3f}  "
              f"{r['jitter']:6.2f}  "
              f"{r['foot_skate']*100:5.1f}%  "
              f"{r['shape']:<26}"
              f"{r['score']:7.4f}")

    print("=" * 130)

    if rows[0]["score"] > 0:
        print(f"\n🏆 Best: {rows[0]['variant']} (score={rows[0]['score']:.4f}, shape={rows[0]['shape']})")
    else:
        print("\n❌ 没有 variant 通过硬约束。需要重新设计参数。")

    print("\n硬约束惩罚（任一触发 → score=0）：")
    print("  ⚠ hit_rate < 5%     : guidance 没起作用")
    print("  ❌ corr < 0          : 时间结构被反向破坏")
    print("  ❌ hit>90% & corr<.15: 塌成静态姿势")
    print("  ❌ jitter > 5°       : 剧烈抖动")
    print("  ❌ delta > 35°       : 过度推力")

    out_json = base / "ablation_results.json"
    json.dump(rows, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {out_json}")

    out_md = base / "ablation_results.md"
    write_markdown_report(rows, posture, out_md)
    print(f"💾 Saved: {out_md}")


if __name__ == "__main__":
    main()