"""
scripts/quantitative_compare.py

量化对比 baseline 和 guided 的差异：
    - 关键关节角的均值 / 中位数 / 标准差
    - 时间曲线相关性（baseline 和 guided 的趋势是否一致）
    - guidance 命中率（多少帧达到了目标值）
    - 整体动作差异（关节坐标 RMSE）

调用：
    python -m scripts.quantitative_compare ./output/comparison_run/comparison.npy
    python -m scripts.quantitative_compare ./output/comparison_run/comparison.npy --sample_idx 0
"""

import os
import sys
import math
import argparse
import numpy as np
import torch

class Logger(object):
    """
    一个简单的日志记录器，用于将 print 的输出同时重定向到终端和文件中。
    """
    def __init__(self, filename="report.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保实时写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from posture_guidance.angle_ops import (
    pelvis_tilt_angle,
    signed_knee_angle,
    spine_kyphosis_angle,
    head_forward_offset,
    three_point_angle,
)
from posture_guidance.registry import POSTURE_REGISTRY, resolve_instruction
from posture_guidance.joint_indices import JOINT_IDX


# =========================================================
# 角度提取
# =========================================================

# 把每个 LossSpec 对应的 angle_fn 连同它的目标值整理出来
def get_metric_specs(posture_instructions):
    """
    根据用户的指令列表，返回需要在量化分析里计算的 (名字, 函数, 目标度数, 单位) 列表。
    """
    specs = []
    for inst in posture_instructions:
        spec_names = resolve_instruction(inst)
        for sn in spec_names:
            s = POSTURE_REGISTRY[sn]
            specs.append({
                "name":        s.name,
                "angle_fn":    s.angle_fn,
                "kwargs":      s.angle_fn_kwargs,
                "target":      s.target_deg,
                "tolerance":   s.tolerance_deg,
                "direction":   s.direction,
                "unit":        s.unit,
            })
    return specs


def xyz_to_qtensor(xyz_np):
    """
    把保存的 xyz (B, J, 3, T) 转成 angle_ops 需要的 (B, T, J, 3) 张量。
    """
    # (B, J, 3, T) → (B, T, J, 3)
    q = torch.from_numpy(xyz_np).permute(0, 3, 1, 2).float()
    return q


def compute_angle_series(q, spec):
    """
    在整个时间序列上计算单个 spec 的角度。
    返回 numpy array (B, T)，单位与 spec.unit 一致（deg 或 米）。
    """
    angle = spec["angle_fn"](q, **spec["kwargs"])  # (B, T)

    if spec["unit"] == "deg":
        angle_np = (angle * 180.0 / math.pi).cpu().numpy()
    else:
        angle_np = angle.cpu().numpy()
    return angle_np


# =========================================================
# 指标计算
# =========================================================

def compute_basic_stats(series, label):
    """对一个 (B, T) 的时间序列计算基础统计量"""
    flat = series.flatten()
    return {
        f"{label}_mean":   float(np.mean(flat)),
        f"{label}_median": float(np.median(flat)),
        f"{label}_std":    float(np.std(flat)),
        f"{label}_min":    float(np.min(flat)),
        f"{label}_max":    float(np.max(flat)),
    }


def compute_hit_rate(series, target, tolerance, direction):
    """
    guidance 命中率：多少帧的角度满足了约束。

    direction:
        greater_than: angle >= target - tolerance 算命中
        less_than:    angle <= target + tolerance 算命中
        equal:        |angle - target| <= tolerance 算命中
    """
    flat = series.flatten()
    if direction == "greater_than":
        hits = (flat >= (target - tolerance)).sum()
    elif direction == "less_than":
        hits = (flat <= (target + tolerance)).sum()
    elif direction == "equal":
        hits = (np.abs(flat - target) <= tolerance).sum()
    else:
        return 0.0
    return float(hits / max(1, len(flat)))


def compute_correlation(s1, s2):
    """两条时间曲线的 Pearson 相关系数（按 batch 平均）"""
    correlations = []
    for b in range(s1.shape[0]):
        a, c = s1[b], s2[b]
        if a.std() < 1e-8 or c.std() < 1e-8:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(a, c)[0, 1]))
    return float(np.mean(correlations))


def compute_xyz_rmse(xyz_base, xyz_guided):
    """整体动作的关节坐标 RMSE，反映动作整体改变了多少"""
    diff = xyz_guided - xyz_base
    rmse_per_joint = np.sqrt(np.mean(diff ** 2, axis=(0, 2, 3)))  # (J,)
    rmse_overall   = float(np.sqrt(np.mean(diff ** 2)))
    return rmse_overall, rmse_per_joint


# =========================================================
# 报告打印
# =========================================================

def print_section_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_metric_row(label, value, unit="", width=40):
    print(f"  {label:<{width}} {value:>10.3f} {unit}")


def print_table_row(name, base_val, guided_val, delta, unit, width=24):
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    print(f"  {name:<{width}} {base_val:>8.2f}{unit}  →  "
          f"{guided_val:>8.2f}{unit}  ({arrow}{abs(delta):>6.2f}{unit})")


# =========================================================
# 主流程
# =========================================================

def analyze(npy_path, sample_idx=None):
    print(f"Loading: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()

    # 元数据
    text_prompt          = data.get("text_prompt", "<unknown>")
    posture_instructions = data.get("posture_instructions", [])
    seed                 = data.get("seed", "<unknown>")

    print_section_header("Experiment metadata")
    print(f"  text_prompt:          {text_prompt}")
    print(f"  posture_instructions: {posture_instructions}")
    print(f"  seed:                 {seed}")

    # 取数据
    xyz_base   = data["motion_xyz"]              # (B, J, 3, T)
    xyz_guided = data["motion_xyz_guided"]

    if sample_idx is not None:
        xyz_base   = xyz_base[sample_idx:sample_idx + 1]
        xyz_guided = xyz_guided[sample_idx:sample_idx + 1]
        print(f"\n  [Filter] only sample index {sample_idx}")

    q_base   = xyz_to_qtensor(xyz_base)
    q_guided = xyz_to_qtensor(xyz_guided)

    print(f"\n  shape: q_base={tuple(q_base.shape)}, "
          f"q_guided={tuple(q_guided.shape)}")

    # ======================================================
    # 1. 关键体态指标对比
    # ======================================================
    if not posture_instructions:
        print("\n[Warning] 无 posture_instructions，跳过角度对比。")
        specs = []
    else:
        specs = get_metric_specs(posture_instructions)

    print_section_header("Posture angle comparison (mean over all frames)")
    print(f"  {'metric':<24} {'baseline':>10}     {'guided':>10}     {'delta':>10}")
    print(f"  {'-' * 64}")

    angle_results = {}
    for spec in specs:
        unit_label = "°" if spec["unit"] == "deg" else "m"

        s_base   = compute_angle_series(q_base,   spec)  # (B, T)
        s_guided = compute_angle_series(q_guided, spec)

        m_base   = float(np.mean(s_base))
        m_guided = float(np.mean(s_guided))
        delta    = m_guided - m_base

        print_table_row(spec["name"], m_base, m_guided, delta, unit_label)

        angle_results[spec["name"]] = {
            "baseline":  s_base,
            "guided":    s_guided,
            "spec":      spec,
        }

    # ======================================================
    # 2. 命中率（guidance 是否真的把角度推到了目标）
    # ======================================================
    print_section_header("Hit rate (frames meeting target ± tolerance)")
    print(f"  {'metric':<24} {'baseline':>10}     {'guided':>10}     {'target':>10}")
    print(f"  {'-' * 64}")

    for name, ar in angle_results.items():
        spec = ar["spec"]
        hr_base = compute_hit_rate(
            ar["baseline"], spec["target"], spec["tolerance"], spec["direction"]
        )
        hr_guided = compute_hit_rate(
            ar["guided"], spec["target"], spec["tolerance"], spec["direction"]
        )
        unit_label = "°" if spec["unit"] == "deg" else "m"
        target_str = f"{spec['target']:.1f}{unit_label}"
        print(f"  {name:<24} {hr_base*100:>8.1f}%     {hr_guided*100:>8.1f}%     "
              f"{target_str:>10}")

    # ======================================================
    # 3. 时间曲线相关性
    # ======================================================
    print_section_header("Temporal correlation (baseline vs guided)")
    print(f"  期望：相关系数高（>0.6）说明 guidance 只是把角度抬高了一个 offset，")
    print(f"        动作的时间结构基本保持。低相关说明 guidance 改变了动作模式。")
    print()

    for name, ar in angle_results.items():
        corr = compute_correlation(ar["baseline"], ar["guided"])
        print(f"  {name:<24} corr = {corr:>+6.3f}")

    # ======================================================
    # 4. 整体动作改变量
    # ======================================================
    print_section_header("Overall motion difference (XYZ RMSE)")
    rmse_overall, rmse_per_joint = compute_xyz_rmse(xyz_base, xyz_guided)
    print(f"  overall RMSE: {rmse_overall:.4f} m")
    print(f"\n  Per-joint RMSE (top 10 most-changed):")
    top_idx = np.argsort(rmse_per_joint)[::-1][:10]
    inv_joint_idx = {v: k for k, v in JOINT_IDX.items()}
    for i in top_idx:
        joint_name = inv_joint_idx.get(int(i), f"joint_{i}")
        print(f"    {joint_name:<20} {rmse_per_joint[i]:.4f} m")

    # ======================================================
    # 5. 健康度判断（自动诊断）
    # ======================================================
    print_section_header("Diagnostic")

    diagnostics = []
    for name, ar in angle_results.items():
        spec    = ar["spec"]
        m_base  = float(np.mean(ar["baseline"]))
        m_guided = float(np.mean(ar["guided"]))
        target   = spec["target"]
        delta    = m_guided - m_base

        # 启发式诊断
        if abs(delta) < 0.5:
            diagnostics.append(
                f"  ✗ [{name}] guidance 几乎没有效果 (Δ={delta:.2f})。"
                f"\n      检查：FK 可微性 / lr 是否过小 / spec 单位是否正确"
            )
        elif spec["direction"] == "greater_than" and m_guided < target * 0.6:
            diagnostics.append(
                f"  ⚠ [{name}] guidance 有效果但未达预期 (现={m_guided:.2f}, 目标={target:.2f})。"
                f"\n      建议：增大 base_weight 或 posture_lbfgs_steps"
            )
        elif spec["direction"] == "greater_than" and m_guided > target * 1.5:
            diagnostics.append(
                f"  ⚠ [{name}] guidance 过强 (现={m_guided:.2f}, 目标={target:.2f})。"
                f"\n      建议：减小 base_weight 或 posture_lr"
            )
        else:
            diagnostics.append(
                f"  ✓ [{name}] guidance 正常工作 (Δ={delta:+.2f})"
            )

    if rmse_overall > 0.5:
        diagnostics.append(
            f"  ⚠ overall RMSE 过大 ({rmse_overall:.3f}m)，"
            f"动作可能严重偏离原始语义。建议降低 lr。"
        )
    elif rmse_overall < 0.01:
        diagnostics.append(
            f"  ✗ overall RMSE 过小 ({rmse_overall:.3f}m)，"
            f"guidance 可能完全没生效。"
        )

    for d in diagnostics:
        print(d)

    print()
    return angle_results, rmse_overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str,
                        help="comparison.npy 文件路径")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="只分析某个 sample（默认对所有 sample 取平均）")
    # 允许用户自定义输出报告的路径
    parser.add_argument("--out_file", type=str, default=None,
                        help="输出文本报告的路径。如果不指定，将与 npy 文件同级存放")
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"Error: {args.npy_path} not found.")
        sys.exit(1)

    # 自动推导输出文件路径
    out_file_path = args.out_file
    if out_file_path is None:
        base_name, _ = os.path.splitext(args.npy_path)
        out_file_path = base_name + "_report.txt"

    # 【核心逻辑】劫持标准输出，开启双写模式
    sys.stdout = Logger(out_file_path)

    try:
        # 执行分析流程
        analyze(args.npy_path, sample_idx=args.sample_idx)
        print("\n" + "=" * 70)
        print(f"📄 分析报告已成功保存至: {out_file_path}")
        print("=" * 70)
    finally:
        # 分析结束后，恢复系统的标准输出，并关闭文件
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()