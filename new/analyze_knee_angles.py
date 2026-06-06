"""
new/analyze_knee_angles.py

全局膝角分布统计：扫描 HumanML3D 训练集，提取行走/站立帧的膝角分布。

角度定义（修正版）：
    使用 three_point_angle(hip, knee, ankle)，即 hip-knee-ankle 三点夹角：
        180°  = 腿完全伸直（膝完全伸展）
        < 180° = 膝弯曲（屈曲）
        物理上限 = 180°（三点角无法超过此值）

    临床超伸（膝向后弯超过 180°）在此定义下会使角度从 180° 回退，
    与普通屈曲无法区分——但这恰恰是训练数据诊断的正确用途：
    我们关心的是 MDM 是否学到"腿接近伸直"的分布。

直立帧过滤（关键修复）：
    signed_knee_angle 在非直立姿势（坐、躺、踢腿等）会产生 300°+ 的错误值。
    本脚本先过滤出"直立行走"帧（hip.y > knee.y > ankle.y，高度差 > 0.5m），
    再计算角度——这样才能反映行走时的膝关节分布。

路线 A 判据：
    若直立帧 P99(three_point_angle) ≥ 175°  →  条件 OOD（行走时膝接近伸直）
        → MDM 分布里有密度，组合式扩散（路线 A）可借力
    若直立帧 P99 < 172°                       →  全局 OOD（训练集行走膝角均较小）
        → 直接跳路线 B（IK + SDEdit）

用法：
    python -m new.analyze_knee_angles --data-dir /path/to/HumanML3D
    python -m new.analyze_knee_angles --data-dir /path/to/HumanML3D --split train
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.angle_ops import signed_knee_angle
from posture_guidance.joint_indices import get_joint_idx
from posture_guidance.phase_detector import PhaseDetector

EPS = 1e-7


# ──────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────

def load_xyz(npy_path: Path):
    """加载 new_joints .npy，返回 (T, 22, 3) float tensor 或 None。"""
    try:
        arr = np.load(npy_path)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[1] >= 22:
            return torch.from_numpy(arr[:, :22, :]).float()
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# 直立帧过滤
# ──────────────────────────────────────────────────────────

def upright_mask(q: torch.Tensor, min_height: float = 0.5) -> torch.Tensor:
    """
    q: (T, 22, 3)，y 轴为垂直方向（HumanML3D 约定）。
    返回 bool mask (T,)：
      - 髋中心 y > 膝中心 y > 踝中心 y （竖直层级正确）
      - 髋踝高度差 > min_height m    （站立高度足够，过滤躺/坐）
    """
    hip_y   = (q[:, get_joint_idx("left_hip"),   1] + q[:, get_joint_idx("right_hip"),   1]) / 2
    knee_y  = (q[:, get_joint_idx("left_knee"),  1] + q[:, get_joint_idx("right_knee"),  1]) / 2
    ankle_y = (q[:, get_joint_idx("left_ankle"), 1] + q[:, get_joint_idx("right_ankle"), 1]) / 2

    return (hip_y > knee_y) & (knee_y > ankle_y) & ((hip_y - ankle_y) > min_height)


# ──────────────────────────────────────────────────────────
# 角度计算
# ──────────────────────────────────────────────────────────

def three_point_angle_np(q: torch.Tensor, ja: str, jb: str, jc: str) -> np.ndarray:
    """
    jb 为顶点，返回 (T,) ndarray（度，范围 [0°, 180°]）。
    180° = 完全伸直，< 180° = 弯曲。
    """
    a = q[:, get_joint_idx(ja), :]
    b = q[:, get_joint_idx(jb), :]
    c = q[:, get_joint_idx(jc), :]
    ba = F.normalize(a - b, dim=-1, eps=EPS)
    bc = F.normalize(c - b, dim=-1, eps=EPS)
    cos_a = (ba * bc).sum(-1).clamp(-1 + EPS, 1 - EPS)
    return torch.acos(cos_a).numpy() * (180.0 / math.pi)


def signed_knee_np(q: torch.Tensor, side: str) -> np.ndarray:
    """返回 (T,) ndarray（度）。直立帧下可检测轻微超伸（>180°）。"""
    with torch.no_grad():
        rad = signed_knee_angle(q, side=side)
    return rad.numpy() * (180.0 / math.pi)


# ──────────────────────────────────────────────────────────
# 报告格式
# ──────────────────────────────────────────────────────────

def print_block(label: str, arr: np.ndarray, thresholds: list):
    print(f"\n  {'─' * 58}")
    print(f"  {label}   帧数: {len(arr):,}")
    print(f"  {'─' * 58}")
    print(f"    均值 ± std :  {arr.mean():.2f}° ± {arr.std():.2f}°")
    print(f"    Min / Max  :  {arr.min():.2f}° / {arr.max():.2f}°")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"    P{p:5.1f}      :  {np.percentile(arr, p):.2f}°")
    print()
    for thr in thresholds:
        count = int((arr > thr).sum())
        pct = 100.0 * count / max(len(arr), 1)
        note = "  ← 接近伸直" if thr >= 175 else ""
        print(f"    > {thr:5.1f}° :  {count:8,} 帧  ({pct:.4f}%){note}")


# ──────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HumanML3D 全局膝角分布统计（路线 A go/no-go 判据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True,
                        help="HumanML3D 根目录（含 new_joints/ 子目录）")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--thresholds", default="165,170,173,175,177,179",
                        help="统计超过这些角度的帧数（度），逗号分隔")
    parser.add_argument("--min-height", type=float, default=0.5,
                        help="直立过滤：髋踝高度差最低阈值（米，默认 0.5）")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="只处理前 N 个 clip（0=全部，调试用）")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    joints_dir = data_dir / "new_joints"

    if not joints_dir.exists():
        print(f"\n❌ 找不到 {joints_dir}")
        print("   请确认 --data-dir 指向 HumanML3D 根目录，且已解压 new_joints/")
        sys.exit(1)

    # clip 列表
    if args.split == "all":
        npy_files = sorted(joints_dir.glob("*.npy"))
    else:
        split_file = data_dir / f"{args.split}.txt"
        if split_file.exists():
            ids = [l.strip() for l in split_file.read_text().split("\n") if l.strip()]
            npy_files = [joints_dir / f"{cid}.npy" for cid in ids if (joints_dir / f"{cid}.npy").exists()]
        else:
            print(f"⚠  找不到 {split_file}，扫描全部 npy")
            npy_files = sorted(joints_dir.glob("*.npy"))

    if args.max_clips > 0:
        npy_files = npy_files[:args.max_clips]

    if not npy_files:
        print("❌ 未找到任何 .npy 文件")
        sys.exit(1)

    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"\n扫描 {len(npy_files)} 个 clip（split={args.split}）...\n")

    # 累积数据
    all_left_raw     = []   # three_point_angle，全部帧
    all_right_raw    = []
    all_left_up      = []   # three_point_angle，直立帧
    all_right_up     = []
    all_left_signed  = []   # signed_knee_angle，直立帧（检测轻微超伸）
    all_right_signed = []
    # ★ 新增：stance-phase 条件分布（锁死 Class 3）
    all_left_stance  = []   # three_point_angle，stance 相帧（仅直立 + stance）
    all_right_stance = []

    # ★ 初始化 PhaseDetector
    detector = PhaseDetector(height_thresh=0.05, vel_thresh=0.10, fps=20)

    skipped = 0
    total_frames  = 0
    upright_frames = 0
    report_every  = max(1, len(npy_files) // 10)

    for i, npy_path in enumerate(npy_files):
        if i > 0 and i % report_every == 0:
            pct = 100 * i // len(npy_files)
            p99_up = (np.percentile(all_left_up, 99) if all_left_up else 0.0)
            print(f"  [{i:5d}/{len(npy_files)}]  {pct}%  "
                  f"直立帧P99={p99_up:.1f}°  "
                  f"直立帧比例={100*upright_frames/max(total_frames,1):.0f}%")

        q = load_xyz(npy_path)
        if q is None:
            skipped += 1
            continue

        T = q.shape[0]
        total_frames += T

        # 全帧无符号角度
        left_raw  = three_point_angle_np(q, "left_hip",  "left_knee",  "left_ankle")
        right_raw = three_point_angle_np(q, "right_hip", "right_knee", "right_ankle")
        all_left_raw.extend(left_raw.tolist())
        all_right_raw.extend(right_raw.tolist())

        # 直立帧过滤
        mask = upright_mask(q, min_height=args.min_height).numpy()
        n_up = int(mask.sum())
        upright_frames += n_up

        if n_up > 0:
            all_left_up.extend(left_raw[mask].tolist())
            all_right_up.extend(right_raw[mask].tolist())

            # signed_knee 只在直立帧上算，误判率大幅降低
            left_sg  = signed_knee_np(q, "left")[mask]
            right_sg = signed_knee_np(q, "right")[mask]
            all_left_signed.extend(left_sg.tolist())
            all_right_signed.extend(right_sg.tolist())

            # ★ Stance-phase filter: 仅保留 stance 概率 > 0.5 的直立帧
            try:
                stance_mask = detector.get_stance_mask(q)  # (T, 2), soft [0,1]
                # 硬阈值：stance 概率 > 0.5
                left_stance = (stance_mask[:, 0] > 0.5).numpy()
                right_stance = (stance_mask[:, 1] > 0.5).numpy()
                # 与直立 mask 取交集
                left_stance_up = left_stance & mask
                right_stance_up = right_stance & mask
                if left_stance_up.sum() > 0:
                    all_left_stance.extend(left_raw[left_stance_up].tolist())
                if right_stance_up.sum() > 0:
                    all_right_stance.extend(right_raw[right_stance_up].tolist())
            except Exception:
                pass  # phase detector 可能在极端姿势失败，静默跳过

    print(f"\n完成。有效 clip: {len(npy_files)-skipped}/{len(npy_files)}")
    print(f"总帧数: {total_frames:,}   直立帧: {upright_frames:,} "
          f"({100*upright_frames/max(total_frames,1):.1f}%)\n")

    # ── 报告 ───────────────────────────────────────────────
    print("=" * 62)
    print("  三点角（hip-knee-ankle），180° = 完全伸直，< 180° = 屈曲")
    print("=" * 62)

    print("\n【全部帧（含坐/躺/踢腿等，仅供参考）】")
    print_block("左膝 (全部)", np.array(all_left_raw,  dtype=np.float32), thresholds)
    print_block("右膝 (全部)", np.array(all_right_raw, dtype=np.float32), thresholds)

    if all_left_up:
        print("\n【直立帧（站立/行走姿势，路线 A 判据依据此处）】")
        arr_l = np.array(all_left_up,  dtype=np.float32)
        arr_r = np.array(all_right_up, dtype=np.float32)
        print_block("左膝 (直立)", arr_l, thresholds)
        print_block("右膝 (直立)", arr_r, thresholds)

        # signed 在直立帧下的表现
        sg_l = np.array(all_left_signed,  dtype=np.float32)
        sg_r = np.array(all_right_signed, dtype=np.float32)
        n_hyperext = int(((sg_l > 180) | (sg_r > 180)).sum())
        print(f"\n  signed_knee_angle > 180°（直立帧中的超伸帧）: "
              f"{n_hyperext:,} 帧 ({100*n_hyperext/max(len(sg_l),1):.3f}%)")
        if n_hyperext > 0:
            max_sg = float(max(sg_l.max(), sg_r.max()))
            p999_sg = float(np.percentile(np.concatenate([sg_l, sg_r]), 99.9))
            print(f"  signed 最大值: {max_sg:.1f}°   P99.9: {p999_sg:.1f}°")

    # ── ★ Stance 相条件分布（锁死 Class 3）──────────────────
    if all_left_stance or all_right_stance:
        print("\n" + "=" * 62)
        print("  站立相膝角分布（three_point_angle, stance-phase conditioned）")
        print("  用于 OOD 三分类 Class 3 诊断")
        print("=" * 62)

        stance_all = np.array(all_left_stance + all_right_stance, dtype=np.float32)
        stance_l   = np.array(all_left_stance,  dtype=np.float32) if all_left_stance else None
        stance_r   = np.array(all_right_stance, dtype=np.float32) if all_right_stance else None

        stance_p50 = float(np.percentile(stance_all, 50))
        stance_p10 = float(np.percentile(stance_all, 10))
        stance_p5  = float(np.percentile(stance_all, 5))
        stance_p1  = float(np.percentile(stance_all, 1))
        stance_p99 = float(np.percentile(stance_all, 99))
        stance_max = float(stance_all.max())
        stance_min = float(stance_all.min())

        print(f"\n  Stance 相帧数: {len(stance_all):,}")
        print(f"  P50={stance_p50:.1f}°  P10={stance_p10:.1f}°  P5={stance_p5:.1f}°  P1={stance_p1:.1f}°")
        print(f"  P99={stance_p99:.1f}°  Max={stance_max:.1f}°  Min={stance_min:.1f}°")

        # Class 3 check: 膝弯曲 125° 在 stance 相是否零密度
        target_knee_flexion = 125.0
        count_below_125 = int((stance_all < target_knee_flexion).sum())
        pct_below_125 = 100.0 * count_below_125 / max(len(stance_all), 1)
        count_below_130 = int((stance_all < 130).sum())
        pct_below_130 = 100.0 * count_below_130 / max(len(stance_all), 1)

        print(f"\n  膝弯曲 125° 条件 OOD 诊断：")
        print(f"    Stance 相 < 125°:  {count_below_125:,} 帧 ({pct_below_125:.3f}%)")
        print(f"    Stance 相 < 130°:  {count_below_130:,} 帧 ({pct_below_130:.3f}%)")

        if pct_below_125 < 0.1:
            print(f"  [Class 3] 条件 OOD 坐实：stance 相膝角 <125° 的帧占比 <0.1%")
            print(f"    → Stance 相膝角集中在 {stance_p10:.0f}-{stance_p99:.0f}°")
            print(f"    → 膝弯曲 125°+stance 是 (角度, 相位) 联合零密度")
            print(f"    → Guidance 将失败（相位约束与角度约束冲突）")
        elif pct_below_125 < 1.0:
            print(f"  [Class 3] 条件 OOD 高度可疑：stance 相膝角 <125° 的帧占比仅 {pct_below_125:.2f}%")
        elif pct_below_125 < 5.0:
            print(f"  [Class 2] 密度尾部但极稀疏：stance 相 <125° 的帧 {pct_below_125:.2f}%")
        else:
            print(f"  [Class 0/2] 有条件密度：stance 相 <125° 的帧 {pct_below_125:.1f}%")

        # Stance-conditioned histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = 60
        ax.hist(stance_all, bins=bins, density=True, alpha=0.7, color='darkred',
                edgecolor='white', linewidth=0.5)
        ax.axvline(target_knee_flexion, color='orange', linestyle='--', linewidth=2,
                   label=f'Knee flexion target {target_knee_flexion}°')
        ax.axvline(stance_p50, color='green', linestyle=':', linewidth=1.5,
                   label=f'Stance P50={stance_p50:.0f}°')
        ax.axvline(stance_p10, color='blue', linestyle=':', linewidth=1.5,
                   label=f'Stance P10={stance_p10:.0f}°')
        ax.axvline(stance_p1, color='purple', linestyle=':', linewidth=1.5,
                   label=f'Stance P1={stance_p1:.0f}°')
        ax.set_xlabel('Knee Angle (three-point, degrees)')
        ax.set_ylabel('Density')
        ax.set_title('Stance-Phase Knee Angle Distribution (three_point_angle)')
        ax.legend(fontsize=8)
        fig.tight_layout()

        out_dir = Path("new_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "knee_angle_stance_distribution.png", dpi=150)
        print(f"\n  Stance 相膝角直方图已保存: new_results/knee_angle_stance_distribution.png")

    # ── 路线 A 判断 ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  路线 A 可行性判断（基于直立帧三点角）")
    print("=" * 62)

    if not all_left_up:
        print("  ❌ 无直立帧，无法判断。检查 --min-height 参数。")
    else:
        arr_up = np.concatenate([
            np.array(all_left_up, dtype=np.float32),
            np.array(all_right_up, dtype=np.float32),
        ])
        p90  = float(np.percentile(arr_up, 90))
        p99  = float(np.percentile(arr_up, 99))
        p999 = float(np.percentile(arr_up, 99.9))
        mx   = float(arr_up.max())

        print(f"\n  直立帧 P90={p90:.2f}°  P99={p99:.2f}°  P99.9={p999:.2f}°  Max={mx:.2f}°\n")

        if p99 >= 175.0:
            print("  ✅ 条件 OOD（行走时膝接近伸直有实质密度）")
            print(f"     P99={p99:.1f}° 已接近 180° 边界")
            print("     组合式扩散（路线 A）理论上可行：text_ood 可把支撑集推向此区域")
            print("     → 下一步：运行 new/retrieve_ood_prompts.py 找 text_ood 候选")
        elif p99 >= 170.0:
            print("  ⚠  边界区域：P99 在 170-175° 之间")
            print("     MDM 有行走时近伸直腿的密度，但距 180° 仍有距离")
            print("     → 路线 A 可尝试（用较大的 w2=1.0-2.0），同时准备路线 B 保底")
        else:
            print(f"  ❌ 全局 OOD：直立帧 P99={p99:.1f}°，训练集行走姿势膝角均较小")
            print("     组合式扩散（路线 A）缺乏密度支撑")
            print("     → 直接跳路线 B（IK + SDEdit）或路线 E（LoRA）")

        # 超伸检测补充说明
        sg_up = np.concatenate([
            np.array(all_left_signed,  dtype=np.float32),
            np.array(all_right_signed, dtype=np.float32),
        ])
        n_hyp = int((sg_up > 180).sum())
        if n_hyp > 0:
            print(f"\n  补充：直立帧里检测到 {n_hyp:,} 帧 signed_knee > 180°（轻微超伸）")
            print("       说明训练集里极少量真实超伸存在，路线 A 支撑更强")
        else:
            print(f"\n  补充：直立帧 signed_knee_angle 均 ≤ 180°")
            print("       训练集行走姿势接近伸直但不超伸，目标 190° 仍需 guidance 推力")

    print("=" * 62)


if __name__ == "__main__":
    main()
