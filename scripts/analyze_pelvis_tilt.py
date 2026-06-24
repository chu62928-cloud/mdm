"""
scripts/analyze_pelvis_tilt.py

骨盆前倾角分布统计：扫描 HumanML3D 训练集，提取行走/站立帧的骨盆倾斜角分布。

角度定义（pelvis_tilt_angle）：
    骨盆在矢状面内与竖直轴的夹角：
        hip_center → spine1 向量，投影到矢状面，atan2(前后, 上下) 取负
        正值 = 骨盆前倾 (Anterior Pelvic Tilt)
        负值 = 骨盆后倾
        0° = 直立中立位

    正常快走均值约 5-7°，病理性前倾 > 15°，目标 20°。

直立帧过滤：
    复用 analyze_knee_angles.py 的 upright_mask：
    hip.y > knee.y > ankle.y，髋踝高度差 > 0.5m。
    骨盆倾斜只在直立/行走姿势下有意义（坐/躺时 pelvis_tilt 无临床含义）。

用法：
    python -m scripts.analyze_pelvis_tilt --data-dir dataset/HumanML3D --split train
    python -m scripts.analyze_pelvis_tilt --data-dir dataset/HumanML3D --split train --max-clips 100
    python -m scripts.analyze_pelvis_tilt --data-dir dataset/HumanML3D \\
        --baseline-dirs output/n15_v2_dps_s40_last_quarter/
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_guidance.joint_indices import get_joint_idx

EPS = 1e-7

# Walking prompt keywords
WALK_KEYWORDS = ["walk", "walking", "stroll", "stride", "pace", "step",
                 " locomotion", "gait", "march", "hike", "jog", "run",
                 "strolls", "striding", "pacing", "stepping", "marches",
                 "marched", "ambulate", "ambulates", "saunter"]


# ──────────────────────────────────────────────────────────
# Walking-prompt clip filter
# ──────────────────────────────────────────────────────────

def filter_walking_clips(data_dir: Path, clip_ids: list) -> list:
    """返回 text prompt 包含 walking 关键词的 clip ID 列表。"""
    texts_dir = data_dir / "texts"
    walk_ids = []
    for cid in clip_ids:
        txt_file = texts_dir / f"{cid}.txt"
        if not txt_file.exists():
            continue
        try:
            text = txt_file.read_text(encoding='utf-8').lower()
            if any(kw in text for kw in WALK_KEYWORDS):
                walk_ids.append(cid)
        except Exception:
            continue
    return walk_ids


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

def pelvis_tilt_np(q: torch.Tensor) -> np.ndarray:
    """返回 (T,) ndarray（度）。正值=骨盆前倾，负值=后倾。"""
    with torch.no_grad():
        rad = pelvis_tilt_angle(q)
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
        # pelvis_tilt can be negative; check for "greater than" semantics
        count_gt = int((arr > thr).sum())
        pct_gt = 100.0 * count_gt / max(len(arr), 1)
        count_lt = int((arr < -thr).sum())
        pct_lt = 100.0 * count_lt / max(len(arr), 1)
        note_gt = "  ← 超过前倾阈值" if thr >= 15 else ""
        print(f"    > {thr:5.1f}° :  {count_gt:8,} 帧  ({pct_gt:.3f}%){note_gt}")
        if count_lt > 0:
            print(f"    < {-thr:5.1f}° :  {count_lt:8,} 帧  ({pct_lt:.3f}%)  [后倾]")


# ──────────────────────────────────────────────────────────
# Baseline 角度收集（用于 overlay）
# ──────────────────────────────────────────────────────────

def collect_baseline_angles(npy_dirs, angle_fn_np):
    """从生成的 baseline comparison.npy 中提取 per-frame 角度序列（度）。"""
    npy_files = []
    for d in npy_dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".npy":
            npy_files.append(p)
        elif p.is_dir():
            npy_files.extend(sorted(p.rglob("comparison.npy")))

    all_angles = []
    for npy_path in npy_files:
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            xyz = data["motion_xyz"][0]  # (22, 3, T)
            q = torch.from_numpy(xyz).float().permute(2, 0, 1)  # (T, 22, 3)
            angles = angle_fn_np(q)  # (T,) in degrees
            all_angles.extend(angles.tolist())
        except Exception as e:
            print(f"  [WARN] 跳过 {npy_path}: {e}")

    return np.array(all_angles) if all_angles else None


# ──────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HumanML3D 骨盆前倾角分布统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True,
                        help="HumanML3D 根目录（含 new_joints/ 子目录）")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--thresholds", default="10,15,18,20,22,25",
                        help="统计超过这些角度的帧数（度），逗号分隔")
    parser.add_argument("--min-height", type=float, default=0.5,
                        help="直立过滤：髋踝高度差最低阈值（米，默认 0.5）")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="只处理前 N 个 clip（0=全部，调试用）")
    parser.add_argument("--target", type=float, default=20.0,
                        help="目标骨盆前倾角度（度，默认 20°）")
    parser.add_argument("--baseline-dirs", nargs="*", default=None,
                        help="含 comparison.npy 的 baseline 生成目录（用于 overlay 图）")
    parser.add_argument("--walking-only", action="store_true",
                        help="仅分析 walking-prompt 子集（锁死 Class 2）")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    joints_dir = data_dir / "new_joints"

    if not joints_dir.exists():
        print(f"\n[ERROR] 找不到 {joints_dir}")
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
            print(f"[WARN] 找不到 {split_file}，扫描全部 npy")
            npy_files = sorted(joints_dir.glob("*.npy"))

    if args.max_clips > 0:
        npy_files = npy_files[:args.max_clips]

    # Walking-only filter
    if args.walking_only:
        walk_ids = set(filter_walking_clips(data_dir, [p.stem for p in npy_files]))
        n_before = len(npy_files)
        npy_files = [p for p in npy_files if p.stem in walk_ids]
        print(f"[Walking-only] {len(npy_files)}/{n_before} clips after prompt filter")

    if not npy_files:
        print("[ERROR] 未找到任何 .npy 文件")
        sys.exit(1)

    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"\n扫描 {len(npy_files)} 个 clip（split={args.split}"
          f"{', walking-only' if args.walking_only else ''}）...\n")

    # 累积数据
    all_tilt_raw = []   # 全帧
    all_tilt_up  = []   # 直立帧

    skipped = 0
    total_frames   = 0
    upright_frames = 0
    report_every   = max(1, len(npy_files) // 10)

    for i, npy_path in enumerate(npy_files):
        if i > 0 and i % report_every == 0:
            pct = 100 * i // len(npy_files)
            p99_up = (np.percentile(all_tilt_up, 99) if all_tilt_up else 0.0)
            print(f"  [{i:5d}/{len(npy_files)}]  {pct}%  "
                  f"直立帧P99={p99_up:.1f}°  "
                  f"直立帧比例={100*upright_frames/max(total_frames,1):.0f}%")

        q = load_xyz(npy_path)
        if q is None:
            skipped += 1
            continue

        T = q.shape[0]
        total_frames += T

        # per-frame 骨盆倾斜角
        tilt = pelvis_tilt_np(q)
        all_tilt_raw.extend(tilt.tolist())

        # 直立帧过滤
        mask = upright_mask(q, min_height=args.min_height).numpy()
        n_up = int(mask.sum())
        upright_frames += n_up

        if n_up > 0:
            all_tilt_up.extend(tilt[mask].tolist())

    print(f"\n完成。有效 clip: {len(npy_files)-skipped}/{len(npy_files)}")
    print(f"总帧数: {total_frames:,}   直立帧: {upright_frames:,} "
          f"({100*upright_frames/max(total_frames,1):.1f}%)\n")

    # ── 报告 ───────────────────────────────────────────────
    print("=" * 62)
    print("  骨盆前倾角 (pelvis_tilt_angle)，正值=前倾，负值=后倾")
    print("=" * 62)

    print("\n【全部帧（含坐/躺/踢腿等，仅供参考）】")
    arr_raw = np.array(all_tilt_raw, dtype=np.float32)
    print_block("骨盆倾斜 (全部)", arr_raw, thresholds)

    if all_tilt_up:
        print("\n【直立帧（站立/行走姿势）】")
        arr_up = np.array(all_tilt_up, dtype=np.float32)
        print_block("骨盆倾斜 (直立)", arr_up, thresholds)

    # ── 三分类判断 ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  OOD 分类判断（基于直立帧 per-frame 分布）")
    print("=" * 62)

    if not all_tilt_up:
        print("  [ERROR] 无直立帧，无法判断。检查 --min-height 参数。")
    else:
        arr_up = np.array(all_tilt_up, dtype=np.float32)
        p50  = float(np.percentile(arr_up, 50))
        p90  = float(np.percentile(arr_up, 90))
        p95  = float(np.percentile(arr_up, 95))
        p99  = float(np.percentile(arr_up, 99))
        p999 = float(np.percentile(arr_up, 99.9))
        mx   = float(arr_up.max())
        mn   = float(arr_up.min())

        print(f"\n  直立帧 per-frame 统计：")
        print(f"  P50={p50:.2f}°  P90={p90:.2f}°  P95={p95:.2f}°  P99={p99:.2f}°  P99.9={p999:.2f}°")
        print(f"  Min={mn:.2f}°  Max={mx:.2f}°")
        print(f"  目标角度：{args.target:.1f}°\n")

        if args.target <= p99:
            print(f"  [Class 2] 密度尾部 (Density Tail)")
            print(f"    目标 {args.target:.1f}° <= 训练集 P99 ({p99:.1f}°)")
            print(f"    → 目标在训练分布支撑集内，但位于低概率尾部")
            print(f"    → Guidance 可成功将分布推向目标，需要适当推力")
        elif args.target <= mx:
            print(f"  [Class 2] 密度尾部 / 边界 (Density Tail / Borderline)")
            print(f"    目标 {args.target:.1f}° 在 [P99={p99:.1f}°, Max={mx:.1f}°] 范围内")
            print(f"    → 训练集中极少帧达到此值，但在分布支撑集内")
            print(f"    → Guidance 可工作但需要较大推力")
        else:
            print(f"  [Class 1] 几何支撑集外 (Geometric Support Outside)")
            print(f"    目标 {args.target:.1f}° > 训练集最大值 ({mx:.1f}°)")
            print(f"    → 训练分布未覆盖此目标，guidance 将失败")

    print("=" * 62)

    # ── 直方图 ───────────────────────────────────────────────
    if all_tilt_up:
        arr_up = np.array(all_tilt_up, dtype=np.float32)

        # 收集 baseline（如果有）
        baseline_arr = None
        if args.baseline_dirs:
            print(f"\n收集 baseline 角度（从 {len(args.baseline_dirs)} 个目录）...")
            baseline_arr = collect_baseline_angles(args.baseline_dirs, pelvis_tilt_np)
            if baseline_arr is not None and len(baseline_arr) > 0:
                print(f"  baseline 帧数: {len(baseline_arr):,}")
                print(f"  baseline 均值: {baseline_arr.mean():.2f}°  std: {baseline_arr.std():.2f}°")
            else:
                print("  [WARN] 未找到有效的 baseline comparison.npy")

        has_baseline = baseline_arr is not None and len(baseline_arr) > 0

        if has_baseline:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

            # Left: training set only
            ax1.hist(arr_up, bins=80, density=True, alpha=0.7, color='steelblue',
                     edgecolor='white', linewidth=0.5)
            ax1.axvline(args.target, color='red', linestyle='--', linewidth=2,
                        label=f'Target {args.target}°')
            ax1.axvline(p50, color='green', linestyle=':', linewidth=1.5,
                        label=f'P50={p50:.1f}°')
            ax1.axvline(p90, color='orange', linestyle=':', linewidth=1.5,
                        label=f'P90={p90:.1f}°')
            ax1.axvline(p99, color='purple', linestyle=':', linewidth=1.5,
                        label=f'P99={p99:.1f}°')
            ax1.set_xlabel('Pelvis Tilt (degrees)')
            ax1.set_ylabel('Density')
            ax1.set_title('Training Set (Upright Frames)')
            ax1.legend(fontsize=8)

            # Right: overlay with baseline
            ax2.hist(arr_up, bins=80, density=True, alpha=0.5, color='steelblue',
                     edgecolor='white', linewidth=0.5, label='Training (upright)')
            ax2.hist(baseline_arr, bins=min(80, max(10, len(baseline_arr)//50)),
                     density=True, alpha=0.5, color='darkorange',
                     edgecolor='white', linewidth=0.5, label='Generated baseline')
            ax2.axvline(args.target, color='red', linestyle='--', linewidth=2,
                        label=f'Target {args.target}°')
            ax2.axvline(p50, color='green', linestyle=':', linewidth=1.5)
            ax2.axvline(p90, color='orange', linestyle=':', linewidth=1.5)
            ax2.axvline(p99, color='purple', linestyle=':', linewidth=1.5)
            ax2.set_xlabel('Pelvis Tilt (degrees)')
            ax2.set_ylabel('Density')
            ax2.set_title('Training vs Generated Baseline')
            ax2.legend(fontsize=8)

            fig.suptitle('Pelvis Tilt (APT) — Distribution Analysis', fontsize=14,
                         fontweight='bold')
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(arr_up, bins=80, density=True, alpha=0.7, color='steelblue',
                    edgecolor='white', linewidth=0.5)
            ax.axvline(args.target, color='red', linestyle='--', linewidth=2,
                       label=f'Target {args.target}°')
            ax.axvline(p50, color='green', linestyle=':', linewidth=1.5,
                       label=f'P50={p50:.1f}°')
            ax.axvline(p90, color='orange', linestyle=':', linewidth=1.5,
                       label=f'P90={p90:.1f}°')
            ax.axvline(p99, color='purple', linestyle=':', linewidth=1.5,
                       label=f'P99={p99:.1f}°')
            ax.set_xlabel('Pelvis Tilt (degrees)')
            ax.set_ylabel('Density')
            ax.set_title('HumanML3D Training Set — Pelvis Tilt Distribution (Upright Frames)')
            ax.legend()
            fig.tight_layout()

        out_dir = Path("new_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix_parts = []
        if has_baseline:
            suffix_parts.append("overlay")
        else:
            suffix_parts.append("training")
        if args.walking_only:
            suffix_parts.append("walking")
        suffix = "_".join(suffix_parts)
        out_path = out_dir / f"pelvis_tilt_distribution_{suffix}.png"
        fig.savefig(out_path, dpi=150)
        print(f"\n直方图已保存: {out_path}")


if __name__ == "__main__":
    main()
