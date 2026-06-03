"""
new/analyze_knee_angles.py

全局膝角分布统计：扫描 HumanML3D 训练集，提取左/右膝角的全局分布。

用途（路线 A go/no-go 判据）：
    若 max_global > 180°（哪怕 0.1% 的帧）→ 条件 OOD
        → MDM 训练集里存在超伸帧，组合式扩散（路线 A）有密度可借
        → 下一步运行 retrieve_ood_prompts.py 找对应的 text_ood

    若 max_global ≤ 175°             → 全局 OOD
        → 整个训练集没有超伸帧，路线 A 无法提供支撑
        → 直接跳路线 B（IK + SDEdit）

用法：
    python -m new.analyze_knee_angles --data-dir /path/to/HumanML3D
    python -m new.analyze_knee_angles --data-dir /path/to/HumanML3D --split train
    python -m new.analyze_knee_angles --data-dir /path/to/HumanML3D --split all --thresholds 170,175,180,185
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.angle_ops import signed_knee_angle


def load_xyz(npy_path: Path):
    """
    加载 HumanML3D new_joints .npy 文件，返回 (T, 22, 3) float tensor。
    不符合格式时返回 None。
    """
    try:
        arr = np.load(npy_path)
        # new_joints 格式：(T, J, 3)，T 可变，J=22
        if arr.ndim == 3 and arr.shape[1] == 22 and arr.shape[2] == 3:
            return torch.from_numpy(arr).float()
        # 有些 clip 存成 (T, 22, 3) 但 J 不一定是 22，宽松匹配
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[1] >= 22:
            return torch.from_numpy(arr[:, :22, :]).float()
        return None
    except Exception:
        return None


def compute_knee_angles_deg(q: torch.Tensor) -> dict:
    """
    q: (T, 22, 3)
    返回 {"left": ndarray(T,), "right": ndarray(T,)}，单位度。
    """
    with torch.no_grad():
        left_rad  = signed_knee_angle(q, side="left")   # (T,) rad
        right_rad = signed_knee_angle(q, side="right")  # (T,) rad
    left_deg  = left_rad.numpy()  * (180.0 / math.pi)
    right_deg = right_rad.numpy() * (180.0 / math.pi)
    return {"left": left_deg, "right": right_deg}


def print_distribution(label: str, vals: list, thresholds: list):
    arr = np.array(vals, dtype=np.float32)
    print(f"  {'─' * 56}")
    print(f"  {label}角分布（度）   帧总数: {len(arr):,}")
    print(f"  {'─' * 56}")
    print(f"    均值 ± std:  {arr.mean():.2f}° ± {arr.std():.2f}°")
    print(f"    Min / Max:   {arr.min():.2f}° / {arr.max():.2f}°")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"    P{p:5.1f}:       {np.percentile(arr, p):.2f}°")
    print()
    for thr in thresholds:
        count = int((arr > thr).sum())
        pct   = 100.0 * count / len(arr)
        flag  = "  ← 超伸" if thr >= 180 else ""
        print(f"    > {thr:5.1f}°:  {count:8,} 帧  ({pct:.4f}%){flag}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="HumanML3D 全局膝角分布统计（路线 A go/no-go 判据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="HumanML3D 根目录（含 new_joints/ 子目录，如 ./dataset/HumanML3D）",
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "val", "test", "all"],
        help="扫描哪个 split（默认 train）",
    )
    parser.add_argument(
        "--thresholds", default="170,175,178,180,182,185,190",
        help="统计超过这些角度的帧数，逗号分隔（度）",
    )
    parser.add_argument(
        "--max-clips", type=int, default=0,
        help="只扫描前 N 个 clip（0 = 全部，调试用）",
    )
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    joints_dir = data_dir / "new_joints"

    if not joints_dir.exists():
        print(f"\n❌ 找不到 {joints_dir}")
        print("   请确认 --data-dir 指向 HumanML3D 根目录，且已解压 new_joints/")
        print("   典型路径：./dataset/HumanML3D")
        sys.exit(1)

    # 确定要扫描的 clip 列表
    if args.split == "all":
        npy_files = sorted(joints_dir.glob("*.npy"))
    else:
        split_file = data_dir / f"{args.split}.txt"
        if not split_file.exists():
            print(f"⚠  找不到 split 文件 {split_file}，改为扫描全部 npy")
            npy_files = sorted(joints_dir.glob("*.npy"))
        else:
            clip_ids  = [l.strip() for l in split_file.read_text().strip().split("\n") if l.strip()]
            npy_files = [joints_dir / f"{cid}.npy" for cid in clip_ids]
            npy_files = [f for f in npy_files if f.exists()]

    if args.max_clips > 0:
        npy_files = npy_files[:args.max_clips]

    if not npy_files:
        print("❌ 未找到任何 .npy 文件，检查路径和 split 文件")
        sys.exit(1)

    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"\n扫描 {len(npy_files)} 个 clip（split={args.split}, data={data_dir}）...\n")

    all_left  = []
    all_right = []
    skipped   = 0
    report_every = max(1, len(npy_files) // 10)

    for i, npy_path in enumerate(npy_files):
        if i > 0 and i % report_every == 0:
            pct = 100 * i // len(npy_files)
            print(f"  [{i:5d}/{len(npy_files)}]  {pct}%  left_max_so_far="
                  f"{max(all_left, default=0):.1f}°  right_max_so_far={max(all_right, default=0):.1f}°")

        q = load_xyz(npy_path)
        if q is None:
            skipped += 1
            continue

        angles = compute_knee_angles_deg(q)
        all_left.extend(angles["left"].tolist())
        all_right.extend(angles["right"].tolist())

    print(f"\n完成。有效 clip: {len(npy_files) - skipped} / {len(npy_files)}，"
          f"跳过 {skipped}（格式不符）。\n")

    print("=" * 60)
    print("  全局膝角分布统计结果")
    print("=" * 60)
    print_distribution("左膝", all_left,  thresholds)
    print_distribution("右膝", all_right, thresholds)

    # 综合判断
    all_vals = np.array(all_left + all_right, dtype=np.float32)
    max_deg  = float(all_vals.max())
    p99      = float(np.percentile(all_vals, 99))
    p999     = float(np.percentile(all_vals, 99.9))

    print("=" * 60)
    print("  路线 A 可行性判断")
    print("=" * 60)
    print(f"  全局最大膝角 = {max_deg:.2f}°")
    print(f"  P99          = {p99:.2f}°")
    print(f"  P99.9        = {p999:.2f}°")
    print()

    if max_deg > 180.0:
        above_180 = int((all_vals > 180).sum())
        above_180_pct = 100.0 * above_180 / len(all_vals)
        print(f"  超伸帧（>180°）: {above_180:,} 帧 ({above_180_pct:.4f}%)")
        print()
        if p99 >= 178.0:
            print("  ✅ 条件 OOD（可能性高）：高膝角在训练集里有实质密度")
            print("     组合式扩散（路线 A）理论上可行")
            print("     → 下一步：运行 new/retrieve_ood_prompts.py 找 text_ood")
        else:
            print("  ⚠  稀疏 OOD：超伸帧极少（P99 仍较低）")
            print("     路线 A 可尝试，但预期效果有限")
            print("     → 建议同时准备路线 B（IK + SDEdit）")
    elif max_deg > 175.0:
        print("  ⚠  接近边界：最大值超过 175° 但未超过 180°")
        print("     路线 A 可能对高膝角走路（而非超伸）有帮助")
        print("     → 同时准备路线 B 作为保底")
    else:
        print("  ❌ 全局 OOD：整个训练集膝角均未超过 175°")
        print("     组合式扩散（路线 A）无法提供密度支撑，任何 text_ood 都无效")
        print("     → 直接跳路线 B（IK + SDEdit）或路线 E（LoRA）")
    print("=" * 60)


if __name__ == "__main__":
    main()
