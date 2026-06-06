"""
new/check_ood_risk.py  —  升级版 (v2)

从二分类 (in/out) 升级为三分类:
  Class 1: 几何支撑集外 (Geometric Support Outside)
  Class 2: 密度尾部 (Density Tail)
  Class 3: 条件 OOD (Conditional OOD)

核心修复: 用 per-FRAME 统计替代 per-file mean 统计。
旧版用 per-file mean 坍缩了 gait cycle 内的帧间方差，
虚假放大了 OOD score (如 APT 20° 被判为 5σ OOD)。

新增 --training-data-dir 模式: 直接从训练集加载 new_joints/ 获取
更准确的 per-frame 百分位统计。

用法:
    # 从 comparison.npy (baseline 生成结果)
    python -m new.check_ood_risk --posture 骨盆前倾 output/n15_v2_dps_s40_last_quarter/

    # 从训练集 + comparison.npy (推荐)
    python -m new.check_ood_risk --posture 骨盆前倾 \\
        --training-data-dir dataset/HumanML3D output/n15_v2_dps_s40_last_quarter/

    # 用训练集 per-frame 统计（不依赖 comparison.npy）
    python -m new.check_ood_risk --posture 膝超伸_左 \\
        --training-data-dir dataset/HumanML3D --training-only
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from posture_guidance.registry import POSTURE_REGISTRY, resolve_instruction
from posture_guidance import angle_ops
from posture_guidance.joint_indices import get_joint_idx


# ============================================================
# 几何上限判定
# ============================================================

GEOMETRIC_MAX = {
    # three_point_angle caps at 180° (acos range [0, pi])
    "three_point_angle": 180.0,
    # signed_knee: 使用 three_point_angle 的 185° 作为物理上限
    # (signed_knee 通过 z-offset sign trick 可超过 180°，但解剖学上
    #  超伸 > 5° 即 185° 极其罕见，训练集 three_point_angle max ~179°)
    "signed_knee_angle": 185.0,
    # angle functions with atan2: range (-90°, 90°) theoretically
    "pelvis_tilt_angle": 90.0,
    "trunk_forward_lean": 90.0,
    # distance-based: no hard geometric cap
    "foot_floor_distance": None,
    "spine_posterior_bulge": None,
    "head_forward_offset": None,
    "pelvis_lateral_tilt": 90.0,
}

# Special handling: for signed_knee_angle, also compute three_point_angle
# stats for geometric verification, since signed_knee has sign artifacts
# that inflate upright-frame percentiles
SIGNED_KNEE_GEOMETRIC_REFERENCE = {
    "signed_knee_angle": "three_point_angle",  # use this fn for geometric check
}


def _get_angle_fn_name(spec) -> str:
    """获取 angle_fn 的名称用于几何上限查表。"""
    fn = spec.angle_fn
    if hasattr(fn, '__name__'):
        return fn.__name__
    return str(fn)


def _get_geometric_max(spec) -> float:
    """返回该角度函数的几何上限（度），None 表示无硬上限。"""
    name = _get_angle_fn_name(spec)
    return GEOMETRIC_MAX.get(name, None)


# ============================================================
# Per-frame 角度提取
# ============================================================

def _per_frame_angles_from_npy(npy_path: Path, angle_fn, angle_fn_kwargs: dict) -> np.ndarray:
    """从单个 comparison.npy 提取 per-FRAME 角度序列（度）。"""
    data = np.load(npy_path, allow_pickle=True).item()
    xyz = data["motion_xyz"][0]  # (22, 3, T)
    q = torch.from_numpy(xyz).float().permute(2, 0, 1)  # (T, 22, 3)
    with torch.no_grad():
        angle = angle_fn(q, **angle_fn_kwargs)  # (T,) in radians
    angle_deg = angle.numpy() * 180.0 / math.pi
    return angle_deg  # (T,) in degrees


def _per_frame_angles_from_training(joints_dir: Path, clip_ids: list,
                                     angle_fn, angle_fn_kwargs: dict,
                                     phase_filter: str = None) -> np.ndarray:
    """从训练集 new_joints/ 提取 per-FRAME 角度序列（度）。

    Args:
        joints_dir: new_joints/ 目录
        clip_ids: clip ID 列表
        angle_fn: 角度函数
        angle_fn_kwargs: 角度函数关键字参数
        phase_filter: 如果非 None，只用 upright 帧（stance 代理）
    Returns:
        (N,) ndarray of per-frame angles in degrees
    """
    all_angles = []
    for cid in clip_ids:
        npy_path = joints_dir / f"{cid}.npy"
        if not npy_path.exists():
            continue
        try:
            arr = np.load(npy_path)
            if arr.ndim != 3 or arr.shape[-1] != 3 or arr.shape[1] < 22:
                continue
            q = torch.from_numpy(arr[:, :22, :]).float()  # (T, 22, 3)
            with torch.no_grad():
                angle = angle_fn(q, **angle_fn_kwargs)  # (T,) in radians
            angle_deg = angle.numpy() * 180.0 / math.pi

            if phase_filter == "stance_proxy":
                # 用 upright_mask 作为 stance 相位代理
                mask = _upright_mask_tensor(q)
                angle_deg = angle_deg[mask.numpy()]

            all_angles.extend(angle_deg.tolist())
        except Exception:
            continue

    return np.array(all_angles) if all_angles else np.array([])


def _upright_mask_tensor(q: torch.Tensor, min_height: float = 0.5) -> torch.Tensor:
    """直立帧 mask（与 analyze_knee_angles.py 一致）。"""
    hip_y   = (q[:, get_joint_idx("left_hip"),   1] + q[:, get_joint_idx("right_hip"),   1]) / 2
    knee_y  = (q[:, get_joint_idx("left_knee"),  1] + q[:, get_joint_idx("right_knee"),  1]) / 2
    ankle_y = (q[:, get_joint_idx("left_ankle"), 1] + q[:, get_joint_idx("right_ankle"), 1]) / 2
    return (hip_y > knee_y) & (knee_y > ankle_y) & ((hip_y - ankle_y) > min_height)


# ============================================================
# 三分类逻辑
# ============================================================

def classify_ood(
    target_deg: float,
    spec,
    train_stats: dict = None,
    baseline_stats: dict = None,
) -> tuple:
    """
    三分类判断。

    策略：
      - 对所有体态，优先用 upright-frame (stance proxy) 统计，因为
        Posture Guidance 目标就是站立/行走姿势。
      - signed_knee_angle 在非直立姿势（坐/躺/踢腿）会产生 300°+ 的极端值，
        使用全局统计会将 OOD 目标误判为分布内。
      - 对于 phase != "always" 的 spec，额外检查条件 OOD。

    Args:
        target_deg: 目标角度（度）
        spec: LossSpec
        train_stats: {'upright_p50', 'upright_p99', 'upright_max', ...} (训练集直立帧)
        baseline_stats: {'p99', 'max', 'mean', 'std', 'n_frames'} (generated baseline, per-frame)

    Returns:
        (class_label: int, class_name: str, detail: str, ood_score: float)
    """
    geometric_max = _get_geometric_max(spec)

    # ---- Resolve which stats to use: prefer upright-frame ----
    use_upright = train_stats and train_stats.get('upright_n_frames', 0) > 100

    if use_upright:
        up50  = train_stats['upright_p50']
        up90  = train_stats.get('upright_p90', up50)
        up95  = train_stats.get('upright_p95', up90)
        up99  = train_stats['upright_p99']
        upmax = train_stats['upright_max']
        upmin = train_stats.get('upright_min', train_stats.get('upright_p1', up99))
        upstd = train_stats.get('upright_std', 1.0)
        up_n  = train_stats['upright_n_frames']

        direction = spec.direction  # greater_than | less_than | equal

        # ---- Class 1: geometric support outside ----
        if geometric_max is not None and target_deg > geometric_max:
            return (1,
                    "几何支撑集外 (Geometric Support Outside)",
                    f"目标 {target_deg:.1f}° > 关节几何上限 {geometric_max}°",
                    float('inf'))
        if target_deg > upmax:
            return (1,
                    "几何支撑集外 (Geometric Support Outside)",
                    f"目标 {target_deg:.1f}° > 训练集直立帧最大值 ({upmax:.1f}°) — 未见任何帧达到此角度",
                    float('inf'))

        # Compute per-frame OOD score
        ood = abs(target_deg - up50) / max(upstd, 1e-3)

        # ---- Class 3: conditional OOD (phase mismatch) ----
        if spec.phase != "always":
            # For greater_than: check if target is far above phase-filtered stats
            # For less_than: check if target is far below phase-filtered stats
            phase_stats_available = train_stats.get('phase_n_frames', 0) > 100
            if phase_stats_available:
                phase_p50 = train_stats.get('phase_p50', up50)
                phase_p99 = train_stats.get('phase_p99', up99)
                phase_p1  = train_stats.get('phase_p1', train_stats.get('phase_min', upmin))
                phase_max = train_stats.get('phase_max', upmax)

                if direction == "greater_than":
                    # target > phase_p99: target is ABOVE what phase-filtered frames normally show
                    if target_deg > phase_max:
                        return (3,
                                "条件 OOD (Conditional OOD)",
                                f"目标 {target_deg:.1f}° > 相位过滤帧最大值 ({phase_max:.1f}°) — 该相位下从未出现此角度",
                                ood)
                    elif target_deg > phase_p99 * 1.1:
                        return (3,
                                "条件 OOD (Conditional OOD)",
                                f"目标 {target_deg:.1f}° >> 相位过滤 P99 ({phase_p99:.1f}°) — 该相位下极其罕见",
                                ood)
                elif direction == "less_than":
                    # target < phase_p1: target is BELOW what phase-filtered frames normally show
                    if target_deg < phase_p1 * 0.9:
                        return (3,
                                "条件 OOD (Conditional OOD)",
                                f"目标 {target_deg:.1f}° << 相位过滤 P1 ({phase_p1:.1f}°) — 该相位下从未如此低",
                                ood)
                    elif target_deg < phase_p50 * 0.5:
                        return (3,
                                "条件 OOD (Conditional OOD)",
                                f"目标 {target_deg:.1f}° << 相位过滤 P50 ({phase_p50:.1f}°) — 该相位下极罕见",
                                ood)

        # ---- Class 0 vs Class 2: in-distribution vs density tail ----
        if direction == "greater_than":
            if target_deg <= up90:
                return (0, "分布内 (In-Distribution)",
                        f"目标 {target_deg:.1f}° <= 直立帧 P90 ({up90:.1f}°) — 分布内的常见值",
                        ood)
            elif target_deg <= up95:
                return (0, "分布内 / 轻度尾部",
                        f"目标 {target_deg:.1f}° 在 [P90={up90:.1f}°, P95={up95:.1f}°] — 轻度密度尾部",
                        ood)
            elif target_deg <= up99:
                return (2, "密度尾部 (Density Tail)",
                        f"目标 {target_deg:.1f}° 在 [P95={up95:.1f}°, P99={up99:.1f}°] — 罕见但 guidance 可到达",
                        ood)
            else:
                return (2, "密度尾部 (Density Tail)",
                        f"目标 {target_deg:.1f}° 在 [P99={up99:.1f}°, Max={upmax:.1f}°] — 极罕见但支撑集内",
                        ood)
        elif direction == "less_than":
            up10 = train_stats.get('upright_p10', up50)
            up5  = train_stats.get('upright_p5',  up10)
            up1  = train_stats.get('upright_p1',  up5)
            if target_deg >= up10:
                return (0, "分布内 (In-Distribution)",
                        f"目标 {target_deg:.1f}° >= 直立帧 P10 ({up10:.1f}°) — 分布内的常见值",
                        ood)
            elif target_deg >= up5:
                return (0, "分布内 / 轻度尾部",
                        f"目标 {target_deg:.1f}° 在 [P5={up5:.1f}°, P10={up10:.1f}°] — 轻度密度尾部",
                        ood)
            elif target_deg >= up1:
                return (2, "密度尾部 (Density Tail)",
                        f"目标 {target_deg:.1f}° 在 [P1={up1:.1f}°, P5={up5:.1f}°] — 罕见但 guidance 可到达",
                        ood)
            else:
                return (2, "密度尾部 (Density Tail)",
                        f"目标 {target_deg:.1f}° 在 [Min={upmin:.1f}°, P1={up1:.1f}°] — 极罕见但支撑集内",
                        ood)
        else:  # equal
            if up1 <= target_deg <= up99:
                return (0, "分布内 (In-Distribution)",
                        f"目标 {target_deg:.1f}° 在 [P1={up1:.1f}°, P99={up99:.1f}°] 范围内",
                        ood)
            else:
                return (2, "密度尾部 (Density Tail)",
                        f"目标 {target_deg:.1f}° 在分布支撑集边界",
                        ood)

    # ---- Fallback: use baseline comparison.npy per-frame stats ----
    if baseline_stats and baseline_stats.get('n_frames', 0) > 10:
        bmax = baseline_stats['max']
        bp99 = baseline_stats['p99']
        bmu  = baseline_stats['mean']
        bstd = baseline_stats['std']

        ood_score = abs(target_deg - bmu) / max(bstd, 1e-3)

        if target_deg > bmax:
            return (1,
                    "几何支撑集外 (Geometric Support Outside — baseline)",
                    f"目标 {target_deg:.1f}° > baseline 生成最大值 ({bmax:.1f}°); OOD={ood_score:.1f}σ",
                    ood_score)
        elif target_deg > bp99:
            return (2,
                    "密度尾部 (Density Tail — baseline)",
                    f"目标 {target_deg:.1f}° 在 baseline [P99={bp99:.1f}°, Max={bmax:.1f}°]; OOD={ood_score:.1f}σ",
                    ood_score)
        else:
            return (0,
                    "分布内 (In-Distribution — baseline)",
                    f"目标 {target_deg:.1f}° <= baseline P99 ({bp99:.1f}°); OOD={ood_score:.1f}σ",
                    ood_score)

    return (-1, "无法判定", "缺少足够的统计数据 (请提供 --training-data-dir)", float('nan'))


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="推理前 OOD 风险预检 (v2) — 三分类 + per-frame 统计。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--posture", required=True, help="体态名称，如 '骨盆前倾'")
    parser.add_argument("dirs", nargs="*", default=[],
                        help="含 comparison.npy 的目录（可选，不提供则只用训练集模式）")
    parser.add_argument("--training-data-dir", default=None,
                        help="HumanML3D 根目录（用于训练集 per-frame 统计）")
    parser.add_argument("--split", default="train",
                        help="训练集 split (train/val/test/all)")
    parser.add_argument("--training-only", action="store_true",
                        help="只用训练集统计，跳过 comparison.npy")
    parser.add_argument("--max-training-clips", type=int, default=0,
                        help="训练集最大 clip 数 (0=全部)")
    parser.add_argument("--unit", default="deg", choices=["deg", "rad"])
    args = parser.parse_args()

    # ── 1. Load spec ────────────────────────────────────────
    try:
        if args.posture in POSTURE_REGISTRY:
            spec = POSTURE_REGISTRY[args.posture]
        else:
            # try resolve
            names = resolve_instruction(args.posture)
            spec = POSTURE_REGISTRY[names[0]]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] {e}")
        print(f"可用体态: {list(POSTURE_REGISTRY.keys())}")
        sys.exit(1)

    angle_fn = spec.angle_fn
    angle_fn_kwargs = spec.angle_fn_kwargs
    target_deg = spec.target_deg
    unit = args.unit or spec.unit

    print(f"\n{'='*70}")
    print(f"  OOD 风险预检 (v2) — {spec.name}")
    print(f"  目标: {target_deg:.1f}°  direction={spec.direction}  phase={spec.phase}")
    print(f"{'='*70}")

    # ── 2. Training set per-frame stats ─────────────────────
    train_stats = None
    if args.training_data_dir:
        data_dir = Path(args.training_data_dir)
        joints_dir = data_dir / "new_joints"
        if not joints_dir.exists():
            print(f"[WARN] 找不到 {joints_dir}")
        else:
            # Load clip IDs
            if args.split == "all":
                clip_ids = [p.stem for p in sorted(joints_dir.glob("*.npy"))]
            else:
                split_file = data_dir / f"{args.split}.txt"
                if split_file.exists():
                    clip_ids = [l.strip() for l in split_file.read_text().split("\n") if l.strip()]
                else:
                    clip_ids = [p.stem for p in sorted(joints_dir.glob("*.npy"))]

            if args.max_training_clips > 0:
                clip_ids = clip_ids[:args.max_training_clips]

            print(f"\n  训练集: {len(clip_ids)} clips (split={args.split})")

            # Global per-frame stats (所有帧，供参考)
            train_angles_all = _per_frame_angles_from_training(
                joints_dir, clip_ids, angle_fn, angle_fn_kwargs, phase_filter=None
            )

            # Upright-frame stats (直立帧，用于分类决策)
            train_angles_upright = _per_frame_angles_from_training(
                joints_dir, clip_ids, angle_fn, angle_fn_kwargs,
                phase_filter="stance_proxy"
            )

            # For signed_knee: also compute three_point_angle as geometric reference
            tpa_upright = None
            fn_name = _get_angle_fn_name(spec)
            if fn_name == "signed_knee_angle":
                side = angle_fn_kwargs.get("side", "left")
                tpa_kwargs = {
                    "joint_a": f"{side}_hip",
                    "joint_b": f"{side}_knee",
                    "joint_c": f"{side}_ankle",
                }
                tpa_upright = _per_frame_angles_from_training(
                    joints_dir, clip_ids, angle_ops.three_point_angle, tpa_kwargs,
                    phase_filter="stance_proxy"
                )
                if len(tpa_upright) > 0:
                    print(f"    三点角参考(直立): N={len(tpa_upright):,}  "
                          f"P99={np.percentile(tpa_upright, 99):.1f}°  "
                          f"Max={tpa_upright.max():.1f}°")

            train_stats = {}
            if len(train_angles_all) > 0:
                print(f"    全帧: N={len(train_angles_all):,}  "
                      f"P50={np.percentile(train_angles_all, 50):.1f}°  "
                      f"P99={np.percentile(train_angles_all, 99):.1f}°  "
                      f"Max={train_angles_all.max():.1f}°")

            if len(train_angles_upright) > 0:
                up = train_angles_upright
                train_stats = {
                    'upright_n_frames': len(up),
                    'upright_p50': float(np.percentile(up, 50)),
                    'upright_p10': float(np.percentile(up, 10)),
                    'upright_p5':  float(np.percentile(up, 5)),
                    'upright_p1':  float(np.percentile(up, 1)),
                    'upright_p90': float(np.percentile(up, 90)),
                    'upright_p95': float(np.percentile(up, 95)),
                    'upright_p99': float(np.percentile(up, 99)),
                    'upright_max': float(up.max()),
                    'upright_min': float(up.min()),
                    'upright_std': float(up.std()),
                }

                # Override upright stats with three_point_angle reference for signed_knee
                # This fixes the geometric support check: three_point_angle caps at 180°
                # so a target of 190° is correctly identified as Class 1
                if tpa_upright is not None and len(tpa_upright) > 0:
                    train_stats['upright_p99'] = float(np.percentile(tpa_upright, 99))
                    train_stats['upright_max'] = float(tpa_upright.max())
                    train_stats['upright_min'] = float(tpa_upright.min())
                    train_stats['_using_tpa_reference'] = True

                # Phase-filtered stats: for spec.phase != "always",
                # upright is the stance proxy, so reuse it
                if spec.phase != "always":
                    train_stats['phase_n_frames'] = len(up)
                    train_stats['phase_p50'] = train_stats['upright_p50']
                    train_stats['phase_p99'] = train_stats['upright_p99']
                    train_stats['phase_p1']  = train_stats['upright_p1']
                    train_stats['phase_max'] = train_stats['upright_max']
                    train_stats['phase_min'] = train_stats['upright_min']

                # 全局统计 as fallback
                train_stats['all_p99'] = float(np.percentile(train_angles_all, 99))
                train_stats['all_max'] = float(train_angles_all.max())
                train_stats['n_frames'] = len(train_angles_all)

                print(f"    直立帧(stance proxy): N={len(up):,}  "
                      f"P50={train_stats['upright_p50']:.1f}°  "
                      f"P90={train_stats['upright_p90']:.1f}°  "
                      f"P95={train_stats['upright_p95']:.1f}°  "
                      f"P99={train_stats['upright_p99']:.1f}°  "
                      f"P1={train_stats['upright_p1']:.1f}°  "
                      f"Max={train_stats['upright_max']:.1f}°")

    # ── 3. Baseline comparison.npy per-frame stats ──────────
    baseline_stats = None
    if not args.training_only and args.dirs:
        npy_files = []
        for d in args.dirs:
            p = Path(d)
            if p.is_file() and p.suffix == ".npy":
                npy_files.append(p)
            elif p.is_dir():
                npy_files.extend(sorted(p.rglob("comparison.npy")))

        if npy_files:
            print(f"\n  Baseline comparison.npy: {len(npy_files)} files")
            all_frame_angles = []
            for npy in npy_files:
                try:
                    angles = _per_frame_angles_from_npy(npy, angle_fn, angle_fn_kwargs)
                    all_frame_angles.extend(angles.tolist())
                except Exception as e:
                    print(f"  [WARN] 跳过 {npy.name}: {e}")

            if all_frame_angles:
                arr = np.array(all_frame_angles)
                baseline_stats = {
                    'p50': float(np.percentile(arr, 50)),
                    'p90': float(np.percentile(arr, 90)),
                    'p95': float(np.percentile(arr, 95)),
                    'p99': float(np.percentile(arr, 99)),
                    'max': float(arr.max()),
                    'min': float(arr.min()),
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'n_frames': len(arr),
                }
                print(f"    Per-frame: N={baseline_stats['n_frames']:,}  "
                      f"mean={baseline_stats['mean']:.2f}°  std={baseline_stats['std']:.2f}°  "
                      f"P99={baseline_stats['p99']:.1f}°  Max={baseline_stats['max']:.1f}°")
        else:
            print(f"\n  [WARN] 未找到 comparison.npy，跳过 baseline 统计")

    # ── 4. Three-class classification ───────────────────────
    class_label, class_name, detail, ood_score = classify_ood(
        target_deg, spec, train_stats, baseline_stats
    )

    print(f"\n{'─'*70}")
    print(f"  判定结果")
    print(f"{'─'*70}")
    print(f"  Class {class_label}: {class_name}")
    print(f"  {detail}")
    if not math.isinf(ood_score) and not math.isnan(ood_score):
        print(f"  OOD score: {ood_score:.2f}σ")

    # ── 5. Guidance prognosis ───────────────────────────────
    print(f"\n  Guidance 预后:")
    if class_label == 0:
        print(f"  → Guidance 可直接有效工作，推力需求低")
    elif class_label == 1:
        print(f"  → Guidance 极可能失败：目标不在分布支撑集内")
        print(f"  → 必须改架构（fine-tune / LoRA / 条件训练）")
    elif class_label == 2:
        print(f"  → Guidance 可成功但需要适当推力（V2_s40 或 V6 default）")
        print(f"  → 建议 schedule='last_quarter'，避免早期步扰动时序")
    elif class_label == 3:
        print(f"  → Guidance 将失败：相位约束与角度约束冲突")
        print(f"  → 模型从未在指定相位下达到目标角度，梯度会被流形顶回")
        print(f"  → 唯一出路：相位条件训练 或 prompt 先验偏移")

    print(f"\n{'='*70}")
    print(f"  OOD 分类参考:")
    print(f"    Class 0: 分布内 — guidance 直接有效")
    print(f"    Class 1: 几何支撑集外 — guidance 极可能失败")
    print(f"    Class 2: 密度尾部 — guidance 可成功 (需足够推力)")
    print(f"    Class 3: 条件 OOD — (角度, 相位) 组合零密度")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
