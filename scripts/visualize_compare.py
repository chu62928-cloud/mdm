"""
scripts/visualize_compare.py  —— 改进版

改进点：
    1. 关节点更小、骨骼更细，骨架比例更接近论文图风格
    2. 自动添加地面平面，把骨架对齐到地面
    3. 用骨架自身高度自动调整坐标范围，避免压扁
    4. 区分上肢/下肢/躯干用不同粗细，提高可读性
    5. 多视角渲染（侧视图 + 等距视图）
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.patches as mpatches

from posture_guidance.registry import POSTURE_REGISTRY, resolve_instruction
from posture_guidance.joint_indices import JOINT_IDX, SMPL_JOINT_NAMES


# ---- 颜色方案 ----
COLOR_BASELINE = "#5F5E5A"
COLOR_GUIDED   = "#D4537E"
COLOR_TARGET   = "#0F6E56"
COLOR_GROUND   = "#D8D6CE"

# ---- SMPL 22-joint 骨骼分组（按身体部位区分线宽和颜色深浅）----
SPINE_BONES = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]      # 中轴：粗线
LEG_BONES   = [(0, 1), (0, 2), (1, 4), (2, 5),
               (4, 7), (5, 8), (7, 10), (8, 11)]                # 腿：中等
ARM_BONES   = [(9, 13), (9, 14), (13, 16), (14, 17),
               (16, 18), (17, 19), (18, 20), (19, 21)]          # 手臂：细线
ALL_BONES   = SPINE_BONES + LEG_BONES + ARM_BONES


# =========================================================
# 辅助函数
# =========================================================

def xyz_to_qtensor(xyz_np):
    """(B, J, 3, T) → (B, T, J, 3) torch tensor"""
    return torch.from_numpy(xyz_np).permute(0, 3, 1, 2).float()


def get_metric_specs(posture_instructions):
    specs = []
    for inst in posture_instructions:
        for sn in resolve_instruction(inst):
            s = POSTURE_REGISTRY[sn]
            specs.append({
                "name":      s.name,
                "angle_fn":  s.angle_fn,
                "kwargs":    s.angle_fn_kwargs,
                "target":    s.target_deg,
                "tolerance": s.tolerance_deg,
                "direction": s.direction,
                "unit":      s.unit,
            })
    return specs


def compute_angle_series(q, spec):
    angle = spec["angle_fn"](q, **spec["kwargs"])
    if spec["unit"] == "deg":
        return (angle * 180.0 / math.pi).cpu().numpy()
    return angle.cpu().numpy()


def get_floor_height(xyz_seq):
    """
    估计地面高度：取所有帧所有关节的 y 坐标最低 5%。
    xyz_seq shape: (J, 3, T) 或 (T, J, 3)
    """
    if xyz_seq.ndim == 3 and xyz_seq.shape[1] == 3:
        # (J, 3, T)
        ys = xyz_seq[:, 1, :].flatten()
    else:
        # (T, J, 3)
        ys = xyz_seq[..., 1].flatten()
    return float(np.percentile(ys, 2))


def setup_3d_axes(ax, xlim, ylim, zlim, view_elev=12, view_azim=45,
                  show_floor=True, floor_y=0.0):
    """统一设置 3D ax 的样式（修复 tightbbox 报错）"""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 把 panes 设为透明而不是不可见
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 关掉网格线
    ax.grid(False)

    # 把 axis line 设为透明（不要 set_visible(False)）
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))   # 透明白线
        # 隐藏 tick 标签和 axis 标签
        axis.set_tick_params(labelleft=False, labelbottom=False, length=0)

    ax.view_init(elev=view_elev, azim=view_azim)

    # 画地面平面
    if show_floor:
        xf = np.linspace(xlim[0], xlim[1], 4)
        yf = np.linspace(ylim[0], ylim[1], 4)
        Xf, Yf = np.meshgrid(xf, yf)
        Zf = np.ones_like(Xf) * floor_y
        ax.plot_surface(Xf, Yf, Zf, color=COLOR_GROUND,
                        alpha=0.35, edgecolor='none',
                        antialiased=True, zorder=-10)


def draw_skeleton(ax, q_frame, color, alpha=1.0, joint_size=10):
    """
    在 3D ax 上画骨架。
    q_frame: (J, 3)，约定 y=高度，x=左右，z=前后
    matplotlib 3D 默认 z 是高度轴，所以画的时候做坐标交换：
      ax.scatter(x, z, y) 对应 ax.set_zlim 是高度
    """
    # 关节点（小）
    ax.scatter(
        q_frame[:, 0], q_frame[:, 2], q_frame[:, 1],
        c=color, s=joint_size, alpha=alpha,
        depthshade=True, zorder=3, edgecolors='none',
    )

    # 骨骼线（按身体部位用不同粗细）
    for (a, b), lw in [
        *[(bone, 2.4) for bone in SPINE_BONES],   # 脊柱粗
        *[(bone, 2.0) for bone in LEG_BONES],     # 腿中等
        *[(bone, 1.6) for bone in ARM_BONES],     # 手臂细
    ]:
        if a < q_frame.shape[0] and b < q_frame.shape[0]:
            ax.plot(
                [q_frame[a, 0], q_frame[b, 0]],
                [q_frame[a, 2], q_frame[b, 2]],
                [q_frame[a, 1], q_frame[b, 1]],
                color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=2,
            )


def compute_uniform_xyz_lim(xyz_list, margin_ratio=0.15):
    """
    根据所有帧所有关节的位置，计算合适的统一坐标范围。
    - 高度方向（y）：地面到头顶
    - 水平方向（x, z）：以 root 轨迹为中心，宽度 = max(轨迹宽度, 骨架自身高度*0.6)
    """
    all_pts = np.concatenate(
        [xyz.reshape(-1, 3) for xyz in xyz_list], axis=0
    )

    floor_y = float(np.percentile(all_pts[:, 1], 2))
    head_y  = float(np.percentile(all_pts[:, 1], 99))
    height  = head_y - floor_y

    # 水平方向：以轨迹中心为原点，宽度自适应
    x_center = float(np.mean(all_pts[:, 0]))
    z_center = float(np.mean(all_pts[:, 2]))
    x_range  = float(np.ptp(all_pts[:, 0]))
    z_range  = float(np.ptp(all_pts[:, 2]))

    half_w = max(x_range, z_range, height * 0.5) / 2 + height * margin_ratio

    xlim = (x_center - half_w, x_center + half_w)
    zlim = (z_center - half_w, z_center + half_w)
    ylim = (floor_y - height * 0.05, floor_y + height * 1.1)

    return xlim, zlim, ylim, floor_y


# =========================================================
# Figure 1: 角度时间曲线
# =========================================================

def plot_angle_curves(data, output_dir, sample_idx=0):
    posture_instructions = data.get("posture_instructions", [])
    if not posture_instructions:
        print("[Figure 1] 跳过：无 posture_instructions")
        return

    specs = get_metric_specs(posture_instructions)
    if not specs:
        return

    q_base   = xyz_to_qtensor(data["motion_xyz"])
    q_guided = xyz_to_qtensor(data["motion_xyz_guided"])

    n = len(specs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), squeeze=False)

    for i, spec in enumerate(specs):
        ax = axes[i, 0]
        s_base   = compute_angle_series(q_base,   spec)[sample_idx]
        s_guided = compute_angle_series(q_guided, spec)[sample_idx]

        frames = np.arange(len(s_base))
        target = spec["target"]
        tol    = spec["tolerance"]
        unit   = "°" if spec["unit"] == "deg" else " m"

        # 目标带
        if spec["direction"] == "greater_than":
            ax.axhspan(target - tol, target * 1.5, alpha=0.08,
                       color=COLOR_TARGET,
                       label=f"target zone (≥{target-tol:.1f}{unit})")
        elif spec["direction"] == "less_than":
            ax.axhspan(target * 0.5, target + tol, alpha=0.08,
                       color=COLOR_TARGET,
                       label=f"target zone (≤{target+tol:.1f}{unit})")
        else:
            ax.axhspan(target - tol, target + tol, alpha=0.08,
                       color=COLOR_TARGET,
                       label=f"target zone ({target:.1f}±{tol:.1f}{unit})")

        ax.axhline(target, color=COLOR_TARGET, linestyle="--",
                   lw=1.0, alpha=0.7, label=f"target = {target:.1f}{unit}")

        ax.plot(frames, s_base,   color=COLOR_BASELINE, lw=1.6,
                label="baseline", alpha=0.9)
        ax.plot(frames, s_guided, color=COLOR_GUIDED,   lw=1.6,
                label="guided",   alpha=0.95)

        ax.set_title(spec["name"], fontsize=12, loc="left")
        ax.set_xlabel("frame")
        ax.set_ylabel(f"angle ({unit.strip()})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "angle_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 1] saved → {out_path}")


# =========================================================
# Figure 2: 多视角骨架关键帧
# =========================================================

def plot_skeleton_keyframes(data, output_dir, sample_idx=0,
                            n_keyframes=4):
    """
    每个关键帧画两行：上排侧视图，下排等距视图。
    每个子图把 baseline 和 guided 重叠绘制。
    """
    xyz_base   = data["motion_xyz"][sample_idx]
    xyz_guided = data["motion_xyz_guided"][sample_idx]
    T = xyz_base.shape[-1]

    keyframes = np.linspace(5, T - 5, n_keyframes).astype(int)

    fig = plt.figure(figsize=(3.5 * n_keyframes, 7))
    gs = GridSpec(2, n_keyframes, figure=fig,
                  hspace=0.05, wspace=0.05)

    xlim, zlim, ylim, floor_y = compute_uniform_xyz_lim(
        [xyz_base, xyz_guided]
    )

    # 上排：侧视图（azim=0）
    # 下排：等距视图（azim=45）
    for i, t in enumerate(keyframes):
        for row, (elev, azim) in enumerate([(8, 0), (12, 45)]):
            ax = fig.add_subplot(gs[row, i], projection="3d")

            q_b = xyz_base[:, :, t]
            q_g = xyz_guided[:, :, t]

            setup_3d_axes(ax, xlim, zlim, ylim,
                          view_elev=elev, view_azim=azim,
                          floor_y=floor_y)

            draw_skeleton(ax, q_b, COLOR_BASELINE, alpha=0.5, joint_size=8)
            draw_skeleton(ax, q_g, COLOR_GUIDED,   alpha=0.95, joint_size=10)

            if row == 0:
                ax.set_title(f"frame {t}", fontsize=11)

    # 添加图例
    handles = [
        mpatches.Patch(color=COLOR_BASELINE, label="baseline"),
        mpatches.Patch(color=COLOR_GUIDED,   label="guided"),
    ]
    fig.legend(handles=handles, loc='upper right',
               fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.99, 0.99))

    plt.suptitle(
        "Side view (top) and isometric view (bottom)",
        fontsize=12, y=1.01,
    )
    out_path = os.path.join(output_dir, "skeleton_keyframes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 2] saved → {out_path}")


# =========================================================
# Figure 3: 关节差异热图
# =========================================================

def plot_joint_diff_heatmap(data, output_dir, sample_idx=0):
    xyz_base   = data["motion_xyz"][sample_idx]
    xyz_guided = data["motion_xyz_guided"][sample_idx]

    diff = xyz_guided - xyz_base
    dist = np.linalg.norm(diff, axis=1)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(dist, aspect="auto", cmap="OrRd",
                   interpolation="nearest")

    ax.set_yticks(range(len(SMPL_JOINT_NAMES)))
    ax.set_yticklabels(SMPL_JOINT_NAMES, fontsize=8)
    ax.set_xlabel("frame")
    ax.set_title("Per-joint per-frame distance (guided − baseline) in meters",
                 fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("distance (m)", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "joint_diff_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 3] saved → {out_path}")


# =========================================================
# Figure 4: 并排动画（含地面）
# =========================================================

def make_side_by_side_animation(data, output_dir, sample_idx=0,
                                fps=20, fmt="mp4"):
    xyz_base   = data["motion_xyz"][sample_idx]
    xyz_guided = data["motion_xyz_guided"][sample_idx]
    T = xyz_base.shape[-1]

    fig = plt.figure(figsize=(11, 5.5))
    ax_b = fig.add_subplot(121, projection="3d")
    ax_g = fig.add_subplot(122, projection="3d")

    xlim, zlim, ylim, floor_y = compute_uniform_xyz_lim(
        [xyz_base, xyz_guided]
    )

    def update(t):
        ax_b.clear()
        ax_g.clear()

        setup_3d_axes(ax_b, xlim, zlim, ylim,
                      view_elev=12, view_azim=45, floor_y=floor_y)
        setup_3d_axes(ax_g, xlim, zlim, ylim,
                      view_elev=12, view_azim=45, floor_y=floor_y)

        ax_b.set_title("baseline", fontsize=12,
                       color=COLOR_BASELINE, weight="bold")
        ax_g.set_title("guided",   fontsize=12,
                       color=COLOR_GUIDED,   weight="bold")

        draw_skeleton(ax_b, xyz_base[:, :, t],   COLOR_BASELINE,
                      alpha=0.95, joint_size=10)
        draw_skeleton(ax_g, xyz_guided[:, :, t], COLOR_GUIDED,
                      alpha=0.95, joint_size=10)

        fig.suptitle(f"frame {t}/{T-1}", fontsize=11, y=0.95)

    anim = FuncAnimation(fig, update, frames=T,
                         interval=1000 / fps, blit=False)

    if fmt == "mp4":
        out_path = os.path.join(output_dir, "comparison_animation.mp4")
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(out_path, writer=writer)
            print(f"[Figure 4] saved → {out_path}")
        except Exception as e:
            print(f"[Figure 4] mp4 失败 ({e}), fallback to gif")
            fmt = "gif"

    if fmt == "gif":
        out_path = os.path.join(output_dir, "comparison_animation.gif")
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        print(f"[Figure 4] saved → {out_path}")

    plt.close()


# =========================================================
# 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_animation", action="store_true")
    parser.add_argument("--anim_fmt", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"Error: {args.npy_path} not found.")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.npy_path)
    )
    output_dir = os.path.join(output_dir, "viz")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading: {args.npy_path}")
    data = np.load(args.npy_path, allow_pickle=True).item()

    print(f"\nGenerating visualizations in: {output_dir}")
    print(f"  sample_idx = {args.sample_idx}")
    print(f"  text_prompt = {data.get('text_prompt', '?')}")
    print(f"  posture     = {data.get('posture_instructions', [])}")
    print()

    plot_angle_curves(data, output_dir, sample_idx=args.sample_idx)
    plot_skeleton_keyframes(data, output_dir, sample_idx=args.sample_idx)
    plot_joint_diff_heatmap(data, output_dir, sample_idx=args.sample_idx)

    if not args.no_animation:
        fps = data.get("fps", 20)
        make_side_by_side_animation(
            data, output_dir,
            sample_idx=args.sample_idx,
            fps=int(fps),
            fmt=args.anim_fmt,
        )

    print(f"\n✓ All figures saved to {output_dir}")


if __name__ == "__main__":
    main()