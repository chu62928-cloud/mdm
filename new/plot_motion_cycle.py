"""
scripts/plot_motion_cycle_v4.py

论文风格的动作周期分解图：
    - 无圆点关节，纯骨架线条
    - 四肢用不同颜色（左肢/右肢/躯干区分）
    - 人物放大，占据图块大部分
    - 严格侧视图

参考风格：MotionCLIP / MDM / Action2Motion 等论文常用样式
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import OrderedDict


# ===== 论文风格配色 =====
# 用 baseline 和 guided 各自一套：左肢深、右肢浅、躯干中等
PALETTE_BASELINE = {
    "spine":     "#1f3a93",   # 深蓝（躯干主色）
    "left_arm":  "#4a90e2",   # 左臂
    "right_arm": "#82b1ff",   # 右臂（更浅）
    "left_leg":  "#1f3a93",   # 左腿
    "right_leg": "#5e8eda",   # 右腿
    "head":      "#1f3a93",
}
PALETTE_GUIDED = {
    "spine":     "#922b3e",   # 深红
    "left_arm":  "#d6486a",
    "right_arm": "#f098a8",
    "left_leg":  "#922b3e",
    "right_leg": "#c3506a",
    "head":      "#922b3e",
}

COLOR_TRAJ      = "#D4A93C"
COLOR_HIGHLIGHT = "#D63D3D"


# ===== 关节索引 =====
J_PELVIS, J_L_HIP, J_R_HIP, J_SPINE1   = 0, 1, 2, 3
J_L_KNEE, J_R_KNEE, J_SPINE2            = 4, 5, 6
J_L_ANKLE, J_R_ANKLE, J_SPINE3          = 7, 8, 9
J_L_FOOT, J_R_FOOT, J_NECK              = 10, 11, 12
J_L_COLLAR, J_R_COLLAR, J_HEAD          = 13, 14, 15
J_L_SHLDR, J_R_SHLDR                    = 16, 17
J_L_ELBOW, J_R_ELBOW                    = 18, 19
J_L_WRIST, J_R_WRIST                    = 20, 21


# ===== 骨骼分组（按身体部位）=====
SPINE_BONES = [
    (J_PELVIS, J_SPINE1), (J_SPINE1, J_SPINE2),
    (J_SPINE2, J_SPINE3), (J_SPINE3, J_NECK),
    (J_NECK,   J_HEAD),
]
LEFT_LEG_BONES = [
    (J_PELVIS,  J_L_HIP), (J_L_HIP,   J_L_KNEE),
    (J_L_KNEE,  J_L_ANKLE), (J_L_ANKLE, J_L_FOOT),
]
RIGHT_LEG_BONES = [
    (J_PELVIS,  J_R_HIP), (J_R_HIP,   J_R_KNEE),
    (J_R_KNEE,  J_R_ANKLE), (J_R_ANKLE, J_R_FOOT),
]
LEFT_ARM_BONES = [
    (J_NECK, J_L_COLLAR), (J_L_COLLAR, J_L_SHLDR),
    (J_L_SHLDR, J_L_ELBOW), (J_L_ELBOW, J_L_WRIST),
]
RIGHT_ARM_BONES = [
    (J_NECK, J_R_COLLAR), (J_R_COLLAR, J_R_SHLDR),
    (J_R_SHLDR, J_R_ELBOW), (J_R_ELBOW, J_R_WRIST),
]


# ===== 朝向估计 =====

def estimate_facing(joints_3d_seq):
    left_hip  = joints_3d_seq[J_L_HIP,  :, :]
    right_hip = joints_3d_seq[J_R_HIP,  :, :]
    lr_mean = (right_hip - left_hip).mean(axis=1)
    lr_mean[1] = 0
    right_axis = lr_mean / (np.linalg.norm(lr_mean) + 1e-8)
    forward = np.cross([0., 1., 0.], right_axis)
    return forward / (np.linalg.norm(forward) + 1e-8)


def project_sagittal(joints_3d, forward):
    return np.stack([joints_3d @ forward, joints_3d[:, 1]], axis=-1)


def normalize_to_origin(joints_3d_seq, forward):
    pelvis_seq = joints_3d_seq[J_PELVIS, :, :]
    pelvis_fwd_t0 = pelvis_seq[:, 0] @ forward
    shift = forward * pelvis_fwd_t0
    return joints_3d_seq - shift[None, :, None]


# ===== 单个骨架绘制（论文风格）=====

def draw_skeleton_paper(ax, coords_2d, palette,
                        x_offset=0, scale=1.0, alpha=1.0,
                        line_width=3.0, head_size=0.06):
    """
    论文风格的骨架：
        - 四肢用不同颜色
        - 无关节圆点
        - 头部用圆圈表示
        - 骨骼线粗、圆角端
    """
    pts = coords_2d.copy() * scale
    pts[:, 0] += x_offset

    # 各部位线宽（spine 最粗，arm 最细）
    bone_groups = [
        (SPINE_BONES,     palette["spine"],     line_width * 1.15),
        (LEFT_LEG_BONES,  palette["left_leg"],  line_width * 1.0),
        (RIGHT_LEG_BONES, palette["right_leg"], line_width * 1.0),
        (LEFT_ARM_BONES,  palette["left_arm"],  line_width * 0.85),
        (RIGHT_ARM_BONES, palette["right_arm"], line_width * 0.85),
    ]

    for bones, color, lw in bone_groups:
        for (a, b) in bones:
            if a < pts.shape[0] and b < pts.shape[0]:
                ax.plot(
                    [pts[a, 0], pts[b, 0]],
                    [pts[a, 1], pts[b, 1]],
                    color=color, lw=lw, alpha=alpha,
                    solid_capstyle='round', zorder=3,
                )

    # 头部圆圈（不填充，仅描边）
    if J_HEAD < pts.shape[0] and J_NECK < pts.shape[0]:
        head_radius = max(
            np.linalg.norm(pts[J_HEAD] - pts[J_NECK]) * 0.5,
            head_size * scale,
        )
        head_y_offset = head_radius * 0.7
        circle = Circle(
            (pts[J_HEAD, 0], pts[J_HEAD, 1] + head_y_offset),
            radius=head_radius,
            facecolor='white', edgecolor=palette["head"],
            linewidth=line_width * 1.0, alpha=alpha,
            zorder=4,
        )
        ax.add_patch(circle)


# ===== 行级绘制 =====

def plot_one_row(ax, motion, label, keyframe_indices,
                 palette, x_spacing, scale, forward,
                 highlight_frame=None, line_width=3.0):
    motion_aligned = normalize_to_origin(motion, forward)
    pelvis_xs, pelvis_ys = [], []

    for i, t in enumerate(keyframe_indices):
        coords_2d = project_sagittal(motion_aligned[:, :, t], forward)
        x_offset = i * x_spacing

        draw_skeleton_paper(
            ax, coords_2d, palette,
            x_offset=x_offset, scale=scale,
            line_width=line_width,
        )

        pelvis_x = coords_2d[J_PELVIS, 0] * scale + x_offset
        pelvis_y = coords_2d[J_PELVIS, 1] * scale
        pelvis_xs.append(pelvis_x)
        pelvis_ys.append(pelvis_y)

        if highlight_frame is not None and i == highlight_frame:
            ymin = coords_2d[:, 1].min() * scale
            ymax = coords_2d[:, 1].max() * scale
            cy = (ymin + ymax) / 2
            radius = (ymax - ymin) * 0.65
            circle = Circle(
                (pelvis_x, cy), radius,
                fill=False, edgecolor=COLOR_HIGHLIGHT,
                linewidth=2.8, zorder=10,
            )
            ax.add_patch(circle)

    # 质心轨迹
    if len(pelvis_xs) > 1:
        ax.plot(pelvis_xs, pelvis_ys,
                color=COLOR_TRAJ, lw=2.0, alpha=0.85, zorder=1)

    # 行标签
    ax.text(
        -x_spacing * 0.5,
        np.mean(pelvis_ys) if pelvis_ys else 0,
        label,
        fontsize=15, ha='right', va='center',
        weight='bold', color='#222222',
    )


def add_top_axis(ax, n_keyframes, x_spacing, axis_label, y_top):
    percentages = np.linspace(0, 100, n_keyframes).astype(int)
    for i, pct in enumerate(percentages):
        x = i * x_spacing
        ax.text(x, y_top, f"{pct}",
                fontsize=12, ha='center', va='bottom',
                color='#222222', weight='bold')

    ax.text(-x_spacing * 0.5, y_top, f"% {axis_label}",
            fontsize=11, ha='right', va='bottom',
            color='#444444', style='italic')

    ax.plot(
        [-x_spacing * 0.4,
          (n_keyframes - 1) * x_spacing + x_spacing * 0.4],
        [y_top - 0.02, y_top - 0.02],
        color='#444444', lw=0.7, clip_on=False,
    )


# ===== 主函数 =====

def plot_motion_cycle_paper(
    motions_dict,
    output_path,
    n_keyframes=7,
    cycle_label="Cycle",
    highlight_frames=None,
    figsize=None,
    skeleton_scale=1.5,        # ★ 骨架放大倍数
):
    if highlight_frames is None:
        highlight_frames = {}

    first_motion = next(iter(motions_dict.values()))
    forward = estimate_facing(first_motion)

    # 计算骨架尺寸
    sample_aligned = normalize_to_origin(first_motion, forward)
    sample_2d = []
    for t in range(sample_aligned.shape[-1]):
        sample_2d.append(project_sagittal(sample_aligned[:, :, t], forward))
    sample_2d = np.concatenate(sample_2d, axis=0)
    body_height = sample_2d[:, 1].max() - sample_2d[:, 1].min()
    body_width  = sample_2d[:, 0].max() - sample_2d[:, 0].min()

    scale = skeleton_scale

    # 帧间距：稍大于骨架宽度，让人物明显
    x_spacing = max(body_width * scale * 1.3,
                    body_height * scale * 0.8)

    n_methods = len(motions_dict)
    if figsize is None:
        figsize = (max(15, n_keyframes * 2.4),
                   body_height * scale * n_methods * 1.6 + 1.0)

    fig, axes = plt.subplots(
        n_methods, 1, figsize=figsize,
        gridspec_kw={'hspace': 0.05},
        squeeze=False,
    )

    # 统一 y 范围
    all_ys = []
    for motion in motions_dict.values():
        aligned = normalize_to_origin(motion, forward)
        for t in range(aligned.shape[-1]):
            c2d = project_sagittal(aligned[:, :, t], forward)
            all_ys.extend(c2d[:, 1] * scale)
    y_min, y_max = min(all_ys), max(all_ys)
    y_pad = (y_max - y_min) * 0.12

    palettes = [PALETTE_BASELINE, PALETTE_GUIDED]
    line_widths = [3.5, 3.5]

    for row_idx, (method_name, motion) in enumerate(motions_dict.items()):
        ax = axes[row_idx, 0]
        T = motion.shape[-1]
        keyframe_indices = np.linspace(0, T - 1, n_keyframes).astype(int)

        plot_one_row(
            ax, motion, method_name, keyframe_indices,
            palette=palettes[row_idx % len(palettes)],
            x_spacing=x_spacing, scale=scale,
            forward=forward,
            highlight_frame=highlight_frames.get(method_name, None),
            line_width=line_widths[row_idx % len(line_widths)],
        )

        ax.set_xlim(-x_spacing * 0.55,
                    (n_keyframes - 1) * x_spacing + x_spacing * 0.55)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if row_idx < n_methods - 1:
            ax.plot(
                [-x_spacing * 0.4,
                 (n_keyframes - 1) * x_spacing + x_spacing * 0.4],
                [y_min - y_pad * 0.5, y_min - y_pad * 0.5],
                color='#888', lw=0.5, linestyle='--', alpha=0.5,
                clip_on=False,
            )

    y_top = y_max + y_pad * 0.4
    add_top_axis(axes[0, 0], n_keyframes, x_spacing, cycle_label, y_top)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"✓ saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str)
    parser.add_argument("--sample_idx",      type=int, default=0)
    parser.add_argument("--n_keyframes",     type=int, default=7)
    parser.add_argument("--cycle_label",     type=str, default="Cycle")
    parser.add_argument("--highlight_frame", type=int, default=None)
    parser.add_argument("--output",          type=str, default=None)
    parser.add_argument("--baseline_label",  type=str, default="Baseline")
    parser.add_argument("--guided_label",    type=str, default="Ours")
    parser.add_argument("--skeleton_scale",  type=float, default=1.5)
    args = parser.parse_args()

    data = np.load(args.npy_path, allow_pickle=True).item()
    baseline = data["motion_xyz"][args.sample_idx]
    guided   = data["motion_xyz_guided"][args.sample_idx]

    motions_dict = OrderedDict([
        (args.baseline_label, baseline),
        (args.guided_label,   guided),
    ])

    highlight_frames = {}
    if args.highlight_frame is not None:
        highlight_frames[args.guided_label] = args.highlight_frame

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.npy_path)),
        "viz", "motion_cycle_paper.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plot_motion_cycle_paper(
        motions_dict=motions_dict,
        output_path=output_path,
        n_keyframes=args.n_keyframes,
        cycle_label=args.cycle_label,
        highlight_frames=highlight_frames,
        skeleton_scale=args.skeleton_scale,
    )


if __name__ == "__main__":
    main()