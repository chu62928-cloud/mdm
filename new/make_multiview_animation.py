"""
scripts/make_multiview_animation_v2.py

论文风格的多视角动画：
    - 无关节圆点
    - 四肢用不同颜色
    - 骨架线条加粗
    - 头部用圆圈表示
    - 三视角：front / side / iso

参考论文风格：MDM、MotionDiffuse、PriorMDM 等
"""

import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

from posture_guidance.angle_ops import pelvis_tilt_angle


# ===== 配色 =====
PALETTE_BASELINE = {
    "spine":     "#1f3a93",
    "left_arm":  "#4a90e2",
    "right_arm": "#82b1ff",
    "left_leg":  "#1f3a93",
    "right_leg": "#5e8eda",
    "head":      "#1f3a93",
}
PALETTE_GUIDED = {
    "spine":     "#922b3e",
    "left_arm":  "#d6486a",
    "right_arm": "#f098a8",
    "left_leg":  "#922b3e",
    "right_leg": "#c3506a",
    "head":      "#922b3e",
}
COLOR_GROUND = "#D8D6CE"


# ===== 骨骼分组 =====
SPINE_BONES = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
LEFT_LEG_BONES  = [(0, 1), (1, 4), (4, 7),  (7, 10)]
RIGHT_LEG_BONES = [(0, 2), (2, 5), (5, 8),  (8, 11)]
LEFT_ARM_BONES  = [(12, 13), (13, 16), (16, 18), (18, 20)]
RIGHT_ARM_BONES = [(12, 14), (14, 17), (17, 19), (19, 21)]


# ===== 视角配置 =====
VIEWS = [
    {"name": "Front",  "elev": 8,  "azim": 90},
    {"name": "Side",   "elev": 8,  "azim": 0},
    {"name": "Iso",    "elev": 15, "azim": 45},
]


def setup_3d_ax(ax, xlim, ylim, zlim, view_elev, view_azim,
                 floor_y=0.0, title="", title_color="black"):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.grid(False)
    ax.view_init(elev=view_elev, azim=view_azim)

    # 地面
    xf = np.linspace(xlim[0], xlim[1], 4)
    yf = np.linspace(ylim[0], ylim[1], 4)
    Xf, Yf = np.meshgrid(xf, yf)
    Zf = np.ones_like(Xf) * floor_y
    ax.plot_surface(Xf, Yf, Zf, color=COLOR_GROUND,
                    alpha=0.4, edgecolor='none', zorder=-10)

    if title:
        ax.set_title(title, fontsize=10, color=title_color, weight='bold')


def draw_skeleton_3d_paper(ax, q_frame, palette, line_width=3.0):
    """
    论文风格 3D 骨架：
        - 无关节圆点
        - 四肢分色
        - 头部圆圈
    """
    # 骨骼线（按部位分色）
    bone_groups = [
        (SPINE_BONES,     palette["spine"],     line_width * 1.15),
        (LEFT_LEG_BONES,  palette["left_leg"],  line_width * 1.0),
        (RIGHT_LEG_BONES, palette["right_leg"], line_width * 1.0),
        (LEFT_ARM_BONES,  palette["left_arm"],  line_width * 0.85),
        (RIGHT_ARM_BONES, palette["right_arm"], line_width * 0.85),
    ]
    for bones, color, lw in bone_groups:
        for (a, b) in bones:
            if a < q_frame.shape[0] and b < q_frame.shape[0]:
                ax.plot(
                    [q_frame[a, 0], q_frame[b, 0]],
                    [q_frame[a, 2], q_frame[b, 2]],
                    [q_frame[a, 1], q_frame[b, 1]],
                    color=color, lw=lw, alpha=1.0,
                    solid_capstyle='round', zorder=2,
                )

    # 头部用 3D 球（用 scatter 一个大点模拟）
    if q_frame.shape[0] > 15:
        head = q_frame[15]
        neck = q_frame[12]
        head_size = max(np.linalg.norm(head - neck) * 25, 60)
        ax.scatter(
            [head[0]], [head[2]], [head[1]],
            s=head_size,
            facecolor='white',
            edgecolor=palette["head"],
            linewidths=2.5,
            zorder=4, depthshade=False,
        )


def get_floor(xyz_seq):
    return float(np.percentile(xyz_seq[:, 1, :].flatten(), 2))


def compute_uniform_lim(xyz_list, scale=1.0, margin=0.15):
    """放大坐标范围让骨架占满图块"""
    all_pts = np.concatenate(
        [xyz.reshape(-1, 3) for xyz in xyz_list], axis=0
    )
    floor_y = float(np.percentile(all_pts[:, 1], 2))
    head_y  = float(np.percentile(all_pts[:, 1], 99))
    height  = head_y - floor_y

    x_center = float(np.mean(all_pts[:, 0]))
    z_center = float(np.mean(all_pts[:, 2]))

    # 缩小水平范围让人物相对放大
    half_w = max(np.ptp(all_pts[:, 0]),
                  np.ptp(all_pts[:, 2]),
                  height * 0.6) / 2 * (1.0 / scale) + height * margin

    xlim = (x_center - half_w, x_center + half_w)
    zlim = (z_center - half_w, z_center + half_w)
    ylim = (floor_y - height * 0.05, floor_y + height * 1.1)

    return xlim, zlim, ylim, floor_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--fmt",        choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--zoom",       type=float, default=1.4,
                         help="人物放大倍数（>1 让骨架占图块更多）")
    parser.add_argument("--line_width", type=float, default=3.5,
                         help="骨架线宽")
    args = parser.parse_args()

    print(f"Loading {args.npy_path}")
    data = np.load(args.npy_path, allow_pickle=True).item()
    fps = int(data.get("fps", 20))

    xyz_base   = data["motion_xyz"][args.sample_idx]
    xyz_guided = data["motion_xyz_guided"][args.sample_idx]
    T = xyz_base.shape[-1]

    xlim, zlim, ylim, floor_y = compute_uniform_lim(
        [xyz_base, xyz_guided], scale=args.zoom,
    )

    # 实时骨盆角
    q_base   = torch.from_numpy(xyz_base).permute(2, 0, 1).float()
    q_guided = torch.from_numpy(xyz_guided).permute(2, 0, 1).float()
    angle_base   = (pelvis_tilt_angle(q_base)   * 180 / math.pi).numpy()
    angle_guided = (pelvis_tilt_angle(q_guided) * 180 / math.pi).numpy()

    fig = plt.figure(figsize=(15, 10))
    axes = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            axes.append(ax)

    # 行标签
    fig.text(0.015, 0.74, "Baseline",
              fontsize=15, weight='bold',
              color=PALETTE_BASELINE["spine"],
              rotation=90, ha='center', va='center')
    fig.text(0.015, 0.28, "Guided",
              fontsize=15, weight='bold',
              color=PALETTE_GUIDED["spine"],
              rotation=90, ha='center', va='center')

    def update(t):
        for ax in axes:
            ax.clear()

        # baseline 三视角
        for i, view in enumerate(VIEWS):
            setup_3d_ax(
                axes[i], xlim, zlim, ylim,
                view_elev=view["elev"], view_azim=view["azim"],
                floor_y=floor_y,
                title=view["name"],
                title_color="#444",
            )
            draw_skeleton_3d_paper(
                axes[i], xyz_base[:, :, t],
                PALETTE_BASELINE,
                line_width=args.line_width,
            )

        # guided 三视角
        for i, view in enumerate(VIEWS):
            setup_3d_ax(
                axes[i + 3], xlim, zlim, ylim,
                view_elev=view["elev"], view_azim=view["azim"],
                floor_y=floor_y,
            )
            draw_skeleton_3d_paper(
                axes[i + 3], xyz_guided[:, :, t],
                PALETTE_GUIDED,
                line_width=args.line_width,
            )

        fig.suptitle(
            f"Frame {t}/{T-1}    "
            f"Baseline: {angle_base[t]:+.1f}°    "
            f"Guided: {angle_guided[t]:+.1f}°",
            fontsize=13, y=0.97,
        )

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    anim = FuncAnimation(fig, update, frames=T,
                          interval=1000 / fps, blit=False)

    if args.output is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.npy_path)), "viz"
        )
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(
            out_dir, f"multiview_paper.{args.fmt}"
        )

    if args.fmt == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2800)
            anim.save(args.output, writer=writer)
        except Exception as e:
            print(f"mp4 失败 ({e})，改 gif")
            args.output = args.output.replace(".mp4", ".gif")
            args.fmt = "gif"

    if args.fmt == "gif":
        writer = PillowWriter(fps=fps)
        anim.save(args.output, writer=writer)

    plt.close()
    print(f"✓ saved → {args.output}")


if __name__ == "__main__":
    main()