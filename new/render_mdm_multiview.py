import os
import math
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# =========================================================
# 1. HumanML / MDM 22-joint skeleton
# =========================================================
# Typical HumanML3D / MDM kinematic chains
T2M_KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],         # right leg
    [0, 1, 4, 7, 10],         # left leg
    [0, 3, 6, 9, 12, 15],     # spine -> head
    [9, 14, 17, 19, 21],      # right arm
    [9, 13, 16, 18, 20],      # left arm
]

CHAIN_COLORS = [
    "#1f77b4",  # right leg
    "#ff7f0e",  # left leg
    "#2ca02c",  # spine
    "#d62728",  # right arm
    "#9467bd",  # left arm
]


# =========================================================
# 2. args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Render side-view / multi-view videos from official MDM results.npy"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to official MDM results.npy")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index in results.npy")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"],
                        help="Vertical axis of motion coordinates")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Output video fps")
    parser.add_argument("--width", type=int, default=900,
                        help="Single-view output width")
    parser.add_argument("--height", type=int, default=700,
                        help="Single-view output height")
    parser.add_argument("--trail", type=int, default=20,
                        help="How many previous root positions to show")
    parser.add_argument("--elev_front", type=float, default=15.0)
    parser.add_argument("--azim_front", type=float, default=-90.0)
    parser.add_argument("--elev_side", type=float, default=15.0)
    parser.add_argument("--azim_side", type=float, default=0.0)
    parser.add_argument("--elev_oblique", type=float, default=20.0)
    parser.add_argument("--azim_oblique", type=float, default=-45.0)
    parser.add_argument("--elev_top", type=float, default=85.0)
    parser.add_argument("--azim_top", type=float, default=-90.0)
    parser.add_argument("--draw_ground", action="store_true",
                        help="Draw a ground plane")
    return parser.parse_args()


# =========================================================
# 3. load official MDM results.npy
# =========================================================
def load_results(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported npy format: {path}")


def extract_motion(data, sample_idx=0):
    if "motion" not in data:
        raise KeyError("'motion' key not found")

    motion = np.asarray(data["motion"])
    if motion.ndim != 4:
        raise ValueError(f"Expected motion shape (N, J, 3, T), got {motion.shape}")

    arr = motion[sample_idx]                 # (J, 3, T)
    arr = np.transpose(arr, (2, 0, 1))      # -> (T, J, 3)

    lengths = data.get("lengths", None)
    if lengths is not None:
        length = int(np.asarray(lengths)[sample_idx])
        arr = arr[:length]
    else:
        length = arr.shape[0]

    prompt = ""
    texts = data.get("text", None)
    if texts is not None and sample_idx < len(texts):
        prompt = str(texts[sample_idx])

    return arr, prompt, length


# =========================================================
# 4. coordinate conversion
# =========================================================
def convert_to_plot_xyz(motion_tjc, up_axis="y"):
    """
    Convert motion coords to plotting coords where Z is vertical.
    Input:  (T, J, 3)
    Output: (T, J, 3) with plotting axes [X_plot, Y_plot, Z_plot]
    """
    if up_axis == "y":
        # original: x, y(up), z
        # plot as  : x, z, y
        out = np.stack([
            motion_tjc[..., 0],
            motion_tjc[..., 2],
            motion_tjc[..., 1],
        ], axis=-1)
    elif up_axis == "z":
        # already z-up
        out = motion_tjc.copy()
    elif up_axis == "x":
        # original x is up -> move to plot z
        out = np.stack([
            motion_tjc[..., 1],
            motion_tjc[..., 2],
            motion_tjc[..., 0],
        ], axis=-1)
    else:
        raise ValueError(f"Unsupported up_axis: {up_axis}")
    return out


def normalize_ground_z(motion_xyz):
    z_min = np.min(motion_xyz[..., 2])
    out = motion_xyz.copy()
    out[..., 2] -= z_min
    return out


# =========================================================
# 5. render helpers
# =========================================================
def fig_to_rgb(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def compute_axis_limits(motion_xyz, padding=0.15):
    xyz = motion_xyz.reshape(-1, 3)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    center = 0.5 * (mins + maxs)
    span = (maxs - mins)
    max_span = max(float(span.max()), 1e-3)

    half = 0.5 * max_span * (1.0 + padding)

    xlim = (center[0] - half, center[0] + half)
    ylim = (center[1] - half, center[1] + half)
    zmin = max(0.0, mins[2] - 0.05 * max_span)
    zmax = maxs[2] + 0.25 * max_span
    zlim = (zmin, zmax)

    return xlim, ylim, zlim


def draw_ground_plane(ax, xlim, ylim, z=0.0):
    xx = np.array([[xlim[0], xlim[1]], [xlim[0], xlim[1]]])
    yy = np.array([[ylim[0], ylim[0]], [ylim[1], ylim[1]]])
    zz = np.array([[z, z], [z, z]])
    ax.plot_surface(xx, yy, zz, alpha=0.08, color="gray", shade=False)


def draw_skeleton(ax, joints_xyz, root_trail=None):
    # root trail
    if root_trail is not None and len(root_trail) > 1:
        ax.plot(
            root_trail[:, 0], root_trail[:, 1], root_trail[:, 2],
            color="black", linewidth=1.2, alpha=0.45
        )

    # chains
    for chain, color in zip(T2M_KINEMATIC_CHAINS, CHAIN_COLORS):
        pts = joints_xyz[chain]
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color=color, linewidth=2.5, alpha=0.95
        )

    # joints
    ax.scatter(
        joints_xyz[:, 0], joints_xyz[:, 1], joints_xyz[:, 2],
        s=18, color="black", alpha=0.95
    )


def render_single_view_frame(
    motion_xyz, frame_idx, prompt, xlim, ylim, zlim,
    elev, azim, width=900, height=700, trail=20, draw_ground=False, view_name=""
):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    joints = motion_xyz[frame_idx]
    root_id = 0
    t0 = max(0, frame_idx - trail)
    root_trail = motion_xyz[t0:frame_idx + 1, root_id, :]

    draw_skeleton(ax, joints, root_trail=root_trail)

    if draw_ground:
        draw_ground_plane(ax, xlim, ylim, z=0.0)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    short_prompt = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    fig.suptitle(f"{view_name} | frame={frame_idx} | {short_prompt}", fontsize=11, y=0.98)

    fig.tight_layout()
    return fig_to_rgb(fig)


def render_multiview_frame(
    motion_xyz, frame_idx, prompt, xlim, ylim, zlim,
    views, width=1400, height=1000, trail=20, draw_ground=False
):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    joints = motion_xyz[frame_idx]
    root_id = 0
    t0 = max(0, frame_idx - trail)
    root_trail = motion_xyz[t0:frame_idx + 1, root_id, :]

    for ax, (name, elev, azim) in zip(axes, views):
        draw_skeleton(ax, joints, root_trail=root_trail)
        if draw_ground:
            draw_ground_plane(ax, xlim, ylim, z=0.0)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(name, fontsize=10)

    short_prompt = prompt if len(prompt) <= 90 else prompt[:87] + "..."
    fig.suptitle(f"frame={frame_idx} | {short_prompt}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig_to_rgb(fig)


def write_video(frames_rgb, out_path, fps=20.0):
    if len(frames_rgb) == 0:
        raise ValueError("No frames to write")

    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    try:
        for fr in frames_rgb:
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


# =========================================================
# 6. main
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = load_results(args.input)
    motion_tjc, prompt, length = extract_motion(data, sample_idx=args.sample_idx)

    motion_xyz = convert_to_plot_xyz(motion_tjc, up_axis=args.up_axis)
    motion_xyz = normalize_ground_z(motion_xyz)

    xlim, ylim, zlim = compute_axis_limits(motion_xyz, padding=0.18)

    views = {
        "front": ("Front view", args.elev_front, args.azim_front),
        "side": ("Side view", args.elev_side, args.azim_side),
        "oblique": ("Oblique view", args.elev_oblique, args.azim_oblique),
        "top": ("Top view", args.elev_top, args.azim_top),
    }

    print("=" * 72)
    print("Rendering views...")
    print(f"Input     : {args.input}")
    print(f"Sample idx: {args.sample_idx}")
    print(f"Frames    : {motion_xyz.shape[0]}")
    print(f"Prompt    : {prompt}")
    print("=" * 72)

    # single views
    for key in ["front", "side", "oblique", "top"]:
        name, elev, azim = views[key]
        out_path = os.path.join(args.outdir, f"{key}.mp4")
        print(f"[Render] {key} -> {out_path}")

        frames = []
        for t in range(motion_xyz.shape[0]):
            fr = render_single_view_frame(
                motion_xyz=motion_xyz,
                frame_idx=t,
                prompt=prompt,
                xlim=xlim, ylim=ylim, zlim=zlim,
                elev=elev, azim=azim,
                width=args.width, height=args.height,
                trail=args.trail,
                draw_ground=args.draw_ground,
                view_name=name
            )
            frames.append(fr)

        write_video(frames, out_path, fps=args.fps)

    # multi-view 2x2
    mv_out = os.path.join(args.outdir, "multiview_2x2.mp4")
    print(f"[Render] multiview -> {mv_out}")

    mv_frames = []
    mv_views = [
        views["front"],
        views["side"],
        views["oblique"],
        views["top"],
    ]
    for t in range(motion_xyz.shape[0]):
        fr = render_multiview_frame(
            motion_xyz=motion_xyz,
            frame_idx=t,
            prompt=prompt,
            xlim=xlim, ylim=ylim, zlim=zlim,
            views=mv_views,
            width=1400,
            height=1000,
            trail=args.trail,
            draw_ground=args.draw_ground
        )
        mv_frames.append(fr)

    write_video(mv_frames, mv_out, fps=args.fps)

    print("=" * 72)
    print("Done.")
    print(f"Outputs saved to: {args.outdir}")
    print("=" * 72)


if __name__ == "__main__":
    main()