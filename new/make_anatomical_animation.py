"""
scripts/make_anatomical_animation_v3.py

修复内容：
    1. 脖子可见：头部圆不再遮住 neck，画一条明显的颈部线段
    2. 驼背视觉更明显：在绘制时放大脊柱曲线的弯曲程度
    3. angle_fn 自动选择（与 v2 一致）
"""

import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.interpolate import CubicSpline

from posture_guidance.angle_ops import (
    pelvis_tilt_angle,
    signed_knee_angle,
    spine_kyphosis_angle,
    head_forward_offset,
)

# 尝试导入新函数（如果还没加到 angle_ops.py 就用 fallback）
try:
    from posture_guidance.angle_ops import spine_posterior_bulge
except ImportError:
    spine_posterior_bulge = None


# ========== 配色 ==========
COLOR_BASELINE_BONE = "#3D4F8F"
COLOR_BASELINE_FILL = "#A0B2DC"
COLOR_GUIDED_BONE   = "#8E3F61"
COLOR_GUIDED_FILL   = "#D4A0B6"
COLOR_GROUND        = "#CFCDC4"

# ========== 关节索引 ==========
J_PELVIS = 0
J_L_HIP  = 1;  J_R_HIP    = 2
J_SPINE1 = 3
J_L_KNEE = 4;  J_R_KNEE   = 5
J_SPINE2 = 6
J_L_ANKLE= 7;  J_R_ANKLE  = 8
J_SPINE3 = 9
J_L_FOOT = 10; J_R_FOOT   = 11
J_NECK   = 12
J_L_COLL = 13; J_R_COLL   = 14
J_HEAD   = 15
J_L_SHLDR= 16; J_R_SHLDR  = 17
J_L_ELBOW= 18; J_R_ELBOW  = 19
J_L_WRIST= 20; J_R_WRIST  = 21


# =========================================================
# angle_fn 自动选择（与 v2 完全一致）
# =========================================================

def _to_tensor(xyz_np):
    return torch.from_numpy(xyz_np).permute(2, 0, 1).float()

def _wrap_pelvis_tilt(q):
    return (pelvis_tilt_angle(q) * 180 / math.pi).numpy()

def _wrap_kyphosis(q):
    if spine_posterior_bulge is not None:
        return spine_posterior_bulge(q).numpy()
    return (spine_kyphosis_angle(q) * 180 / math.pi).numpy()

def _wrap_knee_left(q):
    return (signed_knee_angle(q, side="left") * 180 / math.pi).numpy()

def _wrap_knee_right(q):
    return (signed_knee_angle(q, side="right") * 180 / math.pi).numpy()

def _wrap_knee_both(q):
    l = signed_knee_angle(q, side="left")
    r = signed_knee_angle(q, side="right")
    return ((l + r) / 2 * 180 / math.pi).numpy()

def _wrap_head_forward(q):
    return head_forward_offset(q).numpy()

ANGLE_MAP = {
    "骨盆前倾":      (_wrap_pelvis_tilt,  "Pelvic tilt",       "°"),
    "骨盆前倾_深蹲": (_wrap_pelvis_tilt,  "Pelvic tilt",       "°"),
    "膝超伸":        (_wrap_knee_both,    "Knee angle",        "°"),
    "膝超伸_左":     (_wrap_knee_left,    "Left knee",         "°"),
    "膝超伸_右":     (_wrap_knee_right,   "Right knee",        "°"),
    "驼背":          (_wrap_kyphosis,     "Spinal bulge",      "m"),
    "头前伸":        (_wrap_head_forward, "Head forward",      "m"),
}

def get_angle_fn_and_label(posture_instructions):
    if not posture_instructions:
        return _wrap_pelvis_tilt, "Pelvic tilt", "°"
    primary = posture_instructions[0]
    if primary in ANGLE_MAP:
        return ANGLE_MAP[primary]
    if "骨盆" in primary: return _wrap_pelvis_tilt, "Pelvic tilt", "°"
    if "膝"   in primary: return _wrap_knee_both,   "Knee angle",  "°"
    if "驼背" in primary: return _wrap_kyphosis,    "Spinal bulge","m"
    if "头"   in primary: return _wrap_head_forward,"Head forward","m"
    print(f"[Warning] 未知指令 '{primary}'，使用 pelvis_tilt")
    return _wrap_pelvis_tilt, "Pelvic tilt", "°"


# =========================================================
# 朝向和投影
# =========================================================

def estimate_facing(joints_3d_seq):
    lr = (joints_3d_seq[J_R_HIP,:,:] - joints_3d_seq[J_L_HIP,:,:]).mean(1)
    lr[1] = 0
    right = lr / (np.linalg.norm(lr) + 1e-8)
    fwd   = np.cross([0.,1.,0.], right)
    return fwd / (np.linalg.norm(fwd) + 1e-8)

def project_sagittal(joints_3d, forward):
    return np.stack([joints_3d @ forward, joints_3d[:,1]], axis=-1)


# =========================================================
# 骨架绘制（修复版）
# =========================================================

def draw_lumbar_curve(ax, pts, color, lw=4.5, alpha=1.0,
                      bulge_amplify=1.0):
    """
    脊柱平滑曲线。

    bulge_amplify > 1.0 时，在视觉上放大脊柱的弯曲程度，
    让驼背效果更容易被肉眼识别。
    """
    # 原始脊柱点
    spine_pts = np.array([
        pts[J_PELVIS],
        pts[J_SPINE1],
        pts[J_SPINE2],
        pts[J_SPINE3],
        pts[J_NECK],
    ])

    if bulge_amplify > 1.0:
        # 计算 spine1-neck 基准线，把中间点的偏差放大
        baseline_start = spine_pts[0]   # pelvis
        baseline_end   = spine_pts[-1]  # neck
        baseline_vec   = baseline_end - baseline_start
        baseline_len   = np.linalg.norm(baseline_vec) + 1e-8
        baseline_dir   = baseline_vec / baseline_len

        amplified = spine_pts.copy()
        for i in range(1, len(spine_pts) - 1):
            # 当前点相对基准线的参数位置
            t     = np.dot(spine_pts[i] - baseline_start, baseline_dir) / baseline_len
            # 基准线上对应的点
            proj  = baseline_start + t * baseline_vec
            # 偏差向量
            delta = spine_pts[i] - proj
            # 放大偏差
            amplified[i] = proj + delta * bulge_amplify

        spine_pts = amplified

    diffs    = np.diff(spine_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    t        = np.concatenate([[0], np.cumsum(seg_lens)])

    if t[-1] > 1e-6:
        try:
            cs_x   = CubicSpline(t, spine_pts[:, 0])
            cs_y   = CubicSpline(t, spine_pts[:, 1])
            t_fine = np.linspace(0, t[-1], 50)
            ax.plot(cs_x(t_fine), cs_y(t_fine),
                    color=color, lw=lw, alpha=alpha,
                    solid_capstyle='round', zorder=4)
            return
        except Exception:
            pass

    ax.plot(spine_pts[:, 0], spine_pts[:, 1],
            color=color, lw=lw, alpha=alpha, zorder=4)


def draw_neck(ax, pts, bone_c, lw=3.0, alpha=1.0):
    """
    ★ 修复：单独画颈部线段（neck → head 方向）
    确保脖子可见，不被头部圆遮住。
    """
    neck = pts[J_NECK]
    head = pts[J_HEAD]

    if neck is None or head is None:
        return

    # 颈部向量
    neck_to_head = head - neck
    neck_len = np.linalg.norm(neck_to_head)

    if neck_len < 1e-3:
        return

    # 只画到头部圆边缘（不画进圆圈里）
    body_size  = abs(pts[J_HEAD, 1] - pts[J_PELVIS, 1])
    head_radius = max(neck_len * 0.85, 0.06 * body_size)

    # 颈部线段终点 = 头部圆心下方（圆的边缘）
    neck_dir  = neck_to_head / neck_len
    neck_end  = neck + neck_dir * (neck_len - head_radius * 1.1)

    ax.plot(
        [neck[0], neck_end[0]],
        [neck[1], neck_end[1]],
        color=bone_c, lw=lw, alpha=alpha,
        solid_capstyle='round', zorder=4,
    )


def draw_limb(ax, p1, p2, bone_c, fill_c, width_ratio=0.08, alpha=1.0):
    direction = p2 - p1
    length    = np.linalg.norm(direction)
    if length < 1e-3:
        return
    perp   = np.array([-direction[1], direction[0]]) / length
    half_w = length * width_ratio
    corners = np.array([
        p1 + perp * half_w,
        p1 - perp * half_w,
        p2 - perp * half_w * 0.65,
        p2 + perp * half_w * 0.65,
    ])
    ax.add_patch(Polygon(corners, closed=True,
                          facecolor=fill_c, edgecolor=bone_c,
                          alpha=alpha * 0.65, linewidth=0.8, zorder=3))


def draw_anatomical(ax, coords_2d, bone_c, fill_c,
                    alpha=1.0, bulge_amplify=1.0):
    """
    完整解剖骨架绘制。

    bulge_amplify: 驼背体态传入 >1 的值（如 2.0），
                   骨盆前倾等其他体态传入 1.0
    """
    pts       = coords_2d
    body_size = abs(pts[J_HEAD, 1] - pts[J_PELVIS, 1])

    # 1. 躯干填充
    if all(i < pts.shape[0] for i in [J_L_SHLDR, J_NECK,
                                        J_R_SHLDR, J_R_HIP, J_L_HIP]):
        ax.add_patch(Polygon(
            np.array([pts[J_L_SHLDR], pts[J_NECK], pts[J_R_SHLDR],
                      pts[J_R_HIP],   pts[J_L_HIP]]),
            closed=True, facecolor=fill_c, edgecolor=bone_c,
            alpha=alpha * 0.55, linewidth=1.3, zorder=2,
        ))

    # 2. 骨盆楔形
    center    = (pts[J_L_HIP] + pts[J_R_HIP]) / 2
    top       = center + np.array([0,  0.04 * body_size])
    front     = center + np.array([0.06 * body_size, -0.01 * body_size])
    ax.add_patch(Polygon(
        np.array([top, pts[J_L_HIP], front, pts[J_R_HIP]]),
        closed=True, facecolor=fill_c, edgecolor=bone_c,
        alpha=alpha * 0.78, linewidth=1.4, zorder=2.6,
    ))

    # 3. 四肢
    limbs = [
        (J_L_HIP,    J_L_KNEE,   0.10),
        (J_R_HIP,    J_R_KNEE,   0.10),
        (J_L_KNEE,   J_L_ANKLE,  0.08),
        (J_R_KNEE,   J_R_ANKLE,  0.08),
        (J_L_SHLDR,  J_L_ELBOW,  0.07),
        (J_R_SHLDR,  J_R_ELBOW,  0.07),
        (J_L_ELBOW,  J_L_WRIST,  0.06),
        (J_R_ELBOW,  J_R_WRIST,  0.06),
    ]
    for a, b, w in limbs:
        if a < pts.shape[0] and b < pts.shape[0]:
            draw_limb(ax, pts[a], pts[b], bone_c, fill_c,
                       width_ratio=w, alpha=alpha)

    # 4. 脊柱曲线（驼背时放大弯曲）
    draw_lumbar_curve(ax, pts, bone_c, lw=4.5, alpha=alpha,
                      bulge_amplify=bulge_amplify)

    # 5. ★ 修复：颈部单独绘制
    draw_neck(ax, pts, bone_c, lw=3.2, alpha=alpha)

    # 6. 关节点
    for idx in [J_L_HIP, J_R_HIP, J_L_KNEE, J_R_KNEE,
                J_L_ANKLE, J_R_ANKLE, J_L_SHLDR, J_R_SHLDR,
                J_L_ELBOW, J_R_ELBOW, J_NECK]:
        if idx < pts.shape[0]:
            ax.add_patch(Circle(
                (pts[idx, 0], pts[idx, 1]),
                radius=0.018 * body_size,
                facecolor=bone_c, edgecolor='white',
                linewidth=1.0, alpha=alpha, zorder=5,
            ))

    # 7. ★ 修复：头部圆圈（不再偏移，圆心就在 J_HEAD 位置）
    if J_HEAD < pts.shape[0] and J_NECK < pts.shape[0]:
        neck_head_dist = np.linalg.norm(pts[J_HEAD] - pts[J_NECK])
        head_radius    = max(neck_head_dist * 0.7, 0.055 * body_size)
        ax.add_patch(Circle(
            (pts[J_HEAD, 0], pts[J_HEAD, 1]),   # ← 不再偏移
            radius=head_radius,
            facecolor=fill_c, edgecolor=bone_c,
            linewidth=1.5, alpha=alpha * 0.85, zorder=4.5,
        ))

    # 8. 足部
    for ankle_idx, foot_idx in [(J_L_ANKLE, J_L_FOOT),
                                  (J_R_ANKLE, J_R_FOOT)]:
        if ankle_idx < pts.shape[0] and foot_idx < pts.shape[0]:
            ankle = pts[ankle_idx]
            foot  = pts[foot_idx]
            heel  = np.array([ankle[0] - 0.05 * body_size,
                               min(ankle[1] - 0.015 * body_size, foot[1])])
            ax.add_patch(Polygon(
                np.array([heel, foot, ankle]),
                closed=True, facecolor=fill_c, edgecolor=bone_c,
                linewidth=1.0, alpha=alpha * 0.8, zorder=3,
            ))


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path",   type=str)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output",   type=str, default=None)
    parser.add_argument("--fmt",      choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--posture",  type=str, default=None)
    parser.add_argument("--bulge_amplify", type=float, default=2.5,
                         help="驼背可视化放大倍数，仅对驼背体态生效")
    args = parser.parse_args()

    print(f"Loading {args.npy_path}")
    data = np.load(args.npy_path, allow_pickle=True).item()
    fps  = int(data.get("fps", 20))

    xyz_base   = data["motion_xyz"][args.sample_idx]
    xyz_guided = data["motion_xyz_guided"][args.sample_idx]
    T = xyz_base.shape[-1]

    if args.posture:
        posture_instructions = [args.posture]
    else:
        posture_instructions = data.get("posture_instructions", [])

    print(f"[posture] {posture_instructions}")

    angle_fn, angle_label, angle_unit = get_angle_fn_and_label(
        posture_instructions
    )
    print(f"[angle_fn] {angle_label} ({angle_unit})")

    # 驼背时才启用可视化放大
    is_kyphosis   = any("驼背" in p for p in posture_instructions)
    bulge_amplify = args.bulge_amplify if is_kyphosis else 1.0
    if is_kyphosis:
        print(f"[kyphosis] bulge_amplify={bulge_amplify}x")

    # 朝向和投影
    forward     = estimate_facing(xyz_base)
    coords_base = np.array([project_sagittal(xyz_base[:,:,t], forward)
                             for t in range(T)])
    coords_guided = np.array([project_sagittal(xyz_guided[:,:,t], forward)
                               for t in range(T)])

    # 坐标范围
    all_2d   = np.concatenate([coords_base.reshape(-1,2),
                                coords_guided.reshape(-1,2)], axis=0)
    x_min, x_max = all_2d[:,0].min(), all_2d[:,0].max()
    y_min, y_max = all_2d[:,1].min(), all_2d[:,1].max()
    height   = y_max - y_min
    pad      = height * 0.15
    floor_y  = y_min

    # 角度时间序列
    q_base   = _to_tensor(xyz_base)
    q_guided = _to_tensor(xyz_guided)
    angle_base   = angle_fn(q_base)
    angle_guided = angle_fn(q_guided)

    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(11, 5.5))

    def setup_ax(ax, title, color):
        ax.clear()
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(floor_y - pad * 0.3, y_max + pad)
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(floor_y, color=COLOR_GROUND, lw=1.5)
        ax.set_title(title, fontsize=12, color=color, weight='bold')

    def update(t):
        val_b = angle_base[t]
        val_g = angle_guided[t]

        setup_ax(ax_b,
                  f"Baseline   {angle_label}: "
                  f"{val_b:+.3f}{angle_unit}" if angle_unit == "m"
                  else f"Baseline   {angle_label}: {val_b:+.1f}{angle_unit}",
                  COLOR_BASELINE_BONE)
        setup_ax(ax_g,
                  f"Guided     {angle_label}: "
                  f"{val_g:+.3f}{angle_unit}" if angle_unit == "m"
                  else f"Guided     {angle_label}: {val_g:+.1f}{angle_unit}",
                  COLOR_GUIDED_BONE)

        draw_anatomical(ax_b, coords_base[t],
                         COLOR_BASELINE_BONE, COLOR_BASELINE_FILL,
                         bulge_amplify=1.0)           # baseline 不放大
        draw_anatomical(ax_g, coords_guided[t],
                         COLOR_GUIDED_BONE, COLOR_GUIDED_FILL,
                         bulge_amplify=bulge_amplify)  # guided 根据体态决定

        fig.suptitle(f"Frame {t}/{T-1}", fontsize=11, y=0.96)

    anim = FuncAnimation(fig, update, frames=T,
                          interval=1000/fps, blit=False)

    if args.output is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.npy_path)), "viz"
        )
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir,
                                    f"anatomical_animation.{args.fmt}")

    if args.fmt == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2400)
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