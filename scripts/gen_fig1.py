#!/usr/bin/env python3
"""
Generate Fig 1: sagittal skeleton comparison (baseline vs guided) for APT.

Usage:
    python scripts/gen_fig1.py <mode> <data_dir> [--output <path>] [--seed <int>]
    python scripts/gen_fig1.py both output_0608/n15/apt_both_seed42 --output output_0608/fig1.png
    python scripts/gen_fig1.py joint output_0608/n15/apt_joint_seed42
"""
import argparse, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture_guidance.angle_ops import pelvis_tilt_angle

P, LH, RH, S1 = 0, 1, 2, 3


def sag_2d(pts):
    return np.column_stack([-pts[:, 2], pts[:, 1]])


def draw_skel(ax, p2, bc, fc, lw=3.0):
    spine_i = [P, 3, 6, 9, 12, 15]
    ax.plot(p2[spine_i, 0], p2[spine_i, 1], color=bc, lw=lw, alpha=0.95,
            zorder=4, solid_capstyle='round')
    ax.add_patch(Circle(p2[15], 0.06, fc=fc, ec=bc, lw=1.5, alpha=0.9, zorder=3))
    for i, j in [(1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
                 (12, 13), (13, 16), (16, 18), (18, 20),
                 (12, 14), (14, 17), (17, 19), (19, 21)]:
        ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]],
                color=bc, lw=lw * 0.55, alpha=0.8, zorder=3, solid_capstyle='round')


def apt_vector(pts):
    """Compute sagittal projection of hip_center -> spine1, matching pelvis_tilt_angle."""
    hc = (pts[LH] + pts[RH]) / 2.0
    v = pts[S1] - hc
    lr = pts[RH] - pts[LH]
    lr = lr / (np.linalg.norm(lr) + 1e-8)
    sag = v - np.dot(v, lr) * lr
    return np.array([-sag[2], sag[1]]), hc


def draw_frame(ax, pts_3d, angle_deg, bone_color, fill_color):
    """Draw one skeleton frame with APT vector, vertical reference, and angle arc."""
    p2 = sag_2d(pts_3d)
    sv, hc = apt_vector(pts_3d)
    hc2 = sag_2d(hc.reshape(1, 3))[0]

    draw_skel(ax, p2, bone_color, fill_color)

    # Vertical gravity reference
    ref_len = 0.25
    ax.plot([hc2[0], hc2[0]], [hc2[1], hc2[1] + ref_len],
            '--', color='gray', lw=1.5, alpha=0.5, zorder=2)

    # Pelvis -> spine1 arrow
    v_norm = sv / (np.linalg.norm(sv) + 1e-8)
    v_plot = v_norm * ref_len
    ax.annotate('', xy=(hc2[0] + v_plot[0], hc2[1] + v_plot[1]),
                xytext=(hc2[0], hc2[1]),
                arrowprops=dict(arrowstyle='->', color=bone_color, lw=4.5, alpha=0.95), zorder=6)

    # Angle arc: visual direction is -angle_deg (because sagittal view flips z)
    visual_angle = 90.0 + angle_deg
    arc_r = 0.12
    arc = Arc((hc2[0], hc2[1]), 2 * arc_r, 2 * arc_r, angle=0,
              theta1=min(90, visual_angle), theta2=max(90, visual_angle),
              color=bone_color, lw=2.5, zorder=5)
    ax.add_patch(arc)

    # Degree label
    mid_angle = (90 + visual_angle) / 2.0 * math.pi / 180.0
    ax.text(hc2[0] + arc_r * 1.7 * math.cos(mid_angle),
            hc2[1] + arc_r * 1.7 * math.sin(mid_angle),
            '{:+.0f} deg'.format(angle_deg), fontsize=11, color=bone_color,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.85, ec='none'))

    # Hip center marker
    ax.scatter([hc2[0]], [hc2[1]], s=50, c=bone_color, zorder=7,
               marker='o', edgecolors='white', lw=1.5)

    # View limits
    m = 0.3
    ax.set_xlim(p2[:, 0].min() - m, p2[:, 0].max() + m)
    ax.set_ylim(p2[:, 1].min() - 0.15, p2[:, 1].max() + m)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', help='Directory containing comparison.npy')
    ap.add_argument('--output', '-o', default=None,
                    help='Output PNG path (default: <data_dir>/fig1_skeleton.png)')
    ap.add_argument('--frame', '-f', type=int, default=None,
                    help='Frame to plot (default: frame with max APT delta)')
    args = ap.parse_args()

    npy_path = os.path.join(args.data_dir, 'comparison.npy')
    if not os.path.exists(npy_path):
        print('ERROR: {} not found'.format(npy_path))
        sys.exit(1)

    data = np.load(npy_path, allow_pickle=True).item()
    xyz_b = data['motion_xyz'][0]
    xyz_g = data['motion_xyz_guided'][0]

    qb = torch.from_numpy(xyz_b).permute(2, 0, 1).unsqueeze(0).float()
    qg = torch.from_numpy(xyz_g).permute(2, 0, 1).unsqueeze(0).float()
    apt_b = (pelvis_tilt_angle(qb) * 180.0 / math.pi).numpy()[0]
    apt_g = (pelvis_tilt_angle(qg) * 180.0 / math.pi).numpy()[0]

    if args.frame is None:
        best_frame = int(np.argmax(np.abs(apt_g - apt_b)))
    else:
        best_frame = args.frame

    print('Frame {}: baseline={:.1f} deg, guided={:.1f} deg, delta={:.1f} deg'.format(
        best_frame, apt_b[best_frame], apt_g[best_frame], apt_g[best_frame] - apt_b[best_frame]))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    draw_frame(ax_l, xyz_b[:, :, best_frame], apt_b[best_frame],
               '#3D4F8F', '#A0B2DC')
    ax_l.set_title('Baseline (APT={:+.1f} deg)'.format(apt_b[best_frame]),
                   fontsize=12, fontweight='bold', color='#3D4F8F', pad=4)

    draw_frame(ax_r, xyz_g[:, :, best_frame], apt_g[best_frame],
               '#8E3F61', '#D4A0B6')
    ax_r.set_title('Guided (APT={:+.1f} deg)'.format(apt_g[best_frame]),
                   fontsize=12, fontweight='bold', color='#8E3F61', pad=4)

    fig.suptitle('Fig 1: APT Skeleton Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = args.output or os.path.join(args.data_dir, 'fig1_skeleton.png')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved {}'.format(out_path))


if __name__ == '__main__':
    main()
