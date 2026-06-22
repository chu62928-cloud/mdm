#!/usr/bin/env python3
"""
Generate Fig 2: APT angle curves (baseline vs guided) over time.

Usage:
    python scripts/gen_fig2.py <data_dir1> [data_dir2 ...] [--labels L1 L2 ...] [--output <path>]
    python scripts/gen_fig2.py output_0608/n15/apt_joint_seed42 output_0608/n15/apt_both_seed42         --labels Joint Both --output output_0608/fig2.png
"""
import argparse, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture_guidance.angle_ops import pelvis_tilt_angle
from scipy.stats import pearsonr

COLORS = ['#8E3F61', '#8E3F61', '#8E3F61', '#8E3F61', '#8E3F61', '#8E3F61']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dirs', nargs='+', help='Directories containing comparison.npy')
    ap.add_argument('--labels', '-l', nargs='+', default=None,
                    help='Labels for each curve (default: directory basename)')
    ap.add_argument('--output', '-o', default=None, help='Output PNG path')
    ap.add_argument('--target', '-t', type=float, default=15.0,
                    help='Target APT angle (dashed line)')
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(d) for d in args.data_dirs]

    fig, axes = plt.subplots(1, len(args.data_dirs), figsize=(7 * len(args.data_dirs), 5))
    if len(args.data_dirs) == 1:
        axes = [axes]

    for i, (d, label) in enumerate(zip(args.data_dirs, labels)):
        ax = axes[i]
        npy_path = os.path.join(d, 'comparison.npy')
        if not os.path.exists(npy_path):
            print('ERROR: {} not found'.format(npy_path))
            continue

        data = np.load(npy_path, allow_pickle=True).item()
        xyz_b = data['motion_xyz'][0]
        xyz_g = data['motion_xyz_guided'][0]

        qb = torch.from_numpy(xyz_b).permute(2, 0, 1).unsqueeze(0).float()
        qg = torch.from_numpy(xyz_g).permute(2, 0, 1).unsqueeze(0).float()
        apt_b = (pelvis_tilt_angle(qb) * 180.0 / math.pi).numpy()[0]
        apt_g = (pelvis_tilt_angle(qg) * 180.0 / math.pi).numpy()[0]

        T = len(apt_b)
        t = np.arange(T) / 20.0
        c, _ = pearsonr(apt_b, apt_g)

        color = COLORS[i % len(COLORS)]
        ax.plot(t, apt_b, '#888888', lw=1.5, alpha=0.7, label='Baseline')
        ax.plot(t, apt_g, color, lw=2, alpha=0.9, label='Guided ({})'.format(label))
        ax.axhline(y=args.target, color=color, ls=':', lw=0.8, alpha=0.5)
        ax.axhline(y=0, color='gray', ls='-', lw=0.5, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('APT (deg)')
        ax.set_title('{}  (corr={:+.3f},  mean guided={:+.1f} deg)'.format(
            label, c, apt_g.mean()))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Fig 2: APT Angle Curves — Baseline vs Guided',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = args.output or os.path.join(os.path.dirname(args.data_dirs[0]), 'fig2_angle_curves.png')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved {}'.format(out_path))


if __name__ == '__main__':
    main()
