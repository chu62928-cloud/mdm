#!/usr/bin/env python3
"""
Generate Fig 4: Stride analysis from ankle trajectory.

Usage:
    python new/gen_fig4.py <data_dir> [--output <path>]
"""
import argparse, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--output', '-o', default=None)
    ap.add_argument('--side', default='L', choices=['L', 'R'])
    args = ap.parse_args()
    d = np.load(os.path.join(args.data_dir, 'comparison.npy'), allow_pickle=True).item()
    xyzB = d['motion_xyz'][0]; xyzG = d['motion_xyz_guided'][0]
    ankle_idx = 7 if args.side == 'L' else 8
    az_b = xyzB[ankle_idx, 2, :]; az_g = xyzG[ankle_idx, 2, :]
    T = len(az_b); t = np.arange(T) / 20.0
    min_b, _ = find_peaks(-az_b, distance=10); min_g, _ = find_peaks(-az_g, distance=10)
    sb = np.diff(min_b) if len(min_b) > 1 else [0]; sg = np.diff(min_g) if len(min_g) > 1 else [0]

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 8))
    ax_a.plot(t, az_b, '#3D4F8F', lw=1.5, alpha=0.8, label='Baseline')
    ax_a.plot(t, az_g, '#8E3F61', lw=1.5, alpha=0.9, label='Guided')
    for m in min_b: ax_a.axvline(t[m], color='#3D4F8F', alpha=0.2, lw=1)
    for m in min_g: ax_a.axvline(t[m], color='#8E3F61', alpha=0.2, lw=1)
    ax_a.set_ylabel('Ankle Z (m)'); ax_a.set_title('Ankle Vertical Trajectory ({} side)'.format(args.side))
    ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    ax_b.bar(np.arange(len(sb)) - 0.15, sb, 0.3, color='#3D4F8F', alpha=0.7,
             label='Baseline ({:.1f} frames)'.format(np.mean(sb) if len(sb) > 0 else 0))
    ax_b.bar(np.arange(len(sg)) + 0.15, sg, 0.3, color='#8E3F61', alpha=0.7,
             label='Guided ({:.1f} frames)'.format(np.mean(sg) if len(sg) > 0 else 0))
    ax_b.set_ylabel('Stride (frames)'); ax_b.set_xlabel('Stride #')
    ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3)
    fig.suptitle('Fig 4: Stride Analysis (illustrative)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = args.output or os.path.join(args.data_dir, 'fig4_stride.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved {}'.format(out))
if __name__ == '__main__': main()
