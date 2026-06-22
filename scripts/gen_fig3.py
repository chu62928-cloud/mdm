#!/usr/bin/env python3
"""
Generate Fig 3: APT-relevant muscle group activation change (horizontal bar chart).

Usage:
    python new/gen_fig3.py <data_dir> [--output <path>]
"""
import argparse, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'motion2muscle')
from muscle_guidance_mdm import build_muscle_guidance
from muscle_rollup import get_indices

APT_UP   = ['erector_spinae', 'iliopsoas', 'rectus_femoris']
APT_DOWN = ['rectus_abdominis', 'gluteus_maximus', 'gluteus_medius', 'transversus_abdominis']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--output', '-o', default=None)
    args = ap.parse_args()
    d = np.load(os.path.join(args.data_dir, 'comparison.npy'), allow_pickle=True).item()
    xb = torch.from_numpy(d['motion_hml_tj']).float(); xg = torch.from_numpy(d['motion_hml_tj_guided']).float()
    tm = torch.tensor(np.load('dataset/HumanML3D/Mean.npy')); ts = torch.tensor(np.load('dataset/HumanML3D/Std.npy'))
    mg = build_muscle_guidance(ckpt_path='motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth',
        posture_name='anterior_pelvic_tilt', assets_dir='motion2muscle', mdm_mean=tm, mdm_std=ts, same_normalization=True, device='cpu')
    mg.build_reference(xb)
    with torch.no_grad(): ab = mg._activations(xb).squeeze(0).numpy(); ag = mg._activations(xg).squeeze(0).numpy()
    items = []
    for direction, groups in [('UP', APT_UP), ('DOWN', APT_DOWN)]:
        for g in groups:
            for s in ['_R', '_L']:
                idx = get_indices(g + s, mg.mint_cols)
                if not idx: continue
                bv = float(ab[:, idx].mean()); gv = float(ag[:, idx].mean())
                dg = gv - bv; ok = (dg > 0) if direction == 'UP' else (dg < 0)
                items.append(('{}{} (expect {})'.format(g, s, direction), dg, ok))
    items.reverse()
    labels = [x[0] for x in items]; values = [x[1] for x in items]
    colors = ['#8E3F61' if x[2] else '#CCCCCC' for x in items]
    nc = sum(1 for x in items if x[2])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(values)), values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(values))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean activation change (guided - baseline)', fontsize=11)
    ax.axvline(0, color='black', lw=1); ax.grid(axis='x', alpha=0.3)
    ax.legend(handles=[Patch(facecolor='#8E3F61', label='Correct'), Patch(facecolor='#CCCCCC', label='Wrong')], fontsize=9)
    ax.set_title('Fig 3: APT Muscle Groups ({}/{}, {:.0f}%)'.format(nc, len(items), nc/len(items)*100), fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = args.output or os.path.join(args.data_dir, 'fig3_muscle_bars.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved {}'.format(out))
if __name__ == '__main__': main()
