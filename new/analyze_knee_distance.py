"""Analyze signed_knee_distance_sagittal distribution on training set + comparison.npy."""
import sys, math
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from posture_guidance.angle_ops import signed_knee_distance_sagittal
from posture_guidance.joint_indices import get_joint_idx

# ---- Load training set (500 clips) ----
data_dir = Path("dataset/HumanML3D")
joints_dir = data_dir / "new_joints"
split_file = data_dir / "train.txt"
ids = [l.strip() for l in split_file.read_text().split("\n") if l.strip()][:500]

all_dists = []    # all frames
up_dists = []     # upright frames only
stance_dists = [] # upright + stance (both feet on ground)

from posture_guidance.phase_detector import PhaseDetector
detector = PhaseDetector()

for i, cid in enumerate(ids):
    npy_path = joints_dir / "{}.npy".format(cid)
    if not npy_path.exists():
        continue
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[1] < 22:
        continue
    q = torch.from_numpy(arr[:, :22, :]).float()  # (T, 22, 3)

    # Compute distance for both knees
    with torch.no_grad():
        dl = signed_knee_distance_sagittal(q, side="left")
        dr = signed_knee_distance_sagittal(q, side="right")
    dists = torch.cat([dl, dr]).numpy()  # (2T,)
    all_dists.extend(dists.tolist())

    # Upright filter
    hip_y = (q[:, 1, 1] + q[:, 2, 1]) / 2
    knee_y = (q[:, 4, 1] + q[:, 5, 1]) / 2
    ankle_y = (q[:, 7, 1] + q[:, 8, 1]) / 2
    up_mask = ((hip_y > knee_y) & (knee_y > ankle_y) & ((hip_y - ankle_y) > 0.5)).numpy()
    up_mask_double = np.concatenate([up_mask, up_mask])
    if up_mask.sum() > 0:
        up_dists.extend(dists[up_mask_double].tolist())

    # Stance filter
    try:
        stance = detector.get_stance_mask(q).numpy()
        st_left = (stance[:, 0] > 0.5) & up_mask
        st_right = (stance[:, 1] > 0.5) & up_mask
        if st_left.sum() > 0:
            stance_dists.extend(dl.numpy()[st_left].tolist())
        if st_right.sum() > 0:
            stance_dists.extend(dr.numpy()[st_right].tolist())
    except:
        pass

    if (i+1) % 50 == 0:
        pct = 100 * (i+1) // min(500, len(ids))
        p1 = np.percentile(up_dists, 1) if up_dists else 0
        print("  [{}/{}] {}%  upright P1={:.4f}m".format(i+1, min(500, len(ids)), pct, p1))

print("")

def print_stats(label, arr):
    a = np.array(arr)
    print("{}: N={:,}".format(label, len(a)))
    print("  Mean={:.4f}m  Std={:.4f}m".format(a.mean(), a.std()))
    print("  Min={:.4f}m  Max={:.4f}m".format(a.min(), a.max()))
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print("  P{}={:.4f}m".format(p, np.percentile(a, p)))

print_stats("All frames", all_dists)
print_stats("Upright frames", up_dists)
print_stats("Stance frames", stance_dists)

# Target analysis
target = -0.05
for label, arr in [("upright", up_dists), ("stance", stance_dists)]:
    a = np.array(arr)
    below_target = (a < target).sum()
    below_zero = (a < 0).sum()
    print("{}: below target({:.2f}m)={}/{} ({:.3f}%)  below 0={}/{} ({:.3f}%)".format(
        label, target, below_target, len(a), 100*below_target/len(a),
        below_zero, len(a), 100*below_zero/len(a)))

# Histogram
if stance_dists:
    a = np.array(stance_dists)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(a, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(target, color='red', linestyle='--', linewidth=2, label='Target -0.05m')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, label='Straight knee (dist=0)')
    ax.axvline(np.percentile(a, 50), color='green', linestyle=':', label='P50={:.3f}m'.format(np.percentile(a, 50)))
    ax.axvline(np.percentile(a, 1), color='purple', linestyle=':', label='P1={:.3f}m'.format(np.percentile(a, 1)))
    ax.set_xlabel('Signed Knee Distance (m)')
    ax.set_ylabel('Density')
    ax.set_title('Training Set Stance-Phase Knee Distance Distribution')
    ax.legend()
    fig.tight_layout()
    out_dir = Path("new_results")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "knee_distance_distribution.png", dpi=150)
    print("Histogram saved: new_results/knee_distance_distribution.png")
