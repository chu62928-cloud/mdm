#!/usr/bin/env python3
"""
new/test_apt_reference.py

Test: replace the baseline-PPT reference with a mild-APT (~10°) reference
in L_dense muscle guidance, and check whether the pelvic tilt direction flips.

Core hypothesis:
  L_dense chain component (weight 1.5) has gate_p = relu(ref_p - mean_p).
  When ref_p (gluteus_maximus) comes from a PPT baseline (~-7°), it's elevated,
  causing the chain to push in the wrong direction.
  When ref_p comes from an APT reference, gluteus is naturally low → gate_p=0
  → chain disabled → only ratio mode (correct direction) remains.

Flow:
  1. Load joint-guided APT comparison.npy (baseline ~-7°, guided ~+20°)
  2. Linearly interpolate in MDM feature space to get ~10° APT motion
  3. Verify interpolated pelvis angle ≈ 10° via FK
  4. Extract muscle activations from interpolated motion → build APT reference
  5. Run pure muscle guidance from baseline with APT reference
  6. Compare: pelvic tilt direction, muscle activations, loss components

Usage:
    python new/test_apt_reference.py \
        --joint_apt_npy output_0608/apt_joint_seed42/comparison.npy \
        --target_pelvis_angle 10.0 \
        --muscle_ckpt motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
        --muscle_weight 50 \
        --n_steps 20 \
        --device cuda
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.optim import SGD

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_ASSETS = os.path.join(_ROOT, "motion2muscle")
for p in (_ROOT, _ASSETS):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loaders.humanml.data.dataset import HumanML3D
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from muscle_guidance_mdm.build import build_muscle_guidance
from muscle_rollup import get_indices
from posture_loss import build_reference_from_activations
from posture_guidance.angle_ops import pelvis_tilt_angle

# Key muscle groups for APT diagnosis
KEY_GROUPS = [
    ("iliopsoas",        "high"),
    ("rectus_femoris",   "high"),
    ("erector_spinae",   "high"),
    ("gluteus_maximus",  "low"),
    ("gluteus_medius",   "low"),
]


def group_abs_mean(acts_np, mint_cols, group):
    """Mean activation for a muscle group (R+L pooled)."""
    idx = []
    for side in ("_R", "_L"):
        idx += get_indices(group + side, mint_cols)
    if not idx:
        return None
    return float(np.asarray(acts_np[..., idx]).mean())


def print_muscle_table(acts_np, mint_cols, label):
    """Print key muscle group activations in table format."""
    print(f"\n  {label}:")
    print(f"  {'Group':<20} {'Activation':>12}")
    print(f"  {'-'*32}")
    for group, expect in KEY_GROUPS:
        val = group_abs_mean(acts_np, mint_cols, group)
        if val is not None:
            print(f"  {group:<20} {val:>12.4e}")
    # Flexor/Extensor ratio
    flex = []
    ext = []
    for g in ["iliopsoas", "rectus_femoris"]:
        v = group_abs_mean(acts_np, mint_cols, g)
        if v is not None:
            flex.append(v)
    v = group_abs_mean(acts_np, mint_cols, "gluteus_maximus")
    if v is not None:
        ext.append(v)
    if flex and ext:
        f_mean = np.mean(flex)
        e_mean = np.mean(ext)
        ratio = f_mean / (e_mean + 1e-12)
        print(f"  {'flex/ext ratio':<20} {ratio:>12.3f}")


def make_fk_fn(t2m_dataset, device):
    """Build FK function from t2m_dataset stats."""
    mean = torch.tensor(t2m_dataset.mean, dtype=torch.float32, device=device)
    std = torch.tensor(t2m_dataset.std, dtype=torch.float32, device=device)

    def fk_fn(mu):
        """mu: (B, 263, 1, T) normalized MDM features -> q: (B, T, 22, 3) joints."""
        mu_perm = mu.permute(0, 3, 2, 1)          # (B, T, 1, 263)
        mu_inv = mu_perm * std + mean              # denormalize
        q = recover_from_ric(mu_inv, 22)            # (B, T, 1, 22, 3)
        q = q.squeeze(2)                            # (B, T, 22, 3)
        return q
    return fk_fn


def compute_pelvis_angle(q_xyz):
    """q_xyz: (B, T, 22, 3) -> mean pelvis tilt angle in degrees."""
    ang = pelvis_tilt_angle(q_xyz)  # radians
    return float(torch.rad2deg(ang.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_apt_npy", required=True,
                    help="Path to joint-guided APT comparison.npy")
    ap.add_argument("--target_pelvis_angle", type=float, default=10.0,
                    help="Target pelvis angle for interpolated reference (degrees)")
    ap.add_argument("--muscle_ckpt", required=True)
    ap.add_argument("--muscle_weight", type=float, default=50.0,
                    help="Weight for muscle loss component")
    ap.add_argument("--n_steps", type=int, default=20,
                    help="Number of SGD optimization steps")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset_opt", default="./dataset/humanml_opt.txt")
    ap.add_argument("--abs_path", default=".")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =====================================================================
    # Step 1: Load data and interpolate to ~10° APT
    # =====================================================================
    print("\n" + "="*70)
    print("Step 1: Load joint-guided APT data & interpolate")
    print("="*70)

    data = np.load(args.joint_apt_npy, allow_pickle=True).item()
    hml_base = np.asarray(data["motion_hml_tj"], dtype=np.float32)          # (1,120,263)
    hml_guided = np.asarray(data["motion_hml_tj_guided"], dtype=np.float32)  # (1,120,263)

    # Load t2m dataset for FK
    print("Loading t2m dataset for FK...")
    dataset = HumanML3D(mode="eval", datapath=args.dataset_opt,
                        device="cpu", abs_path=args.abs_path)
    t2m = dataset.t2m_dataset
    fk_fn = make_fk_fn(t2m, device)

    # Compute baseline and guided pelvis angles
    xb = torch.from_numpy(hml_base).float().to(device)      # (1,120,263)
    xg = torch.from_numpy(hml_guided).float().to(device)

    # Reshape for FK: (1,263,1,120)
    xb_fk = xb.permute(0, 2, 1).unsqueeze(2)   # (1,263,1,120)
    xg_fk = xg.permute(0, 2, 1).unsqueeze(2)

    qb = fk_fn(xb_fk)  # (1,120,22,3)
    qg = fk_fn(xg_fk)

    ang_base = compute_pelvis_angle(qb)
    ang_guided = compute_pelvis_angle(qg)
    print(f"  Baseline pelvis angle: {ang_base:+.1f}°")
    print(f"  Joint-guided pelvis angle: {ang_guided:+.1f}°")

    # Interpolation factor
    alpha = (args.target_pelvis_angle - ang_base) / (ang_guided - ang_base + 1e-10)
    alpha = max(0.0, min(1.0, alpha))
    print(f"  Interpolation alpha: {alpha:.4f} (target={args.target_pelvis_angle}°)")

    # Interpolate in MDM feature space
    x_mild = xb + alpha * (xg - xb)  # (1,120,263)
    x_mild_fk = x_mild.permute(0, 2, 1).unsqueeze(2)
    q_mild = fk_fn(x_mild_fk)
    ang_mild = compute_pelvis_angle(q_mild)
    print(f"  Interpolated pelvis angle: {ang_mild:+.1f}°")
    print(f"  (target was {args.target_pelvis_angle:+.0f}°)")

    # =====================================================================
    # Step 2: Build MuscleGuidance & extract APT reference
    # =====================================================================
    print("\n" + "="*70)
    print("Step 2: Build MuscleGuidance & extract APT reference")
    print("="*70)

    mg = build_muscle_guidance(
        ckpt_path=args.muscle_ckpt,
        posture_name="anterior_pelvic_tilt",
        assets_dir=_ASSETS,
        same_normalization=True,
        device=device,
    )

    # Get muscle activations for the interpolated ~10° APT motion
    with torch.no_grad():
        apt_acts_norm = mg._activations(x_mild)  # normalized features -> (1,120,402)
    apt_acts = apt_acts_norm.cpu().numpy()

    # Build APT reference
    apt_ref = build_reference_from_activations(apt_acts, mg.mint_cols)
    print(f"\n  APT reference ({ang_mild:+.1f}° pelvis):")
    for group, expect in KEY_GROUPS:
        for side in ("_R", "_L"):
            key = f"{group}{side}"
            if key in apt_ref:
                print(f"    {key:<25} {apt_ref[key]:.4e}")

    # Also show baseline reference for comparison
    with torch.no_grad():
        base_acts_norm = mg._activations(xb)
    base_acts = base_acts_norm.cpu().numpy()
    base_ref = build_reference_from_activations(base_acts, mg.mint_cols)
    print(f"\n  Baseline reference ({ang_base:+.1f}° pelvis) — for comparison:")
    for group, expect in KEY_GROUPS:
        for side in ("_R", "_L"):
            key = f"{group}{side}"
            if key in apt_ref and key in base_ref:
                print(f"    {key:<25} {base_ref[key]:.4e}  (APT ref: {apt_ref[key]:.4e})")

    # =====================================================================
    # Step 3: Run muscle guidance with APT reference
    # =====================================================================
    print("\n" + "="*70)
    print(f"Step 3: Muscle guidance with APT reference (weight={args.muscle_weight})")
    print("="*70)

    # Set APT reference
    mg.set_reference(apt_ref)

    # Start from baseline features
    x_opt = xb.clone().detach().requires_grad_(True)  # (1,120,263)

    optimizer = SGD([x_opt], lr=args.lr)

    print(f"\n  {'Step':<6} {'Loss':>10} {'Pelvis':>8} {'Flex/Ext':>10}")
    print(f"  {'-'*38}")

    for step in range(args.n_steps + 1):
        with torch.no_grad():
            # Compute pelvis angle
            x_fk = x_opt.permute(0, 2, 1).unsqueeze(2)
            q = fk_fn(x_fk)
            ang = compute_pelvis_angle(q)
            # Muscle activations
            acts = mg._activations(x_opt).cpu().numpy()
            flex, ext = [], []
            for g in ["iliopsoas", "rectus_femoris"]:
                v = group_abs_mean(acts, mg.mint_cols, g)
                if v is not None: flex.append(v)
            v = group_abs_mean(acts, mg.mint_cols, "gluteus_maximus")
            if v is not None: ext.append(v)
            ratio = np.mean(flex) / (np.mean(ext) + 1e-12) if flex and ext else 0

        if step == 0:
            # Just report initial state
            print(f"  {'init':<6} {'---':>10} {ang:>+7.1f}° {ratio:>10.3f}")
            continue

        optimizer.zero_grad()
        loss = mg.loss(x_opt)  # scalar, differentiable
        # To GENERATE APT we MAXIMISE the loss (gradient ASCENT)
        (-loss).backward()
        optimizer.step()

        print(f"  {step:<6} {loss.item():>10.4f} {ang:>+7.1f}° {ratio:>10.3f}")

        if step == args.n_steps:
            final_loss = loss.item()
            final_ang = ang
            final_ratio = ratio
            final_acts = acts

    # =====================================================================
    # Step 4: Summary
    # =====================================================================
    print("\n" + "="*70)
    print("Step 4: Summary")
    print("="*70)

    print(f"\n  Reference pelvis angle: {ang_mild:+.1f}° (mild APT)")
    print(f"  Baseline pelvis angle:  {ang_base:+.1f}° (PPT)")
    print(f"  Guided pelvis angle:    {final_ang:+.1f}°")
    print(f"  Flex/Ext ratio:         {final_ratio:.3f}")

    print(f"\n  Key muscle groups (after guidance):")
    print_muscle_table(final_acts, mg.mint_cols, "Guided activations")

    print(f"\n  Key muscle groups (baseline, for comparison):")
    print_muscle_table(base_acts, mg.mint_cols, "Baseline activations")

    print(f"\n  {'='*70}")
    if final_ang > ang_base:
        print(f"  RESULT: Pelvis moved toward APT (+{final_ang - ang_base:.1f}°)")
        print(f"  INTERPRETATION: APT reference FIXES the direction.")
        print(f"  → H_ref confirmed: baseline reference was the contaminant.")
    elif final_ang > -5:
        print(f"  RESULT: Pelvis near neutral ({final_ang:+.1f}°)")
        print(f"  INTERPRETATION: APT reference reduces the PPT bias but doesn't fully reverse.")
        print(f"  → Partial improvement: reference matters but proxy mapping also contributes.")
    else:
        print(f"  RESULT: Pelvis still PPT ({final_ang:+.1f}°)")
        print(f"  INTERPRETATION: APT reference did NOT fix the direction.")
        print(f"  → H_A confirmed: proxy inverse mapping is the root cause.")

    # Also run verify-style analysis
    print(f"\n  {'='*70}")
    print(f"  verify_reference_vs_proxy style analysis:")
    print(f"  {'Group':<20} {'Expected':<12} {'Baseline':>12} {'Guided':>12} {'Delta':>12}")
    print(f"  {'-'*62}")
    for group, expect in KEY_GROUPS:
        mb = group_abs_mean(base_acts, mg.mint_cols, group)
        mg_val = group_abs_mean(final_acts, mg.mint_cols, group)
        if mb is not None and mg_val is not None:
            d = mg_val - mb
            tag = "high" if expect == "high" else "low"
            print(f"  {group:<20} {tag:<12} {mb:>12.4e} {mg_val:>12.4e} {d:>+12.2e}")

    print(f"\n  Analysis complete.")


if __name__ == "__main__":
    main()
