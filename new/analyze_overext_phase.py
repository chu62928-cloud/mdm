"""G1: Hyperextension vs gait phase — determines paper direction."""
import sys, os, math, csv
sys.path.insert(0, "/root/autodl-tmp/motion-diffusion-model")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from posture_guidance.joint_indices import get_joint_idx
from posture_guidance.phase_detector import PhaseDetector

# ---- True geometric arbiter (same as adjudication) ----
def signed_knee_angle_sagittal(q, side="left"):
    hip   = q[..., get_joint_idx(f"{side}_hip"),   :]
    knee  = q[..., get_joint_idx(f"{side}_knee"),  :]
    ankle = q[..., get_joint_idx(f"{side}_ankle"), :]
    thigh = hip - knee; shank = ankle - knee
    l_hip = q[..., get_joint_idx("left_hip"),  :]
    r_hip = q[..., get_joint_idx("right_hip"), :]
    lateral = r_hip - l_hip
    lat_n = F.normalize(lateral, dim=-1, eps=1e-7)
    def to_sagittal(v):
        return v - (v * lat_n).sum(dim=-1, keepdim=True) * lat_n
    t = to_sagittal(thigh); s = to_sagittal(shank)
    t_n = F.normalize(t, dim=-1, eps=1e-7)
    s_n = F.normalize(s, dim=-1, eps=1e-7)
    cos_a = (t_n * s_n).sum(dim=-1).clamp(-1+1e-7, 1-1e-7)
    base = torch.rad2deg(torch.acos(cos_a))
    cross_ts = torch.linalg.cross(t, s, dim=-1)
    cross_sign = (cross_ts * lat_n).sum(dim=-1)
    return torch.where(cross_sign > 0, base, 360.0 - base)

# ---- Process 5 seeds ----
base_dir = "output/kneedist_v2_last_quarter"
detector = PhaseDetector()
all_rows = []
seed_summaries = []

for sd in sorted(os.listdir(base_dir)):
    if not sd.startswith("seed"): continue
    seed = sd.replace("seed", "")
    npy_path = os.path.join(base_dir, sd, "comparison.npy")
    if not os.path.exists(npy_path): continue

    d = np.load(npy_path, allow_pickle=True).item()

    for key, label in [("motion_xyz", "BASELINE"), ("motion_xyz_guided", "GUIDED")]:
        q = torch.from_numpy(d[key][0]).float().permute(2, 0, 1)
        T = q.shape[0]

        # Sag angle
        sag_L = signed_knee_angle_sagittal(q, "left").numpy()
        sag_R = signed_knee_angle_sagittal(q, "right").numpy()

        # Phase detection
        stance_mask = detector.get_stance_mask(q).numpy()  # (T, 2) soft
        stance_L = stance_mask[:, 0] > 0.5
        stance_R = stance_mask[:, 1] > 0.5
        swing_L = ~stance_L
        swing_R = ~stance_R

        for side, sag_arr, st_arr, sw_arr in [
            ("L", sag_L, stance_L, swing_L),
            ("R", sag_R, stance_R, swing_R),
        ]:
            for f in range(T):
                phase = "stance" if st_arr[f] else "swing"
                overext = 1 if sag_arr[f] > 180 else 0
                all_rows.append({
                    "seed": seed, "type": label, "side": side, "frame": f,
                    "phase": phase, "sag": round(sag_arr[f], 1), "overext": overext,
                })

# ---- Summary ----
guided_rows = [r for r in all_rows if r["type"] == "GUIDED"]
baseline_rows = [r for r in all_rows if r["type"] == "BASELINE"]

print("=" * 60)
print("G1: Hyperextension vs Gait Phase")
print("=" * 60)

for label, rows in [("BASELINE", baseline_rows), ("GUIDED", guided_rows)]:
    oe = [r for r in rows if r["overext"] == 1]
    n_total = len(rows)
    n_oe = len(oe)
    if n_oe == 0:
        print(f"{label}: 0/{n_total} hyperextended frames")
        continue

    oe_stance = [r for r in oe if r["phase"] == "stance"]
    oe_swing  = [r for r in oe if r["phase"] == "swing"]
    all_stance_count = len([r for r in rows if r["phase"] == "stance"])
    all_swing_count  = len([r for r in rows if r["phase"] == "swing"])

    print(f"\n{label}:")
    print(f"  Total frames: {n_total} (stance={all_stance_count}, swing={all_swing_count})")
    print(f"  Hyperextended (sag>180): {n_oe}/{n_total} ({100*n_oe/n_total:.1f}%)")
    print(f"    In STANCE: {len(oe_stance)}/{all_stance_count} stance frames = {100*len(oe_stance)/max(all_stance_count,1):.1f}% of stance")
    print(f"    In SWING:  {len(oe_swing)}/{all_swing_count} swing frames = {100*len(oe_swing)/max(all_swing_count,1):.1f}% of swing")
    if n_oe > 0:
        print(f"    OE frames: {len(oe_stance)/n_oe*100:.0f}% stance / {len(oe_swing)/n_oe*100:.0f}% swing")

    # Per-seed breakdown
    for seed in sorted(set(r["seed"] for r in rows)):
        sr = [r for r in rows if r["seed"] == seed]
        sro = [r for r in sr if r["overext"] == 1]
        if len(sro) == 0: continue
        sro_s = len([r for r in sro if r["phase"] == "stance"])
        print(f"    seed {seed}: {len(sro)} OE frames, {sro_s}/{len(sro)} ({100*sro_s/max(len(sro),1):.0f}%) in stance")

# ---- G1 Verdict ----
print("\n" + "=" * 60)
print("G1 VERDICT")
guided_oe = [r for r in guided_rows if r["overext"] == 1]
if guided_oe:
    oe_stance_pct = len([r for r in guided_oe if r["phase"] == "stance"]) / len(guided_oe) * 100
    print(f"  {oe_stance_pct:.0f}% of hyperextended frames are in STANCE phase")
    if oe_stance_pct > 70:
        print("  -> Case B: Physiological hyperextension (genu recurvatum pattern)")
        print("  -> Continue to Phase C (characterization)")
    elif oe_stance_pct > 40:
        print("  -> AMBIGUOUS: hyperextension distributed across phases")
        print("  -> Needs further investigation before reclassifying")
    else:
        print("  -> Case A: Hyperextension in WRONG phase (phase disruption)")
        print("  -> Jump to Phase G-alt (recast as negative result)")
else:
    print("  No hyperextended frames found at all")

# ---- Save CSV ----
with open("new_results/overext_phase_distribution.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed","type","side","frame","phase","sag","overext"])
    w.writeheader()
    w.writerows(all_rows)
print("\nSaved: new_results/overext_phase_distribution.csv")

# ---- Plot ----
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for idx, sd in enumerate(sorted(set(r["seed"] for r in guided_rows))[:3]):
    ax1, ax2 = axes[idx, 0], axes[idx, 1]
    sr = [r for r in guided_rows if r["seed"] == sd]
    frames = np.arange(len(sr)//2)  # left side only
    sag_L = np.array([r["sag"] for r in sr if r["side"] == "L"])
    phase_L = np.array([1 if r["phase"] == "stance" else 0 for r in sr if r["side"] == "L"])

    # Top: sag angle colored by phase
    colors = ['darkred' if p == 1 else 'steelblue' for p in phase_L]
    ax1.scatter(frames, sag_L, c=colors, s=8, alpha=0.7)
    ax1.axhline(180, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel("Sagittal knee angle (deg)")
    ax1.set_title(f"seed {sd}: guided LEFT knee")
    # legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color='darkred', label='stance'), Patch(color='steelblue', label='swing')], fontsize=7)

    # Bottom: histogram by phase
    sag_stance = [r["sag"] for r in sr if r["side"] == "L" and r["phase"] == "stance"]
    sag_swing  = [r["sag"] for r in sr if r["side"] == "L" and r["phase"] == "swing"]
    bins = np.linspace(150, 210, 40)
    ax2.hist(sag_stance, bins=bins, alpha=0.7, color='darkred', label=f'stance (n={len(sag_stance)})')
    ax2.hist(sag_swing, bins=bins, alpha=0.7, color='steelblue', label=f'swing (n={len(sag_swing)})')
    ax2.axvline(180, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Sagittal knee angle (deg)")
    ax2.set_ylabel("Frames")
    ax2.legend(fontsize=7)

fig.suptitle("G1: Hyperextension vs Gait Phase (3 seeds shown)", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig("new_results/overext_vs_gaitphase.png", dpi=150)
print("Saved: new_results/overext_vs_gaitphase.png")
