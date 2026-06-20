#!/usr/bin/env python3
"""Test gradient direction: descent vs ascent with APT vs PPT reference."""
import sys, os, numpy as np, torch
from torch.optim import SGD

_ROOT = "."
_ASSETS = os.path.join(_ROOT, "motion2muscle")
for p in (_ROOT, _ASSETS):
    sys.path.insert(0, p)

from data_loaders.humanml.data.dataset import HumanML3D
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from muscle_guidance_mdm.build import build_muscle_guidance
from posture_guidance.angle_ops import pelvis_tilt_angle
from posture_loss import build_reference_from_activations
from muscle_rollup import get_indices

data = np.load("output_0608/apt_joint_seed42/comparison.npy", allow_pickle=True).item()
hml_base = np.asarray(data["motion_hml_tj"], dtype=np.float32)
hml_guided = np.asarray(data["motion_hml_tj_guided"], dtype=np.float32)

dataset = HumanML3D(mode="eval", datapath="./dataset/humanml_opt.txt",
                    device="cpu", abs_path=".")
t2m = dataset.t2m_dataset
mean_t = torch.tensor(t2m.mean, dtype=torch.float32).cuda()
std_t = torch.tensor(t2m.std, dtype=torch.float32).cuda()

def fk_fn(mu):
    q = recover_from_ric(mu.permute(0, 3, 2, 1) * std_t + mean_t, 22)
    return q.squeeze(2)

device = torch.device("cuda")
xb = torch.from_numpy(hml_base).float().to(device)
xg = torch.from_numpy(hml_guided).float().to(device)

xb_fk = fk_fn(xb.permute(0, 2, 1).unsqueeze(2))
xg_fk = fk_fn(xg.permute(0, 2, 1).unsqueeze(2))
ang_b = float(torch.rad2deg(pelvis_tilt_angle(xb_fk).mean()))
ang_g = float(torch.rad2deg(pelvis_tilt_angle(xg_fk).mean()))
alpha = (10.0 - ang_b) / (ang_g - ang_b + 1e-10)
x_mild = xb + alpha * (xg - xb)
print(f"Baseline pelvis: {ang_b:+.1f}  Joint APT: {ang_g:+.1f}  Interp={alpha:.4f}")

mg = build_muscle_guidance(
    ckpt_path="motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth",
    posture_name="anterior_pelvic_tilt", assets_dir=_ASSETS,
    same_normalization=True, device=device)

with torch.no_grad():
    apt_acts = mg._activations(x_mild).cpu().numpy()
apt_ref = build_reference_from_activations(apt_acts, mg.mint_cols)

with torch.no_grad():
    base_acts = mg._activations(xb).cpu().numpy()
ppt_ref = build_reference_from_activations(base_acts, mg.mint_cols)

# Show key reference values
print("\nReference comparison:")
for ref_name, ref_dict in [("APT +9.8", apt_ref), ("PPT -7.0", ppt_ref)]:
    ili = 0.5 * (ref_dict.get("iliopsoas_R", 0) + ref_dict.get("iliopsoas_L", 0))
    glu = 0.5 * (ref_dict.get("gluteus_maximus_R", 0) + ref_dict.get("gluteus_maximus_L", 0))
    rf  = 0.5 * (ref_dict.get("rectus_femoris_R", 0) + ref_dict.get("rectus_femoris_L", 0))
    ratio = (ili + rf) / 2 / (glu + 1e-12)
    print(f"  {ref_name}: iliopsoas={ili:.3e}  rect_fem={rf:.3e}  gl_max={glu:.3e}  flex/ext={ratio:.3f}")


def get_stats(mg_obj, x):
    with torch.no_grad():
        q = fk_fn(x.detach().permute(0, 2, 1).unsqueeze(2))
        ang = float(torch.rad2deg(pelvis_tilt_angle(q).mean()))
        acts = mg_obj._activations(x.detach()).cpu().numpy()
        ili_idx = (get_indices("iliopsoas_R", mg_obj.mint_cols) +
                    get_indices("iliopsoas_L", mg_obj.mint_cols))
        glu_idx = (get_indices("gluteus_maximus_R", mg_obj.mint_cols) +
                    get_indices("gluteus_maximus_L", mg_obj.mint_cols))
        rf_idx = (get_indices("rectus_femoris_R", mg_obj.mint_cols) +
                   get_indices("rectus_femoris_L", mg_obj.mint_cols))
        ili = float(acts[..., ili_idx].mean())
        glu = float(acts[..., glu_idx].mean())
        rf = float(acts[..., rf_idx].mean())
        ratio = (ili + rf) / 2 / (glu + 1e-12)
    return ang, ratio


tests = [
    ("APT ref (+9.8)  GRADIENT DESCENT  (toward APT)", apt_ref, 1),
    ("PPT ref (-7.0)  GRADIENT ASCENT   (away from PPT)", ppt_ref, -1),
]

for label, ref, sign in tests:
    mg2 = build_muscle_guidance(
        ckpt_path="motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth",
        posture_name="anterior_pelvic_tilt", assets_dir=_ASSETS,
        same_normalization=True, device=device)
    mg2.set_reference(ref)

    x_opt = xb.clone().detach().requires_grad_(True)
    opt = SGD([x_opt], lr=0.1)

    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"{'='*65}")
    print(f"  {'Step':<6} {'Loss':>8} {'Pelvis':>8} {'Flex/Ext':>10}")

    ang0, ratio0 = get_stats(mg2, x_opt)
    print(f"  {'init':<6} {'---':>8} {ang0:>+7.1f}  {ratio0:>10.2f}")

    for step in range(1, 31):
        opt.zero_grad()
        loss = mg2.loss(x_opt)
        (sign * loss).backward()
        opt.step()
        if step % 5 == 0:
            ang, ratio = get_stats(mg2, x_opt)
            print(f"  {step:<6} {loss.item():>8.4f} {ang:>+7.1f}  {ratio:>10.2f}")

    angf, ratiof = get_stats(mg2, x_opt)
    dir_str = "APT" if angf > ang_b else "PPT"
    print(f"  -> pelvis: {ang_b:+.1f} -> {angf:+.1f}  "
          f"delta={angf-ang_b:+.2f}  ({dir_str})")

# Additional: sweep learning rate with APT ref + descent
print("\n" + "=" * 65)
print("LR sweep: APT ref + gradient descent, 50 steps")
print("=" * 65)
print(f"  {'lr':<8} {'loss':>12} {'pelvis':>10} {'delta':>8} {'flex/ext':>10}")
for lr in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    mg2 = build_muscle_guidance(
        ckpt_path="motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth",
        posture_name="anterior_pelvic_tilt", assets_dir=_ASSETS,
        same_normalization=True, device=device)
    mg2.set_reference(apt_ref)

    x_opt = xb.clone().detach().requires_grad_(True)
    opt = SGD([x_opt], lr=lr)

    ang0, ratio0 = get_stats(mg2, x_opt)
    L0 = mg2.loss(x_opt).item()

    for step in range(1, 51):
        opt.zero_grad()
        loss = mg2.loss(x_opt)
        loss.backward()
        opt.step()

    angf, ratiof = get_stats(mg2, x_opt)
    Lf = mg2.loss(x_opt).item()
    delta = angf - ang0
    print(f"  {lr:<8.2f} {L0:.4f} -> {Lf:.4f}  {ang0:+.1f} -> {angf:+.1f}  {delta:>+.2f}  {ratio0:.2f} -> {ratiof:.2f}")
