# Training-Free Pathological Human Motion Generation via Dual Joint-Angle and Muscle-Activation Guidance

Code release accompanying the paper/poster of the same title.

This repository steers a **frozen** text-to-motion diffusion model (MDM) toward
clinically defined pathological postures (e.g., anterior pelvic tilt, APT) at
**inference time, without any retraining**. Two complementary guidance signals are
injected into the reverse-diffusion loop and fused into a single differentiable
objective with a `joint` / `muscle` / `both` mode switch:

- **Module 1 — Joint-angle (geometric IK) guidance:** differentiable joint-angle
  constraints on skeletons recovered by forward kinematics.
- **Module 2 — Muscle-activation guidance:** a biomechanically grounded posture
  loss back-propagated through a frozen *motion-to-muscle* proxy network.

The repository is a fork of [MDM (Tevet et al., 2023)](https://github.com/GuyTevet/motion-diffusion-model);
our contribution is layered on top of the original MDM source.

---

## Repository layout

### Our contribution
| Path | Description |
|---|---|
| `posture_guidance/` | Module 1: joint-angle / IK guidance and the unified guidance framework (registry, combined loss, guidance variants, controller, gait-phase masks). |
| `muscle_guidance_mdm/` | Module 2 assembly layer: proxy builder, frozen-proxy loader, dense target-matching muscle loss. |
| `motion2muscle/` | Frozen motion→muscle proxy model, clinical posture loss, muscle roll-up to functional groups, muscle name tables. |
| `scripts/` | Reproduction pipelines, muscle-space evaluation, figure generation, and diagnostics. |

### Modified MDM source (part of our contribution)
| Path | Change |
|---|---|
| `diffusion/gaussian_diffusion.py` | Guidance injection hook on the posterior mean during sampling. |
| `sample/generate.py` | Two-pass sampling, mode switch, and guidance CLI flags. |

### Upstream MDM source (unmodified, required to run)
`data_loaders/ diffusion/ eval/ kit/ model/ prepare/ sample/ train/ utils/ visualize/`,
plus `environment.yml`, `DiP.md`, and `LICENSE`.

### Results
`output_0608/` contains the saved baseline/guided motions (`comparison.npy`),
per-run reports, and the poster figures (`fig1_skeleton.png` … `fig4_stride.png`).

---

## Setup

```bash
conda env create -f environment.yml
conda activate mdm   # or the env name defined in environment.yml
```

### Checkpoints (not tracked in git)

Large weights are excluded by `.gitignore` and must be obtained separately:

1. **MDM backbone** — `humanml_trans_dec_512_bert` (DiP 50-step sampler), from the
   [official MDM release](https://github.com/GuyTevet/motion-diffusion-model).
   Point `MODEL_PATH` at its `model*.pt`.
2. **Frozen motion→muscle proxy** — place `net_best_loss.pth` at
   `motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth`
   (see that directory's `README.md` and `motion2muscle/HANDOFF.md`).
   Required only for `muscle` / `both` modes.

You also need the standard MDM/HumanML3D dependencies (`glove/`, `t2m/`, body
models) as described in the upstream MDM README.

---

## Reproducing the results

> Run all commands from the repository root. A GPU is required for generation.

### 1. Three-mode comparison (Tables 1–3, Figs 1–4)
```bash
MODEL_PATH=./save/<mdm_ckpt>/model.pt \
MUSCLE_CKPT=./motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
  bash scripts/run_apt_integrated.sh
```
This runs `joint`, `muscle`, and `both` (the latter two need `MUSCLE_CKPT`) with
the default unified interface `v2_dps`, `s=40`, `schedule=last_quarter` on the
prompt *"a person is walking"*. Override `PROMPT`, `SEED`, `OUT_ROOT`, etc. via
environment variables.

### 2. Joint-space evaluation
```bash
python scripts/evaluate_ablation_v3.py <output_dir>     # per-run joint-angle metrics
python -m scripts.aggregate_seeds <output_dir_1> <output_dir_2> ...  # multi-seed aggregation
```

### 3. Muscle-space evaluation (offline, no re-generation)
```bash
python scripts/evaluate_muscle_space.py <output_root> \
    --muscle_ckpt motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
    --muscle_posture anterior_pelvic_tilt --device cuda
```

### 4. Diagnostics (muscle-pathway repair, Table 5)
```bash
python scripts/check_proxy_norm.py --muscle_ckpt <net_best_loss.pth>   # activation / Jacobian check
python scripts/sanity_muscle_integration.py                            # loss-chain regression
python scripts/probe_proxy_inversion.py ...                            # APT/PPT mapping cross-check
```

### 5. Figures
```bash
python scripts/gen_fig1.py both output_0608/n15/apt_both_seed42
python scripts/gen_fig2.py output_0608/n15/apt_joint_seed42 output_0608/n15/apt_both_seed42 --labels Joint Both
python scripts/gen_fig3.py output_0608/n15/apt_both_seed42
python scripts/gen_fig4.py output_0608/n15/apt_both_seed42
```

---

## Key configuration

| Flag / env var | Meaning |
|---|---|
| `--guidance_mode {joint,muscle,both}` | Select the active guidance term(s). |
| `--joint_weight`, `--muscle_weight` | Module weights $w_J$, $w_M$. |
| `GUIDANCE_VARIANT` | Injection variant (default `v2_dps`; closed-loop PID `v6` also supported). |
| `GUIDANCE_KWARGS_JSON` | Variant kwargs, e.g. `{"s":40,"schedule":"last_quarter"}`. |
| `MUSCLE_MARGIN` | Dense muscle-loss directional margin $m$ (default 0.3). |
| `MUSCLE_CKPT` / `--muscle_ckpt` | Path to the frozen proxy weights. |

---

## Notes
- `scripts/run_calibrated_ablation_v3.sh` is a development-time hyperparameter
  sweep, **not** required for the reported results.
- `README.md` (Chinese) is the original engineering hand-off log with the full
  development history and per-posture findings.

## License & attribution
The upstream MDM code is released under its original license (see `LICENSE`).
Please cite MDM (Tevet et al., 2023), HumanML3D (Guo et al., CVPR 2022),
DPS (Chung et al., NeurIPS 2022), and the Muscles-in-Time motion-to-muscle proxy
alongside this work.
