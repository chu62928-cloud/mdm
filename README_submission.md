# Training-Free Pathological Human Motion Generation via Dual Joint-Angle and Muscle-Activation Guidance

Code release accompanying the paper of the same title.

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
| `scripts/` | Reproduction pipelines, muscle-space evaluation and figure generation. |

### Modified MDM source (part of our contribution)
| Path | Change |
|---|---|
| `diffusion/gaussian_diffusion.py` | Guidance injection hook on the posterior mean during sampling. |
| `sample/generate.py` | Two-pass sampling, mode switch, and guidance CLI flags. |

### Upstream MDM source (unmodified, required to run)
`data_loaders/ diffusion/ eval/ kit/ model/ prepare/ sample/ train/ utils/ visualize/`,
plus `environment.yml`, `DiP.md`, and `LICENSE`.

---

## Results

Headline numbers on the APT walking task (prompt *"a person is walking"*, target
$\tau_{\text{APT}}=15°$, $w_M=22$), reported post-fix over $N=15$ random seeds.

**Table 1 — APT in angle space** (band = $[13°, 17°]$). `joint` reaches the
anterior target on every seed; `both` is the only mode with non-zero in-band
precision and, among anterior-producing modes, the best temporal correlation;
`muscle`-only moves posterior due to the proxy inversion (see below).

| Mode | Guided APT | Δ | Hit (band / loose) | Corr | Shape |
|---|---|---|---|---|---|
| joint (single seed) | +15.5° | +22.5° | 88.3% / 100% | +0.53 | – |
| joint | +19.5° | +30.4 ± 2.6° | 0% / 100% | +0.136 | 12 / 15 |
| muscle | −16.5° | −5.6 ± 3.7° | 0% / 0% | +0.595 | 0 / 15 |
| **both** | +18.6° | +28.1 ± 2.3° | **21.8 ± 12.2% / 91.8%** | +0.254 | 12 / 15 |

**Table 2 — Muscle-space evaluation.** Directionality counts functional groups
moving in the clinically expected APT direction (out of 20). The inflated
`muscle` ratio is a denominator artifact (over-suppressed rectus abdominis), not
genuine APT patterning; `both` is the practical operating point.

| Mode | Clinical loss ratio | Direction (/20) |
|---|---|---|
| joint | 3 ± 1× | 6.1 ± 1.2 |
| muscle | 622 ± 314× | 14.0 ± 1.7 |
| both | 160 ± 178× | 9.8 ± 1.9 |

The muscle pathway repair restores a usable guidance gradient
(norm 0.0017 → 0.279). See the paper for the full analysis.

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
   `motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth`.
   Required only for `muscle` / `both` modes.

You also need the standard MDM/HumanML3D dependencies (`glove/`, `t2m/`, body
models) as described in the upstream MDM README.

---

## Reproducing the results

> Run all commands from the repository root. A GPU is required for generation.

### 1. Three-mode comparison (generates the motions behind Table 1)
```bash
MODEL_PATH=./save/<mdm_ckpt>/model.pt \
MUSCLE_CKPT=./motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
  bash scripts/run_apt_integrated.sh
```
This runs `joint`, `muscle`, and `both` (the latter two need `MUSCLE_CKPT`) with
the default unified interface `v2_dps`, `s=40`, `schedule=last_quarter` on the
prompt *"a person is walking"*. Override `PROMPT`, `SEED`, `OUT_ROOT`, etc. via
environment variables.

### 2. Joint-space evaluation (Table 1)
```bash
python scripts/evaluate_ablation.py <output_dir>     # per-run joint-angle metrics
python -m scripts.aggregate_seeds <output_dir_1> <output_dir_2> ...  # multi-seed aggregation
```

### 3. Muscle-space evaluation (Table 2; offline, no re-generation)
```bash
python scripts/evaluate_muscle_space.py <output_root> \
    --muscle_ckpt motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
    --muscle_posture anterior_pelvic_tilt --device cuda
```

### 4. Figures
The paper's **Figure 1** is a framework schematic (not script-generated).
**Figure 2** (qualitative angle) and **Figure 3** (muscle-space + emergent stride)
are assembled from the panels below; `<dir>` is an experiment directory containing
a `comparison.npy`, e.g. `output_0608/n15/apt_both_seed42`.

```bash
# Figure 2 — sagittal key-frames + per-frame APT trajectories
python scripts/gen_fig1.py <both_dir>                                 # skeleton key-frames
python scripts/gen_fig2.py <joint_dir> <both_dir> --labels Joint Both # APT angle curves

# Figure 3 — muscle-group activation change + emergent stride
python scripts/gen_fig3.py <both_dir>                                 # APT-relevant muscle bars
python scripts/gen_fig4.py <both_dir>                                 # stride / gait-phase
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


## License & attribution
The upstream MDM code is released under its original license (see `LICENSE`).
Please cite MDM (Tevet et al., 2023), HumanML3D (Guo et al., CVPR 2022),
DPS (Chung et al., NeurIPS 2022), and the Muscles-in-Time motion-to-muscle proxy
alongside this work.
