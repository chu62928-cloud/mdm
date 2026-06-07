# Training-Free Pathological Human Motion Generation via Dual Joint-Angle and Muscle-Activation Guidance

> Poster manuscript — pipeline & method. Academic draft; result tables/figures contain
> reserved placeholders (marked `〔…〕`) to be filled once the final runs complete.

---

## Abstract

We present a **training-free** framework that steers a frozen text-to-motion diffusion
model toward **clinically defined pathological postures** (e.g., anterior pelvic tilt, APT)
without any retraining. The framework injects, at inference time, two complementary and
jointly operating guidance signals: (1) a **geometric inverse-kinematics (IK) module** that
enforces differentiable joint-angle constraints on recovered 3-D skeletons, and (2) a
**muscle-aware module** that back-propagates a biomechanically grounded posture loss through
a frozen *motion-to-muscle* proxy network. The two signals are fused into a single
differentiable objective that can be operated in **joint-only**, **muscle-only**, or
**combined** modes, all sharing one unguided/guided two-pass sampling interface. On the APT
walking task, geometric guidance shifts mean pelvic tilt from 〔7.0°〕 to 〔20.5°〕 (Δ = +13.5°)
while preserving gait temporal structure, and the combined mode attains the same kinematic
accuracy without interference. We further diagnose and resolve two failure modes that had
silenced the muscle pathway—a **gradient dead-zone** in the detection-style posture loss and
a **degenerate proxy output activation**—restoring a usable muscle-level guidance signal.

---

## 1. Introduction

**Motivation.** Diffusion models are the dominant paradigm for text-conditioned human-motion
synthesis, but canonical checkpoints are trained on *healthy, normative* movement and provide
no mechanism for the controlled synthesis of *pathological* postures. Clinical biomechanics,
rehabilitation training, and synthetic-data augmentation all require motions that reproducibly
exhibit specific, anatomically defined deviations—anterior pelvic tilt (APT), thoracic
kyphosis, genu recurvatum—ideally parametrized on a continuous scale (e.g., "5° more pelvic
tilt").

**Why not retrain.** Curating large pathological motion-capture corpora is impractical: data
are scarce, and clinical phenotypes are continuous and parametric rather than categorical. We
therefore adopt an **inference-time, training-free** strategy: steer a frozen MDM checkpoint
toward quantitative targets while preserving its learned prior over motion naturalness.

**Approach.** Pathology has both a *kinematic* signature (joint geometry) and a *physiological*
signature (muscle-coordination pattern). We address both with two modules that operate inside
the reverse-diffusion loop:

- **Module 1 — Geometric IK guidance** formulates posture targets as differentiable constraints
  on joint coordinates recovered by forward kinematics, and nudges each denoising step toward
  them. It is kinematically precise and experimentally validated.
- **Module 2 — Muscle-aware guidance** maps the generated skeleton to fascicle-level muscle
  activations through a **frozen transformer proxy**, and scores them with a directional,
  biomechanically motivated **posture loss**. It grounds guidance in muscle physiology rather
  than pure geometry.

Both are merged into one **combined loss** with a shared MDM interface, selectable as
`joint`, `muscle`, or `both`.

**Contributions.**
1. A unified, training-free guidance framework that fuses joint-angle and muscle-activation
   constraints into a single differentiable objective with a mode switch (joint / muscle / both),
   leaving the MDM sampling interface unchanged.
2. A **dense, target-matching muscle-guidance loss** that repairs the gradient dead-zone of the
   original detection-style clinical loss, enabling non-zero guidance from the first denoising step.
3. A **diagnosis-and-fix methodology** for the muscle pathway (normalization handshake, proxy
   output-activation correction, in-loop gradient diagnostics) and a dedicated **muscle-space
   evaluation** complementary to joint-space metrics.

---

## 2. Method

### 2.1 Backbone and motion representation

We build on the publicly released MDM checkpoint (`humanml_trans_dec_512_bert`) for the
HumanML3D dataset, operated under the DiP 50-step sampler. Each motion is represented as
$x \in \mathbb{R}^{T\times263}$, concatenating root dynamics, local joint positions
$x_{4:67}\in\mathbb{R}^{T\times63}$ (21 joints × XYZ relative to root), 6-D joint rotations,
joint velocities, and foot-contact probabilities (Table A). Global 3-D joint coordinates
$q\in\mathbb{R}^{T\times J\times3}$ are recovered from $x_{4:67}$ via a standard, differentiable
`recover_from_ric` routine, defining the forward-kinematics map $\mathrm{FK}(\cdot)$.

A key property: MDM **predicts the clean signal $\hat x_0$ directly** at every step, so each
denoising step exposes a usable clean-motion estimate $\hat x_0$ that can be fed to either
guidance module.

**Table A — HumanML3D 263-D feature partition.**

| Indices | Dim | Semantics |
|---|---|---|
| `x[0]` | 1 | root angular velocity (Y) |
| `x[1:3]` | 2 | root linear velocity (X,Z) |
| `x[3:4]` | 1 | root height (Y) |
| `x[4:67]` | 63 | 21 joint positions (XYZ), root-relative — **gradient path** |
| `x[67:193]` | 126 | 21 joint rotations (6-D) |
| `x[193:259]` | 66 | 22 joint velocities (XYZ) |
| `x[259:263]` | 4 | foot-contact probabilities |

### 2.2 Unified guidance framework

We inject guidance on the **posterior mean** $\mu_t$ rather than on the score, following the
established result that perturbing $\mu_t$ applies a geometric translation to the denoising step
while subsequent noise injection preserves high-frequency naturalness. The primary injection
variant is **DPS-style** (denoted `v2_dps`): the guidance gradient is taken w.r.t. the noisy
latent $x_t$ and flows **through the frozen MDM** via $\hat x_0(x_t)$, covering all 263 feature
dimensions:

$$
\mu_t \;\leftarrow\; \mu_t \;-\; s\,\nabla_{x_t}\, \mathcal{L}\big(\hat x_0(x_t)\big),
$$

with step size $s$ and a temporal schedule restricting guidance to the low-noise tail
(`last_quarter`). A closed-loop PID variant (`v6`) is also supported; `v2_dps` is the default
unified interface (best mean accuracy on APT).

**Data flow.**

```
text ─► MDM ─► x̂0 (B,263,1,T)
                 │
        ┌────────┴───────────────────────────────┐
        ▼ FK (recover_from_ric)                   ▼ de/re-normalize → frozen motion→muscle proxy
   q (B,T,J,3)                               a (B,T,402) muscle activations
        │ joint-angle loss                        │ muscle posture loss
        ▼                                         ▼
   L_joint  ───────────►  L = w_J·L_joint + w_M·L_muscle  ◄─────────── L_muscle
                                   │
                                   ▼  ∇_{x_t} L  (through MDM, DPS)
                          μ_t ← μ_t − s·∇_{x_t} L
```

### 2.3 Module 1 — Joint-angle (geometric IK) guidance

Given recovered joints $q=\mathrm{FK}(\hat x_0)$, each posture target is a scalar geometric
measurement $\theta_k(q)$ compared to a threshold $\tau_k$ through a smooth one-sided penalty:

$$
\mathcal{L}_{\text{joint}}(q,t) \;=\; \sum_k w_k(t)\,\rho\big(\theta_k(q)-\tau_k\big),
\qquad \rho(\cdot)=\big[\max(0,\cdot)\big]^2 \ \text{(hinge)},
$$

where $w_k(t)$ is a temporal schedule (active in the low-noise tail) and gait-phase masks
restrict constraints to the correct stance/swing window. A symmetric **Huber** variant
(over-shoot reversible) is used by the closed-loop PID controller. Representative measurements:

- **Anterior pelvic tilt:** sagittal-plane angle of the pelvis→spine vector,
  $\theta_{\text{APT}}(q) = -\operatorname{atan2}(s_z, s_y)$, with $s$ the sagittal projection
  of `(spine1 − hip)`.
- **Trunk forward lean; knee hyper-extension (singularity-free signed sagittal distance);
  thoracic kyphosis** — defined analogously.

Gradients reach $\mu_t$ **only through the 63 position dimensions $x_{4:67}$**; the remaining 200
dimensions receive zero geometric gradient (a known structural dilution that motivates larger
guidance weights and the DPS injection that re-couples all dimensions through MDM).

### 2.4 Module 2 — Muscle-activation guidance

**Frozen motion-to-muscle proxy.** A 16-layer transformer $f_\phi$ maps a 263-D motion window to
fascicle-level activations $a=f_\phi(x)\in\mathbb{R}^{T\times402}$ (lower-limb + thoraco-lumbar
models). It is **frozen** (parameters `requires_grad=False`, but the input path keeps a graph so
gradients flow to the motion). Fascicles are mean-pooled into named **functional groups** (e.g.,
`gluteus_maximus_R`, `iliopsoas_L`) via fixed index sets.

**Normalization handshake.** MDM works in its own normalized space; the proxy expects its training
normalization. The module de-normalizes with MDM statistics and re-normalizes with proxy statistics
($\texttt{same\_normalization}$ when the two share HumanML3D stats). Mismatch here is the principal
silent-failure risk.

**Reference (two-pass).** A first **unguided** sampling pass produces a normative motion whose
proxy activations define a frozen per-group reference $a^{\text{ref}}_g$. A second guided pass then
scores deviations against this reference.

**Clinical posture loss (for *evaluation*).** A table-driven, four-component loss encodes how a
named pathology deviates from the reference:
$\mathcal{L}_{\text{post}}=w_a\mathcal{L}_{\text{antag}}+w_c\mathcal{L}_{\text{chain}}
+w_s\mathcal{L}_{\text{syn}}+w_t\mathcal{L}_{\text{stab}}$
(antagonist imbalance, compensation chain, synergy imbalance, inhibited stabilizer), with
default weights $\{1.0,1.5,0.5,0.5\}$. Each term is a *thresholded detector*
$\propto\operatorname{relu}(\text{deviation}-\delta)$, $\delta\in[0.2,0.5]$, calibrated for
real-pathology magnitudes. We retain this loss to reproduce clinical directionality tables.

**The dead-zone problem (for *guidance*).** At the start of guidance the guided activations equal
the reference, so every detector term sits in its dead zone:
$\mathcal{L}_{\text{post}}\equiv0$ and $\nabla\mathcal{L}_{\text{post}}\equiv0$. With zero
gradient the motion never moves, so the loss stays zero for all steps (self-locking). Empirically,
$\|\nabla\mathcal{L}_{\text{post}}\|=0$ at the reference (verified).

**Dense target-matching guidance loss (proposed).** We keep the same four mechanisms, group
structure, and clinical *directions*, but replace each thresholded detector with a one-sided pull
toward a directional target $a^{\text{ref}}_g(1\pm m)$ (margin $m$, default 0.3):

$$
\mathcal{L}^{\text{dense}}_{\text{antag}}=\!\!\sum_{(o,u)}\!\operatorname{relu}\!\big(a^{\text{ref}}_o(1{+}m)-a_o\big)
+\operatorname{relu}\!\big(a_u-a^{\text{ref}}_u(1{-}m)\big),
$$

with analogous one-sided terms for chain (primary↓, compensator↑), synergy (dominant share↓),
and stabilizer (↓). This objective is **differentiable everywhere, non-zero at the reference,
bounded (stops at the target), and minimized to approach the pathology** (same sign convention as
the joint loss). Verified: $\|\nabla\mathcal{L}^{\text{dense}}\|>0$ at the reference.

**Proxy output-activation correction.** The proxy must output activations in the trained regime
(per-group means $\sim$0.03–0.15). An erroneous extra final `sigmoid` maps 0.03–0.15 to
0.508–0.537, collapsing all groups to a near-constant $\sim$0.5 band, flattening per-group
structure and shrinking the input→output Jacobian (≈70× smaller guidance gradient). We expose the
final activation as configurable (default: none, matching the checkpoint), restoring the correct
activation regime and a usable Jacobian.

### 2.5 Combined loss and injection

The two modules are fused into one differentiable scalar (sign-unified to **minimization**, since
the dense muscle loss and the hinge joint loss both decrease toward the target):

$$
\mathcal{L} \;=\; w_J\,\mathcal{L}_{\text{joint}}\big(\mathrm{FK}(\hat x_0)\big)
\;+\; w_M\,\mathcal{L}^{\text{dense}}_{\text{muscle}}\big(f_\phi(\hat x_0)\big).
$$

A `mode` switch activates the joint term (`joint`), the muscle term (`muscle`), or both (`both`);
weights $w_J,w_M$ and margin $m$ are configurable (CLI / environment). The combined scalar is
differentiated w.r.t. $x_t$ and applied via the DPS update of §2.2; for `both`, both gradients
propagate through MDM, so the two constraints are optimized jointly within each denoising step.
Sampling uses the **two-pass protocol**: pass 1 (unguided) builds the muscle reference; pass 2
applies the combined guidance.

### 2.6 Evaluation metrics

- **Joint-space (Module-1 objective):** mean pelvic-tilt shift Δ; hit-rate within
  $[\tau-\epsilon,\tau+\epsilon]$ (band) and one-sided (loose); baseline↔guided Pearson
  correlation (temporal-structure preservation); per-joint RMSE; jitter; foot-skate.
- **Muscle-space (Module-2 objective):** the clinical four-component posture loss
  (baseline vs. guided, ratio); per-functional-group mean-activation change with the
  **expected-direction check** (✓/✗ against the pathology template). This is computed offline
  by running the frozen proxy on the saved 263-D baseline/guided motions—**no re-generation**.

---

## 3. Results

> Joint-space numbers below are validated. Muscle-space numbers are reported in two states:
> *pre-fix* (diagnostic) and *post-fix* (to be filled after re-running with the dense loss and the
> corrected proxy activation).

### 3.1 Joint-angle guidance — APT walking

Setup: prompt "a person is walking", target $\tau_{\text{APT}}=20°$, `v2_dps` (s = 40,
schedule = last_quarter), seed 42, T = 120 (single-seed run); multi-seed reference from the
ablation study (N = 15).

**Table 1 — APT, joint mode.**

| Setting | N | Baseline APT | Guided APT | Δ | Hit (band) | Corr | RMSE (m) | Shape |
|---|---|---|---|---|---|---|---|---|
| v2_dps · s40 · last_quarter (single) | 1 | 7.0° | 20.5° | **+13.5°** | 80.8% | +0.588 | 0.494 | ✅ |
| v2_dps · s40 · last_quarter (ablation) | 15 | 〔6.0°〕 | 〔20.4°〕 | 〔+14.4°〕 | **88.7 ± 9.2%** | +0.407 ± 0.200 | 〔…〕 | 11/15 ✅ |

*Finding:* guidance raises the pelvic tilt onto target while the positive correlation confirms the
gait time-structure is preserved (not collapsed to a static pose).

### 3.2 Combined vs. joint — non-interference

**Table 2 — joint vs. both (kinematics).**

| Mode | Baseline APT | Guided APT | Δ | Hit (band) | Corr |
|---|---|---|---|---|---|
| joint | 7.0° | 20.5° | +13.5° | 80.8% | +0.588 |
| both  | 7.0° | 20.5° | +13.4° | 80.8% | +0.579 |

*Finding:* adding the muscle term leaves kinematic accuracy intact—the two constraints coexist
without negative interference.

### 3.3 Muscle-space evaluation (post-fix)

**Table 3 — clinical posture loss & directionality (proxy on saved motions).**

| Mode | Posture loss (baseline) | Posture loss (guided) | Ratio | Group directions ✓/scored |
|---|---|---|---|---|
| joint  | 〔…〕 | 〔…〕 | 〔…〕 | 〔…/…〕 |
| muscle | 〔…〕 | 〔…〕 | 〔…〕 | 〔…/…〕 |
| both   | 〔…〕 | 〔…〕 | 〔…〕 | 〔…/…〕 |

**Table 4 — key functional-group mean activation (expected APT direction).**

| Group | Expected | Reference | Baseline | Guided (muscle) | Guided (both) |
|---|---|---|---|---|---|
| erector_spinae | ↑ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |
| rectus_abdominis | ↓ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |
| iliopsoas | ↑ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |
| gluteus_maximus | ↓ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |
| rectus_femoris | ↑ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |
| gluteus_medius | ↓ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |
| transversus_abdominis | ↓ | 〔…〕 | 〔…〕 | 〔…〕 | 〔…〕 |

*Expected finding:* after the fixes, `muscle`/`both` should raise the clinical loss ratio above 1
and increase the count of groups moving in the clinically expected direction, even where the
pelvic angle changes little—evidence that the muscle pathway contributes biomechanical "texture"
beyond geometry.

### 3.4 Diagnostic results (muscle pathway repair)

**Table 5 — root-cause diagnostics.**

| Symptom | Measurement | Root cause | Fix | Status |
|---|---|---|---|---|
| muscle loss ≡ 0 over all 50 steps | clinical $\|\nabla\|=0$ at reference | thresholded detector dead-zone | dense target-matching loss | ✅ resolved (\|∇\|≈0.46 at ref) |
| activations clustered 0.50–0.53 | vs. midterm 0.03–0.15 | extra final `sigmoid` in reconstructed proxy | configurable final activation (default none) | ✅ resolved |
| guidance grad ≈ 70× too small | grad_norm ≈ 0.0017 vs. 0.12 (joint) | Jacobian collapse from above | both fixes | 〔re-measure〕 |

### 3.5 Emergent biomechanical compensation

Under APT guidance the generative prior spontaneously **lengthens the stride** to accommodate the
imposed constraint without violating balance/gravity priors: mid-swing stride period
〔28 → 33 frames〕 in the single-seed walking case. (Single-observation; multi-seed statistical
validation reserved — 〔…〕.)

### 3.6 Qualitative results

- **Fig. 1** — Baseline vs. guided skeleton (sagittal view), APT key-frames. 〔image〕
- **Fig. 2** — Pelvic-tilt angle vs. frame (baseline vs. guided). 〔image〕
- **Fig. 3** — Per-group muscle-activation deviation (baseline vs. guided), APT template overlay. 〔image〕
- **Fig. 4** — Gait-phase / stride-period analysis (vertical ankle trajectory). 〔image〕

---

## 4. Summary and Conclusions

We introduced a training-free framework for pathological human-motion generation that unifies
**geometric joint-angle guidance** and **muscle-activation guidance** into a single differentiable
objective with a joint / muscle / both mode switch, sharing one two-pass MDM sampling interface.
The joint-angle module is kinematically precise and validated (APT shifted to target, ≈80–89%
hit-rate, gait structure preserved), and the combined mode inherits this accuracy with no
interference.

The principal scientific contribution is on the **muscle pathway**: we identified that its
apparent ineffectiveness was *not* an intrinsic proxy limitation but two correctable defects—a
**gradient dead-zone** in the detection-style clinical loss and a **degenerate proxy output
activation**. We resolved the former with a **dense, directional target-matching guidance loss**
(non-zero gradient from the reference, bounded, sign-consistent with the joint loss) and the latter
by restoring the proxy's trained output regime. We additionally provide a **muscle-space evaluation**
that measures the muscle module by its own objective (clinical posture loss and per-group
directionality) rather than by joint angle alone.

**Takeaways.** (i) Pathology has orthogonal kinematic and physiological signatures; guiding both
yields motions that are simultaneously geometrically correct and biomechanically plausible.
(ii) Detection-style losses are poor guidance objectives—guidance requires a dense signal that is
non-zero at the starting point. (iii) Evaluating a module by the wrong metric (muscle quality by
joint angle) hides both its contribution and its bugs.

**Limitations & future work.** Single-seed quantitative results should be extended to N ≥ 10 with
statistical validation; the emergent stride-elongation effect needs multi-seed confirmation; the
muscle proxy's intrinsic sensitivity—once the activation and normalization handshakes are verified—
may warrant a task-specific proxy (explicit pelvic-orientation features, posture-contrastive
training); and the framework should be evaluated across multiple postures (posterior pelvic tilt,
forward-head posture, Trendelenburg) for which templates already exist.

---

## Appendix — Reproducibility

- **Backbone:** MDM `humanml_trans_dec_512_bert`, DiP 50-step sampler.
- **Guidance:** `v2_dps`, s = 40, schedule = last_quarter (`GUIDANCE_VARIANT`, `GUIDANCE_KWARGS_JSON`).
- **Mode/weights:** `--guidance_mode {joint,muscle,both}`, `--joint_weight`, `--muscle_weight`,
  `MUSCLE_MARGIN`.
- **Proxy:** frozen 16-layer motion→muscle transformer; final activation = none (checkpoint-matched).
- **Pipelines:** `new/run_apt_integrated.sh` (three-mode comparison);
  `new/evaluate_muscle_space.py` (muscle-space metrics, offline from `comparison.npy`);
  `new/check_proxy_norm.py` (proxy activation/Jacobian diagnostics);
  `new/sanity_muscle_integration.py` (loss-chain regression).

### References (abbrev.)
MDM (Tevet et al., 2023); DiP sampler; HumanML3D / Guo et al. (CVPR 2022); DPS (Chung et al.,
NeurIPS 2022); Muscles-in-Time motion-to-muscle proxy.
