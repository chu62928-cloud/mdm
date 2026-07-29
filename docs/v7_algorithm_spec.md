# V7 Auto-DPS Algorithm Specification

> **Status**: FROZEN (2026-07-29)
> **Branch**: `feature/v7-auto-dps`
> **Commit**: `840896bc93aef96603ffbe75f3589e4b3a30e729`
>
> This document defines the V7 algorithm mathematically. Code implementations must conform to this spec. Changes require a revision of this document.

---

## 1. Control Variable

V7 proposes and validates updates in the noisy latent space:

```
x_t in R^(B x C x 1 x T)
```

where B = batch size, C = channels (= 263 for HumanML3D), T = motion frames.

V7 does NOT directly edit `pred_xstart` or apply unvalidated `delta` to `mu_t`.

---

## 2. Primary Constraint

### 2.1 New LossSpec Fields

Two new fields on `LossSpec`:

```python
is_primary: bool = False       # True for the single V7 constraint
control_type: str = "equality" # equality | lower_bound | upper_bound
```

For APT auto-calibration, the spec must use:

```python
is_primary = True
control_type = "equality"
```

V7 selects the spec where `is_primary == True`. If zero or multiple primary specs exist, V7 raises an error.

### 2.2 Physical Residual

Even if the legacy loss uses `direction="greater_than"` (hinge), the V7 physical residual is:

```
r_i = summary_value_i - target_i
```

where:
- `summary_value_i` = mean angle over effective (active AND valid) frames for sample i
- `target_i` = target_deg converted to radians
- `r_i in R` (signed, in radians)

---

## 3. Merit Function

Per sample:

```
Phi(r) = 0.5 * r^2
```

Target band definition:

```
|r| <= tolerance  ->  "in band"
```

---

## 4. Current State Measurement

```python
out_current = p_mean_variance(model, x_t, t, clip_denoised, denoised_fn, model_kwargs)
x0_current = out_current["pred_xstart"]
q_current = fk_fn(x0_current)
measurement = guidance.measure_primary_constraint(q_current, t, T)
```

This yields: `r_current`, `active_mask`, `valid_mask`, `effective_mask`.

**Mask freezing**: The first measurement in a timestep freezes `active_mask` and `valid_mask`. All subsequent trials within the same timestep reuse these frozen masks. This prevents the model from changing which frames are active to evade control.

---

## 5. Jacobian Computation

```python
x_t_work = x_t.detach().requires_grad_(True)
out_current = predict_fn(x_t_work)
# ... FK, measurement ...
r_sum = measurement.summary_residual.sum()  # scalar
g = torch.autograd.grad(outputs=r_sum, inputs=x_t_work,
                        create_graph=False, retain_graph=False)[0]
```

Per-sample validity check:
- `g` must be finite
- `sum(g^2)` must exceed gradient floor (default: 1e-20)
- measurement must have sufficient effective frames (valid_fraction > 0)

If any check fails, the sample is skipped (zero update) for this timestep.

---

## 6. Gauss-Newton Step Computation

### 6.1 Raw Delta

Per sample (flatten spatial dims):

```python
g_flat = g.flatten(start_dim=1)          # (B, C*1*T)
g_sq_sum = (g_flat ** 2).sum(dim=1)      # (B,)
scale = -r_current / (g_sq_sum + damping) # (B,)
delta_raw = scale.view(B, 1, 1, 1) * g   # (B, C, 1, T)
```

where `damping = 1e-8`.

**Prohibition**: Do NOT use `abs(r) / grad_rms` as step length. The correct GN step uses `sum(g^2)` which accounts for latent dimensionality.

### 6.2 Trust Region

The trust radius is based on diffusion noise level:

```python
noise_level_t = sqrt(1 - alpha_bar_t)
radius_rms = clamp(radius_scale * noise_level_t, min_radius_rms, max_radius_rms)
```

Defaults:
- `radius_scale = 0.05`
- `min_radius_rms = 1e-4`
- `max_radius_rms = 0.10`

### 6.3 RMS Clipping

```python
delta_rms = delta_raw.flatten(1).pow(2).mean(1).sqrt()  # (B,)
clip_factor = torch.clamp_max(radius_rms / (delta_rms + eps), 1.0)
delta = delta_raw * clip_factor.view(B, 1, 1, 1)
hit_boundary = (delta_rms > radius_rms)
```

---

## 7. Trial Evaluation

### 7.1 Candidate

```python
x_t_trial = x_t.detach() + delta
out_trial = predict_fn(x_t_trial)  # no_grad, same p_mean_variance as main sampler
```

### 7.2 Trial Measurement

Uses FROZEN masks from current state (Section 4):

```python
q_trial = fk_fn(out_trial["pred_xstart"])
measurement_trial = guidance.measure_primary_constraint(
    q_trial, t, T,
    frozen_active_mask=current_active_mask,
    frozen_valid_mask=current_valid_mask,
)
r_trial = measurement_trial.summary_residual
```

### 7.3 Merit and Rho

```python
r_pred = r_current + (g_flat * delta_flat).sum(dim=1)  # linear prediction
Phi_current = 0.5 * r_current^2
Phi_pred    = 0.5 * r_pred^2
Phi_trial   = 0.5 * r_trial^2

pred_reduction   = Phi_current - Phi_pred
actual_reduction = Phi_current - Phi_trial
rho = actual_reduction / (pred_reduction + eps)
```

### 7.4 Acceptance Criteria

All must be true:

1. `pred_reduction > 0`
2. `actual_reduction > 0`
3. `rho >= rho_accept` (default: 0.10)
4. Trial finite (no NaN/Inf)
5. Trial `valid_fraction` not significantly degraded vs current
6. Trial effective frame count >= minimum

---

## 8. Backtracking

If a trial is rejected:

```python
radius_scale = radius_scale * shrink_factor      # shrink_factor = 0.5
delta = delta * shrink_factor                     # re-clip raw delta
```

Re-evaluate trial. Maximum `max_backtracks = 3` per timestep.

Radius update after final decision:

```python
if rho < rho_shrink:              # 0.25
    radius_scale *= shrink_factor
elif rho > rho_grow and hit_boundary:  # 0.75
    radius_scale *= grow_factor        # 1.5
radius_scale = clamp(radius_scale, 0.1, 10.0)
```

If all backtracking fails: return `out_current` unchanged.

---

## 9. Accepted Application

On acceptance:

```python
selected_out = out_trial    # complete posterior from p_mean_variance
```

On rejection:

```python
selected_out = {k: v.detach() for k, v in out_current.items()}
```

Then sample:

```python
noise_z = <same noise draw used for this timestep>
nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

sample = selected_out["mean"] \
    + nonzero_mask * torch.exp(0.5 * selected_out["log_variance"]) * noise_z
```

### Noise Fairness

The same `noise_z` is used regardless of accept/reject. Do NOT re-draw noise per backtrack. This ensures reproducible and fair comparison between accepted and rejected paths.

---

## 10. Target-Band Stopping with Hysteresis

Per-sample state machine:

```
State IN_BAND:    |r| <= tolerance
State ACTIVE:     |r| > tolerance

Transition ACTIVE -> IN_BAND:  |r| <= tolerance
Transition IN_BAND -> ACTIVE:  |r| > hysteresis * tolerance
```

Default `hysteresis = 1.5`.

While IN_BAND: no joint update proposed (delta = 0), posterior unchanged, state persists until residual exceeds wider threshold.

---

## 11. Schedule Control

```
schedule = "second_half"  # t < T/2 are active (default)
schedule = "always"       # all t active
```

When a timestep is outside the schedule window, V7 returns `out_current` unchanged.

---

## 12. State Management

### 12.1 Per-Sample Controller State

```python
@dataclass
class AutoDPSState:
    radius_scale: Tensor     # (B,) init 1.0
    in_band: Tensor          # (B,) init False
    accepted_steps: Tensor   # (B,) count
    rejected_steps: Tensor   # (B,) count
    total_backtracks: Tensor # (B,) count
```

### 12.2 Lifecycle

- **Reset**: At the start of each new sampling trajectory (t = T-1)
- **Persist**: Controller state persists across timesteps within a trajectory
- **Independent**: Each batch sample has its own state; no cross-contamination

---

## 13. Default Global Configuration

```json
{
  "schedule": "second_half",
  "radius_scale": 0.05,
  "min_radius_rms": 0.0001,
  "max_radius_rms": 0.10,
  "damping": 1e-8,
  "rho_accept": 0.10,
  "rho_shrink": 0.25,
  "rho_grow": 0.75,
  "shrink_factor": 0.5,
  "grow_factor": 1.5,
  "max_backtracks": 3,
  "band_hysteresis": 1.5,
  "min_valid_fraction": 0.8,
  "trace": true
}
```

These values are the first-smoke defaults. They may be adjusted ONCE on tuning seeds (10 seeds). After that, they are locked for all targets and all test seeds. No per-target or per-seed tuning is allowed.

---

## 14. Diagnostics Per Timestep

```json
{
  "seed": 42,
  "t": 7,
  "noise_level": 0.12,
  "value_before_deg": 13.4,
  "target_deg": 20.0,
  "residual_before_deg": -6.6,
  "tolerance_deg": 2.0,
  "merit_before": 0.0066,
  "grad_rms": 0.0031,
  "g_sq_sum": 0.0025,
  "delta_raw_rms": 0.42,
  "radius_rms": 0.06,
  "delta_rms": 0.06,
  "hit_boundary": true,
  "predicted_residual_deg": -3.2,
  "trial_residual_deg": -3.8,
  "pred_reduction": 1.23e-4,
  "actual_reduction": 1.10e-4,
  "rho": 0.74,
  "backtracks": 1,
  "accepted": true,
  "in_band": false,
  "reject_reason": null,
  "valid_fraction": 0.95,
  "active_count": 57
}
```

---

## 15. V7 Integration Point in Sampler

V7 requires a **special branch** in `p_sample_loop_progressive()`, NOT just an entry in `_apply_guidance_variant()`.

The special branch:
1. Replaces the full posterior (`mean`, `variance`, `log_variance`, `pred_xstart`)
2. Uses the same `noise_z` for accept and reject paths
3. Calls `p_mean_variance()` via a closure for trials (outside the guidance dispatch)
4. Resets controller state at start of new trajectory (t = T-1)
