"""
V7 Auto-DPS Single-Step Algorithm.

Connects MDM forward/backward, constraint measurement, and the
trust-region controller into a single-step interface used by the sampler.

Design (from revised execution plan Section 2-3):
  - Trial in x_t space → accepted = use full out_trial posterior
  - predict_fn closure calls p_mean_variance() — no replicated MDM logic
  - Frozen masks across trials within same timestep
  - Per-sample independent accept/reject
  - Returns selected_out dict + diagnostics dict
"""

import math
import time
import torch
from typing import Callable, Optional

from .constraint_measurement import (
    ConstraintMeasurement,
    measure_primary_constraint,
)
from .auto_dps_controller import (
    AutoDPSConfig,
    AutoDPSState,
    TrustRegionAutoDPSController,
)


def _check_schedule(t_int: int, T_total: int, schedule: str) -> bool:
    """Check if guidance is active at this timestep.

    schedule="second_half": active when t < T/2 (earlier steps, more noise)
    schedule="always": always active
    """
    if schedule == "always":
        return True
    if schedule == "second_half":
        return t_int < T_total / 2
    if schedule == "last_quarter":
        return t_int < T_total / 4
    if schedule == "never":
        return False
    return True


def compute_constraint_jacobian(
    x_t: torch.Tensor,                # (B, C, 1, T) with requires_grad
    residual_sum: torch.Tensor,       # scalar
) -> torch.Tensor:
    """Compute Jacobian of residual sum w.r.t x_t.

    Args:
        x_t: noisy latent with requires_grad=True.
        residual_sum: scalar = measurement.summary_residual.sum().

    Returns:
        grad: (B, C, 1, T) gradient tensor.
    """
    grad = torch.autograd.grad(
        outputs=residual_sum,
        inputs=x_t,
        create_graph=False,
        retain_graph=False,
    )[0]
    return grad


def apply_v7_step(
    *,
    x_t: torch.Tensor,                         # (B, C, 1, T)
    t_int: int,
    t_tensor: torch.Tensor,                    # (B,) long
    T_total: int,
    predict_fn: Callable,                      # x -> dict with mean, variance, log_variance, pred_xstart
    fk_fn: Callable,                           # x0 -> q
    guidance,                                   # PostureGuidance with is_primary spec
    controller: TrustRegionAutoDPSController,
    controller_state: AutoDPSState,
    noise_level: torch.Tensor,                 # (B,) sqrt(1 - alpha_bar_t)
    config: AutoDPSConfig = None,
) -> tuple[dict, dict, AutoDPSState]:
    """Execute one V7 Auto-DPS step.

    Args:
        x_t: Current noisy latent (B, C, 1, T).
        t_int: Integer timestep index.
        t_tensor: Long tensor (B,).
        T_total: Total diffusion timesteps.
        predict_fn: Closure calling p_mean_variance(model, x, t, ...).
        fk_fn: Forward kinematics function (x0_pred -> joint coords).
        guidance: PostureGuidance instance.
        controller: TrustRegionAutoDPSController instance.
        controller_state: Current per-sample state.
        noise_level: (B,) current diffusion noise level.
        config: AutoDPSConfig (uses controller.config if None).

    Returns:
        (selected_out, diagnostics, new_controller_state)
        selected_out: dict with mean, variance, log_variance, pred_xstart.
        diagnostics: dict with per-timestep trace data.
    """
    cfg = config or controller.config
    B, C, _, T_frames = x_t.shape
    device = x_t.device
    dtype = x_t.dtype

    diag = {
        "t": t_int,
        "noise_level": None,
        "value_before_deg": None,
        "target_deg": None,
        "residual_before_deg": None,
        "tolerance_deg": None,
        "merit_before": None,
        "grad_rms": None,
        "g_sq_sum": None,
        "delta_raw_rms": None,
        "radius_rms": None,
        "delta_rms": None,
        "hit_boundary": None,
        "predicted_residual_deg": None,
        "trial_residual_deg": None,
        "merit_trial": None,
        "pred_reduction": None,
        "actual_reduction": None,
        "rho": None,
        "backtracks": 0,
        "accepted": False,
        "in_band": False,
        "reject_reason": None,
        "valid_fraction": None,
        "active_count": None,
        "extra_forwards": 0,
    }

    # ----------------------------------------------------------------
    # Step 0: Schedule check
    # ----------------------------------------------------------------
    if not _check_schedule(t_int, T_total, cfg.schedule):
        # Not active — return current posterior unchanged
        with torch.no_grad():
            out_current = predict_fn(x_t)
        return {k: v.detach() for k, v in out_current.items()}, diag, controller_state

    # ----------------------------------------------------------------
    # Step 1: Current state measurement (with grad for Jacobian)
    # ----------------------------------------------------------------
    x_t_work = x_t.detach().requires_grad_(True)
    out_current = predict_fn(x_t_work)
    x0_current = out_current["pred_xstart"]
    q_current = fk_fn(x0_current)

    measurement = guidance.measure_primary_constraint(q_current, t_int, T_total)

    # Freeze masks for all trials in this timestep
    frozen_active_mask = measurement.active_mask.detach().clone()
    frozen_valid_mask = measurement.valid_mask.detach().clone()

    residual_before = measurement.summary_residual                   # (B,)
    merit_before = measurement.merit                                # (B,)

    # Check per-sample validity
    valid_measurement = (
        torch.isfinite(residual_before)
        & (measurement.effective_count > 0)
    )

    # ----------------------------------------------------------------
    # Step 2: Compute Jacobian
    # ----------------------------------------------------------------
    r_sum = residual_before.sum()  # scalar — only uses valid samples
    grad = compute_constraint_jacobian(x_t_work, r_sum)

    # ----------------------------------------------------------------
    # Step 3: Controller proposes delta
    # ----------------------------------------------------------------
    tolerance = measurement.tolerance
    proposal = controller.propose(
        residual_before, grad, noise_level.to(device),
        tolerance, valid_measurement, controller_state,
    )

    # ----------------------------------------------------------------
    # Step 4: Trial evaluation loop (with backtracking)
    # ----------------------------------------------------------------
    accepted = torch.zeros(B, device=device, dtype=torch.bool)
    rho = torch.zeros(B, device=device, dtype=torch.float32)
    reject_reason = torch.zeros(B, device=device, dtype=torch.int32)
    out_trial = None
    current_proposal = proposal
    current_state = controller_state
    trial_residual = residual_before.clone()

    for backtrack in range(cfg.max_backtracks + 1):
        # Check if any sample still needs evaluation
        needs_trial = current_proposal.valid & ~accepted
        if not needs_trial.any():
            break

        # Only evaluate samples that need it
        # Build trial x_t
        x_t_trial = x_t.detach() + current_proposal.delta
        # Apply delta only for valid samples (others unchanged)
        trial_mask = needs_trial.view(B, 1, 1, 1)
        x_t_trial = torch.where(trial_mask, x_t_trial, x_t.detach())

        # Trial forward (no grad)
        with torch.no_grad():
            out_trial = predict_fn(x_t_trial)
            q_trial = fk_fn(out_trial["pred_xstart"])
            # Use FROZEN masks
            measurement_trial = guidance.measure_primary_constraint(
                q_trial, t_int, T_total,
                frozen_active_mask=frozen_active_mask,
                frozen_valid_mask=frozen_valid_mask,
            )

        trial_residual = measurement_trial.summary_residual
        valid_trial = (
            torch.isfinite(trial_residual)
            & (measurement_trial.effective_count > 0)
        )

        # Accept/reject
        step_accepted, step_rho, step_reason = controller.check_acceptance(
            residual_before,
            trial_residual,
            current_proposal.predicted_reduction,
            valid_trial,
            measurement_trial.valid_fraction,
            measurement.valid_fraction,
        )

        # Only mark newly accepted samples
        newly_accepted = step_accepted & needs_trial & ~accepted
        accepted = accepted | newly_accepted
        rho = torch.where(newly_accepted, step_rho, rho)
        reject_reason = torch.where(newly_accepted & ~step_accepted, step_reason, reject_reason)

        diag["extra_forwards"] += needs_trial.sum().item()

        if accepted.all() or backtrack >= cfg.max_backtracks:
            break

        # Backtrack: shrink radius for rejected samples
        new_proposal, new_state = controller.apply_backtrack(current_proposal, current_state)
        current_proposal = new_proposal
        current_state = new_state

    diag["backtracks"] = min(backtrack, cfg.max_backtracks)

    # ----------------------------------------------------------------
    # Step 5: Select output (accepted → trial, rejected → current)
    # ----------------------------------------------------------------
    # We need final trial output for accepted samples
    if out_trial is not None and accepted.any():
        # out_trial already computed in the last trial
        pass
    elif out_trial is None:
        # No trial ever ran — use current
        with torch.no_grad():
            out_current_detached = predict_fn(x_t.detach())
        out_trial = {k: v.detach() for k, v in out_current_detached.items()}

    # Build selected_out: per-sample mix of trial and current
    accepted_mask = accepted.view(B, 1, 1, 1)
    selected_out = {}
    out_current_detached = {k: v.detach() for k, v in out_current.items()}
    for key in ["mean", "variance", "log_variance", "pred_xstart"]:
        if key in out_current_detached:
            trial_val = out_trial.get(key, out_current_detached[key])
            selected_out[key] = torch.where(
                accepted_mask, trial_val, out_current_detached[key]
            )

    # ----------------------------------------------------------------
    # Step 6: Update controller state
    # ----------------------------------------------------------------
    # Update radius
    current_state = controller.update_radius(
        current_state, rho, accepted, current_proposal.hit_boundary
    )
    # Update band state
    # Use residual_before for band entry/exit check
    current_state = controller.update_band_state(
        current_state, residual_before, tolerance
    )

    # ----------------------------------------------------------------
    # Step 7: Build diagnostics
    # ----------------------------------------------------------------
    rad_to_deg = 180.0 / math.pi
    diag["noise_level"] = noise_level.mean().item()
    if valid_measurement.any():
        diag["value_before_deg"] = measurement.summary_value[valid_measurement].mean().item() * rad_to_deg
        diag["target_deg"] = measurement.target[valid_measurement].mean().item() * rad_to_deg
        diag["residual_before_deg"] = residual_before[valid_measurement].mean().item() * rad_to_deg
        diag["tolerance_deg"] = tolerance[valid_measurement].mean().item() * rad_to_deg
        diag["merit_before"] = merit_before[valid_measurement].mean().item()
        diag["valid_fraction"] = measurement.valid_fraction[valid_measurement].mean().item()
        diag["active_count"] = measurement.effective_count[valid_measurement].mean().item()
    diag["grad_rms"] = grad.flatten(1).pow(2).mean(1).sqrt().mean().item()
    diag["g_sq_sum"] = grad.flatten(1).pow(2).sum(1).mean().item()
    diag["delta_raw_rms"] = proposal.delta_raw.flatten(1).pow(2).mean(1).sqrt().mean().item()
    diag["radius_rms"] = proposal.radius_rms.mean().item()
    diag["delta_rms"] = current_proposal.delta.flatten(1).pow(2).mean(1).sqrt().mean().item()
    diag["hit_boundary"] = current_proposal.hit_boundary.float().mean().item()
    if accepted.any():
        diag["rho"] = rho[accepted].mean().item()
        diag["trial_residual_deg"] = trial_residual[accepted].mean().item() * rad_to_deg
        diag["predicted_residual_deg"] = current_proposal.predicted_residual[accepted].mean().item() * rad_to_deg
        diag["pred_reduction"] = current_proposal.predicted_reduction[accepted].mean().item()
        actual_red = (0.5 * residual_before[accepted]**2 - 0.5 * trial_residual[accepted]**2)
        diag["actual_reduction"] = actual_red.mean().item()
    diag["accepted"] = accepted.float().mean().item() > 0.5
    diag["in_band"] = current_state.in_band.float().mean().item() > 0.5
    diag["reject_reason"] = reject_reason[~accepted].float().mean().item() if not accepted.all() else 0.0

    # Clean up graph
    del x_t_work

    return selected_out, diag, current_state
