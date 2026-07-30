"""
V7 Auto-DPS Single-Step Algorithm (V7.1 — extended R3 trace diagnostics).

Connects MDM forward/backward, constraint measurement, and the
trust-region controller into a single-step interface used by the sampler.

R3 trace fields (Phase 0): schedule_active, proposal_valid, proposal_skip_reason,
  measurement_valid, gradient_floor_triggered, clip_factor, boundary_hit,
  proposal_count, accepted_proposal_count, remaining_active_steps
"""

import math, json, os, time
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
    if schedule == "always":
        return True
    if schedule == "second_half":
        return t_int < T_total / 2
    if schedule == "last_quarter":
        return t_int < T_total / 4
    if schedule == "never":
        return False
    return True


def _count_remaining_steps(t_int: int, T_total: int, schedule: str) -> int:
    """Count remaining schedule-active diffusion steps."""
    n = 0
    for tt in range(t_int - 1, -1, -1):
        if _check_schedule(tt, T_total, schedule):
            n += 1
    return n


def compute_constraint_jacobian(
    x_t: torch.Tensor,
    residual_sum: torch.Tensor,
) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=residual_sum,
        inputs=x_t,
        create_graph=False,
        retain_graph=False,
    )[0]
    return grad


# ---- Per-seed JSONL trace writer ----

_TRACE_DIR = None
_TRACE_SEED = None
_TRACE_FILE = None

def init_trace(run_dir: str, seed: int):
    """Initialize JSONL trace writer for a new sampling trajectory."""
    global _TRACE_DIR, _TRACE_SEED, _TRACE_FILE
    _TRACE_DIR = run_dir
    _TRACE_SEED = seed
    os.makedirs(run_dir, exist_ok=True)
    trace_path = os.path.join(run_dir, "v7_trace.jsonl")
    _TRACE_FILE = open(trace_path, "w")
    return trace_path


def write_trace(diag: dict, seed: int):
    """Write one timestep's diagnostics to JSONL."""
    global _TRACE_FILE
    if _TRACE_FILE is None:
        return
    record = {"seed": seed}
    for k, v in diag.items():
        if isinstance(v, torch.Tensor):
            record[k] = v.item() if v.numel() == 1 else v.tolist()
        elif isinstance(v, (int, float, bool, str, type(None))):
            record[k] = v
        else:
            record[k] = str(v)
    _TRACE_FILE.write(json.dumps(record) + "\n")
    _TRACE_FILE.flush()


def close_trace():
    global _TRACE_FILE
    if _TRACE_FILE is not None:
        _TRACE_FILE.close()
        _TRACE_FILE = None


def apply_v7_step(
    *,
    x_t: torch.Tensor,
    t_int: int,
    t_tensor: torch.Tensor,
    T_total: int,
    predict_fn: Callable,
    fk_fn: Callable,
    guidance,
    controller: TrustRegionAutoDPSController,
    controller_state: AutoDPSState,
    noise_level: torch.Tensor,
    config: AutoDPSConfig = None,
    trace_seed: int = None,
) -> tuple[dict, dict, AutoDPSState]:
    cfg = config or controller.config
    B, C, _, T_frames = x_t.shape
    device = x_t.device
    dtype = x_t.dtype

    rad_to_deg = 180.0 / math.pi
    eps = 1e-12

    # ---- Extended diagnostics (R3 fields) ----
    diag = {
        "t": t_int,
        "noise_level": None,
        "value_before_deg": None, "target_deg": None,
        "residual_before_deg": None, "tolerance_deg": None,
        "merit_before": None,
        "grad_rms": None, "g_sq_sum": None,
        "delta_raw_rms": None, "radius_rms": None, "delta_rms": None,
        "hit_boundary": None,
        "predicted_residual_deg": None, "trial_residual_deg": None,
        "merit_trial": None,
        "pred_reduction": None, "actual_reduction": None, "rho": None,
        "backtracks": 0, "accepted": False, "in_control_band": False,
        "reject_reason": None, "valid_fraction": None, "active_count": None,
        "extra_forwards": 0,
        # ---- V7.2 three-band trace fields ----
        "control_tolerance_deg": None,
        "evaluation_tolerance_deg": None,
        "hysteresis_exit_deg": None,
        "in_control_band_before": False,
        "in_control_band_after": False,
        "control_stop_triggered": False,
        "reactivation_triggered": False,
        "distance_to_target_deg": None,
        # ---- R3 extended ----
        "schedule_active": True,
        "proposal_valid": False, "proposal_skip_reason": "",
        "measurement_valid": False, "gradient_floor_triggered": False,
        "clip_factor": None, "boundary_hit": False,
        "proposal_count": 0, "accepted_proposal_count": 0,
        "remaining_active_steps": None, "radius_scale_value": None,
    }

    # ---- Step 0: Schedule check ----
    diag["remaining_active_steps"] = _count_remaining_steps(t_int, T_total, cfg.schedule)

    if not _check_schedule(t_int, T_total, cfg.schedule):
        diag["schedule_active"] = False
        diag["proposal_skip_reason"] = "schedule_inactive"
        with torch.no_grad():
            out_current = predict_fn(x_t)
        if trace_seed is not None:
            write_trace(diag, trace_seed)
        return {k: v.detach() for k, v in out_current.items()}, diag, controller_state

    # ---- Step 1: Current state measurement ----
    x_t_work = x_t.detach().requires_grad_(True)
    out_current = predict_fn(x_t_work)
    x0_current = out_current["pred_xstart"]
    q_current = fk_fn(x0_current)

    measurement = guidance.measure_primary_constraint(q_current, t_int, T_total)

    frozen_active_mask = measurement.active_mask.detach().clone()
    frozen_valid_mask = measurement.valid_mask.detach().clone()

    residual_before = measurement.summary_residual
    merit_before = measurement.merit

    valid_measurement = (
        torch.isfinite(residual_before)
        & (measurement.effective_count > 0)
    )
    diag["measurement_valid"] = valid_measurement.any().item()

    # ---- FIX (Branch C): Update band state BEFORE proposal ----
    # Previously this was at Step 6 (after proposal), causing stale in_control_band
    # to block proposals even when residual had drifted out of tolerance.
    if not cfg.band_order_bug:
        was_in_control = controller_state.in_control_band.clone()
        # V7.2: use three-band logic with separate control/evaluation/hysteresis
        residual_deg = residual_before * rad_to_deg
        controller_state = controller.update_control_state(
            state=controller_state,
            residual_deg=residual_deg,
            control_tolerance_deg=cfg.control_tolerance_deg,
            hysteresis_exit_deg=cfg.hysteresis_exit_deg,
        )
        diag["band_just_entered"] = (controller_state.in_control_band & ~was_in_control).any().item()
        diag["band_just_exited"] = (~controller_state.in_control_band & was_in_control).any().item()
        diag["in_control_band_before"] = was_in_control.any().item()
        diag["in_control_band_after"] = controller_state.in_control_band.any().item()
        diag["control_stop_triggered"] = (~was_in_control & controller_state.in_control_band).any().item()
        diag["reactivation_triggered"] = (was_in_control & ~controller_state.in_control_band).any().item()
        diag["control_tolerance_deg"] = cfg.control_tolerance_deg
        diag["evaluation_tolerance_deg"] = cfg.evaluation_tolerance_deg
        diag["hysteresis_exit_deg"] = cfg.hysteresis_exit_deg
        diag["distance_to_target_deg"] = residual_deg[valid_measurement].mean().item() if valid_measurement.any() else None

    if not valid_measurement.any():
        diag["proposal_skip_reason"] = "measurement_invalid"
        if trace_seed is not None:
            write_trace(diag, trace_seed)
        with torch.no_grad():
            out_detached = predict_fn(x_t.detach())
        return {k: v.detach() for k, v in out_detached.items()}, diag, controller_state

    # ---- Step 2: Jacobian ----
    r_sum = residual_before.sum()
    grad = compute_constraint_jacobian(x_t_work, r_sum)

    # R3: gradient floor check
    grad_flat = grad.flatten(start_dim=1)
    g_sq_per_sample = (grad_flat ** 2).sum(dim=1)
    diag["gradient_floor_triggered"] = (g_sq_per_sample <= 1e-20).any().item()

    # ---- Step 3: Controller proposal ----
    tolerance = measurement.tolerance
    proposal = controller.propose(
        residual_before, grad, noise_level.to(device),
        tolerance, valid_measurement, controller_state,
    )

    # R3: proposal-level diagnostics
    diag["proposal_valid"] = proposal.valid.any().item()
    diag["proposal_count"] = int(proposal.valid.sum().item())
    diag["boundary_hit"] = proposal.hit_boundary.any().item()
    diag["radius_scale_value"] = controller_state.radius_scale.mean().item()

    if proposal.valid.any():
        raw_rms = proposal.delta_raw[proposal.valid].flatten(1).pow(2).mean(1).sqrt()
        clipped_rms = proposal.delta[proposal.valid].flatten(1).pow(2).mean(1).sqrt()
        cf = (clipped_rms / (raw_rms + eps)).mean().item()
        diag["clip_factor"] = cf

    if not proposal.valid.any():
        skip_reasons = []
        if not valid_measurement.any():
            skip_reasons.append("measurement_invalid")
        if controller_state.in_control_band.any():
            skip_reasons.append("in_control_band")
        if diag.get("gradient_floor_triggered", False):
            skip_reasons.append("gradient_floor")
        diag["proposal_skip_reason"] = "+".join(skip_reasons) if skip_reasons else "proposal_invalid"

    # ---- Step 4: Trial evaluation loop ----
    accepted = torch.zeros(B, device=device, dtype=torch.bool)
    rho = torch.zeros(B, device=device, dtype=torch.float32)
    reject_reason = torch.zeros(B, device=device, dtype=torch.int32)
    out_trial = None
    current_proposal = proposal
    current_state = controller_state
    trial_residual = residual_before.clone()
    n_proposals_evaluated = 0

    if cfg.disable_trial:
        # Ablation: accept all valid proposals without candidate evaluation
        x_t_trial = x_t.detach() + proposal.delta
        valid_mask = proposal.valid.view(B, 1, 1, 1)
        x_t_trial = torch.where(valid_mask, x_t_trial, x_t.detach())
        with torch.no_grad():
            out_trial = predict_fn(x_t_trial)
        accepted = proposal.valid.clone()
        rho = torch.ones(B, device=device, dtype=torch.float32)
        diag["backtracks"] = 0
        diag["extra_forwards"] = int(proposal.valid.sum().item())
        diag["accepted_proposal_count"] = int(accepted.sum().item())
    else:
        for backtrack in range(cfg.max_backtracks + 1):
            needs_trial = current_proposal.valid & ~accepted
            if not needs_trial.any():
                break

            x_t_trial = x_t.detach() + current_proposal.delta
            trial_mask = needs_trial.view(B, 1, 1, 1)
            x_t_trial = torch.where(trial_mask, x_t_trial, x_t.detach())

            with torch.no_grad():
                out_trial = predict_fn(x_t_trial)
                q_trial = fk_fn(out_trial["pred_xstart"])
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

            step_accepted, step_rho, step_reason = controller.check_acceptance(
                residual_before, trial_residual,
                current_proposal.predicted_reduction,
                valid_trial, measurement_trial.valid_fraction,
                measurement.valid_fraction,
            )

            newly_accepted = step_accepted & needs_trial & ~accepted
            accepted = accepted | newly_accepted
            rho = torch.where(newly_accepted, step_rho, rho)
            reject_reason = torch.where(newly_accepted & ~step_accepted, step_reason, reject_reason)

            n_proposals_evaluated += needs_trial.sum().item()

            if accepted.all() or backtrack >= cfg.max_backtracks:
                break

            new_proposal, new_state = controller.apply_backtrack(current_proposal, current_state)
            current_proposal = new_proposal
            current_state = new_state

        diag["backtracks"] = min(backtrack, cfg.max_backtracks)
        diag["extra_forwards"] = n_proposals_evaluated
        diag["accepted_proposal_count"] = int(accepted.sum().item())

    # ---- Step 5: Select output ----
    if out_trial is None:
        with torch.no_grad():
            out_current_detached = predict_fn(x_t.detach())
        out_trial = {k: v.detach() for k, v in out_current_detached.items()}

    accepted_mask = accepted.view(B, 1, 1, 1)
    selected_out = {}
    out_current_detached = {k: v.detach() for k, v in out_current.items()}
    for key in ["mean", "variance", "log_variance", "pred_xstart"]:
        if key in out_current_detached:
            trial_val = out_trial.get(key, out_current_detached[key])
            selected_out[key] = torch.where(
                accepted_mask, trial_val, out_current_detached[key]
            )

    # ---- Step 6: Update controller state ----
    current_state = controller.update_radius(
        current_state, rho, accepted, current_proposal.hit_boundary
    )
    if cfg.band_order_bug:
        residual_deg_bug = residual_before * rad_to_deg
        current_state = controller.update_control_state(
            state=current_state,
            residual_deg=residual_deg_bug,
            control_tolerance_deg=cfg.control_tolerance_deg,
            hysteresis_exit_deg=cfg.hysteresis_exit_deg,
        )

    # ---- Step 7: Build diagnostics ----
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
    diag["g_sq_sum"] = g_sq_per_sample.mean().item()
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
    diag["in_control_band"] = current_state.in_control_band.float().mean().item() > 0.5
    if not accepted.all():
        diag["reject_reason"] = int(reject_reason[~accepted].float().mean().item()) if (~accepted).any() else 0

    # ---- Trace ----
    if trace_seed is not None:
        write_trace(diag, trace_seed)

    del x_t_work
    return selected_out, diag, current_state
