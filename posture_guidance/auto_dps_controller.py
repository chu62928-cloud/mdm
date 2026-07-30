"""
Trust-Region Auto-DPS Controller (pure — no MDM dependency).

Implements the per-sample Gauss-Newton step computation with
diffusion-aware trust region, backtracking, and band hysteresis.

Key equations (from frozen algorithm spec):
  delta_raw = -r / (sum(g^2) + damping) * g           (GN step)
  radius_rms = clamp(radius_scale * noise_level, min, max)
  delta = clip_by_rms(delta_raw, radius_rms)
  rho = actual_reduction / (pred_reduction + eps)

This controller is PURE — it works on residual scalars and
gradient tensors, and never touches the MDM model.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


# ---- Configuration ----

@dataclass
class AutoDPSConfig:
    """Global V7 configuration. Locked after tuning-seed calibration."""
    schedule: str = "second_half"
    radius_scale: float = 0.05
    min_radius_rms: float = 1e-4
    max_radius_rms: float = 0.10
    damping: float = 1e-8
    rho_accept: float = 0.10
    rho_shrink: float = 0.25
    rho_grow: float = 0.75
    shrink_factor: float = 0.5
    grow_factor: float = 1.5
    max_backtracks: int = 3
    band_hysteresis: float = 1.5
    min_valid_fraction: float = 0.8
    trace: bool = True
    disable_band_stop: bool = False
    disable_trial: bool = False
    band_order_bug: bool = False
    # ---- V7.2 three-band target redesign ----
    control_tolerance_deg: float = 2.0
    evaluation_tolerance_deg: float = 2.0
    hysteresis_exit_deg: float = 3.0
    soft_taper_enabled: bool = False
    soft_taper_width_deg: float = 1.0


# ---- State ----

    def __post_init__(self):
        if self.control_tolerance_deg <= 0:
            raise ValueError(f"control_tolerance_deg must be > 0, got {self.control_tolerance_deg}")
        if self.control_tolerance_deg > self.evaluation_tolerance_deg:
            raise ValueError(
                f"control_tolerance_deg ({self.control_tolerance_deg}) must be <= "
                f"evaluation_tolerance_deg ({self.evaluation_tolerance_deg})"
            )
        if self.evaluation_tolerance_deg >= self.hysteresis_exit_deg:
            raise ValueError(
                f"evaluation_tolerance_deg ({self.evaluation_tolerance_deg}) must be < "
                f"hysteresis_exit_deg ({self.hysteresis_exit_deg})"
            )


@dataclass
class AutoDPSState:
    """Per-sample controller state. Reset at start of each trajectory (t=T-1)."""
    radius_scale: torch.Tensor    # (B,) current trust-radius multiplier
    in_control_band: torch.Tensor  # (B,) bool — is this sample in target band?
    accepted_steps: torch.Tensor  # (B,) cumulative accepted step count
    rejected_steps: torch.Tensor  # (B,) cumulative rejected step count
    total_backtracks: torch.Tensor  # (B,) cumulative backtrack count


# ---- Proposal ----

@dataclass
class AutoDPSProposal:
    """Output of controller.propose()."""
    delta_raw: torch.Tensor       # (B, C, 1, T) raw GN step
    delta: torch.Tensor           # (B, C, 1, T) trust-region-clipped step
    predicted_residual: torch.Tensor       # (B,) r + <g, delta>
    predicted_reduction: torch.Tensor      # (B,) Phi_current - Phi_pred
    delta_rms: torch.Tensor       # (B,)
    radius_rms: torch.Tensor      # (B,)
    hit_boundary: torch.Tensor    # (B,) bool — delta was clipped
    valid: torch.Tensor           # (B,) bool — proposal is usable


class TrustRegionAutoDPSController:
    """Per-sample trust-region controller for V7 Auto-DPS.

    Usage per timestep:
        state = controller.reset(batch_size, device, dtype)  # once at t=T-1
        proposal = controller.propose(residual, grad, noise_level, tolerance, valid)
        # ... evaluate trial ...
        state = controller.update_radius(rho, accepted, proposal.hit_boundary)
        state = controller.update_band_state(residual, tolerance)
    """

    def __init__(self, config: AutoDPSConfig = None):
        self.config = config or AutoDPSConfig()

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def reset(self, batch_size: int, device, dtype) -> AutoDPSState:
        """Reset state at start of a new sampling trajectory (t == T-1)."""
        return AutoDPSState(
            radius_scale=torch.ones(batch_size, device=device, dtype=torch.float32),
            in_control_band=torch.zeros(batch_size, device=device, dtype=torch.bool),
            accepted_steps=torch.zeros(batch_size, device=device, dtype=torch.int32),
            rejected_steps=torch.zeros(batch_size, device=device, dtype=torch.int32),
            total_backtracks=torch.zeros(batch_size, device=device, dtype=torch.int32),
        )

    # ----------------------------------------------------------------
    # Step proposal (Section 6 of algorithm spec)
    # ----------------------------------------------------------------

    def propose(
        self,
        residual: torch.Tensor,            # (B,) signed residual
        grad: torch.Tensor,                # (B, C, 1, T) Jacobian
        noise_level: torch.Tensor,         # (B,) or float — sqrt(1 - alpha_bar_t)
        tolerance: torch.Tensor,           # (B,)
        valid: torch.Tensor,               # (B,) bool — measurement valid
        state: AutoDPSState,
    ) -> AutoDPSProposal:
        """Compute Gauss-Newton proposal with trust-region clipping.

        Args:
            residual: (B,) signed summary residual (radians or meters).
            grad: (B, C, 1, T) gradient of residual.sum() w.r.t x_t.
            noise_level: (B,) current diffusion noise level.
            tolerance: (B,) per-sample tolerance.
            valid: (B,) bool — whether measurement is valid.
            state: Current per-sample controller state.

        Returns:
            AutoDPSProposal with delta_raw, delta (clipped), and validity flags.
        """
        cfg = self.config
        B = residual.shape[0]
        device = residual.device
        dtype = grad.dtype

        # Ensure noise_level is (B,)
        if isinstance(noise_level, (int, float)):
            noise_level = torch.full((B,), float(noise_level), device=device, dtype=torch.float32)
        elif noise_level.ndim == 0:
            noise_level = noise_level.expand(B)

        # ---- Compute raw GN step ----
        grad_flat = grad.flatten(start_dim=1)                  # (B, C*1*T)
        g_sq_sum = (grad_flat ** 2).sum(dim=1)                 # (B,)
        damping = torch.full((B,), cfg.damping, device=device, dtype=dtype)

        scale = -residual / (g_sq_sum + damping)               # (B,)
        delta_raw = scale.view(B, 1, 1, 1) * grad              # (B, C, 1, T)

        # ---- Trust region radius ----
        radius_scale = state.radius_scale.to(dtype)
        radius_rms = radius_scale * noise_level.to(dtype)
        radius_rms = torch.clamp(radius_rms,
                                 cfg.min_radius_rms,
                                 cfg.max_radius_rms)

        # ---- RMS clipping ----
        delta_rms = delta_raw.flatten(1).pow(2).mean(1).sqrt()  # (B,)
        eps = 1e-12
        clip_factor = torch.clamp_max(radius_rms / (delta_rms + eps), 1.0)
        delta = delta_raw * clip_factor.view(B, 1, 1, 1)
        hit_boundary = (delta_rms > radius_rms)

        # ---- Linear prediction ----
        delta_flat = delta.flatten(start_dim=1)
        predicted_residual = residual + (grad_flat * delta_flat).sum(dim=1)

        # ---- Predicted reduction in merit ----
        Phi_current = 0.5 * residual ** 2
        Phi_pred = 0.5 * predicted_residual ** 2
        predicted_reduction = Phi_current - Phi_pred

        # ---- Validity checks ----
        # Valid if: measurement valid, residual finite, grad finite, grad not zero,
        #           not in band, radius positive
        valid_proposal = valid.clone()
        valid_proposal = valid_proposal & torch.isfinite(residual)
        valid_proposal = valid_proposal & torch.isfinite(g_sq_sum)
        valid_proposal = valid_proposal & (g_sq_sum > 1e-20)
        if not self.config.disable_band_stop:
            valid_proposal = valid_proposal & ~state.in_control_band
        valid_proposal = valid_proposal & (radius_rms > 0)
        valid_proposal = valid_proposal & torch.isfinite(predicted_reduction)
        valid_proposal = valid_proposal & (predicted_reduction > 0)

        # Zero out invalid proposals
        delta = torch.where(valid_proposal.view(B, 1, 1, 1), delta,
                            torch.zeros_like(delta))
        delta_raw = torch.where(valid_proposal.view(B, 1, 1, 1), delta_raw,
                                torch.zeros_like(delta_raw))
        predicted_residual = torch.where(valid_proposal, predicted_residual,
                                         residual)
        predicted_reduction = torch.where(valid_proposal, predicted_reduction,
                                          torch.zeros_like(predicted_reduction))

        return AutoDPSProposal(
            delta_raw=delta_raw,
            delta=delta,
            predicted_residual=predicted_residual,
            predicted_reduction=predicted_reduction,
            delta_rms=delta_rms,
            radius_rms=radius_rms.to(dtype),
            hit_boundary=hit_boundary,
            valid=valid_proposal,
        )

    # ----------------------------------------------------------------
    # Trial evaluation (Section 7 of algorithm spec)
    # ----------------------------------------------------------------

    def evaluate(
        self,
        proposal: AutoDPSProposal,
        residual_trial: torch.Tensor,     # (B,) trial residual
        valid_trial: torch.Tensor,        # (B,) trial measurement valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rho (actual vs predicted reduction) for a trial.

        Returns:
            (accepted, rho) — both (B,) tensors.
        """
        cfg = self.config
        eps = 1e-12

        # Actual reduction
        # residual is implicit — we compute from proposal validity
        actual_reduction = proposal.predicted_reduction.clone()
        # We need residual_before. The proposal doesn't store it explicitly.
        # We reconstruct: since predicted_reduction = 0.5*r^2 - 0.5*r_pred^2,
        # we need the actual residual.
        # Instead, we'll have the caller pass residual_before explicitly.

        # For now, this is a placeholder — the full evaluation happens in
        # accept_or_backtrack which receives residual_before from caller.
        return torch.zeros_like(proposal.valid), torch.zeros_like(proposal.valid)

    def compute_rho(
        self,
        residual_before: torch.Tensor,       # (B,)
        residual_trial: torch.Tensor,        # (B,)
        predicted_reduction: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Compute rho = actual_reduction / predicted_reduction."""
        eps = 1e-12
        Phi_before = 0.5 * residual_before ** 2
        Phi_trial = 0.5 * residual_trial ** 2
        actual_reduction = Phi_before - Phi_trial
        return actual_reduction / (predicted_reduction + eps)

    def check_acceptance(
        self,
        residual_before: torch.Tensor,       # (B,)
        residual_trial: torch.Tensor,        # (B,)
        predicted_reduction: torch.Tensor,   # (B,)
        valid_trial: torch.Tensor,           # (B,)
        valid_fraction_trial: torch.Tensor,  # (B,)
        valid_fraction_current: torch.Tensor,# (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check acceptance criteria for each sample.

        Returns:
            (accepted, rho, reject_reason) — all (B,)
            reject_reason: 0=none, 1=pred_reduction<=0, 2=actual_reduction<=0,
                           3=rho<accept, 4=not_finite, 5=valid_fraction_low
        """
        cfg = self.config
        B = residual_before.shape[0]
        device = residual_before.device
        eps = 1e-12

        rho = self.compute_rho(residual_before, residual_trial, predicted_reduction)

        Phi_before = 0.5 * residual_before ** 2
        Phi_trial = 0.5 * residual_trial ** 2
        actual_reduction = Phi_before - Phi_trial

        accepted = torch.ones(B, device=device, dtype=torch.bool)
        reason = torch.zeros(B, device=device, dtype=torch.int32)

        # Criterion 1: predicted_reduction > 0
        mask = predicted_reduction <= 0
        accepted = accepted & ~mask
        reason = torch.where(mask, torch.tensor(1, device=device, dtype=torch.int32), reason)

        # Criterion 2: actual_reduction > 0
        mask = actual_reduction <= 0
        accepted = accepted & ~mask
        reason = torch.where(mask & (reason == 0),
                             torch.tensor(2, device=device, dtype=torch.int32), reason)

        # Criterion 3: rho >= rho_accept
        mask = rho < cfg.rho_accept
        accepted = accepted & ~mask
        reason = torch.where(mask & (reason == 0),
                             torch.tensor(3, device=device, dtype=torch.int32), reason)

        # Criterion 4: trial finite
        mask = ~(torch.isfinite(residual_trial) & torch.isfinite(rho))
        accepted = accepted & ~mask
        reason = torch.where(mask & (reason == 0),
                             torch.tensor(4, device=device, dtype=torch.int32), reason)

        # Criterion 5: valid_fraction not significantly degraded
        frac_ratio = valid_fraction_trial / (valid_fraction_current + eps)
        mask = (frac_ratio < cfg.min_valid_fraction) & valid_trial
        accepted = accepted & ~mask
        reason = torch.where(mask & (reason == 0),
                             torch.tensor(5, device=device, dtype=torch.int32), reason)

        # Also require trial measurement valid
        accepted = accepted & valid_trial

        return accepted, rho, reason

    # ----------------------------------------------------------------
    # Acceptance & backtracking (Section 8 of algorithm spec)
    # ----------------------------------------------------------------

    def update_radius(
        self,
        state: AutoDPSState,
        rho: torch.Tensor,              # (B,)
        accepted: torch.Tensor,         # (B,)
        hit_boundary: torch.Tensor,     # (B,)
    ) -> AutoDPSState:
        """Update radius_scale per-sample based on rho and boundary hit.

        - rho < rho_shrink: shrink radius
        - rho > rho_grow and hit_boundary: grow radius
        - Otherwise: keep (slight decay toward 1.0 over time handled elsewhere)
        """
        cfg = self.config
        new_radius_scale = state.radius_scale.clone().to(torch.float32)

        shrink_mask = accepted & (rho < cfg.rho_shrink)
        grow_mask = accepted & (rho > cfg.rho_grow) & hit_boundary

        new_radius_scale = torch.where(
            shrink_mask,
            new_radius_scale * cfg.shrink_factor,
            new_radius_scale,
        )
        new_radius_scale = torch.where(
            grow_mask,
            new_radius_scale * cfg.grow_factor,
            new_radius_scale,
        )
        new_radius_scale = torch.clamp(new_radius_scale, 0.1, 10.0)

        # Update counters
        new_accepted = state.accepted_steps.clone()
        new_rejected = state.rejected_steps.clone()
        new_accepted = torch.where(accepted, new_accepted + 1, new_accepted)
        new_rejected = torch.where(~accepted & (state.radius_scale > 0), new_rejected + 1, new_rejected)

        return AutoDPSState(
            radius_scale=new_radius_scale,
            in_control_band=state.in_control_band,
            accepted_steps=new_accepted,
            rejected_steps=new_rejected,
            total_backtracks=state.total_backtracks,
        )

    def apply_backtrack(
        self,
        proposal: AutoDPSProposal,
        state: AutoDPSState,
    ) -> tuple[AutoDPSProposal, AutoDPSState]:
        """Shrink radius and re-clip delta for backtracking.

        Returns updated proposal and state with shrunk radius.
        """
        cfg = self.config
        B = proposal.delta.shape[0]
        dtype = proposal.delta.dtype

        new_radius_scale = state.radius_scale.clone().to(torch.float32) * cfg.shrink_factor
        new_radius_scale = torch.clamp(new_radius_scale, 0.1, 10.0)
        new_total_backtracks = state.total_backtracks + 1

        # Re-clip delta_raw with new radius
        noise_level = proposal.radius_rms.to(torch.float32) / (state.radius_scale.to(torch.float32) + 1e-12)
        new_radius_rms = torch.clamp(
            new_radius_scale * noise_level.to(torch.float32),
            cfg.min_radius_rms,
            cfg.max_radius_rms,
        )

        delta_rms = proposal.delta_raw.flatten(1).pow(2).mean(1).sqrt()
        eps = 1e-12
        clip_factor = torch.clamp_max(new_radius_rms.to(dtype) / (delta_rms + eps), 1.0)
        new_delta = proposal.delta_raw * clip_factor.view(B, 1, 1, 1)
        new_hit_boundary = (delta_rms > new_radius_rms.to(dtype))

        # Recompute linear prediction with new delta
        # For simplicity, recalculate predicted_reduction
        new_proposal = AutoDPSProposal(
            delta_raw=proposal.delta_raw,
            delta=new_delta,
            predicted_residual=proposal.predicted_residual,  # Needs recalc — handled in V7 step
            predicted_reduction=proposal.predicted_reduction,  # Ditto
            delta_rms=delta_rms * clip_factor,
            radius_rms=new_radius_rms.to(dtype),
            hit_boundary=new_hit_boundary,
            valid=proposal.valid & (new_radius_rms.to(dtype) > cfg.min_radius_rms),
        )

        new_state = AutoDPSState(
            radius_scale=new_radius_scale.to(torch.float32),
            in_control_band=state.in_control_band,
            accepted_steps=state.accepted_steps,
            rejected_steps=state.rejected_steps,
            total_backtracks=new_total_backtracks,
        )

        return new_proposal, new_state

    # ----------------------------------------------------------------
    # Band hysteresis (Section 10 of algorithm spec)
    # ----------------------------------------------------------------

    def update_control_state(
        self,
        state: AutoDPSState,
        residual_deg: torch.Tensor,       # (B,) in degrees
        control_tolerance_deg: float,
        hysteresis_exit_deg: float,
    ) -> AutoDPSState:
        """V7.2: Update in_control_band status with three-band hysteresis.

        Enter control band:  |r| <= control_tolerance
        Leave control band:  |r| > hysteresis_exit
        In between:          state unchanged (hysteresis)

        Unlike V7.1, evaluation tolerance is SEPARATE from control tolerance.
        This function only manages the control-band state machine.

        Backward compat: when control_tolerance_deg == evaluation_tolerance_deg == 2.0
        and hysteresis_exit_deg == 2.0 * band_hysteresis (3.0), behavior matches V7.1.
        """
        abs_r = residual_deg.abs()

        new_in_band = state.in_control_band.clone()

        # Enter control band
        enter = (abs_r <= control_tolerance_deg) & ~new_in_band
        new_in_band = new_in_band | enter

        # Leave control band (wider threshold)
        leave = (abs_r > hysteresis_exit_deg) & new_in_band
        new_in_band = new_in_band & ~leave

        return AutoDPSState(
            radius_scale=state.radius_scale,
            in_control_band=new_in_band,
            accepted_steps=state.accepted_steps,
            rejected_steps=state.rejected_steps,
            total_backtracks=state.total_backtracks,
        )

    def update_band_state(
        self,
        state: AutoDPSState,
        residual: torch.Tensor,
        tolerance: torch.Tensor,
    ) -> AutoDPSState:
        """Legacy V7.1 interface — delegates to update_control_state with
        control_tol = eval_tol = tolerance, hysteresis = tolerance * band_hysteresis.

        Kept for backward compatibility only. V7.2 callers should use
        update_control_state() directly.
        """
        tol_deg = tolerance.mean().item() * 180.0 / 3.141592653589793  # rad -> deg
        exit_deg = tol_deg * self.config.band_hysteresis
        return self.update_control_state(
            state=state,
            residual_deg=torch.rad2deg(residual),
            control_tolerance_deg=tol_deg,
            hysteresis_exit_deg=exit_deg,
        )
        """Update in_band status with hysteresis.

        Enter band:  |r| <= tolerance
        Leave band:  |r| > hysteresis * tolerance
        """
        cfg = self.config
        abs_r = residual.abs()

        new_in_band = state.in_control_band.clone()

        # Enter band
        enter = (abs_r <= tolerance) & ~new_in_band
        new_in_band = new_in_band | enter

        # Leave band (wider threshold)
        leave_threshold = cfg.band_hysteresis * tolerance
        leave = (abs_r > leave_threshold) & new_in_band
        new_in_band = new_in_band & ~leave

        return AutoDPSState(
            radius_scale=state.radius_scale,
            in_control_band=new_in_band,
            accepted_steps=state.accepted_steps,
            rejected_steps=state.rejected_steps,
            total_backtracks=state.total_backtracks,
        )
