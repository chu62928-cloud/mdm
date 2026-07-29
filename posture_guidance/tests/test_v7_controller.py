"""Task 2: Auto-DPS Controller unit tests.

Tests:
  1. 1D linear model — GN step should reach target in one step
  2. Nonlinear toy — bad trial triggers radius shrink
  3. Unit invariance — rad/deg produce same delta
  4. Batch independence — no cross-contamination
  5. Trust radius scales with noise level
  6. Band hysteresis — enter/exit behavior
  7. NaN/zero-grad safety — zero update
  8. Backtracking — shrinks radius and delta
"""

import math
import torch
import pytest

from posture_guidance.auto_dps_controller import (
    AutoDPSConfig,
    AutoDPSState,
    TrustRegionAutoDPSController,
)


@pytest.fixture
def ctrl():
    return TrustRegionAutoDPSController()


# ---- Test 1: 1D linear model ----

def test_linear_one_step(ctrl):
    """For r(x) = a*x - b, one GN step should reach the target.

    In 1D: residual r = a*x - b, gradient g = a.
    GN step: delta = -r/(g^2 + damping) * g = -(a*x-b)/(a^2) * a = (b-a*x)/a = target - x
    So one step should reach target exactly (modulo damping).
    """
    B = 2
    a_val = 3.0
    b_val = 6.0  # target value

    # Current x yields residual = 3*2 - 6 = 0 (at target for x=2)
    # Let's start away: x=10 → r = 3*10 - 6 = 24, target x=2
    x_current = torch.tensor([10.0, 10.0])
    r = a_val * x_current - b_val  # = [24, 24]

    # Gradient: dr/dx = a = 3
    g = torch.full((B, 263, 1, 60), a_val / 100.0)  # small per-element
    # But we need sum(g^2) = a^2 = 9 (across all elements)
    # Number of elements = 263 * 1 * 60 = 15780
    n_elem = 263 * 60
    g_val = math.sqrt(a_val**2 / n_elem)
    g = torch.full((B, 263, 1, 60), g_val)

    nl = torch.tensor([0.3, 0.3])
    tol = torch.tensor([0.001, 0.001])
    valid = torch.tensor([True, True])

    state = ctrl.reset(B, r.device, g.dtype)
    prop = ctrl.propose(r, g, nl, tol, valid, state)

    # The GN step should reduce residual close to zero
    assert prop.valid.all()
    # predicted_residual ≈ 0
    assert (prop.predicted_residual.abs() < 0.01).all(), \
        f"Expected near-zero pred residual, got {prop.predicted_residual}"


# ---- Test 2: Nonlinear toy — bad trial shrinks radius ----

def test_nonlinear_radius_shrink(ctrl):
    """Simulate a nonlinear function where the linear model is poor.
    When rho is low, radius should shrink."""
    B = 2

    # Simulate: residual_before, we set up a contrived scenario
    r_before = torch.tensor([0.5, 0.3])
    r_trial = torch.tensor([0.48, 0.28])  # small improvement
    pred_reduction = torch.tensor([0.05, 0.04])  # predicted much larger
    valid_trial = torch.tensor([True, True])
    vfrac_trial = torch.tensor([0.95, 0.95])
    vfrac_current = torch.tensor([0.95, 0.95])

    accepted, rho, reason = ctrl.check_acceptance(
        r_before, r_trial, pred_reduction, valid_trial, vfrac_trial, vfrac_current,
    )

    # rho should be small
    assert (rho < ctrl.config.rho_shrink).any(), \
        f"Expected rho < {ctrl.config.rho_shrink}, got {rho}"

    # Update radius: should shrink for small rho
    state = ctrl.reset(B, r_before.device, rho.dtype)
    hit_boundary = torch.tensor([False, False])
    new_state = ctrl.update_radius(state, rho, accepted, hit_boundary)

    # Samples with low rho should have shrunk radius
    for i in range(B):
        if accepted[i] and rho[i] < ctrl.config.rho_shrink:
            assert new_state.radius_scale[i] < state.radius_scale[i], \
                f"Radius should shrink for low rho sample {i}"


# ---- Test 3: Unit invariance (radians vs degrees) ----

def test_unit_invariance(ctrl):
    """Using radians or degrees should produce the same physical delta.

    If we scale residual and gradient by the same factor (e.g., deg = rad * 180/pi),
    the GN step delta should be approximately invariant.
    """
    B = 1
    rad_to_deg = 180.0 / math.pi

    # Radians scenario
    r_rad = torch.tensor([0.35])  # ~20 deg in radians
    n_elem = 263 * 60
    g_val_rad = 0.001
    g_rad = torch.full((B, 263, 1, 60), g_val_rad, dtype=torch.float64)

    nl = torch.tensor([0.3], dtype=torch.float64)
    tol_rad = torch.tensor([0.035], dtype=torch.float64)  # 2 deg
    valid = torch.tensor([True])

    state_rad = ctrl.reset(B, r_rad.device, torch.float64)
    prop_rad = ctrl.propose(r_rad, g_rad, nl, tol_rad, valid, state_rad)

    # Degrees scenario
    r_deg = r_rad * rad_to_deg
    g_deg = g_rad * rad_to_deg  # gradient also scales
    tol_deg = tol_rad * rad_to_deg

    state_deg = ctrl.reset(B, r_deg.device, torch.float64)
    prop_deg = ctrl.propose(r_deg, g_deg, nl, tol_deg, valid, state_deg)

    # GN step is scale-invariant: delta_deg == delta_rad (same physical update)
    # Because: r_deg = s*r_rad, g_deg = s*g_rad
    #   scale_deg = -r_deg/sum(g_deg^2) = -s*r / (s^2*sum(g^2)) = scale_rad / s
    #   delta_deg = scale_deg * g_deg = (scale_rad/s) * (s*g) = delta_rad
    delta_rad_rms = prop_rad.delta.flatten(1).pow(2).mean(1).sqrt()
    delta_deg_rms = prop_deg.delta.flatten(1).pow(2).mean(1).sqrt()

    ratio = delta_deg_rms / (delta_rad_rms + 1e-12)
    assert torch.allclose(ratio.float(), torch.tensor([1.0]), rtol=0.02), \
        f"Expected ratio ~1.0 (scale-invariant GN), got {ratio.item():.4f}"


# ---- Test 4: Batch independence ----

def test_batch_independence(ctrl):
    """Samples with different states (in-band, large residual, zero grad)
    must not contaminate each other."""
    B = 4
    device = "cpu"
    dtype = torch.float32

    # Sample 0: already in band → proposal should be invalid
    # Sample 1: large residual → valid proposal
    # Sample 2: zero gradient → invalid proposal
    # Sample 3: NaN residual → invalid proposal
    r = torch.tensor([0.01, 0.5, 0.3, float('nan')])
    tol = torch.full((B,), 0.035)

    n_elem = 263 * 60
    g = torch.randn(B, 263, 1, 60) * 0.001
    g[2] = 0.0  # zero grad for sample 2

    valid = torch.tensor([True, True, True, False])

    state = ctrl.reset(B, device, dtype)
    # Set sample 0 as in-band
    state.in_band[0] = True

    nl = torch.full((B,), 0.3)
    prop = ctrl.propose(r, g, nl, tol, valid, state)

    # Sample 0: in-band → not valid
    assert not prop.valid[0], "Sample in-band should have invalid proposal"
    # Sample 1: everything fine → valid
    assert prop.valid[1], "Sample with valid measurement should be valid"
    # Sample 2: zero grad → not valid
    assert not prop.valid[2], "Sample with zero grad should be invalid"
    # Sample 3: invalid measurement → not valid
    assert not prop.valid[3], "Sample with invalid measurement should be invalid"

    # delta for invalid samples should be zero
    assert (prop.delta[0].abs().sum() == 0)
    assert (prop.delta[2].abs().sum() == 0)
    assert (prop.delta[3].abs().sum() == 0)
    # delta for valid sample should be nonzero
    assert (prop.delta[1].abs().sum() > 0)


# ---- Test 5: Trust radius scales with noise level ----

def test_trust_radius_noise_scaling(ctrl):
    """Higher noise level should produce larger trust radius."""
    B = 2
    dtype = torch.float32

    r = torch.tensor([0.5, 0.5])
    g = torch.randn(B, 263, 1, 60) * 0.001
    tol = torch.tensor([0.035, 0.035])
    valid = torch.tensor([True, True])

    state = ctrl.reset(B, r.device, dtype)

    # Low noise
    nl_low = torch.tensor([0.01, 0.01])
    prop_low = ctrl.propose(r, g, nl_low, tol, valid, state)

    # High noise
    nl_high = torch.tensor([0.5, 0.5])
    prop_high = ctrl.propose(r, g, nl_high, tol, valid, state)

    assert (prop_high.radius_rms > prop_low.radius_rms).all(), \
        f"High-noise radius {prop_high.radius_rms} should exceed low-noise {prop_low.radius_rms}"


# ---- Test 6: Band hysteresis ----

def test_band_hysteresis(ctrl):
    """Enter band at |r| <= tol, leave at |r| > hysteresis * tol."""
    B = 1
    state = ctrl.reset(B, "cpu", torch.float32)
    tol = torch.tensor([0.035])  # ~2 deg

    # Initially not in band
    assert not state.in_band[0]

    # Residual within tolerance → enter band
    r_small = torch.tensor([0.01])
    state = ctrl.update_band_state(state, r_small, tol)
    assert state.in_band[0], "Should enter band"

    # Residual slightly above tolerance but below hysteresis → stay in band
    r_medium = torch.tensor([0.04])  # > 0.035 but < 1.5*0.035 = 0.0525
    state = ctrl.update_band_state(state, r_medium, tol)
    assert state.in_band[0], "Should stay in band below hysteresis threshold"

    # Residual above hysteresis threshold → leave band
    r_large = torch.tensor([0.06])  # > 1.5 * 0.035 = 0.0525
    state = ctrl.update_band_state(state, r_large, tol)
    assert not state.in_band[0], "Should leave band above hysteresis threshold"


# ---- Test 7: NaN and Inf safety ----

def test_nan_safety(ctrl):
    """NaN or Inf in residual, gradient, or measurement should produce zero update."""
    B = 2
    dtype = torch.float32

    # NaN residual
    r_nan = torch.tensor([float('nan'), 0.5])
    g = torch.randn(B, 263, 1, 60) * 0.001
    tol = torch.tensor([0.035, 0.035])
    valid = torch.tensor([True, True])
    nl = torch.tensor([0.3, 0.3])

    state = ctrl.reset(B, r_nan.device, dtype)
    prop = ctrl.propose(r_nan, g, nl, tol, valid, state)

    # NaN sample should be invalid (not cause crash)
    assert not prop.valid[0], "NaN residual should be invalid"
    assert (prop.delta[0].abs().sum() == 0), "NaN sample delta should be zero"

    # Valid sample should still work
    assert prop.valid[1], "Valid sample should be unaffected"


# ---- Test 8: Backtracking shrinks radius and delta ----

def test_backtrack_shrinks(ctrl):
    """apply_backtrack should reduce radius_scale and delta magnitude."""
    B = 2
    dtype = torch.float32

    r = torch.tensor([0.5, 0.5])
    g = torch.randn(B, 263, 1, 60) * 0.001
    tol = torch.tensor([0.035, 0.035])
    valid = torch.tensor([True, True])
    nl = torch.tensor([0.3, 0.3])

    state = ctrl.reset(B, r.device, dtype)
    prop = ctrl.propose(r, g, nl, tol, valid, state)

    old_radius_scale = state.radius_scale.clone()
    old_delta_rms = prop.delta.flatten(1).pow(2).mean(1).sqrt().clone()

    new_prop, new_state = ctrl.apply_backtrack(prop, state)

    # Radius should shrink
    assert (new_state.radius_scale < old_radius_scale).all()
    # Total backtracks incremented
    assert (new_state.total_backtracks > state.total_backtracks).all()
    # Delta should be smaller (or equal if not on boundary)
    # At minimum it shouldn't be larger
    new_delta_rms = new_prop.delta.flatten(1).pow(2).mean(1).sqrt()
    assert (new_delta_rms <= old_delta_rms + 1e-6).all(), \
        f"Backtrack should not increase delta: {old_delta_rms} -> {new_delta_rms}"


# ---- Main ----

if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
