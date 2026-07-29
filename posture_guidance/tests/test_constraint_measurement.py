"""Task 1: Constraint Measurement API unit tests.

Tests:
  1. phase=always → all frames active
  2. stance_left → mask reduces active fraction
  3. mask all zero → valid=False handling
  4. APT residual sign (positive when pelvis tilted forward)
  5. PPT residual sign (negative when pelvis tilted backward)
  6. degree ↔ radian consistency
  7. Manual angle increase → residual monotonic change
  8. Frozen mask → trial mask exactly matches
  9. Batch independence
  10. find_primary_spec errors
"""

import math
import sys
import json
import torch
import pytest

# Setup path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from posture_guidance.registry import LossSpec, POSTURE_REGISTRY
from posture_guidance.phase_detector import PhaseDetector, PHASE_FUNCTIONS
from posture_guidance.constraint_measurement import (
    ConstraintMeasurement,
    masked_batch_mean,
    spec_target_tensor,
    spec_tolerance_tensor,
    measure_spec,
    find_primary_spec,
    measure_primary_constraint,
)
from posture_guidance.controller import PostureGuidance


# ---- Fixtures ----

@pytest.fixture
def detector():
    return PhaseDetector()


@pytest.fixture
def q_apt():
    """Simulate APT: pelvis tilted forward (~+25 deg). B=2, N=60, J=22, 3D coords."""
    B, N, J = 2, 60, 22
    q = torch.randn(B, N, J, 3)
    # Set pelvis joint indices (0=pelvis) with forward tilt signature
    # For testing sign: set y-coordinates to create a detectable tilt
    q[:, :, 0, 0] = 1.5   # pelvis x (forward)
    q[:, :, 0, 1] = 2.5   # pelvis y (up) — tilted forward means higher
    q[:, :, 12, 0] = 0.5  # mid-spine x behind pelvis
    q[:, :, 12, 1] = 1.5  # mid-spine y lower
    # Second sample slightly different
    q[1, :, 0, 1] += 0.3
    return q


# ---- Test 1: phase=always → all frames active ----

def test_phase_always(detector):
    """phase=always should produce all-ones mask."""
    q = torch.randn(2, 60, 22, 3)
    mask = PHASE_FUNCTIONS["always"](detector, q)
    assert mask.shape == (2, 60)
    assert torch.allclose(mask, torch.ones_like(mask))


# ---- Test 2: stance_left → mask reduces active fraction ----

def test_stance_left(detector, q_apt):
    """stance_left should have active_fraction in (0, 1]. Not all frames are stance."""
    q = torch.randn(2, 60, 22, 3)
    # Use left_foot index from PhaseDetector (matching joint_indices)
    foot_idx_l = detector.foot_idx["left"]
    # Create alternating stance/swing: foot low (stance) then high (swing)
    pattern = torch.zeros(60)
    pattern[0:15] = -0.1   # well below height_thresh=0.05 (stance)
    pattern[15:30] = 0.5   # well above (swing)
    pattern[30:45] = -0.1  # stance
    pattern[45:60] = 0.5   # swing
    q[:, :, foot_idx_l, 1] = pattern.unsqueeze(0)

    mask = PHASE_FUNCTIONS["stance_left"](detector, q)
    active = (mask > 0.5).float().mean(dim=1)
    # Should be between 0 and 1
    assert (active > 0.0).all(), f"Expected some active frames, got {active}"
    assert (active < 1.0).all(), f"Expected some inactive frames, got {active}"


# ---- Test 3: mask all zero → valid handling ----

def test_mask_all_zero(detector):
    """When effective_mask is all zero, summary returns reasonable values, and
    valid_fraction indicates no valid frames."""
    q = torch.randn(2, 60, 22, 3) * 0.001  # Tiny coords → all height below threshold?
    # Force a zero mask scenario
    frozen_zero = torch.zeros(2, 60, device=q.device, dtype=q.dtype)

    # Use an equality spec for simple testing
    spec = LossSpec(
        name="test",
        angle_fn=lambda q: q[:, :, 0, 0],  # Simple projection
        target_deg=0.0,
        direction="equal",
        tolerance_deg=1.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )

    meas = measure_spec(spec, q, detector, t=0, T=50,
                        frozen_active_mask=frozen_zero)
    # With zero mask, effective_count should be very small
    assert (meas.effective_count < 1.0).all()


# ---- Test 4: APT residual sign ----

def test_apt_residual_sign(detector):
    """For APT with target=20deg (0.349 rad), if the measured angle exceeds target,
    residual should be positive (value > target)."""
    target_deg = 20.0
    target_rad = target_deg * math.pi / 180.0

    # We test sign logic using a synthetic angle_fn that returns known values
    spec = LossSpec(
        name="test_apt",
        angle_fn=lambda q, **kw: torch.ones(2, 60) * 0.5,  # 0.5 rad ≈ 28.6 deg
        target_deg=target_deg,
        direction="greater_than",
        tolerance_deg=2.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )
    q = torch.randn(2, 60, 22, 3)
    meas = measure_spec(spec, q, detector, t=0, T=50)

    # All frames produce 0.5 rad, target is 0.349 rad → residual ≈ +0.151 rad
    assert (meas.summary_residual > 0).all(), f"Expected positive residual, got {meas.summary_residual}"
    assert (meas.summary_value > target_rad - 0.01).all()


# ---- Test 5: Residual sign for angle below target ----

def test_below_target_residual_sign(detector):
    """When value < target, residual should be negative."""
    target_deg = 20.0
    target_rad = target_deg * math.pi / 180.0

    spec = LossSpec(
        name="test_below",
        angle_fn=lambda q, **kw: torch.ones(2, 60) * 0.1,  # 0.1 rad ≈ 5.7 deg
        target_deg=target_deg,
        direction="greater_than",
        tolerance_deg=2.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )
    q = torch.randn(2, 60, 22, 3)
    meas = measure_spec(spec, q, detector, t=0, T=50)

    # 0.1 rad < 0.349 rad → residual should be negative
    assert (meas.summary_residual < 0).all(), f"Expected negative residual, got {meas.summary_residual}"


# ---- Test 6: degree ↔ radian consistency ----

def test_degree_radian_consistency(detector):
    """If spec has unit='deg', target and tolerance should be internally
    converted to radians before residual computation."""
    q = torch.randn(2, 60, 22, 3)

    spec = LossSpec(
        name="test_consistency",
        angle_fn=lambda q, **kw: torch.ones(2, 60) * (30.0 * math.pi / 180.0),  # 30 deg in rad
        target_deg=20.0,
        direction="equal",
        tolerance_deg=2.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )
    meas = measure_spec(spec, q, detector, t=0, T=50)

    # value = 30 deg = 0.5236 rad, target = 20 deg = 0.3491 rad
    # residual ≈ 0.1745 rad ≈ 10 deg
    expected_residual_rad = (30.0 - 20.0) * math.pi / 180.0
    assert torch.allclose(meas.summary_residual,
                          torch.tensor(expected_residual_rad).expand(2),
                          atol=1e-4)


# ---- Test 7: Manual angle increase → residual monotonic change ----

def test_residual_monotonic(detector):
    """As angle increases, residual should increase monotonically."""
    spec = LossSpec(
        name="test_mono",
        angle_fn=lambda q, offset=0.0, **kw: torch.ones(2, 60) * offset,
        angle_fn_kwargs={"offset": 0.0},
        target_deg=20.0,
        direction="equal",
        tolerance_deg=2.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )
    q = torch.randn(2, 60, 22, 3)

    residuals = []
    for offset_deg in [10.0, 15.0, 20.0, 25.0, 30.0]:
        spec.angle_fn_kwargs = {"offset": offset_deg * math.pi / 180.0}
        meas = measure_spec(spec, q, detector, t=0, T=50)
        residuals.append(meas.summary_residual.mean().item())

    for i in range(len(residuals) - 1):
        assert residuals[i] < residuals[i + 1], \
            f"Residuals should be monotonic: {residuals[i]} < {residuals[i+1]}"


# ---- Test 8: Frozen mask ----

def test_frozen_mask(detector):
    """When frozen masks are passed, the output masks must match exactly."""
    q = torch.randn(2, 60, 22, 3)

    spec = LossSpec(
        name="test_frozen",
        angle_fn=lambda q, **kw: q[:, :, 0, 0],
        target_deg=0.0,
        direction="equal",
        tolerance_deg=1.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )

    # First measurement — get the masks
    meas1 = measure_spec(spec, q, detector, t=0, T=50)

    # Second measurement — freeze masks
    meas2 = measure_spec(
        spec, q, detector, t=0, T=50,
        frozen_active_mask=meas1.active_mask,
        frozen_valid_mask=meas1.valid_mask,
    )

    # Masks must match
    assert torch.allclose(meas1.active_mask, meas2.active_mask)
    assert torch.allclose(meas1.valid_mask, meas2.valid_mask)
    assert torch.allclose(meas1.effective_mask, meas2.effective_mask)


# ---- Test 9: Batch independence ----

def test_batch_independence(detector):
    """Samples in a batch must have independent measurements."""
    N = 60
    # Create two very different samples
    q0 = torch.randn(N, 22, 3)
    q1 = torch.randn(N, 22, 3) + 5.0  # Large offset
    q = torch.stack([q0, q1], dim=0)  # (2, N, J, 3)

    def angle_at_idx(q, idx=0, **kw):
        return q[:, :, idx, 0]

    spec = LossSpec(
        name="test_batch_indep",
        angle_fn=angle_at_idx,
        target_deg=0.0,
        direction="equal",
        tolerance_deg=1.0,
        phase="always",
        schedule="always",
        base_weight=1.0,
        is_primary=True,
        control_type="equality",
    )

    meas = measure_spec(spec, q, detector, t=0, T=50)

    # Summary values should be different
    assert not torch.allclose(meas.summary_value[0:1], meas.summary_value[1:2]), \
        "Batch samples should have different summary values"
    assert not torch.allclose(meas.summary_residual[0:1], meas.summary_residual[1:2]), \
        "Batch samples should have different residuals"


# ---- Test 10: find_primary_spec errors ----

def test_find_primary_spec_errors():
    """find_primary_spec should raise ValueError for zero or multiple primary specs."""
    spec_a = LossSpec(name="a", angle_fn=lambda q: q[:, :, 0, 0],
                      is_primary=False, control_type="equality")
    spec_b = LossSpec(name="b", angle_fn=lambda q: q[:, :, 0, 0],
                      is_primary=False, control_type="equality")

    # Zero primary specs
    with pytest.raises(ValueError, match="exactly one spec"):
        find_primary_spec([spec_a, spec_b])

    # Multiple primary specs
    spec_a.is_primary = True
    spec_b.is_primary = True
    with pytest.raises(ValueError, match="exactly one spec"):
        find_primary_spec([spec_a, spec_b])

    # Exactly one
    spec_b.is_primary = False
    found = find_primary_spec([spec_a, spec_b])
    assert found is spec_a


# ---- Test 11: measure_primary_constraint via PostureGuidance ----

def test_measure_primary_via_guidance(detector):
    """PostureGuidance.measure_primary_constraint should work end-to-end."""
    q = torch.randn(2, 60, 22, 3)
    guidance = PostureGuidance(["anterior_pelvic_tilt"], phase_detector=detector)
    meas = guidance.measure_primary_constraint(q, t=25, T=50)

    assert isinstance(meas, ConstraintMeasurement)
    assert meas.spec_name == "骨盆前倾"
    assert meas.constraint_type == "equality"
    assert meas.unit == "deg"
    assert meas.summary_residual.shape == (2,)
    assert meas.summary_value.shape == (2,)
    assert meas.merit.shape == (2,)


# ---- Main ----

if __name__ == "__main__":
    # Run tests and save output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
