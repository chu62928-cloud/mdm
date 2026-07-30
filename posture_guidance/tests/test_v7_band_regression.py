"""Permanent regression tests for the band-order fix (Branch C, commit f414856).

These tests lock the invariant: `update_band_state()` must run BEFORE
`controller.propose()` inside `apply_v7_step`. If someone reverts the
band-order fix, these tests fail.

Historical context:
  V7.0 had a bug where band state was updated at Step 6 (after proposal),
  using stale in_band from previous timestep. When posterior noise pushed
  residual out of band, the stale in_band=True caused proposal to be
  permanently skipped. Median 19/25 timesteps wasted per trajectory.

The fix moves update_band_state to Step 1.5 (right after measurement,
before proposal). See docs/v7_algorithm_spec.md Section 10 and Phase 0
diagnostics report.
"""

import os
import sys
import pytest
import torch

from posture_guidance.auto_dps_controller import (
    AutoDPSConfig,
    AutoDPSState,
    TrustRegionAutoDPSController,
)


@pytest.fixture
def ctrl():
    return TrustRegionAutoDPSController()


# ---- Test 1: Enter band on |r| <= tol ----

def test_enter_band(ctrl):
    """All samples with |r| <= tolerance must transition to in_band=True."""
    B, tol = 6, 0.035
    state = ctrl.reset(B, "cpu", torch.float32)
    r = torch.tensor([0.02, 0.01, 0.03, 0.005, 0.015, 0.034])
    state = ctrl.update_band_state(state, r, torch.full((B,), tol))
    assert state.in_band.all(), f"All samples should enter band, got {state.in_band}"


# ---- Test 2: Hysteresis exit on |r| > 1.5*tol ----

def test_hysteresis_exit(ctrl):
    """|r| > 1.5*tol exits; |r| in [tol, 1.5*tol] stays in band."""
    B, tol = 6, 0.035
    state = ctrl.reset(B, "cpu", torch.float32)
    # First enter band
    state = ctrl.update_band_state(state,
        torch.tensor([0.02, 0.01, 0.03, 0.005, 0.015, 0.034]),
        torch.full((B,), tol))
    assert state.in_band.all()

    # Now test exits: 0.06 and 0.07 > 1.5*0.035=0.0525
    state = ctrl.update_band_state(state,
        torch.tensor([0.06, 0.01, 0.04, 0.07, 0.02, 0.06]),
        torch.full((B,), tol))
    # Expected: 0 out, 1 in, 2 in (hysteresis), 3 out, 4 in, 5 out
    expected = [False, True, True, False, True, False]
    assert state.in_band.tolist() == expected, \
        f"Hysteresis exit wrong: expected {expected}, got {state.in_band.tolist()}"


# ---- Test 3: Re-entry after exit ----

def test_reentry_after_exit(ctrl):
    """After exiting band, sample re-enters when |r| <= tol again."""
    B, tol = 4, 0.035
    state = ctrl.reset(B, "cpu", torch.float32)
    # Enter
    state = ctrl.update_band_state(state,
        torch.tensor([0.02, 0.01, 0.03, 0.02]),
        torch.full((B,), tol))
    assert state.in_band.all()
    # Exit
    state = ctrl.update_band_state(state,
        torch.tensor([0.06, 0.01, 0.08, 0.07]),
        torch.full((B,), tol))
    # Sample 0, 2, 3 out; 1 in
    # Re-enter for samples 0 and 3
    state = ctrl.update_band_state(state,
        torch.tensor([0.01, 0.01, 0.08, 0.01]),
        torch.full((B,), tol))
    assert state.in_band[0], "Sample 0 should re-enter band"
    assert state.in_band[1], "Sample 1 should stay in band"
    assert not state.in_band[2], "Sample 2 should stay out (0.08 > 1.5*tol)"
    assert state.in_band[3], "Sample 3 should re-enter"


# ---- Test 4: Proposal is valid only for out-of-band samples ----

def test_proposal_gated_by_band(ctrl):
    """After update_band_state, controller.propose must:
    - Reject in_band samples (proposal.valid[i] == False)
    - Accept out-of-band samples (proposal.valid[i] == True)
    """
    B, tol = 6, 0.035
    tol_t = torch.full((B,), tol)
    state = ctrl.reset(B, "cpu", torch.float32)
    # Mixed band states
    state = ctrl.update_band_state(state,
        torch.tensor([0.02, 0.01, 0.03, 0.005, 0.015, 0.034]),
        tol_t)  # all in
    state = ctrl.update_band_state(state,
        torch.tensor([0.06, 0.01, 0.04, 0.07, 0.02, 0.06]),
        tol_t)
    # [F, T, T, F, T, F]

    # Now propose with current (fresh) band state
    r = torch.tensor([0.06, 0.01, 0.04, 0.07, 0.02, 0.06])
    g = torch.randn(B, 263, 1, 60) * 0.001
    nl = torch.full((B,), 0.3)
    valid = torch.ones(B, dtype=torch.bool)

    prop = ctrl.propose(r, g, nl, tol_t, valid, state)

    for i in range(B):
        if state.in_band[i]:
            assert not prop.valid[i], f"Sample {i} in-band should get invalid proposal"
        else:
            assert prop.valid[i], f"Sample {i} out-of-band should get valid proposal"


# ---- Test 5: Old permanent-skip bug does NOT recur ----

def test_no_permanent_skip_after_exit(ctrl):
    """The V7.0 bug: enter band → noise pushes out → old in_band=True
    was checked in propose(), permanently blocking new proposals.

    Fix: update_band_state before propose() ensures fresh in_band.
    """
    tol = 0.035
    tol_t = torch.full((2,), tol)
    state = ctrl.reset(2, "cpu", torch.float32)

    # Timestep 1: enter band (|r| very small)
    state = ctrl.update_band_state(state,
        torch.tensor([-0.01, -0.005]), tol_t)
    assert state.in_band.all(), "Both should enter band"

    # Timestep 2: posterior noise pushes sample 0 out (|r| > 1.5*tol)
    state = ctrl.update_band_state(state,
        torch.tensor([-0.06, -0.002]), tol_t)
    assert not state.in_band[0], "Sample 0 should exit band"
    assert state.in_band[1], "Sample 1 stays in band"

    # Timestep 3: propose — sample 0 MUST get valid proposal
    r = torch.tensor([-0.06, -0.002])
    g = torch.randn(2, 263, 1, 60) * 0.001
    prop = ctrl.propose(r, g, torch.full((2,), 0.3), tol_t,
                        torch.ones(2, dtype=torch.bool), state)

    assert prop.valid[0], "OLD BUG: sample 0 got no proposal after band exit"
    assert not prop.valid[1], "Sample 1 in-band should get no proposal"


# ---- Test 6: Batch independence of band state ----

def test_batch_independent_band(ctrl):
    """Different samples in one batch must have independent band states."""
    tol = 0.035
    state = ctrl.reset(4, "cpu", torch.float32)
    r_mixed = torch.tensor([0.01, 0.06, 0.02, 0.07])
    state = ctrl.update_band_state(state, r_mixed, torch.full((4,), tol))

    # Sample 0, 2: |r| <= tol → in band
    # Sample 1, 3: |r| > tol (fresh entry, but not > 1.5*tol yet) — actually
    #   on first update, if |r| > tol they stay out.
    expected_in_band = [True, False, True, False]
    assert state.in_band.tolist() == expected_in_band, \
        f"Expected {expected_in_band}, got {state.in_band.tolist()}"

    # Independence: modifying one sample's state must not affect others
    state2 = ctrl.update_band_state(state,
        torch.tensor([0.08, 0.01, 0.02, 0.01]),  # sample 0 exit, 1&3 enter, 2 stay
        torch.full((4,), tol))
    assert not state2.in_band[0]
    assert state2.in_band[1]
    assert state2.in_band[2]
    assert state2.in_band[3]


# ---- Test 7: Source code invariant — update_band_state BEFORE propose ----

def test_source_order_invariant():
    """Structural check: in apply_v7_step, update_band_state() text position
    must precede controller.propose() text position.

    This test catches accidental reverts of the Branch C fix (commit f414856).
    """
    import posture_guidance.v7_auto_dps as v7_module

    src_path = v7_module.__file__
    with open(src_path) as f:
        content = f.read()

    # Find first occurrence of each inside apply_v7_step function
    apply_start = content.find("def apply_v7_step(")
    assert apply_start > 0, "apply_v7_step function not found"

    # Look for update_band_state and controller.propose after apply_v7_step start
    band_pos = content.find("update_band_state", apply_start)
    propose_pos = content.find("controller.propose(", apply_start)

    assert band_pos > 0, "update_band_state call not found in apply_v7_step"
    assert propose_pos > 0, "controller.propose call not found in apply_v7_step"

    assert band_pos < propose_pos, (
        f"REGRESSION: update_band_state at byte {band_pos} must appear BEFORE "
        f"controller.propose at byte {propose_pos}. This is the Branch C fix "
        f"(commit f414856) that prevents permanent proposal skipping after "
        f"band re-exit. If this test fails, the V7.0 bug has been reintroduced."
    )


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
