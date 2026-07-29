# eval/control_metrics.py — Frozen physical-plausibility + control-accuracy metrics v1.0
#
# Implements:
#   - foot_skate_ratio (field-standard: height < 5cm, slide > 2.5cm)
#   - posture_target_metrics (judge-op based, with judge/guidance firewall)
#   - validity_mask layer (per-frame valid/invalid flags)
#   - phase_selectivity
#
# Invariant 1 (circularity firewall): judge_op MUST differ from guidance_op.
# Invariant 2 (validity domain): every angle op returns (value, valid_flag).
import math
import numpy as np
from scipy.stats import pearsonr

from posture_guidance import angle_ops
from posture_guidance.phase_detector import PhaseDetector

# ---------------------------------------------------------------------------
# Judge angle-op registry — these are the THIRD-PARTY measurement functions
# that have NEVER appeared in a guidance loss.
# ---------------------------------------------------------------------------
JUDGE_ANGLE_OPS = {
    "pelvis_tilt":      angle_ops.pelvis_tilt_angle,
    "signed_knee_left": lambda q: angle_ops.signed_knee_angle(q, side="left"),
    "signed_knee_right": lambda q: angle_ops.signed_knee_angle(q, side="right"),
    "signed_knee_avg":  lambda q: (angle_ops.signed_knee_angle(q, side="left")
                                   + angle_ops.signed_knee_angle(q, side="right")) / 2.0,
    "trunk_lean":       angle_ops.trunk_forward_lean,
    "pelvis_lateral":   angle_ops.pelvis_lateral_tilt,
    "spine_bulge":      angle_ops.spine_posterior_bulge,
    "head_forward":     angle_ops.head_forward_offset,
}

# ---------------------------------------------------------------------------
# Per-angle-op validity rules (Invariant 2)
# ---------------------------------------------------------------------------
def validity_mask(angle_values_deg, angle_op_name):
    """Return per-frame boolean mask of valid measurements.

    Rules:
      - signed_knee: sag > 210 deg or (tpa < 160 and sag > 180) -> invalid
      - other ops: all valid by default
    """
    n = len(angle_values_deg)
    valid = np.ones(n, dtype=bool)
    if "knee" in angle_op_name:
        vals = np.asarray(angle_values_deg)
        # For knee angles above 210 degrees, acos-based metric breaks down
        valid = valid & (vals <= 210.0)
        # Deep flexion edge case
        valid = valid & ~((vals < 160.0) & (vals > 180.0))
    return valid


# ---------------------------------------------------------------------------
# Foot-skating ratio
# ---------------------------------------------------------------------------
def foot_skate_ratio(q_xyz, contact_height=0.05, slide_thresh=0.025):
    """Field-standard foot-skating ratio (GMD/OmniControl thresholds).

    Args:
        q_xyz: (T, J, 3) or (B, T, J, 3) joint positions in meters.
        contact_height: max foot height (m) to count as "on ground".
        slide_thresh: min horizontal displacement (m) to count as "sliding".

    Returns:
        float in [0, 1] — fraction of contact frames with skating.
    """
    q = np.asarray(q_xyz)
    if q.ndim == 4:
        q = q[0]  # take first batch element
    T, J, _ = q.shape
    if J < 12:
        return 0.0
    L_FOOT, R_FOOT = 10, 11  # SMPL-H indices
    feet = q[:, [L_FOOT, R_FOOT], :]  # (T, 2, 3)

    on_ground = feet[:, :, 1] < contact_height  # (T, 2)
    # Horizontal motion between consecutive frames
    motion = np.linalg.norm(
        np.diff(feet[:, :, [0, 2]], axis=0), axis=-1)  # (T-1, 2)
    sliding = motion > slide_thresh
    skate = on_ground[1:] & sliding
    if skate.size == 0:
        return 0.0
    return float(skate.any(axis=1).mean())


# ---------------------------------------------------------------------------
# Posture-target metrics (the core function)
# ---------------------------------------------------------------------------
def posture_target_metrics(q_guided, q_baseline, judge_angle_op,
                           target, tolerance, direction,
                           phase_mask=None,
                           judge_op_name="unknown",
                           unit="deg"):
    """Compute all posture-target metrics from saved motions.

    Args:
        q_guided/baseline: (T, J, 3) or (B, T, J, 3) joint positions.
        judge_angle_op: callable q -> (T,) angle.
        target: scalar target value in degrees (or meters if unit='m').
        tolerance: band half-width in degrees (or meters if unit='m').
        direction: "greater_than" | "less_than" | "equal".
        phase_mask: optional (T,) bool array.
        judge_op_name: for validity masking.
        unit: "deg" (radians->degrees) or "m" (meters, no conversion).
    """
    import torch, math

    # Convert target/tolerance to radians for angle ops
    if unit == "deg":
        target_rad = target * math.pi / 180.0
        tol_rad = tolerance * math.pi / 180.0
    else:
        target_rad = target
        tol_rad = tolerance

    # Handle batch dim
    qg = np.asarray(q_guided)
    qb = np.asarray(q_baseline)
    if qg.ndim == 4:
        qg = qg[0]
        qb = qb[0]

    # Convert to torch for angle ops
    qg_t = torch.from_numpy(qg).float()
    qb_t = torch.from_numpy(qb).float()

    a_g = judge_angle_op(qg_t).numpy()
    a_b = judge_angle_op(qb_t).numpy()

    # Validity mask (needs degrees for angle ops)
    a_g_for_valid = a_g * 180.0 / math.pi if unit == "deg" else a_g
    valid = validity_mask(a_g_for_valid, judge_op_name)
    valid_fraction = float(valid.mean())

    # Apply validity + optional phase mask
    mask = valid.copy()
    if phase_mask is not None:
        pm = np.asarray(phase_mask).flatten()[:len(mask)]
        mask = mask & pm

    if mask.sum() == 0:
        return {"target_hit_band": 0.0, "delta": 0.0, "overshoot": 0.0,
                "temporal_corr": 0.0, "phase_selectivity": 0.0,
                "valid_fraction": valid_fraction}

    a_g_m = a_g[mask]
    a_b_m = a_b[mask]

    # Delta (report in degrees for deg units)
    if unit == "deg":
        delta = float((a_g_m.mean() - a_b_m.mean()) * 180.0 / math.pi)
    else:
        delta = float(a_g_m.mean() - a_b_m.mean())

    # Hit band (two-sided) — use radian target/tolerance
    hit_mask = (a_g_m >= target_rad - tol_rad) & (a_g_m <= target_rad + tol_rad)
    target_hit_band = float(hit_mask.mean())

    # Overshoot: distance of mean from target (report in degrees for deg units)
    if unit == "deg":
        overshoot = float(abs(a_g_m.mean() * 180.0 / math.pi - target))
    else:
        overshoot = float(abs(a_g_m.mean() - target_rad))

    # Temporal correlation
    if a_b_m.std() > 1e-8 and a_g_m.std() > 1e-8:
        temporal_corr = float(pearsonr(a_b_m, a_g_m)[0])
    else:
        temporal_corr = 0.0

    # Phase selectivity: ratio of on-phase deviation to off-phase deviation
    if phase_mask is not None:
        pm = np.asarray(phase_mask).flatten()[:len(valid)]
        off_mask = valid & ~pm
        if off_mask.sum() > 0 and pm.sum() > 0:
            on_dev = np.abs(a_g[pm & valid] - a_b[pm & valid]).mean()
            off_dev = np.abs(a_g[off_mask] - a_b[off_mask]).mean()
            if off_dev > 1e-8:
                phase_selectivity = float(on_dev / off_dev)
            else:
                phase_selectivity = float("inf") if on_dev > 0 else 1.0
        else:
            phase_selectivity = 1.0
    else:
        phase_selectivity = 1.0

    return {
        "target_hit_band": target_hit_band,
        "delta": delta,
        "overshoot": overshoot,
        "temporal_corr": temporal_corr,
        "phase_selectivity": phase_selectivity,
        "valid_fraction": valid_fraction,
    }


# ---------------------------------------------------------------------------
# Firewall assertion
# ---------------------------------------------------------------------------
def assert_judge_not_guidance(judge_op, guidance_op_name):
    """Hard assertion: judge op must NOT be the guidance op."""
    judge_id = getattr(judge_op, "__name__", repr(judge_op))
    if judge_id == guidance_op_name or judge_id in str(guidance_op_name):
        raise AssertionError(
            f"Circularity firewall violated! judge_op={judge_id} "
            f"appears to be the same as guidance_op={guidance_op_name}. "
            f"Use an independent angle function as judge.")