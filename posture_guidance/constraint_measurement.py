"""
Unified constraint measurement for V7 Auto-DPS.

Separates physical value/target/residual/mask computation from
the weighted hinge/Huber loss code. V7 uses this to get raw
per-sample residuals for Gauss-Newton step computation.

Key design rules (from revised execution plan Section 2):
- Per-sample reduction (no cross-batch pooling)
- Frozen mask support for trial evaluation
- Primary constraint must be explicitly marked (is_primary=True)
- APT uses control_type="equality" (not greater_than hinge)
"""

import math
from dataclasses import dataclass
import torch

from .registry import LossSpec
from .phase_detector import PhaseDetector, PHASE_FUNCTIONS


@dataclass
class ConstraintMeasurement:
    """Per-sample constraint measurement with frozen-mask support."""

    # Per-frame
    value_per_frame: torch.Tensor       # (B, N) raw angle/geometry per frame
    residual_per_frame: torch.Tensor    # (B, N) signed (value - target)

    # Per-sample summary (mean over effective frames)
    summary_value: torch.Tensor         # (B,)
    summary_residual: torch.Tensor      # (B,)
    merit: torch.Tensor                 # (B,) = 0.5 * r^2

    target: torch.Tensor                # (B,) target in physical units
    tolerance: torch.Tensor             # (B,) tolerance in physical units

    # Masks
    active_mask: torch.Tensor           # (B, N) phase mask from detector
    valid_mask: torch.Tensor            # (B, N) all-ones (reserved for future)
    effective_mask: torch.Tensor        # (B, N) = active_mask * valid_mask

    # Per-sample validity
    active_fraction: torch.Tensor       # (B,)
    valid_fraction: torch.Tensor        # (B,)
    effective_count: torch.Tensor       # (B,) number of effective frames

    spec_name: str
    constraint_type: str                # "equality" | "lower_bound" | "upper_bound"
    unit: str                           # "deg" | "meter"


def masked_batch_mean(
    value: torch.Tensor,       # (B, N) or (B, N, ...)
    mask: torch.Tensor,        # (B, N)
) -> torch.Tensor:
    """Compute per-sample mean over masked frames. No cross-batch pooling.

    Args:
        value: (B, N) or (B, N, ...) — extra dims averaged too
        mask: (B, N)
    Returns:
        per_sample_mean: (B,)
    """
    # Ensure mask broadcasts with value
    if mask.dim() < value.dim():
        mask_expanded = mask
        while mask_expanded.dim() < value.dim():
            mask_expanded = mask_expanded.unsqueeze(-1)
    else:
        mask_expanded = mask

    masked_sum = (value * mask_expanded).flatten(start_dim=1).sum(dim=1)
    mask_sum = mask.flatten(start_dim=1).sum(dim=1).clamp(min=1.0)
    return masked_sum / mask_sum


def spec_target_tensor(
    spec: LossSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a spec's target to a (B,) tensor placeholder — actual B is set later."""
    if spec.unit == "deg":
        target_val = spec.target_deg * math.pi / 180.0
    else:
        target_val = spec.target_deg
    return torch.tensor(target_val, device=device, dtype=dtype)


def spec_tolerance_tensor(
    spec: LossSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a spec's tolerance to a (B,) tensor placeholder."""
    if spec.unit == "deg":
        tol_val = spec.tolerance_deg * math.pi / 180.0
    else:
        tol_val = spec.tolerance_deg
    return torch.tensor(tol_val, device=device, dtype=dtype)


def measure_spec(
    spec: LossSpec,
    q: torch.Tensor,                     # (B, N, J, 3) or (N, J, 3)
    detector: PhaseDetector,
    t: int,
    T: int,
    frozen_active_mask: torch.Tensor = None,
    frozen_valid_mask: torch.Tensor = None,
) -> ConstraintMeasurement:
    """Measure one LossSpec on a batch of motions.

    Args:
        spec: The LossSpec to measure.
        q: Joint coordinates.
        detector: PhaseDetector instance.
        t, T: Current/total diffusion timesteps (for phase functions).
        frozen_active_mask: If provided, reuse this mask instead of computing fresh.
        frozen_valid_mask: If provided, reuse this mask instead of computing fresh.

    Returns:
        ConstraintMeasurement with per-sample summary statistics.
    """
    # Ensure batch dim
    squeeze_batch = q.dim() == 3
    if squeeze_batch:
        q = q.unsqueeze(0)

    B, N = q.shape[0], q.shape[1]

    # 1. Compute angle/geometry per frame
    value_per_frame = spec.angle_fn(q, **spec.angle_fn_kwargs)  # (B, N) typically

    # 2. Phase mask (active_mask)
    if frozen_active_mask is not None:
        active_mask = frozen_active_mask
    else:
        active_mask = PHASE_FUNCTIONS[spec.phase](detector, q)  # (B, N)
        # Ensure float type consistent with value
        if active_mask.dtype != value_per_frame.dtype:
            active_mask = active_mask.to(value_per_frame.dtype)

    # Ensure shape compatibility
    if active_mask.dim() < value_per_frame.dim():
        while active_mask.dim() < value_per_frame.dim():
            active_mask = active_mask.unsqueeze(-1)
    elif active_mask.dim() > value_per_frame.dim():
        active_mask = active_mask.squeeze(-1)

    # 3. Valid mask (reserved for future use, e.g., occlusion)
    if frozen_valid_mask is not None:
        valid_mask = frozen_valid_mask
    else:
        valid_mask = torch.ones(B, N, device=q.device, dtype=value_per_frame.dtype)
        if valid_mask.dim() < value_per_frame.dim():
            while valid_mask.dim() < value_per_frame.dim():
                valid_mask = valid_mask.unsqueeze(-1)
        valid_mask = valid_mask.expand_as(value_per_frame)

    # 4. Effective mask
    effective_mask = active_mask * valid_mask

    # 5. Per-sample summary (mean over effective frames)
    summary_value = masked_batch_mean(value_per_frame, effective_mask)

    # 6. Target and tolerance
    target_scalar = spec_target_tensor(spec, q.device, q.dtype)
    tolerance_scalar = spec_tolerance_tensor(spec, q.device, q.dtype)
    target = target_scalar.expand(B)
    tolerance = tolerance_scalar.expand(B)

    # 7. Residual
    summary_residual = summary_value - target

    # 8. Merit: 0.5 * r^2
    merit = 0.5 * summary_residual ** 2

    # 9. Per-sample fractions
    frame_count = N
    effective_count = effective_mask.flatten(start_dim=1).sum(dim=1)  # (B,)
    active_fraction = active_mask.flatten(start_dim=1).sum(dim=1) / frame_count
    valid_fraction = valid_mask.flatten(start_dim=1).sum(dim=1) / frame_count

    if squeeze_batch:
        value_per_frame = value_per_frame.squeeze(0)
        effective_mask = effective_mask.squeeze(0)
        if active_mask.dim() >= 2:
            active_mask = active_mask.squeeze(0)
        if valid_mask.dim() >= 2:
            valid_mask = valid_mask.squeeze(0)

    return ConstraintMeasurement(
        value_per_frame=value_per_frame,
        residual_per_frame=value_per_frame - target.view(-1, *([1] * (value_per_frame.dim() - 1))),
        summary_value=summary_value,
        summary_residual=summary_residual,
        merit=merit,
        target=target,
        tolerance=tolerance,
        active_mask=active_mask,
        valid_mask=valid_mask,
        effective_mask=effective_mask,
        active_fraction=active_fraction,
        valid_fraction=valid_fraction,
        effective_count=effective_count,
        spec_name=spec.name,
        constraint_type=spec.control_type,
        unit=spec.unit,
    )


def find_primary_spec(specs: list) -> LossSpec:
    """Find the single primary spec. Errors if zero or multiple."""
    primary = [s for s in specs if s.is_primary]
    if len(primary) == 0:
        raise ValueError(
            "V7 requires exactly one spec with is_primary=True. "
            f"Found 0 among {[s.name for s in specs]}."
        )
    if len(primary) > 1:
        raise ValueError(
            f"V7 requires exactly one spec with is_primary=True. "
            f"Found {len(primary)}: {[s.name for s in primary]}."
        )
    return primary[0]


def measure_primary_constraint(
    guidance,          # PostureGuidance instance
    q: torch.Tensor,
    t: int,
    T: int,
    frozen_active_mask: torch.Tensor = None,
    frozen_valid_mask: torch.Tensor = None,
) -> ConstraintMeasurement:
    """Measure the primary constraint from a PostureGuidance instance.

    Finds the spec with is_primary=True and measures it.

    Args:
        guidance: PostureGuidance with at least one is_primary spec.
        q: Joint coordinates.
        t, T: Current/total diffusion timesteps.
        frozen_active_mask: Reuse mask for trial evaluation.
        frozen_valid_mask: Reuse mask for trial evaluation.

    Returns:
        ConstraintMeasurement for the primary spec.
    """
    primary_spec = find_primary_spec(guidance.specs)
    return measure_spec(
        spec=primary_spec,
        q=q,
        detector=guidance.detector,
        t=t,
        T=T,
        frozen_active_mask=frozen_active_mask,
        frozen_valid_mask=frozen_valid_mask,
    )
