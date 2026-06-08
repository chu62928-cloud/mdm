"""
muscle_guidance_mdm — MDM side integration of motion2muscle proxy + posture loss.

Named muscle_guidance_mdm (not muscle_guidance) to avoid collision with
motion2muscle/muscle_guidance.py. The internal module is imported via
sys.path manipulation in build_muscle_guidance.

Primary API:
    build_muscle_guidance(...) -> motion2muscle.MuscleGuidance (with build_reference)
    load_frozen_proxy(...)     -> frozen nn.Module
    muscle_guidance_loss(...)  -> differentiable muscle loss (default: dense guidance)
"""
import os as _os, sys as _sys
# Ensure motion2muscle/ is importable before loading dense_loss (which imports posture_loss_torch)
_M2M = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "motion2muscle")
if _M2M not in _sys.path:
    _sys.path.insert(0, _M2M)

from .build import build_muscle_guidance, load_mint_cols
from .loader import load_frozen_proxy
from .dense_loss import dense_posture_guidance_loss


def muscle_guidance_loss(mg, x_btc, kind: str = "guidance", margin: float = 0.3):
    """
    Unified muscle loss entry for CombinedGuidance / other callers.

    Args:
        mg     : motion2muscle.MuscleGuidance instance (with build_reference() called)
        x_btc  : (B,T,263) MDM-normalized motion, requires grad.
        kind   :
            "guidance" — dense target-matching loss (differentiable, grad non-zero at ref);
                          **minimise = toward pathology**. Use during inference.
            "clinical" — original detection loss (>0 = pathology).
                          Use for evaluation / midterm Table 3.
        margin : dense mode pathological target margin.
    Returns:
        scalar tensor.
    """
    if kind == "clinical":
        return mg.loss(x_btc)
    if kind != "guidance":
        raise ValueError(f"kind must be 'guidance' or 'clinical', got {kind!r}")
    assert mg.reference_acts is not None, "call mg.build_reference() / set_reference() first"
    acts = mg._activations(x_btc)
    return dense_posture_guidance_loss(
        acts, mg.group_index, mg.posture_name, mg.reference_acts,
        component_weights=mg.component_weights, margin=margin)


__all__ = [
    "build_muscle_guidance", "load_mint_cols", "load_frozen_proxy",
    "dense_posture_guidance_loss", "muscle_guidance_loss",
]
