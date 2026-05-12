"""
posture_guidance/mdm_integration_v2.py

Updated apply_posture_guidance that supports all 5 variants.

REPLACES the existing apply_posture_guidance function in
posture_guidance/mdm_integration.py.

Key API change: now accepts `variant` and `variant_kwargs` parameters.
Backward-compatible: if variant=None, uses original V1 behavior.
"""

import torch
from typing import Optional

from posture_guidance.guidance_variants import get_guidance_fn


def apply_posture_guidance(
    mu_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    T: int,
    fk_fn,
    guidance_loss_fn,
    *,
    variant: str = "v1_mu_sgd",
    variant_kwargs: Optional[dict] = None,
    model=None,
    model_kwargs: Optional[dict] = None,
    sigma_t: Optional[torch.Tensor] = None,
    posterior_mean_fn=None,
    diagnostic: bool = False,
) -> torch.Tensor:
    """
    Apply IK-style posture guidance using one of 5 variants.

    Args:
        mu_t : the posterior mean from MDM, shape (B, 263, 1, T_frames)
        x_t  : the noisy sample at step t (needed for V2/V5)
        t    : current diffusion step (int)
        T    : total diffusion steps (50 for DiP)
        fk_fn         : forward kinematics, mu -> joint xyz
        guidance_loss_fn : (q, t, T) -> scalar loss
        variant       : one of GUIDANCE_VARIANTS keys
        variant_kwargs: kwargs to forward to the variant function
        model         : MDM denoiser (needed for V2/V3/V5)
        model_kwargs  : kwargs for MDM forward (text emb etc.)
        sigma_t       : noise std at step t (needed for V2/V5)
        posterior_mean_fn : (x0_hat, x_t, t) -> mu_t (needed for V3)
        diagnostic    : if True, print |Delta mu| per step
    """
    variant_kwargs = variant_kwargs or {}
    model_kwargs = model_kwargs or {}

    fn = get_guidance_fn(variant)

    full_kwargs = dict(variant_kwargs)
    if variant in ("v2_dps", "v3_x0_direct", "v5_lgd"):
        full_kwargs["model"] = model
        full_kwargs["model_kwargs"] = model_kwargs
    if variant in ("v2_dps", "v5_lgd"):
        full_kwargs["sigma_t"] = sigma_t
    if variant == "v3_x0_direct":
        full_kwargs["posterior_mean_fn"] = posterior_mean_fn

    mu_t_new = fn(
        mu_t=mu_t, x_t=x_t, t=t, T=T,
        fk_fn=fk_fn, guidance_loss_fn=guidance_loss_fn,
        **full_kwargs,
    )

    if diagnostic:
        delta = (mu_t_new - mu_t).norm().item()
        print(f"[guidance variant={variant} t={t}] |Δμ|={delta:.4f}")

    return mu_t_new


# ==============================================================
#  Helper: build posterior_mean_fn that closes over diffusion coefs
# ==============================================================
def make_posterior_mean_fn(diffusion):
    """
    Build a posterior_mean_fn(x0_hat, x_t, t) -> mu_t closure
    that uses the diffusion's stored coefficients.

    Used by V3 (x0_direct).
    """
    def posterior_mean_fn(x0_hat, x_t, t):
        coef1 = diffusion.posterior_mean_coef1[t]
        coef2 = diffusion.posterior_mean_coef2[t]
        if isinstance(coef1, float):
            coef1 = torch.tensor(coef1, device=x_t.device)
            coef2 = torch.tensor(coef2, device=x_t.device)
        return coef1 * x0_hat + coef2 * x_t
    return posterior_mean_fn


# ==============================================================
#  Helper: extract sigma_t from diffusion at step t
# ==============================================================
def get_sigma_t(diffusion, t: int, x_ref: torch.Tensor) -> torch.Tensor:
    """
    Extract noise std for step t from diffusion.posterior_variance.
    Returns a tensor on the same device as x_ref.
    """
    var = diffusion.posterior_variance[t]
    sigma = float(var) ** 0.5
    return torch.tensor(sigma, device=x_ref.device, dtype=x_ref.dtype)