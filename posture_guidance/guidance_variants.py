"""
posture_guidance/guidance_variants.py

Five guidance injection strategies for IK-style posture guidance on MDM/DiP.

Each variant takes the same inputs (model, x_t, mu_t, t, fk_fn, guidance_loss)
and returns the modified mu_t to use for sampling x_{t-1}.

Variants:
    V1_mu_sgd       : SGD on detached mu_t                    (baseline / current)
    V2_dps          : DPS-style, gradient through MDM via x0   (theoretically correct)
    V3_x0_direct    : direct edit of predicted x0              (tests GMD's warning)
    V4_omni         : OmniControl-style iterative perturbation (dynamic Ke<<Kl)
    V5_lgd          : LGD-style with Monte Carlo smoothing     (variance reduction)
"""

import torch
import torch.nn.functional as F
from torch.optim import SGD
from typing import Callable, Optional


# ==============================================================
#  V1 — SGD on mu_t (current method, baseline for comparison)
# ==============================================================
def guidance_v1_mu_sgd(
    mu_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    T: int,
    fk_fn: Callable,
    guidance_loss_fn: Callable,
    *,
    n_inner_steps: int = 15,
    lr: float = 0.5,
    schedule: str = "final",
    base_weight: float = 30.0,
    **kwargs,
) -> torch.Tensor:
    """
    V1: Current method. Detach mu_t, run SGD on FK loss.
    Gradient does NOT pass through MDM network.
    """
    if not _schedule_active(schedule, t, T):
        return mu_t

    mu_var = mu_t.detach().clone().contiguous().requires_grad_(True)
    optimizer = SGD([mu_var], lr=lr)

    for _ in range(n_inner_steps):
        optimizer.zero_grad()
        q = fk_fn(mu_var)
        loss = base_weight * guidance_loss_fn(q, t, T)
        loss.backward()
        optimizer.step()

    return mu_var.detach()


# ==============================================================
#  V2 — DPS-style: gradient flows through MDM via x0_hat
# ==============================================================
def guidance_v2_dps(
    mu_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    T: int,
    fk_fn: Callable,
    guidance_loss_fn: Callable,
    *,
    model: Callable,
    model_kwargs: dict,
    sigma_t: torch.Tensor,
    posterior_mean_fn: Callable,
    s: float = 1.0,
    schedule: str = "always",
    base_weight: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    V2: DPS (Chung et al., NeurIPS 2022).

    Computes ∇_{x_t} L(FK(x̂_0(x_t))) by backpropagating through MDM.
    Per Dhariwal: mu_t ← mu_t - s · sigma_t^2 · ∇_{x_t} L

    Cost: ONE full MDM forward+backward per diffusion step.
    Benefit: gradient covers ALL 263 dimensions (no dilution).
    """
    if not _schedule_active(schedule, t, T):
        return mu_t

    x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)

    x0_hat = model(x_t_var, _build_t_tensor(t, x_t_var), **model_kwargs)

    q = fk_fn(x0_hat)
    loss = base_weight * guidance_loss_fn(q, t, T)
    grad = torch.autograd.grad(loss, x_t_var, retain_graph=False)[0]

    mu_t_new = mu_t.detach() - s * (sigma_t ** 2) * grad
    return mu_t_new


# ==============================================================
#  V3 — direct edit of x0_hat, then recompose mu_t
# ==============================================================
def guidance_v3_x0_direct(
    mu_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    T: int,
    fk_fn: Callable,
    guidance_loss_fn: Callable,
    *,
    model: Callable,
    model_kwargs: dict,
    posterior_mean_fn: Callable,
    n_inner_steps: int = 5,
    lr: float = 0.05,
    schedule: str = "always",
    base_weight: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    V3: Direct gradient descent on x̂_0, then recompose mu_t.

    This tests GMD's Appendix A warning that x_0 models can "undo"
    guidance late in sampling. If it works on 50-step DiP, it would
    actually contradict GMD's claim for this regime.

    Procedure:
        x̂_0 ← MDM(x_t)
        for i in range(K):
            x̂_0 ← x̂_0 - lr · ∇_{x̂_0} L(FK(x̂_0))
        mu_t ← posterior_mean(x̂_0_modified, x_t)
    """
    if not _schedule_active(schedule, t, T):
        return mu_t

    with torch.no_grad():
        x0_hat = model(x_t, _build_t_tensor(t, x_t), **model_kwargs)

    x0_var = x0_hat.detach().clone().contiguous().requires_grad_(True)
    optimizer = SGD([x0_var], lr=lr)

    for _ in range(n_inner_steps):
        optimizer.zero_grad()
        q = fk_fn(x0_var)
        loss = base_weight * guidance_loss_fn(q, t, T)
        loss.backward()
        optimizer.step()

    mu_t_new = posterior_mean_fn(x0_var.detach(), x_t, t)
    return mu_t_new


# ==============================================================
#  V4 — OmniControl-style: dynamic iterative perturbation of mu_t
# ==============================================================
def guidance_v4_omni(
    mu_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    T: int,
    fk_fn: Callable,
    guidance_loss_fn: Callable,
    *,
    K_early: int = 1,
    K_late: int = 10,
    T_split: int = None,
    lr: float = 0.5,
    base_weight: float = 30.0,
    schedule: str = "always",
    **kwargs,
) -> torch.Tensor:
    """
    V4: OmniControl (Xie et al., ICLR 2024) style.

    Use FEW iterations early (large t) and MANY iterations late (small t).
    This matches OmniControl's empirical finding that K_e << K_l works best.

    For 50-step DiP: T_split=15 means t>=15 gets K_early=1, t<15 gets K_late=10.
    """
    if not _schedule_active(schedule, t, T):
        return mu_t

    if T_split is None:
        T_split = T // 3   # default: top third gets K_late

    n_steps = K_late if t < T_split else K_early
    if n_steps == 0:
        return mu_t

    mu_var = mu_t.detach().clone().contiguous().requires_grad_(True)
    optimizer = SGD([mu_var], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        q = fk_fn(mu_var)
        loss = base_weight * guidance_loss_fn(q, t, T)
        loss.backward()
        optimizer.step()

    return mu_var.detach()


# ==============================================================
#  V5 — LGD-style: gradient on x_0_hat with Monte Carlo smoothing
# ==============================================================
def guidance_v5_lgd(
    mu_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    T: int,
    fk_fn: Callable,
    guidance_loss_fn: Callable,
    *,
    model: Callable,
    model_kwargs: dict,
    sigma_t: torch.Tensor,
    n_mc: int = 4,
    s: float = 1.0,
    schedule: str = "always",
    base_weight: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    V5: LGD-style (Song et al., ICML 2023) with MC smoothing.

    Sample n_mc noisy versions of x̂_0, average their gradients.
    This reduces variance and tends to be more stable than pure DPS.

        x̂_0_i = MDM(x_t) + sigma_t * eps_i,  eps_i ~ N(0, I)
        grad = (1/n_mc) * sum_i ∇_{x_t} L(FK(x̂_0_i))
    """
    if not _schedule_active(schedule, t, T):
        return mu_t

    grads = []
    for _ in range(n_mc):
        x_t_var = x_t.detach().clone().contiguous().requires_grad_(True)
        x0_hat = model(x_t_var, _build_t_tensor(t, x_t_var), **model_kwargs)

        eps = torch.randn_like(x0_hat) * sigma_t * 0.1
        x0_perturbed = x0_hat + eps

        q = fk_fn(x0_perturbed)
        loss = base_weight * guidance_loss_fn(q, t, T)
        g = torch.autograd.grad(loss, x_t_var, retain_graph=False)[0]
        grads.append(g)

    grad_mean = torch.stack(grads).mean(dim=0)
    mu_t_new = mu_t.detach() - s * (sigma_t ** 2) * grad_mean
    return mu_t_new


# ==============================================================
#  Helpers
# ==============================================================
def _schedule_active(schedule: str, t: int, T: int) -> bool:
    """Return True if guidance should fire at diffusion step t."""
    if schedule == "always":
        return True
    if schedule == "decay":
        return True
    if schedule == "last_quarter":
        return t < T // 4
    if schedule == "second_half":
        return t < T // 2
    if schedule == "final":
        return t < 5
    return True


def _build_t_tensor(t: int, x_ref: torch.Tensor) -> torch.Tensor:
    B = x_ref.shape[0]
    return torch.full((B,), t, device=x_ref.device, dtype=torch.long)


# ==============================================================
#  Dispatch table
# ==============================================================
GUIDANCE_VARIANTS = {
    "v1_mu_sgd":   guidance_v1_mu_sgd,
    "v2_dps":      guidance_v2_dps,
    "v3_x0_direct": guidance_v3_x0_direct,
    "v4_omni":     guidance_v4_omni,
    "v5_lgd":      guidance_v5_lgd,
}


def get_guidance_fn(variant: str) -> Callable:
    """Look up a guidance variant by name."""
    if variant not in GUIDANCE_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(GUIDANCE_VARIANTS.keys())}"
        )
    return GUIDANCE_VARIANTS[variant]