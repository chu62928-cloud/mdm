"""
muscle_guidance.py
==================
Integration layer between MDM and the frozen motion2muscle proxy + posture loss.

This is the ONE object the MDM side talks to. It hides the proxy, the
normalisation handshake, the muscle roll-up and the (differentiable) posture loss.

PIPELINE
--------
    MDM x0_hat (normalised, B,T,263)
        -> de-normalise to raw HumanML3D  (MDM stats)
        -> re-normalise to proxy input    (proxy stats)
        -> frozen proxy  -> muscle activations (B,T,402) in [0,1]
        -> compute_posture_loss_torch(... reference_acts ...)  -> scalar L
        -> autograd grad of L w.r.t. x0_hat
    Guidance MAXIMISES L (gradient ASCENT) to push generation toward the pathology.

TWO-PHASE USAGE (matches the design: "first MDM batch = reference, then guide")
    g = MuscleGuidance(proxy, mint_cols, "anterior_pelvic_tilt", ...stats...)
    # Phase 1: generate a normal batch with vanilla MDM, then:
    g.build_reference(x0_ref_norm)          # freezes reference_acts (no grad)
    # Phase 2: generate again, calling g.guidance_grad(x0_hat) inside the loop.

NOTE the three handshake args that MUST be correct (see HANDOFF.md):
    mdm_mean/std    : the Mean.npy/Std.npy MDM was trained with
    proxy_mean/std  : the Mean.npy/Std.npy the proxy was trained with
    If they are the SAME files, pass same_normalization=True and skip the stats.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from posture_loss import build_reference_from_activations
from posture_loss_torch import build_group_index, compute_posture_loss_torch


class MuscleGuidance:
    def __init__(self,
                 proxy: nn.Module,
                 mint_cols: List[str],
                 posture_name: str,
                 mdm_mean: Optional[torch.Tensor] = None,
                 mdm_std: Optional[torch.Tensor] = None,
                 proxy_mean: Optional[torch.Tensor] = None,
                 proxy_std: Optional[torch.Tensor] = None,
                 same_normalization: bool = False,
                 component_weights: Optional[Dict[str, float]] = None,
                 device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.posture_name = posture_name
        self.mint_cols = mint_cols
        self.component_weights = component_weights
        self.same_norm = same_normalization

        # --- freeze the proxy: params don't train, but the input path must build a graph
        self.proxy = proxy.to(self.device).eval()
        for p in self.proxy.parameters():
            p.requires_grad_(False)

        # normalisation stats (shape broadcastable to (1,1,263))
        def _prep(x):
            return None if x is None else x.to(self.device).view(1, 1, -1)
        self.mdm_mean, self.mdm_std = _prep(mdm_mean), _prep(mdm_std)
        self.proxy_mean, self.proxy_std = _prep(proxy_mean), _prep(proxy_std)

        # precompute group->index map for the chosen posture (done once)
        self.group_index = build_group_index(mint_cols, posture_name, device=self.device)
        self.reference_acts: Optional[Dict[str, float]] = None

    # -- normalisation handshake -------------------------------------------
    def _mdm_to_proxy(self, x0_norm: torch.Tensor) -> torch.Tensor:
        """x0_norm: MDM-normalised motion (B,T,263) -> proxy-normalised input."""
        if self.same_norm:
            return x0_norm
        raw = x0_norm * self.mdm_std + self.mdm_mean            # de-normalise (MDM)
        return (raw - self.proxy_mean) / self.proxy_std          # re-normalise (proxy)

    def _activations(self, x0_norm: torch.Tensor) -> torch.Tensor:
        """Run the frozen proxy. Returns (B,T,402) in [0,1], with grad to x0_norm."""
        proxy_in = self._mdm_to_proxy(x0_norm)
        a = self.proxy(proxy_in)
        # If your proxy does not already clamp to [0,1], uncomment:
        # a = a.clamp(0.0, 1.0)
        return a

    # -- phase 1: build & freeze the reference -----------------------------
    @torch.no_grad()
    def build_reference(self, x0_ref_norm: torch.Tensor) -> Dict[str, float]:
        """x0_ref_norm: a NORMAL MDM batch (B,T,263). Builds {group: scalar} and
        stores it. These become constant targets for phase 2 (no grad)."""
        a_ref = self._activations(x0_ref_norm.to(self.device))     # (B,T,402)
        self.reference_acts = build_reference_from_activations(
            a_ref.detach().cpu().numpy(), self.mint_cols)
        return self.reference_acts

    def set_reference(self, reference_acts: Dict[str, float]) -> None:
        """Alternatively, inject a reference you built/cached elsewhere."""
        self.reference_acts = reference_acts

    # -- phase 2: the differentiable loss & its guidance gradient ----------
    def loss(self, x0_norm: torch.Tensor) -> torch.Tensor:
        """Differentiable scalar. total==0 -> matches reference; >0 -> toward pathology."""
        assert self.reference_acts is not None, "call build_reference() / set_reference() first"
        a = self._activations(x0_norm)
        return compute_posture_loss_torch(
            a, self.group_index, self.posture_name, self.reference_acts,
            component_weights=self.component_weights)

    def guidance_grad(self, x0_norm: torch.Tensor):
        """Returns (grad, loss_value). `grad` has the same shape as x0_norm.
        To GENERATE the pathology you MAXIMISE the loss, so step in the +grad
        direction:  x0_guided = x0_norm + scale * grad."""
        x = x0_norm.detach().requires_grad_(True)
        with torch.enable_grad():
            L = self.loss(x)
        grad = torch.autograd.grad(L, x)[0]
        return grad, float(L.detach())


# ===========================================================================
# WHERE THIS HOOKS INTO MDM  (pseudocode -- adapt to GuyTevet/motion-diffusion-model)
# ===========================================================================
#
# MDM predicts x0 directly (not epsilon), so x0_hat is available every step.
# Cheapest + most stable: x0-space ("reconstruction") guidance. We nudge x0_hat
# using the proxy gradient, WITHOUT backprop through the diffusion network.
#
#   guide = MuscleGuidance(proxy, mint_cols, "anterior_pelvic_tilt", ...)
#
#   # ---- Phase 1: reference (vanilla sampling, no guidance) ----
#   x0_ref = mdm.sample(model_kwargs)          # (B,263,1,T) -> permute to (B,T,263)
#   guide.build_reference(x0_ref.permute(0,3,2,1).squeeze(-1))   # match your layout!
#
#   # ---- Phase 2: guided sampling ----
#   def p_sample_with_guidance(model, x_t, t, scale, t_start_guidance, **kw):
#       out = diffusion.p_mean_variance(model, x_t, t, **kw)   # gives pred_xstart
#       x0_hat = out["pred_xstart"]                            # (B,263,1,T)
#       if int(t[0]) <= t_start_guidance:                      # only low-noise steps
#           x0_btc = x0_hat.permute(0, 3, 2, 1).squeeze(-1)    # -> (B,T,263)
#           grad, Lval = guide.guidance_grad(x0_btc)           # +grad = toward pathology
#           grad = grad.unsqueeze(2).permute(0, 3, 2, 1)       # back to (B,263,1,T)
#           x0_hat = x0_hat + scale * grad                     # gradient ASCENT
#           out["pred_xstart"] = x0_hat
#           # recompute the posterior mean from the nudged x0_hat:
#           out["mean"], _, _ = diffusion.q_posterior_mean_variance(x0_hat, x_t, t)
#       noise = torch.randn_like(x_t)
#       nonzero = (t != 0).float().view(-1, *([1]*(x_t.dim()-1)))
#       return out["mean"] + nonzero * (0.5 * out["log_variance"]).exp() * noise
#
# Practical knobs (start here, then tune):
#   scale            : guidance strength. Start ~ a few; sweep on a log scale.
#   t_start_guidance : only guide once x0_hat is meaningful, e.g. last ~30-50% of steps.
#   Clamp x0_hat to the valid HumanML3D range if your pipeline expects it.
#
# Alternative (stronger, costlier): x_t-space guidance -- let grad flow through MDM
# too (do NOT detach before p_mean_variance, take grad w.r.t. x_t). Only needed if
# x0-space guidance is too weak.
