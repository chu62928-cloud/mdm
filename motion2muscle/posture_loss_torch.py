"""
posture_loss_torch.py
=====================
Differentiable (torch) port of `posture_loss.compute_posture_loss`.

WHY THIS FILE EXISTS
--------------------
The original `posture_loss.py` is numpy-only. For MDM guidance we need gradients
to flow from the loss, back through the (frozen) motion2muscle proxy, and into the
generated motion x. This module reimplements the exact same four loss components in
torch so `loss.backward()` works.

It reuses the *definitions* (ROLLUP_GROUPS, POSTURE_PRIORS) from the numpy modules so
there is a single source of truth for the muscle groups and the bad-posture templates.
Only the math is reimplemented (np -> torch, np.maximum(x,0) -> clamp(x,min=0)).

SEMANTICS PRESERVED FROM THE NUMPY VERSION (verified component by component):
  - antagonist: ratio mode if ref_under >= 0.01 else abs-diff mode; temporal mean first
  - chain:      gated product, both gates from temporal means
  - synergy:    dominant share of (dominant+compensator), reference-derived normal share
  - stabilizer: per-frame penalty (NOT temporally aggregated -- matches the original)
  - normalize_by_count and the default weights {antag 1.0, chain 1.5, syn 0.5, stab 0.5}

INPUT CONVENTION
----------------
activations : torch.FloatTensor, shape (B, T, 402) or (T, 402), values in [0, 1].
              The proxy output. Must carry grad if you want guidance.
reference_acts : Dict[str, float]   {group_name: scalar mean activation}
              Built ONCE from the reference (normal) sample, then frozen. These are
              plain python floats -- constants, no grad.
group_index : Dict[str, LongTensor]  {group_name: indices into the 402 axis}
              Precompute once with build_group_index(...) so we don't rebuild it every
              denoising step.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import torch

from muscle_rollup import get_indices
from posture_loss import POSTURE_PRIORS, get_posture_relevant_groups

MIN_REF_FOR_RATIO = 0.01
DEFAULT_WEIGHTS = {"antagonist": 1.0, "chain": 1.5, "synergy": 0.5, "stabilizer": 0.5}


# ---------------------------------------------------------------------------
# Setup helpers (call once)
# ---------------------------------------------------------------------------
def build_group_index(mint_cols: List[str],
                      posture_name: str,
                      device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
    """Precompute {group_name: LongTensor of column indices} for every group the
    posture references. Do this ONCE (not every diffusion step)."""
    idx_map: Dict[str, torch.Tensor] = {}
    for name in get_posture_relevant_groups(posture_name):
        idx = get_indices(name, mint_cols)
        if idx:
            idx_map[name] = torch.as_tensor(idx, dtype=torch.long, device=device)
    return idx_map


def _expand_sides(name: str, bilateral: bool) -> List[str]:
    return [f"{name}_R", f"{name}_L"] if bilateral else [name]


# ---------------------------------------------------------------------------
# Core differentiable building blocks
# ---------------------------------------------------------------------------
def _group_act(activations: torch.Tensor,
               group_index: Dict[str, torch.Tensor],
               name: str) -> Optional[torch.Tensor]:
    """Mean-pool a group's fascicles. Returns (..., T) i.e. one value per frame."""
    idx = group_index.get(name)
    if idx is None or idx.numel() == 0:
        return None
    return activations.index_select(-1, idx).mean(dim=-1)   # (B, T) or (T,)


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def _batch_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean over all remaining axes -> scalar tensor (keeps grad)."""
    return x.mean() if x.numel() > 0 else x.new_zeros(())


# ---------------------------------------------------------------------------
# Public: the differentiable loss
# ---------------------------------------------------------------------------
def compute_posture_loss_torch(activations: torch.Tensor,
                               group_index: Dict[str, torch.Tensor],
                               posture_name: str,
                               reference_acts: Dict[str, float],
                               component_weights: Optional[Dict[str, float]] = None,
                               normalize_by_count: bool = True,
                               eps: float = 1e-6,
                               return_components: bool = False) -> Any:
    """Returns a scalar torch tensor (the weighted total loss) that you can
    .backward() through. Set return_components=True to also get detached
    per-component values for logging.

    DIRECTIONALITY (unchanged from numpy version):
      total == 0  -> activations match the reference (normal pattern)
      total >  0  -> activations deviate TOWARD the named bad posture
    To GENERATE the bad posture during guidance you MAXIMISE this loss
    (gradient ASCENT), or equivalently descend on -total.
    """
    if posture_name not in POSTURE_PRIORS:
        raise ValueError(f"Unknown posture '{posture_name}'. Available: {list(POSTURE_PRIORS)}")

    cfg = POSTURE_PRIORS[posture_name]
    bilateral = cfg.get("bilateral", True)
    weights = dict(DEFAULT_WEIGHTS)
    if component_weights:
        weights.update(component_weights)

    zero = activations.new_zeros(())

    # ---- 1. antagonist imbalance -----------------------------------------
    L_antag, n_antag = zero.clone(), 0
    for over, under, delta in cfg.get("antagonist_imbalances", []):
        for s_over, s_under in zip(_expand_sides(over, bilateral),
                                   _expand_sides(under, bilateral)):
            a_over = _group_act(activations, group_index, s_over)
            a_under = _group_act(activations, group_index, s_under)
            ref_o, ref_u = reference_acts.get(s_over), reference_acts.get(s_under)
            if a_over is None or a_under is None or ref_o is None or ref_u is None:
                continue
            mean_over = a_over.mean(dim=-1)    # temporal mean -> (B,) or scalar
            mean_under = a_under.mean(dim=-1)
            if ref_u >= MIN_REF_FOR_RATIO:                      # RATIO mode
                r_threshold = (ref_o / (ref_u + eps)) * (1.0 + delta)
                r_current = mean_over / (mean_under + eps)
                term = _batch_mean(_relu(r_current - r_threshold))
            else:                                               # ABS-DIFF mode
                threshold = ref_o * (1.0 + delta)
                term = _batch_mean(_relu(mean_over - threshold))
            L_antag = L_antag + term
            n_antag += 1

    # ---- 2. compensation chain (gated product) ---------------------------
    L_chain, n_chain = zero.clone(), 0
    for primary, comp, delta in cfg.get("compensation_chains", []):
        for s_p, s_c in zip(_expand_sides(primary, bilateral),
                            _expand_sides(comp, bilateral)):
            a_p = _group_act(activations, group_index, s_p)
            a_c = _group_act(activations, group_index, s_c)
            ref_p, ref_c = reference_acts.get(s_p), reference_acts.get(s_c)
            if a_p is None or a_c is None or ref_p is None or ref_c is None:
                continue
            mean_p = a_p.mean(dim=-1)
            mean_c = a_c.mean(dim=-1)
            gate_p = _relu(ref_p - mean_p)
            gate_c = _relu(mean_c - ref_c * (1.0 + delta))
            term = _batch_mean(gate_p * gate_c)
            L_chain = L_chain + term
            n_chain += 1

    # ---- 3. synergy imbalance --------------------------------------------
    L_syn, n_syn = zero.clone(), 0
    for syn in cfg.get("synergy_imbalances", []):
        sides = ["_R", "_L"] if bilateral else [""]
        for side in sides:
            dom_acts = [a for n in syn["dominant"]
                        if (a := _group_act(activations, group_index, n + side)) is not None]
            comp_acts = [a for n in syn["compensator"]
                         if (a := _group_act(activations, group_index, n + side)) is not None]
            if not dom_acts or not comp_acts:
                continue
            dom_mean = torch.stack([a.mean(dim=-1) for a in dom_acts]).sum(dim=0)
            comp_mean = torch.stack([a.mean(dim=-1) for a in comp_acts]).sum(dim=0)
            dom_share = dom_mean / (dom_mean + comp_mean + eps)
            ref_dom = sum(reference_acts.get(n + side, 0.0) for n in syn["dominant"])
            ref_comp = sum(reference_acts.get(n + side, 0.0) for n in syn["compensator"])
            if ref_dom + ref_comp > eps:
                normal_share = ref_dom / (ref_dom + ref_comp + eps)
            else:
                normal_share = syn["normal_dominant_share"]
            target_dom = normal_share - syn["shift"]
            term = _batch_mean(_relu(target_dom - dom_share))
            L_syn = L_syn + term
            n_syn += 1

    # ---- 4. inhibited stabiliser (per-frame, NOT temporally aggregated) ---
    L_stab, n_stab = zero.clone(), 0
    for name, fraction_below in cfg.get("inhibited_stabilizers", []):
        for s in _expand_sides(name, bilateral):
            a = _group_act(activations, group_index, s)
            ref = reference_acts.get(s)
            if a is None or ref is None:
                continue
            threshold = ref * (1.0 - fraction_below)
            term = _batch_mean(_relu(threshold - a))
            L_stab = L_stab + term
            n_stab += 1

    if normalize_by_count:
        if n_antag > 0: L_antag = L_antag / n_antag
        if n_chain > 0: L_chain = L_chain / n_chain
        if n_syn > 0:   L_syn = L_syn / n_syn
        if n_stab > 0:  L_stab = L_stab / n_stab

    total = (weights["antagonist"] * L_antag + weights["chain"] * L_chain
             + weights["synergy"] * L_syn + weights["stabilizer"] * L_stab)

    if not return_components:
        return total
    return total, {
        "antagonist": float(L_antag.detach()),
        "chain":      float(L_chain.detach()),
        "synergy":    float(L_syn.detach()),
        "stabilizer": float(L_stab.detach()),
        "rule_counts": {"antagonist": n_antag, "chain": n_chain,
                        "synergy": n_syn, "stabilizer": n_stab},
    }
