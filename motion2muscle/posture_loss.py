"""
posture_loss.py
================
Generalisable, table-driven loss for detecting "bad-posture" patterns from
fascicle-level muscle activations (after roll-up to functional groups).

Why table-driven?
-----------------
Adding a new bad posture should require **no new code**, only a new entry in
POSTURE_PRIORS that references the functional groups defined in
muscle_rollup.ROLLUP_GROUPS. The four loss components below cover every
mechanism described in the literature for postural compensations:
  1. antagonist_imbalances   ->  agonist over-active relative to its antagonist
  2. compensation_chains     ->  primary mover inhibited AND a substitute
                                 muscle takes over (gated product penalty)
  3. synergy_imbalances      ->  within a synergistic group, the dominant
                                 muscle's share of total activation drops
  4. inhibited_stabilizers   ->  deep stabiliser drops below normal level

Loss interpretation
-------------------
loss == 0   -> activations match the reference (i.e. "normal" pattern)
loss > 0    -> activations deviate from the reference IN THE DIRECTION OF the
               named bad-posture template

The loss is **directional**, not symmetric. This is exactly what we want for
MDM guidance: minimising the loss with reversed sign (or maximising it) pushes
generation toward the bad posture, while a sample that already exhibits the
bad posture produces a high score.

Validation property
-------------------
For a *normal-population* sample whose mean was used to build `reference_acts`,
the mean loss across that sample should be near zero (only natural temporal
variation contributes). If it is not, the design is broken.

Compatibility
-------------
Inputs are numpy arrays. Math is restricted to +, -, *, /, max(., 0), mean,
sum -- so a torch port is mechanical (replace np with torch and
np.maximum(x, 0.) with torch.clamp(x, min=0.)). For MDM guidance you will
want the torch version so gradients flow.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
import numpy as np

from muscle_rollup import ROLLUP_GROUPS, get_indices


# ---------------------------------------------------------------------------
# POSTURE PRIOR TABLE
# ---------------------------------------------------------------------------
# Each entry is a complete description of the bad-posture muscle pattern.
# Group names are SIDE-AGNOSTIC ("iliopsoas", not "iliopsoas_R") because almost
# all postural compensations are bilateral. The loss expands them to _R / _L.
# Set "bilateral": False to keep them as-is (e.g. for unilateral patterns).
# ---------------------------------------------------------------------------

POSTURE_PRIORS: Dict[str, Dict[str, Any]] = {

    "anterior_pelvic_tilt": {
        "description": (
            "APT: hip flexors (iliopsoas, rectus femoris) and lumbar extensors "
            "are over-active; gluteus maximus and abdominal wall (especially "
            "deep core) are inhibited."
        ),
        # (over_active, under_active, severity_delta)
        # severity_delta = how much the over/under ratio should EXCEED normal
        # before the loss starts firing. Larger delta -> more permissive.
        "antagonist_imbalances": [
            ("erector_spinae",  "rectus_abdominis", 0.40),
            ("iliopsoas",       "gluteus_maximus",  0.50),
            ("rectus_femoris",  "hamstrings",       0.20),
        ],
        # (inhibited_primary, over_active_compensator, severity_delta)
        "compensation_chains": [
            ("gluteus_maximus", "hamstrings", 0.35),
            ("gluteus_medius",  "tfl",        0.40),
        ],
        "synergy_imbalances": [
            {
                "name":                  "hip_abduction",
                "dominant":              ["gluteus_medius", "gluteus_minimus"],
                "compensator":           ["tfl"],
                "normal_dominant_share": 0.75,   # ~75% of group activation
                "shift":                 0.20,   # threshold = 0.55
            },
        ],
        "inhibited_stabilizers": [
            # (group, fraction_below_ref): if activation < ref * (1 - frac), penalise
            ("transversus_abdominis", 0.30),
            ("lumbar_multifidus",     0.20),
        ],
        "bilateral": True,
    },

    "posterior_pelvic_tilt": {
        "description": (
            "PPT: glutes and abdominal wall over-active; hip flexors and "
            "lumbar extensors inhibited (flat-back / posterior tilt)."
        ),
        "antagonist_imbalances": [
            ("rectus_abdominis", "erector_spinae", 0.40),
            ("gluteus_maximus",  "iliopsoas",      0.50),
            ("hamstrings",       "rectus_femoris", 0.20),
        ],
        "compensation_chains": [],
        "synergy_imbalances": [],
        "inhibited_stabilizers": [
            ("erector_spinae", 0.20),
        ],
        "bilateral": True,
    },

    "forward_head_posture": {
        "description": (
            "FHP: SCM, scalenes and superficial neck extensors over-active; "
            "deep cervical flexors (longus colli) inhibited."
        ),
        "antagonist_imbalances": [
            ("sternocleidomastoid", "deep_neck_flexors", 0.50),
            ("neck_extensors",      "deep_neck_flexors", 0.40),
        ],
        "compensation_chains": [
            ("deep_neck_flexors", "sternocleidomastoid", 0.40),
            ("deep_neck_flexors", "scalenus",            0.30),
        ],
        "synergy_imbalances": [],
        "inhibited_stabilizers": [
            ("deep_neck_flexors", 0.30),
        ],
        "bilateral": True,
    },

    "trendelenburg": {
        "description": (
            "Trendelenburg: hip-abductor weakness (glmed/glmin) on the stance "
            "side leads to TFL/QL over-recruitment and contralateral pelvic "
            "drop. Treated bilaterally for symmetry."
        ),
        "antagonist_imbalances": [
            ("tfl",                "gluteus_medius",     0.30),
            ("quadratus_lumborum", "gluteus_medius",     0.30),
        ],
        "compensation_chains": [
            ("gluteus_medius",  "tfl",                0.40),
            ("gluteus_medius",  "quadratus_lumborum", 0.30),
        ],
        "synergy_imbalances": [
            {
                "name":                  "hip_abduction",
                "dominant":              ["gluteus_medius", "gluteus_minimus"],
                "compensator":           ["tfl"],
                "normal_dominant_share": 0.75,
                "shift":                 0.25,
            },
        ],
        "inhibited_stabilizers": [
            ("gluteus_medius",  0.30),
            ("gluteus_minimus", 0.30),
        ],
        "bilateral": True,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _activation_for_group(activations: np.ndarray,
                          mint_cols: List[str],
                          group_name: str) -> Optional[np.ndarray]:
    """Mean-pool activations across a group's fascicles.
    Works for both (T, M) and (B, T, M) inputs (selects the LAST axis)."""
    if group_name not in ROLLUP_GROUPS:
        return None
    idx = get_indices(group_name, mint_cols)
    if not idx:
        return None
    return activations[..., idx].mean(axis=-1)


def _expand_sides(name: str, bilateral: bool) -> List[str]:
    return [f"{name}_R", f"{name}_L"] if bilateral else [name]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _safe_mean(x: np.ndarray) -> float:
    """Mean over all axes -> python float, NaN-safe."""
    if x.size == 0:
        return 0.0
    return float(np.nanmean(x))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reference_from_activations(activations: np.ndarray,
                                     mint_cols: List[str],
                                     groups: Optional[List[str]] = None
                                     ) -> Dict[str, float]:
    """Build {group_name: scalar mean activation} from a normal sample.

    Parameters
    ----------
    activations : np.ndarray, shape (T, 402) or (B, T, 402)
        Activations from a NORMAL-population sample.
    mint_cols : list[str]
        Ordered list of MinT column names.
    groups : list[str] | None
        Which groups to compute; default = all in ROLLUP_GROUPS.

    Returns
    -------
    dict
        {group_name: float}
    """
    if groups is None:
        groups = list(ROLLUP_GROUPS.keys())
    ref: Dict[str, float] = {}
    for g in groups:
        a = _activation_for_group(activations, mint_cols, g)
        if a is not None:
            ref[g] = float(a.mean())
    return ref


def compute_posture_loss(activations: np.ndarray,
                         mint_cols: List[str],
                         posture_name: str,
                         reference_acts: Dict[str, float],
                         component_weights: Optional[Dict[str, float]] = None,
                         normalize_by_count: bool = True
                         ) -> Dict[str, Any]:
    """Compute the posture-deviation loss.

    Parameters
    ----------
    activations : np.ndarray, shape (T, 402) or (B, T, 402)
        Candidate activations to score.
    mint_cols : list[str]
        Ordered list of MinT column names.
    posture_name : str
        Key in POSTURE_PRIORS.
    reference_acts : dict
        {group_name: scalar} -- typically built from a normal sample via
        build_reference_from_activations().
    component_weights : dict | None
        Override weights for {antagonist, chain, synergy, stabilizer}.
        Defaults: {antagonist=1.0, chain=1.5, synergy=0.5, stabilizer=0.5}.
    normalize_by_count : bool
        If True, divide each component's accumulated loss by the number of
        rules contributing to it. Recommended -- makes components comparable
        across postures with different numbers of rules.

    Returns
    -------
    dict with keys:
        total                  -- weighted sum, scalar
        components             -- {component: unweighted scalar}
        weighted_components    -- {component: weight * unweighted}
        breakdown              -- per-rule details (for debugging / viz)
        weights, posture
    """
    if posture_name not in POSTURE_PRIORS:
        raise ValueError(
            f"Unknown posture '{posture_name}'. "
            f"Available: {list(POSTURE_PRIORS)}"
        )

    cfg = POSTURE_PRIORS[posture_name]
    bilateral = cfg.get("bilateral", True)

    weights = {"antagonist": 1.0, "chain": 1.5, "synergy": 0.5, "stabilizer": 0.5}
    if component_weights:
        weights.update(component_weights)

    eps = 1e-6
    breakdown: Dict[str, list] = {"antagonist": [], "chain": [],
                                  "synergy": [], "stabilizer": []}

    # ------------------------------------------------------------------ 1
    # Antagonist imbalance: penalise when the agonist is abnormally dominant
    # relative to its antagonist.
    #
    # TWO MODES depending on whether the under-muscle has a meaningful
    # baseline activation:
    #
    #  RATIO mode  (ref_u >= MIN_REF_FOR_RATIO = 0.01):
    #    loss = relu(r_current - r_normal * (1 + delta))
    #    where r = mean_over / mean_under (temporal means).
    #    Ratio is the right measure when both muscles are meaningfully active.
    #
    #  ABS-DIFF mode  (ref_u < MIN_REF_FOR_RATIO):
    #    The under-muscle baseline is near-zero (e.g. rectus_abdominis during
    #    locomotion / acting). Ratios become meaningless (515:1 is "normal").
    #    Instead we penalise the ABSOLUTE EXCESS of mean_over above a
    #    threshold = ref_o * (1 + delta).
    #    loss = relu(mean_over - ref_o * (1 + delta))
    #    This still fires when the over-muscle is abnormally elevated relative
    #    to ITS OWN reference, regardless of the antagonist's value.
    #
    # In both cases we aggregate over time before computing the loss (same
    # rationale as before: posture is a sustained pattern).
    # ------------------------------------------------------------------
    MIN_REF_FOR_RATIO = 0.01   # below this, ratio mode is unreliable

    L_antag, n_antag = 0.0, 0
    for over, under, delta in cfg.get("antagonist_imbalances", []):
        for s_over, s_under in zip(_expand_sides(over, bilateral),
                                   _expand_sides(under, bilateral)):
            a_over  = _activation_for_group(activations, mint_cols, s_over)
            a_under = _activation_for_group(activations, mint_cols, s_under)
            ref_o, ref_u = reference_acts.get(s_over), reference_acts.get(s_under)

            if a_over is None or a_under is None or ref_o is None or ref_u is None:
                breakdown["antagonist"].append({"name": f"{s_over}/{s_under}",
                                                "loss": None, "skipped": True})
                continue

            mean_over  = np.nanmean(a_over,  axis=-1)
            mean_under = np.nanmean(a_under, axis=-1)

            if ref_u >= MIN_REF_FOR_RATIO:
                # RATIO mode
                r_normal    = ref_o / (ref_u + eps)
                r_threshold = r_normal * (1.0 + delta)
                r_current   = mean_over / (mean_under + eps)
                term        = _safe_mean(_relu(r_current - r_threshold))
                mode        = "ratio"
                breakdown["antagonist"].append({
                    "name": f"{s_over}/{s_under}",
                    "loss": float(term),
                    "mode": mode,
                    "r_current_mean": float(np.nanmean(r_current)),
                    "r_normal":       float(ref_o / (ref_u + eps)),
                    "r_threshold":    float(r_threshold),
                    "delta":          float(delta),
                })
            else:
                # ABS-DIFF mode: under-muscle baseline too small for ratio
                threshold = ref_o * (1.0 + delta)
                term      = _safe_mean(_relu(mean_over - threshold))
                mode      = "abs_diff"
                breakdown["antagonist"].append({
                    "name": f"{s_over}/{s_under}",
                    "loss": float(term),
                    "mode": mode,
                    "mean_over":      float(np.nanmean(mean_over)),
                    "ref_over":       float(ref_o),
                    "threshold":      float(threshold),
                    "ref_under":      float(ref_u),
                    "note": f"ref_under={ref_u:.5f} < {MIN_REF_FOR_RATIO} -> abs_diff mode",
                })

            L_antag += term
            n_antag += 1

    # ------------------------------------------------------------------ 2
    # Compensation chain: gated product (after temporal aggregation).
    #   gate_p = relu(ref_p - mean_a_p)            primary BELOW normal
    #   gate_c = relu(mean_a_c - ref_c*(1+delta))  compensator ABOVE normal
    #   loss   = gate_p * gate_c
    # The product means BOTH conditions must hold at the sample level for
    # the loss to fire. Temporal aggregation again avoids spurious firing
    # on noisy normal data.
    # ------------------------------------------------------------------
    L_chain, n_chain = 0.0, 0
    for primary, comp, delta in cfg.get("compensation_chains", []):
        for s_p, s_c in zip(_expand_sides(primary, bilateral),
                            _expand_sides(comp, bilateral)):
            a_p = _activation_for_group(activations, mint_cols, s_p)
            a_c = _activation_for_group(activations, mint_cols, s_c)
            ref_p, ref_c = reference_acts.get(s_p), reference_acts.get(s_c)

            if a_p is None or a_c is None or ref_p is None or ref_c is None:
                breakdown["chain"].append({"name": f"{s_p}->{s_c}",
                                           "loss": None, "skipped": True})
                continue

            mean_p = np.nanmean(a_p, axis=-1)
            mean_c = np.nanmean(a_c, axis=-1)

            gate_p = _relu(ref_p - mean_p)
            gate_c = _relu(mean_c - ref_c * (1.0 + delta))
            term   = _safe_mean(gate_p * gate_c)

            L_chain += term
            n_chain += 1
            breakdown["chain"].append({
                "name": f"{s_p}->{s_c}",
                "loss": float(term),
                "primary_mean": float(np.nanmean(mean_p)),
                "primary_ref":  float(ref_p),
                "comp_mean":    float(np.nanmean(mean_c)),
                "comp_ref":     float(ref_c),
                "delta":        float(delta),
            })

    # ------------------------------------------------------------------ 3
    # Synergy imbalance: dominant share of total group activation drops.
    # As with antagonist, we use temporal-mean activations to avoid per-
    # frame noise inflating the loss on normal samples.
    #
    # We derive the "normal" dominant share from the reference if available
    # (so that a normal sample whose mean was used as reference gives
    # exactly zero loss), and fall back to the literature value in
    # `normal_dominant_share` only when the reference doesn't cover the
    # required groups.
    # ------------------------------------------------------------------
    L_syn, n_syn = 0.0, 0
    for syn in cfg.get("synergy_imbalances", []):
        sides = ["_R", "_L"] if bilateral else [""]
        for side in sides:
            dom_acts = [a for n in syn["dominant"]
                          if (a := _activation_for_group(activations, mint_cols, n + side)) is not None]
            comp_acts = [a for n in syn["compensator"]
                          if (a := _activation_for_group(activations, mint_cols, n + side)) is not None]
            if not dom_acts or not comp_acts:
                breakdown["synergy"].append({"name": f"{syn['name']}{side}",
                                             "loss": None, "skipped": True})
                continue

            # Aggregate over time first
            dom_mean   = np.stack([np.nanmean(a, axis=-1) for a in dom_acts]).sum(axis=0)
            comp_mean  = np.stack([np.nanmean(a, axis=-1) for a in comp_acts]).sum(axis=0)
            total      = dom_mean + comp_mean + eps
            dom_share  = dom_mean / total

            # Prefer reference-based normal share when possible
            ref_dom_total  = sum(reference_acts.get(n + side, 0.0) for n in syn["dominant"])
            ref_comp_total = sum(reference_acts.get(n + side, 0.0) for n in syn["compensator"])
            if ref_dom_total + ref_comp_total > eps:
                normal_share_used = ref_dom_total / (ref_dom_total + ref_comp_total + eps)
                share_source      = "reference"
            else:
                normal_share_used = syn["normal_dominant_share"]
                share_source      = "literature"

            target_dom = normal_share_used - syn["shift"]
            term = _safe_mean(_relu(target_dom - dom_share))

            L_syn += term
            n_syn += 1
            breakdown["synergy"].append({
                "name": f"{syn['name']}{side}",
                "loss": float(term),
                "dom_share_mean":     float(np.nanmean(dom_share)),
                "target_dom_share":   float(target_dom),
                "normal_dom_share":   float(normal_share_used),
                "share_source":       share_source,
            })

    # ------------------------------------------------------------------ 4
    # Inhibited stabiliser: activation falls BELOW ref * (1 - fraction_below)
    # ------------------------------------------------------------------
    L_stab, n_stab = 0.0, 0
    for name, fraction_below in cfg.get("inhibited_stabilizers", []):
        for s in _expand_sides(name, bilateral):
            a = _activation_for_group(activations, mint_cols, s)
            ref = reference_acts.get(s)
            if a is None or ref is None:
                breakdown["stabilizer"].append({"name": s,
                                                "loss": None, "skipped": True})
                continue

            threshold = ref * (1.0 - fraction_below)
            term      = _safe_mean(_relu(threshold - a))

            L_stab += term
            n_stab += 1
            breakdown["stabilizer"].append({
                "name": s,
                "loss": float(term),
                "current_mean": float(np.nanmean(a)),
                "ref":          float(ref),
                "threshold":    float(threshold),
            })

    # ----- Normalisation by rule count ----------------------------------
    if normalize_by_count:
        if n_antag > 0: L_antag /= n_antag
        if n_chain > 0: L_chain /= n_chain
        if n_syn   > 0: L_syn   /= n_syn
        if n_stab  > 0: L_stab  /= n_stab

    components = {"antagonist": L_antag, "chain": L_chain,
                  "synergy": L_syn,      "stabilizer": L_stab}
    weighted   = {k: float(weights[k] * v) for k, v in components.items()}
    total      = float(sum(weighted.values()))

    return {
        "total":                total,
        "components":           {k: float(v) for k, v in components.items()},
        "weighted_components":  weighted,
        "breakdown":            breakdown,
        "weights":              dict(weights),
        "posture":              posture_name,
        "rule_counts":          {"antagonist": n_antag, "chain": n_chain,
                                 "synergy": n_syn, "stabilizer": n_stab},
    }


# ---------------------------------------------------------------------------
# Synthetic distortion (for testing / data augmentation)
# ---------------------------------------------------------------------------

def make_synthetic_distortion(activations: np.ndarray,
                              mint_cols: List[str],
                              posture_name: str,
                              severity: float = 0.5,
                              clip_max: float = 1.0,
                              clip_min: float = 0.0
                              ) -> np.ndarray:
    """Apply a synthetic distortion to NORMAL activations to mimic the named
    bad posture.

    The distortion is derived directly from the same POSTURE_PRIORS entry --
    so adding a new posture automatically supports synthetic distortion
    without extra code.

    Parameters
    ----------
    activations : np.ndarray, shape (T, 402) or (B, T, 402)
    mint_cols   : list[str]
    posture_name: str
    severity    : float in [0, 1] -- 0 = no distortion, 1 = strong distortion
    clip_max,
    clip_min    : output clipping range (activations are typically in [0,1])

    Returns
    -------
    distorted : np.ndarray of the same shape as activations
    """
    if posture_name not in POSTURE_PRIORS:
        raise ValueError(f"Unknown posture '{posture_name}'.")

    cfg = POSTURE_PRIORS[posture_name]
    bilateral = cfg.get("bilateral", True)

    n_cols = activations.shape[-1]
    scale = np.ones(n_cols, dtype=np.float64)

    def _scale_group(group_name: str, factor: float) -> None:
        if group_name not in ROLLUP_GROUPS:
            return
        for i in get_indices(group_name, mint_cols):
            scale[i] *= factor

    s = severity

    # 1) antagonists: over-active up, under-active down
    for over, under, _delta in cfg.get("antagonist_imbalances", []):
        for so in _expand_sides(over, bilateral):
            _scale_group(so, 1.0 + 0.6 * s)
        for su in _expand_sides(under, bilateral):
            _scale_group(su, 1.0 - 0.4 * s)

    # 2) compensation chains: primary down, compensator up
    for primary, comp, _delta in cfg.get("compensation_chains", []):
        for sp in _expand_sides(primary, bilateral):
            _scale_group(sp, 1.0 - 0.5 * s)
        for sc in _expand_sides(comp, bilateral):
            _scale_group(sc, 1.0 + 0.7 * s)

    # 3) synergy: dominant down, compensator up
    for syn in cfg.get("synergy_imbalances", []):
        sides = ["_R", "_L"] if bilateral else [""]
        for side in sides:
            for n in syn["dominant"]:
                _scale_group(n + side, 1.0 - 0.4 * s)
            for n in syn["compensator"]:
                _scale_group(n + side, 1.0 + 0.8 * s)

    # 4) inhibited stabilisers: down
    for name, frac in cfg.get("inhibited_stabilizers", []):
        for sn in _expand_sides(name, bilateral):
            _scale_group(sn, 1.0 - frac * s)

    distorted = activations * scale[..., :]   # broadcast on last axis
    if clip_min is not None or clip_max is not None:
        distorted = np.clip(distorted, clip_min, clip_max)
    return distorted.astype(activations.dtype, copy=False)


def get_posture_relevant_groups(posture_name: str) -> List[str]:
    """Return the list of (side-aware) group names referenced by the posture
    template. Useful for visualisation -- you only want to plot the groups
    that the loss actually cares about."""
    if posture_name not in POSTURE_PRIORS:
        raise ValueError(f"Unknown posture '{posture_name}'.")
    cfg = POSTURE_PRIORS[posture_name]
    bilateral = cfg.get("bilateral", True)

    names: List[str] = []
    for over, under, _ in cfg.get("antagonist_imbalances", []):
        names += _expand_sides(over, bilateral) + _expand_sides(under, bilateral)
    for p, c, _ in cfg.get("compensation_chains", []):
        names += _expand_sides(p, bilateral) + _expand_sides(c, bilateral)
    for syn in cfg.get("synergy_imbalances", []):
        sides = ["_R", "_L"] if bilateral else [""]
        for side in sides:
            for n in syn["dominant"] + syn["compensator"]:
                names.append(n + side)
    for n, _ in cfg.get("inhibited_stabilizers", []):
        names += _expand_sides(n, bilateral)

    # de-duplicate while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen and n in ROLLUP_GROUPS:
            seen.add(n)
            out.append(n)
    return out


if __name__ == "__main__":
    print("Available postures:")
    for k, v in POSTURE_PRIORS.items():
        print(f"  - {k}: {v['description']}")
        for g in get_posture_relevant_groups(k):
            print(f"      {g}")