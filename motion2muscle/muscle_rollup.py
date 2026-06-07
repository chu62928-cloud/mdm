"""
muscle_rollup.py
================
Maps MinT's 402 fascicle-level muscle columns to biomechanically meaningful
functional groups.  Every group is a list of raw MinT column names that can be
mean-pooled to produce a single scalar activation for that group.

Design principles
-----------------
- Groups are defined at the *functional unit* level, not the anatomical muscle
  level, because that is the right granularity for loss computation (e.g.
  "hamstrings" as one antagonist, not four separate muscles).
- Left and right sides are always separate entries so that asymmetry losses
  can be computed without extra logic.
- The 402 columns are ordered: indices 0-79 are LU (lower body), 80-401 are TL
  (thoracolumbar).  Index lookup is provided by `get_indices()`.
- Rollup is mean-pooling by default: group_activation = mean(fascicle activations).

Usage
-----
    from muscle_rollup import ROLLUP_GROUPS, get_indices, rollup

    # get the column names for one group
    cols = ROLLUP_GROUPS["gluteus_maximus_R"]

    # get their integer indices in the 402-dim vector
    idx = get_indices("gluteus_maximus_R", mint_col_list)

    # roll up a (T, 402) numpy array to (T, N_groups) array
    grouped = rollup(activation_array, mint_col_list)
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

# ---------------------------------------------------------------------------
# LU  — Lower-body muscles (Lai & Arnold 2017, 40 fascicles per side)
# ---------------------------------------------------------------------------

_LU: Dict[str, List[str]] = {
    # ── Gluteal group ───────────────────────────────────────────────────────
    "gluteus_maximus_R": ["LU_glmax1_r", "LU_glmax2_r", "LU_glmax3_r"],
    "gluteus_maximus_L": ["LU_glmax1_l", "LU_glmax2_l", "LU_glmax3_l"],

    "gluteus_medius_R": ["LU_glmed1_r", "LU_glmed2_r", "LU_glmed3_r"],
    "gluteus_medius_L": ["LU_glmed1_l", "LU_glmed2_l", "LU_glmed3_l"],

    "gluteus_minimus_R": ["LU_glmin1_r", "LU_glmin2_r", "LU_glmin3_r"],
    "gluteus_minimus_L": ["LU_glmin1_l", "LU_glmin2_l", "LU_glmin3_l"],

    # ── Hip flexors ─────────────────────────────────────────────────────────
    # iliopsoas = iliacus + psoas (LU model only has iliacus; psoas is in TL)
    "iliacus_R": ["LU_iliacus_r"],
    "iliacus_L": ["LU_iliacus_l"],

    # Rectus femoris: hip flexor AND knee extensor — kept separate
    "rectus_femoris_R": ["LU_recfem_r"],
    "rectus_femoris_L": ["LU_recfem_l"],

    # TFL: key compensation muscle for hip abduction
    "tfl_R": ["LU_tfl_r"],
    "tfl_L": ["LU_tfl_l"],

    "sartorius_R": ["LU_sart_r"],
    "sartorius_L": ["LU_sart_l"],

    "piriformis_R": ["LU_piri_r"],
    "piriformis_L": ["LU_piri_l"],

    # ── Hamstrings ──────────────────────────────────────────────────────────
    # Long head BF crosses hip AND knee → most important for APT
    "biceps_femoris_lh_R": ["LU_bflh_r"],
    "biceps_femoris_lh_L": ["LU_bflh_l"],
    "biceps_femoris_sh_R": ["LU_bfsh_r"],
    "biceps_femoris_sh_L": ["LU_bfsh_l"],
    "semitendinosus_R": ["LU_semiten_r"],
    "semitendinosus_L": ["LU_semiten_l"],
    "semimembranosus_R": ["LU_semimem_r"],
    "semimembranosus_L": ["LU_semimem_l"],

    # Pooled hamstrings group (for synergy / compensation losses)
    "hamstrings_R": ["LU_bflh_r", "LU_bfsh_r", "LU_semiten_r", "LU_semimem_r"],
    "hamstrings_L": ["LU_bflh_l", "LU_bfsh_l", "LU_semiten_l", "LU_semimem_l"],

    # ── Quadriceps ──────────────────────────────────────────────────────────
    "vastus_medialis_R": ["LU_vasmed_r"],
    "vastus_medialis_L": ["LU_vasmed_l"],
    "vastus_lateralis_R": ["LU_vaslat_r"],
    "vastus_lateralis_L": ["LU_vaslat_l"],
    "vastus_intermedius_R": ["LU_vasint_r"],
    "vastus_intermedius_L": ["LU_vasint_l"],

    # Pooled quads (recfem included)
    "quadriceps_R": ["LU_recfem_r", "LU_vasmed_r", "LU_vaslat_r", "LU_vasint_r"],
    "quadriceps_L": ["LU_recfem_l", "LU_vasmed_l", "LU_vaslat_l", "LU_vasint_l"],

    # ── Adductors ───────────────────────────────────────────────────────────
    "adductor_magnus_R": [
        "LU_addmagDist_r", "LU_addmagIsch_r",
        "LU_addmagMid_r",  "LU_addmagProx_r",
    ],
    "adductor_magnus_L": [
        "LU_addmagDist_l", "LU_addmagIsch_l",
        "LU_addmagMid_l",  "LU_addmagProx_l",
    ],
    "adductors_R": [
        "LU_addbrev_r", "LU_addlong_r",
        "LU_addmagDist_r", "LU_addmagIsch_r",
        "LU_addmagMid_r", "LU_addmagProx_r",
        "LU_grac_r",
    ],
    "adductors_L": [
        "LU_addbrev_l", "LU_addlong_l",
        "LU_addmagDist_l", "LU_addmagIsch_l",
        "LU_addmagMid_l", "LU_addmagProx_l",
        "LU_grac_l",
    ],
    "gracilis_R": ["LU_grac_r"],
    "gracilis_L": ["LU_grac_l"],

    # ── Ankle / lower leg ───────────────────────────────────────────────────
    "gastrocnemius_R": ["LU_gaslat_r", "LU_gasmed_r"],
    "gastrocnemius_L": ["LU_gaslat_l", "LU_gasmed_l"],
    "soleus_R": ["LU_soleus_r"],
    "soleus_L": ["LU_soleus_l"],
    "plantarflexors_R": [
        "LU_gaslat_r", "LU_gasmed_r", "LU_soleus_r",
        "LU_fdl_r",    "LU_fhl_r",
        "LU_perbrev_r","LU_perlong_r",
    ],
    "plantarflexors_L": [
        "LU_gaslat_l", "LU_gasmed_l", "LU_soleus_l",
        "LU_fdl_l",    "LU_fhl_l",
        "LU_perbrev_l","LU_perlong_l",
    ],
    "tibialis_anterior_R": ["LU_tibant_r"],
    "tibialis_anterior_L": ["LU_tibant_l"],
    "dorsiflexors_R": ["LU_tibant_r", "LU_edl_r", "LU_ehl_r"],
    "dorsiflexors_L": ["LU_tibant_l", "LU_edl_l", "LU_ehl_l"],
    "peroneals_R": ["LU_perbrev_r", "LU_perlong_r"],
    "peroneals_L": ["LU_perbrev_l", "LU_perlong_l"],
}

# ---------------------------------------------------------------------------
# TL  — Thoracolumbar muscles (Bruno et al. 2015, 161 fascicles per side)
# ---------------------------------------------------------------------------

_TL: Dict[str, List[str]] = {
    # ── Abdominal / core ────────────────────────────────────────────────────
    "rectus_abdominis_R": ["TL_rect_abd_r"],
    "rectus_abdominis_L": ["TL_rect_abd_l"],

    "external_oblique_R": [
        "TL_E0_R5_r",  "TL_E0_R6_r",  "TL_E0_R7_r",  "TL_E0_R8_r",
        "TL_E0_R9_r",  "TL_E0_R10_r", "TL_E0_R11_r", "TL_E0_R12_r",
    ],
    "external_oblique_L": [
        "TL_E0_R5_l",  "TL_E0_R6_l",  "TL_E0_R7_l",  "TL_E0_R8_l",
        "TL_E0_R9_l",  "TL_E0_R10_l", "TL_E0_R11_l", "TL_E0_R12_l",
    ],

    "internal_oblique_R": [
        "TL_IO1_r", "TL_IO2_r", "TL_IO3_r",
        "TL_IO4_r", "TL_IO5_r", "TL_IO6_r",
    ],
    "internal_oblique_L": [
        "TL_IO1_l", "TL_IO2_l", "TL_IO3_l",
        "TL_IO4_l", "TL_IO5_l", "TL_IO6_l",
    ],

    "transversus_abdominis_R": [
        "TL_TR1_r", "TL_TR2_r", "TL_TR3_r", "TL_TR4_r", "TL_TR5_r",
    ],
    "transversus_abdominis_L": [
        "TL_TR1_l", "TL_TR2_l", "TL_TR3_l", "TL_TR4_l", "TL_TR5_l",
    ],

    # ── Erector spinae — iliocostalis ────────────────────────────────────────
    "iliocostalis_R": [
        "TL_IL_L1_r",  "TL_IL_L2_r",  "TL_IL_L3_r",  "TL_IL_L4_r",
        "TL_IL_R5_r",  "TL_IL_R6_r",  "TL_IL_R7_r",  "TL_IL_R8_r",
        "TL_IL_R9_r",  "TL_IL_R10_r", "TL_IL_R11_r", "TL_IL_R12_r",
    ],
    "iliocostalis_L": [
        "TL_IL_L1_l",  "TL_IL_L2_l",  "TL_IL_L3_l",  "TL_IL_L4_l",
        "TL_IL_R5_l",  "TL_IL_R6_l",  "TL_IL_R7_l",  "TL_IL_R8_l",
        "TL_IL_R9_l",  "TL_IL_R10_l", "TL_IL_R11_l", "TL_IL_R12_l",
    ],

    # ── Erector spinae — longissimus (thoracis + lumbar) ────────────────────
    "longissimus_thoracis_R": [
        "TL_LTpT_T1_r",  "TL_LTpT_T2_r",  "TL_LTpT_T3_r",
        "TL_LTpT_T4_r",  "TL_LTpT_T5_r",  "TL_LTpT_T6_r",
        "TL_LTpT_T7_r",  "TL_LTpT_T8_r",  "TL_LTpT_T9_r",
        "TL_LTpT_T10_r", "TL_LTpT_T11_r", "TL_LTpT_T12_r",
        "TL_LTpT_R4_r",  "TL_LTpT_R5_r",  "TL_LTpT_R6_r",
        "TL_LTpT_R7_r",  "TL_LTpT_R8_r",  "TL_LTpT_R9_r",
        "TL_LTpT_R10_r", "TL_LTpT_R11_r", "TL_LTpT_R12_r",
        "TL_LTpL_L5_r",  "TL_LTpL_L4_r",  "TL_LTpL_L3_r",
        "TL_LTpL_L2_r",  "TL_LTpL_L1_r",
    ],
    "longissimus_thoracis_L": [
        "TL_LTpT_T1_l",  "TL_LTpT_T2_l",  "TL_LTpT_T3_l",
        "TL_LTpT_T4_l",  "TL_LTpT_T5_l",  "TL_LTpT_T6_l",
        "TL_LTpT_T7_l",  "TL_LTpT_T8_l",  "TL_LTpT_T9_l",
        "TL_LTpT_T10_l", "TL_LTpT_T11_l", "TL_LTpT_T12_l",
        "TL_LTpT_R4_l",  "TL_LTpT_R5_l",  "TL_LTpT_R6_l",
        "TL_LTpT_R7_l",  "TL_LTpT_R8_l",  "TL_LTpT_R9_l",
        "TL_LTpT_R10_l", "TL_LTpT_R11_l", "TL_LTpT_R12_l",
        "TL_LTpL_L5_l",  "TL_LTpL_L4_l",  "TL_LTpL_L3_l",
        "TL_LTpL_L2_l",  "TL_LTpL_L1_l",
    ],

    # Pooled erector spinae (iliocostalis + longissimus) — used in loss directly
    "erector_spinae_R": [
        "TL_IL_L1_r",  "TL_IL_L2_r",  "TL_IL_L3_r",  "TL_IL_L4_r",
        "TL_IL_R5_r",  "TL_IL_R6_r",  "TL_IL_R7_r",  "TL_IL_R8_r",
        "TL_IL_R9_r",  "TL_IL_R10_r", "TL_IL_R11_r", "TL_IL_R12_r",
        "TL_LTpT_T1_r",  "TL_LTpT_T2_r",  "TL_LTpT_T3_r",
        "TL_LTpT_T4_r",  "TL_LTpT_T5_r",  "TL_LTpT_T6_r",
        "TL_LTpT_T7_r",  "TL_LTpT_T8_r",  "TL_LTpT_T9_r",
        "TL_LTpT_T10_r", "TL_LTpT_T11_r", "TL_LTpT_T12_r",
        "TL_LTpT_R4_r",  "TL_LTpT_R5_r",  "TL_LTpT_R6_r",
        "TL_LTpT_R7_r",  "TL_LTpT_R8_r",  "TL_LTpT_R9_r",
        "TL_LTpT_R10_r", "TL_LTpT_R11_r", "TL_LTpT_R12_r",
        "TL_LTpL_L5_r",  "TL_LTpL_L4_r",  "TL_LTpL_L3_r",
        "TL_LTpL_L2_r",  "TL_LTpL_L1_r",
    ],
    "erector_spinae_L": [
        "TL_IL_L1_l",  "TL_IL_L2_l",  "TL_IL_L3_l",  "TL_IL_L4_l",
        "TL_IL_R5_l",  "TL_IL_R6_l",  "TL_IL_R7_l",  "TL_IL_R8_l",
        "TL_IL_R9_l",  "TL_IL_R10_l", "TL_IL_R11_l", "TL_IL_R12_l",
        "TL_LTpT_T1_l",  "TL_LTpT_T2_l",  "TL_LTpT_T3_l",
        "TL_LTpT_T4_l",  "TL_LTpT_T5_l",  "TL_LTpT_T6_l",
        "TL_LTpT_T7_l",  "TL_LTpT_T8_l",  "TL_LTpT_T9_l",
        "TL_LTpT_T10_l", "TL_LTpT_T11_l", "TL_LTpT_T12_l",
        "TL_LTpT_R4_l",  "TL_LTpT_R5_l",  "TL_LTpT_R6_l",
        "TL_LTpT_R7_l",  "TL_LTpT_R8_l",  "TL_LTpT_R9_l",
        "TL_LTpT_R10_l", "TL_LTpT_R11_l", "TL_LTpT_R12_l",
        "TL_LTpL_L5_l",  "TL_LTpL_L4_l",  "TL_LTpL_L3_l",
        "TL_LTpL_L2_l",  "TL_LTpL_L1_l",
    ],

    # ── Multifidus — three regions ────────────────────────────────────────────
    # Lumbar multifidus is the deep stabilizer targeted in APT
    "lumbar_multifidus_R": [
        "TL_MF_m1t_1_r", "TL_MF_m1t_2_r", "TL_MF_m1t_3_r",
        "TL_MF_m1s_r",   "TL_MF_m2s_r",   "TL_MF_m2t_1_r",
        "TL_MF_m2t_2_r", "TL_MF_m2t_3_r", "TL_MF_m3s_r",
        "TL_MF_m3t_1_r", "TL_MF_m3t_2_r", "TL_MF_m3t_3_r",
        "TL_MF_m4s_r",   "TL_MF_m4t_1_r", "TL_MF_m4t_2_r",
        "TL_MF_m4t_3_r", "TL_MF_m5s_r",   "TL_MF_m5t_1_r",
        "TL_MF_m5t_2_r", "TL_MF_m5t_3_r",
        "TL_MF_m1_laminar_r", "TL_MF_m2_laminar_r", "TL_MF_m3_laminar_r",
        "TL_MF_m4_laminar_r", "TL_MF_m5_laminar_r",
    ],
    "lumbar_multifidus_L": [
        "TL_MF_m1t_1_l", "TL_MF_m1t_2_l", "TL_MF_m1t_3_l",
        "TL_MF_m1s_l",   "TL_MF_m2s_l",   "TL_MF_m2t_1_l",
        "TL_MF_m2t_2_l", "TL_MF_m2t_3_l", "TL_MF_m3s_l",
        "TL_MF_m3t_1_l", "TL_MF_m3t_2_l", "TL_MF_m3t_3_l",
        "TL_MF_m4s_l",   "TL_MF_m4t_1_l", "TL_MF_m4t_2_l",
        "TL_MF_m4t_3_l", "TL_MF_m5s_l",   "TL_MF_m5t_1_l",
        "TL_MF_m5t_2_l", "TL_MF_m5t_3_l",
        "TL_MF_m1_laminar_l", "TL_MF_m2_laminar_l", "TL_MF_m3_laminar_l",
        "TL_MF_m4_laminar_l", "TL_MF_m5_laminar_l",
    ],

    # ── Psoas (appears in TL model — 11 fascicles per side) ──────────────────
    "psoas_R": [
        "TL_Ps_L1_VB_r",     "TL_Ps_L1_TP_r",    "TL_Ps_L1_L2_IVD_r",
        "TL_Ps_L2_TP_r",     "TL_Ps_L2_L3_IVD_r","TL_Ps_L3_TP_r",
        "TL_Ps_L3_L4_IVD_r", "TL_Ps_L4_TP_r",    "TL_Ps_L4_L5_IVD_r",
        "TL_Ps_L5_TP_r",     "TL_Ps_L5_VB_r",
    ],
    "psoas_L": [
        "TL_Ps_L1_VB_l",     "TL_Ps_L1_TP_l",    "TL_Ps_L1_L2_IVD_l",
        "TL_Ps_L2_TP_l",     "TL_Ps_L2_L3_IVD_l","TL_Ps_L3_TP_l",
        "TL_Ps_L3_L4_IVD_l", "TL_Ps_L4_TP_l",    "TL_Ps_L4_L5_IVD_l",
        "TL_Ps_L5_TP_l",     "TL_Ps_L5_VB_l",
    ],

    # ── Quadratus lumborum (lateral stabilizer) ────────────────────────────
    "quadratus_lumborum_R": [
        "TL_QL_post_I_1-L3_r",  "TL_QL_post_I_2-L4_r",
        "TL_QL_post_I_2-L3_r",  "TL_QL_post_I_2-L2_r",
        "TL_QL_post_I_3-L1_r",  "TL_QL_post_I_3-L2_r",
        "TL_QL_post_I_3-L3_r",  "TL_QL_mid_L3-12_3_r",
        "TL_QL_mid_L3-12_2_r",  "TL_QL_mid_L3-12_1_r",
        "TL_QL_mid_L2-12_1_r",  "TL_QL_mid_L4-12_3_r",
        "TL_QL_ant_I_2-T12_r",  "TL_QL_ant_I_3-T12_r",
        "TL_QL_ant_I_2-12_1_r", "TL_QL_ant_I_3-12_1_r",
        "TL_QL_ant_I_3-12_2_r", "TL_QL_ant_I_3-12_3_r",
    ],
    "quadratus_lumborum_L": [
        "TL_QL_post_I_1-L3_l",  "TL_QL_post_I_2-L4_l",
        "TL_QL_post_I_2-L3_l",  "TL_QL_post_I_2-L2_l",
        "TL_QL_post_I_3-L1_l",  "TL_QL_post_I_3-L2_l",
        "TL_QL_post_I_3-L3_l",  "TL_QL_mid_L3-12_3_l",
        "TL_QL_mid_L3-12_2_l",  "TL_QL_mid_L3-12_1_l",
        "TL_QL_mid_L2-12_1_l",  "TL_QL_mid_L4-12_3_l",
        "TL_QL_ant_I_2-T12_l",  "TL_QL_ant_I_3-T12_l",
        "TL_QL_ant_I_2-12_1_l", "TL_QL_ant_I_3-12_1_l",
        "TL_QL_ant_I_3-12_2_l", "TL_QL_ant_I_3-12_3_l",
    ],

    # ── Neck extensors ──────────────────────────────────────────────────────
    "splenius_capitis_R": ["TL_splen_cap_sklT3"],
    "splenius_capitis_L": ["TL_splen_cap_sklT3_L"],
    "splenius_cervicis_R": [
        "TL_splen_cerv_c3_T3", "TL_splen_cerv_c3_T4",
        "TL_splen_cerv_c3_T5", "TL_splen_cerv_c3_T6",
    ],
    "splenius_cervicis_L": [
        "TL_splen_cerv_c3_T3_L", "TL_splen_cerv_c3_T4_L",
        "TL_splen_cerv_c3_T5_L", "TL_splen_cerv_c3_T6_L",
    ],
    "semispinalis_capitis_R": ["TL_semi_cap_sklthx"],
    "semispinalis_capitis_L": ["TL_semi_cap_sklthx_L"],
    "semispinalis_cervicis_R": ["TL_semi_cerv_c3thx"],
    "semispinalis_cervicis_L": ["TL_semi_cerv_c3thx_L"],
    "longissimus_cervicis_R": ["TL_longissi_cerv_c4thx"],
    "longissimus_cervicis_L": ["TL_longissi_cerv_c4thx_L"],
    "iliocostalis_cervicis_R": ["TL_iliocost_cerv_c5rib"],
    "iliocostalis_cervicis_L": ["TL_iliocost_cerv_c5rib_L"],
    "levator_scapulae_R": ["TL_levator_scap"],
    "levator_scapulae_L": ["TL_levator_scap_L"],

    # Pooled neck extensors
    "neck_extensors_R": [
        "TL_semi_cap_sklthx",      "TL_splen_cap_sklT3",
        "TL_semi_cerv_c3thx",      "TL_longissi_cerv_c4thx",
        "TL_splen_cerv_c3_T3",     "TL_splen_cerv_c3_T4",
        "TL_splen_cerv_c3_T5",     "TL_splen_cerv_c3_T6",
        "TL_iliocost_cerv_c5rib",
    ],
    "neck_extensors_L": [
        "TL_semi_cap_sklthx_L",    "TL_splen_cap_sklT3_L",
        "TL_semi_cerv_c3thx_L",    "TL_longissi_cerv_c4thx_L",
        "TL_splen_cerv_c3_T3_L",   "TL_splen_cerv_c3_T4_L",
        "TL_splen_cerv_c3_T5_L",   "TL_splen_cerv_c3_T6_L",
        "TL_iliocost_cerv_c5rib_L",
    ],

    # ── Neck flexors / deep cervical ─────────────────────────────────────────
    "sternocleidomastoid_R": ["TL_stern_mast", "TL_cleid_mast"],
    "sternocleidomastoid_L": ["TL_stern_mast_L", "TL_cleid_mast_L"],
    "scalenus_R": ["TL_scal_ant", "TL_scal_med", "TL_scal_post"],
    "scalenus_L": ["TL_scal_ant_L", "TL_scal_med_L", "TL_scal_post_L"],
    "longus_colli_R": [
        "TL_long_col_VB_C3C4", "TL_long_col_VB_C4C5",
        "TL_long_col_VB_C5C6", "TL_long_col_VB_C6C7",
        "TL_long_col_VB_C7T1", "TL_long_col_obliq_sup",
        "TL_long_col_obliq_inf",
    ],
    "longus_colli_L": [
        "TL_long_col_VB_C3C4_L", "TL_long_col_VB_C4C5_L",
        "TL_long_col_VB_C5C6_L", "TL_long_col_VB_C6C7_L",
        "TL_long_col_VB_C7T1_L", "TL_long_col_obliq_sup_L",
        "TL_long_col_obliq_inf_L",
    ],

    # Pooled deep neck flexors (key for FHP)
    "deep_neck_flexors_R": [
        "TL_long_col_VB_C3C4", "TL_long_col_VB_C4C5",
        "TL_long_col_VB_C5C6", "TL_long_col_VB_C6C7",
        "TL_long_col_VB_C7T1", "TL_long_col_obliq_sup",
        "TL_long_col_obliq_inf",
    ],
    "deep_neck_flexors_L": [
        "TL_long_col_VB_C3C4_L", "TL_long_col_VB_C4C5_L",
        "TL_long_col_VB_C5C6_L", "TL_long_col_VB_C6C7_L",
        "TL_long_col_VB_C7T1_L", "TL_long_col_obliq_sup_L",
        "TL_long_col_obliq_inf_L",
    ],
}

# ---------------------------------------------------------------------------
# Master dict — single entry point used by loss functions
# ---------------------------------------------------------------------------

ROLLUP_GROUPS: Dict[str, List[str]] = {**_LU, **_TL}

# ---------------------------------------------------------------------------
# Convenience aliases used directly in the prior table
# ---------------------------------------------------------------------------

ROLLUP_ALIASES: Dict[str, List[str]] = {
    # Hip extensor synergy (for APT loss)
    # 原来的可能有问题：
    "hip_extensors_R": list(ROLLUP_GROUPS["gluteus_maximus_R"]) + list(ROLLUP_GROUPS["hamstrings_R"]),
    "hip_extensors_L": (
        ROLLUP_GROUPS["gluteus_maximus_L"] + ROLLUP_GROUPS["hamstrings_L"]
    ),
    # Hip abductor synergy (for APT valgus loss)
    "hip_abductors_R": (
        ROLLUP_GROUPS["gluteus_medius_R"]
        + ROLLUP_GROUPS["gluteus_minimus_R"]
        + ROLLUP_GROUPS["tfl_R"]
    ),
    "hip_abductors_L": (
        ROLLUP_GROUPS["gluteus_medius_L"]
        + ROLLUP_GROUPS["gluteus_minimus_L"]
        + ROLLUP_GROUPS["tfl_L"]
    ),
    # Deep core stabilizers
    "deep_core_R": (
        ROLLUP_GROUPS["transversus_abdominis_R"]
        + ROLLUP_GROUPS["lumbar_multifidus_R"]
    ),
    "deep_core_L": (
        ROLLUP_GROUPS["transversus_abdominis_L"]
        + ROLLUP_GROUPS["lumbar_multifidus_L"]
    ),
    # Superficial abdominals (global movers)
    "superficial_abdominals_R": (
        ROLLUP_GROUPS["rectus_abdominis_R"]
        + ROLLUP_GROUPS["external_oblique_R"]
    ),
    "superficial_abdominals_L": (
        ROLLUP_GROUPS["rectus_abdominis_L"]
        + ROLLUP_GROUPS["external_oblique_L"]
    ),
    # Iliopsoas (combined hip flexor from both LU and TL models)
    "iliopsoas_R": ROLLUP_GROUPS["iliacus_R"] + ROLLUP_GROUPS["psoas_R"],
    "iliopsoas_L": ROLLUP_GROUPS["iliacus_L"] + ROLLUP_GROUPS["psoas_L"],
}

ROLLUP_GROUPS.update(ROLLUP_ALIASES)


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------

def get_indices(group_name: str, mint_columns: List[str]) -> List[int]:
    """Return integer indices of a group's fascicles within mint_columns.

    Parameters
    ----------
    group_name : str
        Key in ROLLUP_GROUPS.
    mint_columns : list[str]
        Ordered list of all 402 MinT column names (from the DataFrame header).

    Returns
    -------
    list[int]
        Indices into the 402-dim activation vector.
    """
    col_to_idx = {c: i for i, c in enumerate(mint_columns)}
    fascicles = ROLLUP_GROUPS[group_name]
    indices = []
    for f in fascicles:
        if f in col_to_idx:
            indices.append(col_to_idx[f])
        else:
            import warnings
            warnings.warn(f"Fascicle '{f}' not found in mint_columns — skipped.")
    return indices


def rollup(
    activation: np.ndarray,
    mint_columns: List[str],
    groups: List[str] | None = None,
) -> tuple[np.ndarray, List[str]]:
    """Mean-pool a (T, 402) activation array into a (T, G) grouped array.

    Parameters
    ----------
    activation : np.ndarray, shape (T, 402)
    mint_columns : list[str]  — ordered column names
    groups : list[str] | None
        Which groups to include (default = all ROLLUP_GROUPS keys).

    Returns
    -------
    grouped : np.ndarray, shape (T, G)
    group_names : list[str]  — same order as grouped columns
    """
    if groups is None:
        groups = list(ROLLUP_GROUPS.keys())

    out = []
    for g in groups:
        idx = get_indices(g, mint_columns)
        if len(idx) == 0:
            out.append(np.zeros(activation.shape[0]))
        else:
            out.append(activation[:, idx].mean(axis=1))

    return np.stack(out, axis=1), groups


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total = sum(len(v) for v in ROLLUP_GROUPS.values())
    print(f"Number of functional groups defined: {len(ROLLUP_GROUPS)}")
    for name, fascicles in sorted(ROLLUP_GROUPS.items()):
        print(f"  {name:40s}  {len(fascicles):3d} fascicles")
