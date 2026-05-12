#!/usr/bin/env bash
# new/run_param_sweep.sh
#
# Parameter sweep over s (guidance strength) and schedule for the BEST
# variant identified by run_ablation.sh.
#
# Recommended after you've picked a winner. Set BEST_VARIANT before running.
#
# Usage:
#   BEST_VARIANT=v2_dps bash new/run_param_sweep.sh

set -e

BEST_VARIANT="${BEST_VARIANT:-v1_mu_sgd}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
SEED="${SEED:-42}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TAG="${TAG:-sweep}"

OUT_BASE="./output/sweep_${BEST_VARIANT}_${TAG}_seed${SEED}"
mkdir -p "${OUT_BASE}"

echo "============================================================"
echo "  Parameter sweep on variant: ${BEST_VARIANT}"
echo "  Prompt : ${TEXT_PROMPT}"
echo "  Posture: ${POSTURE}"
echo "============================================================"

# ============================================================
#  Sweep 1: schedule
# ============================================================
echo -e "\n[Sweep 1/2] schedule ..."
for SCHED in always decay last_quarter second_half final; do
  echo -e "\n  -- schedule=${SCHED} --"

  case "${BEST_VARIANT}" in
    v1_mu_sgd|v4_omni)
      KW='{"n_inner_steps":15,"lr":0.5,"schedule":"'${SCHED}'","base_weight":30.0}'
      ;;
    v2_dps|v5_lgd)
      KW='{"s":1.0,"schedule":"'${SCHED}'","base_weight":1.0}'
      ;;
    v3_x0_direct)
      KW='{"n_inner_steps":5,"lr":0.05,"schedule":"'${SCHED}'","base_weight":1.0}'
      ;;
  esac

  GUIDANCE_VARIANT="${BEST_VARIANT}" \
  GUIDANCE_KWARGS_JSON="${KW}" \
  TEXT_PROMPT="${TEXT_PROMPT}" \
  POSTURE="${POSTURE}" \
  SEED="${SEED}" \
  MOTION_LENGTH="${MOTION_LENGTH}" \
  OUTPUT_DIR="${OUT_BASE}/sched_${SCHED}" \
  MAKE_ANIMATION="" \
  ./new/run_posture_pipeline.sh
done

# ============================================================
#  Sweep 2: s / base_weight (different scales for diff variants)
# ============================================================
echo -e "\n[Sweep 2/2] guidance strength ..."

case "${BEST_VARIANT}" in
  v1_mu_sgd|v4_omni)
    STRENGTHS=(5.0 10.0 30.0 50.0 100.0)
    KW_KEY="base_weight"
    ;;
  v2_dps|v5_lgd)
    STRENGTHS=(0.5 1.0 5.0 10.0 50.0)
    KW_KEY="s"
    ;;
  v3_x0_direct)
    STRENGTHS=(0.01 0.05 0.1 0.5 1.0)
    KW_KEY="lr"
    ;;
esac

for S in "${STRENGTHS[@]}"; do
  echo -e "\n  -- ${KW_KEY}=${S} --"

  case "${BEST_VARIANT}" in
    v1_mu_sgd|v4_omni)
      KW='{"n_inner_steps":15,"lr":0.5,"schedule":"final","base_weight":'${S}'}'
      ;;
    v2_dps)
      KW='{"s":'${S}',"schedule":"always","base_weight":1.0}'
      ;;
    v5_lgd)
      KW='{"n_mc":4,"s":'${S}',"schedule":"always","base_weight":1.0}'
      ;;
    v3_x0_direct)
      KW='{"n_inner_steps":5,"lr":'${S}',"schedule":"always","base_weight":1.0}'
      ;;
  esac

  GUIDANCE_VARIANT="${BEST_VARIANT}" \
  GUIDANCE_KWARGS_JSON="${KW}" \
  TEXT_PROMPT="${TEXT_PROMPT}" \
  POSTURE="${POSTURE}" \
  SEED="${SEED}" \
  MOTION_LENGTH="${MOTION_LENGTH}" \
  OUTPUT_DIR="${OUT_BASE}/${KW_KEY}_${S}" \
  MAKE_ANIMATION="" \
  ./new/run_posture_pipeline.sh
done

echo -e "\n============================================================"
echo "  Param sweep done. Evaluate:"
echo "    python -m new.evaluate_sweep ${OUT_BASE}"
echo "============================================================"