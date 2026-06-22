#!/bin/bash
# =============================================================================
# run_posture_pipeline_v2.sh
# Extended pipeline: supports joint / muscle / both modes.
# Each invocation generates ONE sample (baseline + guided comparison).
#
# Usage:
#   MODE=joint   POSTURE=anterior_pelvic_tilt SEED=42 bash new/run_posture_pipeline_v2.sh
#   MODE=muscle  POSTURE=anterior_pelvic_tilt SEED=42 MUSCLE_CKPT=path/to/net_best_loss.pth bash new/run_posture_pipeline_v2.sh
#   MODE=both    POSTURE=anterior_pelvic_tilt SEED=42 MUSCLE_CKPT=path/to/net_best_loss.pth bash new/run_posture_pipeline_v2.sh
# =============================================================================

set -e

# ---- User config ----
MODE="${MODE:-joint}"
POSTURE="${POSTURE:-anterior_pelvic_tilt}"
SEED="${SEED:-42}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

# ---- Paths ----
PROJECT_ROOT="/root/autodl-tmp/motion-diffusion-model"
MODEL_PATH="${MODEL_PATH:-./save/humanml_trans_dec_512_bert/model000600000.pt}"
MUSCLE_CKPT="${MUSCLE_CKPT:-motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth}"
# Both joint and muscle modules now use the same English posture name via registry aliases
MUSCLE_POSTURE="${MUSCLE_POSTURE:-${POSTURE}}"
MUSCLE_ASSETS_DIR="${MUSCLE_ASSETS_DIR:-motion2muscle}"

# ---- Guidance config ----
export GUIDANCE_VARIANT="${GUIDANCE_VARIANT:-v2_dps}"
export GUIDANCE_KWARGS_JSON="${GUIDANCE_KWARGS_JSON:-{\"s\":40,\"schedule\":\"last_quarter\"}}"
export GUIDANCE_DIAGNOSTIC="${GUIDANCE_DIAGNOSTIC:-1}"

# ---- Output ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-./output/posture_pipeline/${MODE}_${POSTURE}_seed${SEED}_${TIMESTAMP}}"

# ---- Viz config ----
MAKE_ANIMATION="${MAKE_ANIMATION:-mp4}"
MAKE_MULTIVIEW="${MAKE_MULTIVIEW:-yes}"
MAKE_ANATOMICAL="${MAKE_ANATOMICAL:-yes}"

# ---- Internal ----
cd "${PROJECT_ROOT}"
VIZ_DIR="${OUTPUT_DIR}/viz"
NP_PATH="${OUTPUT_DIR}/comparison.npy"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
mkdir -p "${OUTPUT_DIR}" "${VIZ_DIR}"

log() { echo -e "[$(date '+%H:%M:%S')] $1" | tee -a "${LOG_FILE}"; }

echo ""
echo "============================================================"
echo "  Posture Pipeline V2"
echo "  MODE=${MODE}  POSTURE=${POSTURE}  SEED=${SEED}"
echo "  OUTPUT=${OUTPUT_DIR}"
echo "============================================================"
echo ""

log "Config: MODE=${MODE} POSTURE=${POSTURE} SEED=${SEED}"
log "Config: MODEL=${MODEL_PATH}"
log "Config: GUIDANCE_VARIANT=${GUIDANCE_VARIANT}"

# =============================================================================
# Step 1: Generate comparison.npy (baseline + guided)
# =============================================================================
log "Step 1: Generating comparison data (mode=${MODE})..."

MUSCLE_FLAGS=""
if [ "${MODE}" = "muscle" ] || [ "${MODE}" = "both" ]; then
    MUSCLE_FLAGS="--muscle_ckpt ${MUSCLE_CKPT} --muscle_posture ${MUSCLE_POSTURE} --muscle_assets_dir ${MUSCLE_ASSETS_DIR}"
    log "  Muscle flags: ${MUSCLE_FLAGS}"
fi

python -m scripts.run_posture_comparison \
    --model_path "${MODEL_PATH}" \
    --text_prompt "${TEXT_PROMPT}" \
    --posture_instructions ${POSTURE} \
    --guidance_mode "${MODE}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_repetitions 1 \
    --motion_length "${MOTION_LENGTH}" \
    --seed "${SEED}" \
    --dataset humanml \
    --device 0 \
    --output_dir "${OUTPUT_DIR}" \
    --comparison_output "comparison.npy" \
    --posture_lbfgs_steps 8 \
    --posture_lr 0.05 \
    ${MUSCLE_FLAGS} \
    2>&1 | tee -a "${LOG_FILE}"

[ -f "${NP_PATH}" ] || { log "ERROR: comparison.npy not found!"; exit 1; }
log "Step 1 OK: ${NP_PATH}"

# =============================================================================
# Step 2: Quantitative comparison
# =============================================================================
log "Step 2: Quantitative comparison..."
python -m scripts.quantitative_compare "${NP_PATH}" 2>&1 | tee -a "${LOG_FILE}" | tee "${OUTPUT_DIR}/comparison_report.txt"
log "Step 2 OK"

# =============================================================================
# Step 3: Angle curves visualization
# =============================================================================
log "Step 3: Angle curves + animation..."
ANIM_FLAG=""
[ -n "${MAKE_ANIMATION}" ] && ANIM_FLAG="--anim_fmt ${MAKE_ANIMATION}"
python -m scripts.visualize_compare "${NP_PATH}" --output_dir "${OUTPUT_DIR}" ${ANIM_FLAG} 2>&1 | tee -a "${LOG_FILE}"
log "Step 3 OK"

# =============================================================================
# Step 4: Multiview animation
# =============================================================================
if [ "${MAKE_MULTIVIEW}" = "yes" ] && [ -n "${MAKE_ANIMATION}" ]; then
    log "Step 4: Multiview animation..."
    python -m scripts.make_multiview_animation \
        "${NP_PATH}" \
        --output "${VIZ_DIR}/multiview.${MAKE_ANIMATION}" \
        --fmt "${MAKE_ANIMATION}" \
        --zoom 1.4 --line_width 3.5 \
        2>&1 | tee -a "${LOG_FILE}"
    log "Step 4 OK"
fi

# =============================================================================
# Step 5: Anatomical animation
# =============================================================================
if [ "${MAKE_ANATOMICAL}" = "yes" ] && [ -n "${MAKE_ANIMATION}" ]; then
    log "Step 5: Anatomical animation..."
    python -m scripts.make_anatomical_animation \
        "${NP_PATH}" \
        --output "${VIZ_DIR}/anatomical.${MAKE_ANIMATION}" \
        --fmt "${MAKE_ANIMATION}" \
        2>&1 | tee -a "${LOG_FILE}"
    log "Step 5 OK"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "  Output: ${OUTPUT_DIR}"
echo "  Files:"
find "${OUTPUT_DIR}" -type f | sort | while read f; do echo "    $f"; done
echo "============================================================"
