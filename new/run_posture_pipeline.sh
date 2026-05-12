#!/bin/bash
# =============================================================================
# run_posture_pipeline.sh  (v3)
#
# 改进：
#   1. 修复 viz/viz/ 嵌套问题（visualize_compare 内部会拼一次 viz/）
#   2. step 4 改用论文风格的 plot_motion_cycle_v4（彩色四肢 + 大骨架）
#   3. step 6 改用论文风格的 make_multiview_animation_v2（无圆点 + 彩色）
#   4. 新增 SKELETON_SCALE / LINE_WIDTH 等参数控制可视化风格
# =============================================================================

set -e

# =============================================================================
# 用户配置
# =============================================================================
export GUIDANCE_VARIANT="${GUIDANCE_VARIANT:-v1_mu_sgd}"
export GUIDANCE_KWARGS_JSON="${GUIDANCE_KWARGS_JSON}"
export DIAGNOSTIC="${DIAGNOSTIC:-0}"

PROJECT_ROOT="/root/autodl-tmp/motion-diffusion-model"
MODEL_PATH="./save/humanml_trans_dec_512_bert/model000200000.pt"
DATASET="humanml"

TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
SEED="${SEED:-42}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
LBFGS_STEPS="${LBFGS_STEPS:-8}"
LR="${LR:-0.05}"

OUTPUT_DIR="${OUTPUT_DIR:-./new_results/posture_exp_$(date +%Y%m%d_%H%M%S)}"

N_KEYFRAMES="${N_KEYFRAMES:-7}"
CYCLE_LABEL="${CYCLE_LABEL:-Gait Cycle}"
BASELINE_LABEL="${BASELINE_LABEL:-Baseline}"
GUIDED_LABEL="${GUIDED_LABEL:-Ours}"
HIGHLIGHT_FRAME="${HIGHLIGHT_FRAME:-4}"

# 可视化风格参数
SKELETON_SCALE="${SKELETON_SCALE:-1.5}"   # 骨架放大倍数
LINE_WIDTH="${LINE_WIDTH:-3.5}"            # 骨骼线宽
ZOOM="${ZOOM:-1.4}"                        # 多视角动画的放大倍数

DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"

MAKE_ANIMATION="${MAKE_ANIMATION:-mp4}"
MAKE_MULTIVIEW="${MAKE_MULTIVIEW:-yes}"
MAKE_ANATOMICAL="${MAKE_ANATOMICAL:-yes}"

# =============================================================================
# 内部变量
# =============================================================================

NP_PATH="${OUTPUT_DIR}/comparison.npy"
VIZ_DIR="${OUTPUT_DIR}/viz"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"; }
log_ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}" | tee -a "${LOG_FILE}"; }
log_warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}" | tee -a "${LOG_FILE}"; }
log_err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}" | tee -a "${LOG_FILE}"; }

check_file() {
    [ -f "$1" ] || { log_err "文件不存在: $1"; exit 1; }
}

# =============================================================================
# Step 0: 初始化
# =============================================================================

echo ""
echo "============================================================"
echo "  Posture Guidance Pipeline (v3)"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${VIZ_DIR}"
echo "Pipeline started at $(date)" > "${LOG_FILE}"

log "配置信息："
log "  TEXT_PROMPT      = ${TEXT_PROMPT}"
log "  POSTURE          = ${POSTURE}"
log "  SEED             = ${SEED}"
log "  MOTION_LENGTH    = ${MOTION_LENGTH}s"
log "  OUTPUT_DIR       = ${OUTPUT_DIR}"
log "  LBFGS_STEPS      = ${LBFGS_STEPS}"
log "  LR               = ${LR}"
log "  SKELETON_SCALE   = ${SKELETON_SCALE}"
log "  LINE_WIDTH       = ${LINE_WIDTH}"

cd "${PROJECT_ROOT}"
check_file "${MODEL_PATH}"

# =============================================================================
# Step 1: 生成 comparison.npy
# =============================================================================

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Step 1/7: 生成 comparison.npy"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

IFS=' ' read -r -a POSTURE_ARRAY <<< "${POSTURE}"
POSTURE_ARGS=""
for p in "${POSTURE_ARRAY[@]}"; do
    POSTURE_ARGS="${POSTURE_ARGS} ${p}"
done

${PYTHON} -m new.run_posture_comparison \
    --model_path "${MODEL_PATH}" \
    --text_prompt "${TEXT_PROMPT}" \
    --posture_instructions ${POSTURE_ARGS} \
    --num_samples "${NUM_SAMPLES}" \
    --num_repetitions 1 \
    --motion_length "${MOTION_LENGTH}" \
    --seed "${SEED}" \
    --dataset "${DATASET}" \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}" \
    --comparison_output "comparison.npy" \
    --posture_lbfgs_steps "${LBFGS_STEPS}" \
    --posture_lr "${LR}" \
    2>&1 | tee -a "${LOG_FILE}"

check_file "${NP_PATH}"
log_ok "comparison.npy → ${NP_PATH}"

# =============================================================================
# Step 2: 量化分析
# =============================================================================

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Step 2/7: 量化对比分析"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

REPORT_PATH="${OUTPUT_DIR}/comparison_report.txt"

${PYTHON} -m new.quantitative_compare \
    "${NP_PATH}" \
    2>&1 | tee -a "${LOG_FILE}" | tee "${REPORT_PATH}"

log_ok "量化报告 → ${REPORT_PATH}"

# =============================================================================
# Step 3: 角度时间曲线 + 标准动画
# =============================================================================

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Step 3/7: 角度时间曲线 + 标准动画"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "${MAKE_ANIMATION}" ]; then
    ANIM_FLAG="--no_animation"
else
    ANIM_FLAG="--anim_fmt ${MAKE_ANIMATION}"
fi

# ★ 修复：传 OUTPUT_DIR 而不是 VIZ_DIR，避免 visualize_compare 内部再拼一次 viz/
${PYTHON} -m new.visualize_compare \
    "${NP_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    ${ANIM_FLAG} \
    2>&1 | tee -a "${LOG_FILE}"

log_ok "角度曲线图 → ${VIZ_DIR}/angle_curves.png"

# =============================================================================
# Step 4: 论文风格动作周期图（v4，彩色四肢 + 放大）
# =============================================================================

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Step 4/7: 论文风格动作周期图"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${HIGHLIGHT_FRAME}" -ge 0 ] 2>/dev/null; then
    HIGHLIGHT_FLAG="--highlight_frame ${HIGHLIGHT_FRAME}"
else
    HIGHLIGHT_FLAG=""
fi

MOTION_CYCLE_OUTPUT="${VIZ_DIR}/motion_cycle_paper.png"

${PYTHON} -m new.plot_motion_cycle \
    "${NP_PATH}" \
    --n_keyframes "${N_KEYFRAMES}" \
    --cycle_label "${CYCLE_LABEL}" \
    --baseline_label "${BASELINE_LABEL}" \
    --guided_label "${GUIDED_LABEL}" \
    --output "${MOTION_CYCLE_OUTPUT}" \
    --skeleton_scale "${SKELETON_SCALE}" \
    ${HIGHLIGHT_FLAG} \
    2>&1 | tee -a "${LOG_FILE}"

check_file "${MOTION_CYCLE_OUTPUT}"
log_ok "动作周期图 → ${MOTION_CYCLE_OUTPUT}"

# =============================================================================
# Step 5: 关节差异热图
# =============================================================================

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Step 5/7: 关节差异热图"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

HEATMAP_OUTPUT="${VIZ_DIR}/joint_diff_heatmap.png"
export NP_PATH HEATMAP_OUTPUT

${PYTHON} - << 'PYEOF'
import os, numpy as np, matplotlib.pyplot as plt

npy_path = os.environ["NP_PATH"]
out_path = os.environ["HEATMAP_OUTPUT"]

SMPL_JOINT_NAMES = [
    "pelvis","left_hip","right_hip","spine1","left_knee","right_knee",
    "spine2","left_ankle","right_ankle","spine3","left_foot","right_foot",
    "neck","left_collar","right_collar","head","left_shoulder",
    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
]

data       = np.load(npy_path, allow_pickle=True).item()
xyz_base   = data["motion_xyz"][0]
xyz_guided = data["motion_xyz_guided"][0]

diff = xyz_guided - xyz_base
dist = np.linalg.norm(diff, axis=1)

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(dist, aspect="auto", cmap="OrRd", interpolation="nearest")
ax.set_yticks(range(len(SMPL_JOINT_NAMES)))
ax.set_yticklabels(SMPL_JOINT_NAMES, fontsize=8)
ax.set_xlabel("frame", fontsize=11)
ax.set_title("Per-joint per-frame distance: guided − baseline (meters)",
             fontsize=12)
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("distance (m)", fontsize=10)
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ saved → {out_path}")
PYEOF

log_ok "差异热图 → ${HEATMAP_OUTPUT}"

# =============================================================================
# Step 6: 论文风格多视角动画（v2）
# =============================================================================

if [ -n "${MAKE_ANIMATION}" ] && [ "${MAKE_MULTIVIEW}" = "yes" ]; then
    echo ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Step 6/7: 论文风格多视角动画"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    MULTIVIEW_OUTPUT="${VIZ_DIR}/multiview_paper.${MAKE_ANIMATION}"

    ${PYTHON} -m new.make_multiview_animation \
        "${NP_PATH}" \
        --output "${MULTIVIEW_OUTPUT}" \
        --fmt "${MAKE_ANIMATION}" \
        --zoom "${ZOOM}" \
        --line_width "${LINE_WIDTH}" \
        2>&1 | tee -a "${LOG_FILE}"

    [ -f "${MULTIVIEW_OUTPUT}" ] && \
        log_ok "多视角动画 → ${MULTIVIEW_OUTPUT}" || \
        log_warn "多视角动画生成失败"
else
    echo ""
    log "Step 6/7: 多视角动画（已跳过：MAKE_MULTIVIEW=${MAKE_MULTIVIEW}）"
fi

# =============================================================================
# Step 7: 解剖风格双视图动画
# =============================================================================

if [ -n "${MAKE_ANIMATION}" ] && [ "${MAKE_ANATOMICAL}" = "yes" ]; then
    echo ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Step 7/7: 解剖风格双视图动画"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ANATOMICAL_OUTPUT="${VIZ_DIR}/anatomical_animation.${MAKE_ANIMATION}"

    ${PYTHON} -m new.make_anatomical_animation \
        "${NP_PATH}" \
        --output "${ANATOMICAL_OUTPUT}" \
        --fmt "${MAKE_ANIMATION}" \
        2>&1 | tee -a "${LOG_FILE}"

    [ -f "${ANATOMICAL_OUTPUT}" ] && \
        log_ok "解剖动画 → ${ANATOMICAL_OUTPUT}" || \
        log_warn "解剖动画生成失败"
else
    echo ""
    log "Step 7/7: 解剖动画（已跳过：MAKE_ANATOMICAL=${MAKE_ANATOMICAL}）"
fi

# =============================================================================
# 完成
# =============================================================================

echo ""
echo "============================================================"
echo -e "${GREEN}  Pipeline 完成${NC}"
echo "============================================================"
echo ""
echo "  输出目录：${OUTPUT_DIR}"
echo ""
echo "  生成文件："
echo "    📊 comparison.npy"
echo "    📄 comparison_report.txt"
echo "    📈 viz/angle_curves.png"
echo "    🦴 viz/motion_cycle_paper.png        (论文风格周期图)"
echo "    🌡  viz/joint_diff_heatmap.png"
echo "    🎬 viz/skeleton_keyframes.png"
if [ -n "${MAKE_ANIMATION}" ]; then
echo "    🎥 viz/comparison_animation.${MAKE_ANIMATION}    (标准动画)"
[ "${MAKE_MULTIVIEW}" = "yes" ] && \
echo "    🎥 viz/multiview_paper.${MAKE_ANIMATION}         (论文风格多视角)"
[ "${MAKE_ANATOMICAL}" = "yes" ] && \
echo "    🧍 viz/anatomical_animation.${MAKE_ANIMATION}    (解剖风格)"
fi
echo ""
echo "  查看报告：cat ${REPORT_PATH}"
echo ""

echo "" >> "${LOG_FILE}"
echo "Pipeline finished at $(date)" >> "${LOG_FILE}"