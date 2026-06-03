#!/usr/bin/env bash
# new/run_sdedit_sweep.sh
#
# 路线 B（IK + SDEdit）t0 扫描实验：
#   对膝超伸 190°（全局 OOD），用 IK 注入超伸结构 + SDEdit 重去噪，
#   扫 t0_ratio ∈ {0.3,0.5,0.7} 找「保住超伸 ↔ 自然性」甜点。
#   每个 t0 跑两个 variant：纯 SDEdit / SDEdit+V6 PID hybrid。
#
# 用法：
#   bash new/run_sdedit_sweep.sh [N_SEEDS]
#       N_SEEDS 默认 5
#
# 可调环境变量：
#   T0_RATIOS="0.3 0.5 0.7"   要扫的 t0_ratio 列表
#   TARGET_DEG=190            IK 目标 signed 膝角
#   MIN_BASE_DEG=150          仅超伸三点角 > 此值的帧
#   TEXT_PROMPT / MOTION_LENGTH / MODEL_PATH
#
# 输出：
#   ./output/sdedit_knee_<variant>_t<ratio>/seed<seed>/comparison.npy
# 聚合：
#   python -m new.aggregate_seeds ./output/sdedit_knee_*

set -e

N_SEEDS="${1:-5}"
MODEL_PATH="${MODEL_PATH:-./save/humanml_trans_dec_512_bert/model000200000.pt}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TARGET_DEG="${TARGET_DEG:-190}"
MIN_BASE_DEG="${MIN_BASE_DEG:-150}"
DATASET="${DATASET:-humanml}"
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"

read -r -a T0_RATIOS <<< "${T0_RATIOS:-0.3 0.5 0.7}"

ALL_SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

# 纯 SDEdit 用 v1 占位（不传 posture_instructions → guidance 不触发）；
# hybrid 用 v6_closed_loop（需传 posture_instructions 膝超伸）。
V6_KWARGS='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter"}'

echo "================================================="
echo "  路线 B SDEdit 扫描：膝超伸 ${TARGET_DEG}°"
echo "  t0_ratios = ${T0_RATIOS[*]}    N_SEEDS = ${N_SEEDS}"
echo "  model = ${MODEL_PATH}"
echo "================================================="

run_one () {
    local VARIANT_NAME="$1" T0="$2" SEED="$3"
    local OUT="./output/sdedit_knee_${VARIANT_NAME}_t${T0}/seed${SEED}"
    if [ -f "${OUT}/comparison.npy" ]; then
        echo "   [${VARIANT_NAME} t0=${T0} seed=${SEED}] CACHED, skip"; return
    fi
    mkdir -p "${OUT}"
    echo "   [${VARIANT_NAME} t0=${T0} seed=${SEED}] running..."

    if [ "${VARIANT_NAME}" = "pure" ]; then
        # 纯 SDEdit：不传 posture_instructions
        ${PYTHON} -m new.sdEdit_ood \
            --model_path "${MODEL_PATH}" --text_prompt "${TEXT_PROMPT}" \
            --t0_ratio "${T0}" --target_signed_deg "${TARGET_DEG}" \
            --min_base_deg "${MIN_BASE_DEG}" \
            --num_samples 1 --motion_length "${MOTION_LENGTH}" \
            --seed "${SEED}" --dataset "${DATASET}" --device "${DEVICE}" \
            --output_dir "${OUT}"
    else
        # SDEdit + V6 PID hybrid
        GUIDANCE_VARIANT=v6_closed_loop GUIDANCE_KWARGS_JSON="${V6_KWARGS}" \
        ${PYTHON} -m new.sdEdit_ood \
            --model_path "${MODEL_PATH}" --text_prompt "${TEXT_PROMPT}" \
            --posture_instructions 膝超伸 \
            --t0_ratio "${T0}" --target_signed_deg "${TARGET_DEG}" \
            --min_base_deg "${MIN_BASE_DEG}" \
            --num_samples 1 --motion_length "${MOTION_LENGTH}" \
            --seed "${SEED}" --dataset "${DATASET}" --device "${DEVICE}" \
            --output_dir "${OUT}"
    fi
}

for T0 in "${T0_RATIOS[@]}"; do
    for VAR in pure v6hybrid; do
        for SEED in "${SEEDS[@]}"; do
            run_one "${VAR}" "${T0}" "${SEED}"
        done
    done
done

echo ""
echo "================================================="
echo "  SDEdit 扫描完成。聚合："
echo "    python -m new.aggregate_seeds ./output/sdedit_knee_*"
echo ""
echo "  关注指标（每个 comparison.npy 的 final_knee_deg / motion_xyz_guided）："
echo "    - 实际膝角是否保住 > 185°（vs 纯 guidance 的 ~175°）"
echo "    - 动作自然性（FID / 肉眼）随 t0_ratio 的变化"
echo "    - v6hybrid vs pure：PID 是否在 SDEdit 基础上进一步保住超伸"
echo "================================================="
