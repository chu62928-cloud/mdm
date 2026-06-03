#!/usr/bin/env bash
# new/run_sdedit_stability_sweep.sh
#
# 路线 B 第二阶段：稳定性修复 + Route C 扫描
#
# 第一轮结果（t0 sweep）关键发现：
#   - 纯 SDEdit：完全失败（MDM 去噪抹掉 IK 结构）
#   - v6hybrid t0=0.3：hit_band=14.8%（最佳）但 2/5 seeds 过度推力，CV(Δ)=22.1%
#   - v6hybrid t0=0.5：稳定（CV=8.8%）但 hit_band=1.7%（过低）
#   - v6hybrid t0=0.7：NaN，发散
#
# 根本原因（t0=0.3 不稳定）：
#   spec_schedule_override="last_quarter" 在 SDEdit(300 步)中覆盖 250/300=83% 步，
#   高噪声步（t=250-300）梯度方向不可靠 → PID 积分失控 → 过推。
#
# 本轮实验矩阵：
#   t0 = 0.30（不稳定区）× 6 变体
#        0.35 / 0.40（甜点探索）× 1 变体（stable）
#        0.50（稳定基线参照）× 1 变体（stable）
#
# 变体说明：
#   base        — 复现第一轮 v6hybrid（对照）
#   delta_max   — +delta_max=5.0（单步限幅）
#   sigma_cut   — +sigma_cutoff=0.4（跳过高噪步）
#   stable      — delta_max + sigma_cutoff（组合）
#   sup01       — stable + score_suppress_ratio=0.1（Route C 弱）
#   sup02       — stable + score_suppress_ratio=0.2（Route C 中）
#
# 用法：
#   bash new/run_sdedit_stability_sweep.sh [N_SEEDS]
#       N_SEEDS 默认 8
#
# 输出：
#   ./output/sdedit_stab_<variant>_t<t0>/seed<seed>/comparison.npy
# 聚合：
#   python -m new.aggregate_seeds ./output/sdedit_stab_*

set -e

N_SEEDS="${1:-8}"
MODEL_PATH="${MODEL_PATH:-./save/humanml_trans_dec_512_bert/model000200000.pt}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TARGET_DEG="${TARGET_DEG:-190}"
MIN_BASE_DEG="${MIN_BASE_DEG:-150}"
DATASET="${DATASET:-humanml}"
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"

ALL_SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

# ---- V6 kwargs（用 declare -A 避免 JSON 冒号与字段分隔符冲突） ----
declare -A KWARGS_MAP

KWARGS_MAP["base"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter"}'

KWARGS_MAP["delta_max"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":5.0}'

KWARGS_MAP["sigma_cut"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","sigma_cutoff":0.4}'

KWARGS_MAP["stable"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":5.0,"sigma_cutoff":0.4}'

KWARGS_MAP["sup01"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":5.0,"sigma_cutoff":0.4,"score_suppress_ratio":0.1}'

KWARGS_MAP["sup02"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":5.0,"sigma_cutoff":0.4,"score_suppress_ratio":0.2}'

# ---- 实验矩阵：(variant, t0) pairs ----
# 格式：每个元素 = "variant_name t0_value"
EXPERIMENTS=(
    "base    0.30"
    "delta_max 0.30"
    "sigma_cut 0.30"
    "stable  0.30"
    "sup01   0.30"
    "sup02   0.30"
    "stable  0.35"
    "stable  0.40"
    "stable  0.50"
)

echo "================================================="
echo "  路线 B 第二阶段：稳定性修复 + Route C 扫描"
echo "  目标膝角 = ${TARGET_DEG}°"
echo "  N_SEEDS = ${N_SEEDS}"
echo "  实验数 = ${#EXPERIMENTS[@]} × ${N_SEEDS} seeds = $(( ${#EXPERIMENTS[@]} * N_SEEDS ))"
echo "================================================="

run_one () {
    local VNAME="$1" T0="$2" SEED="$3"
    local KWARGS="${KWARGS_MAP[$VNAME]}"
    local T0TAG="${T0/./}"
    local OUT="./output/sdedit_stab_${VNAME}_t${T0TAG}/seed${SEED}"

    if [ -f "${OUT}/comparison.npy" ]; then
        echo "   [${VNAME} t0=${T0} seed=${SEED}] CACHED, skip"; return
    fi
    mkdir -p "${OUT}"
    echo "   [${VNAME} t0=${T0} seed=${SEED}] running..."

    GUIDANCE_VARIANT=v6_closed_loop GUIDANCE_KWARGS_JSON="${KWARGS}" \
    ${PYTHON} -m new.sdEdit_ood \
        --model_path "${MODEL_PATH}" --text_prompt "${TEXT_PROMPT}" \
        --posture_instructions 膝超伸 \
        --t0_ratio "${T0}" --target_signed_deg "${TARGET_DEG}" \
        --min_base_deg "${MIN_BASE_DEG}" \
        --num_samples 1 --motion_length "${MOTION_LENGTH}" \
        --seed "${SEED}" --dataset "${DATASET}" --device "${DEVICE}" \
        --output_dir "${OUT}"
}

for EXP in "${EXPERIMENTS[@]}"; do
    read -r VNAME T0 <<< "${EXP}"
    echo ""
    echo "-- [${VNAME} t0=${T0}]  kwargs=${KWARGS_MAP[$VNAME]:0:60}..."
    for SEED in "${SEEDS[@]}"; do
        run_one "${VNAME}" "${T0}" "${SEED}"
    done
done

echo ""
echo "================================================="
echo "  稳定性扫描完成。聚合："
echo "    python -m new.aggregate_seeds ./output/sdedit_stab_*"
echo ""
echo "  关注指标："
echo "    - CV(Δ) < 15%     → 稳定性达标（第一轮 base t03 = 22.1%）"
echo "    - hit_band ≥ 10%  → 角度目标可达（第一轮 base t03 = 14.8%）"
echo "    - NaN 比例 = 0    → 完全稳定"
echo ""
echo "  预期最优：stable_t030 或 sup01_t030"
echo "    若 score_suppress 提升 hit_band → Route C 对 OOD 生成有效"
echo "    若 stable_t030 CV < 10% 且 hit_band ≈ base → 稳定+性能两全"
echo "================================================="
