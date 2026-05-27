#!/usr/bin/env bash
# new/run_cross_posture.sh
#
# 跨体态评估：V2_dps_s40_always vs V6_closed_loop_second_half 在多个体态上的表现。
# 自动按 spec.unit 调整 huber_delta（deg→0.05rad, meter→0.01m）。
#
# 用法：
#   bash new/run_cross_posture.sh 膝超伸 [N_SEEDS]
#   bash new/run_cross_posture.sh 驼背    [N_SEEDS]
#
#   POSTURE 必填（位置参数 1），N_SEEDS 默认 5
#
# 体态-单位-参数对照：
#   膝超伸  deg  huber_delta=0.05 rad  ⚠ OOD（190° 不在训练分布内）
#   膝弯曲  deg  huber_delta=0.05 rad  ✓ 分布内（慢走 125° 常见）
#   驼背    m    huber_delta=0.01 m   ⚠ 弱可表征体态（见 POSTURE_REPRESENTABILITY.md）
#   头前伸  m    huber_delta=0.01 m   ⚠ 弱可表征
#   骨盆前倾 deg  huber_delta=0.05 rad（同 N=15 实验）
#
# 输出：
#   ./output/cross_<posture>_<variant>/<seed>/comparison.npy
# 聚合：
#   python -m new.aggregate_seeds ./output/cross_<posture>_*

set -e

if [ -z "$1" ]; then
    echo "用法: bash new/run_cross_posture.sh <体态> [N_SEEDS]"
    echo "支持体态: 膝超伸 | 膝弯曲 | 膝弯曲_A | 膝弯曲_B | 骨盆前倾 | 驼背 | 头前伸"
    exit 1
fi

POSTURE="$1"
N_SEEDS="${2:-5}"

# 单位自动检测 + huber_delta 选择
case "$POSTURE" in
    驼背|头前伸)
        UNIT_TAG="meter"
        HUBER_DELTA="0.01"
        echo "⚠  $POSTURE 是 meter 单位，且为'弱可表征'体态（见 POSTURE_REPRESENTABILITY.md）"
        echo "   预期 hit_band/corr 比强可表征体态差，仅作参照。"
        ;;
    膝超伸|膝弯曲|膝弯曲_A|膝弯曲_B|骨盆前倾|骨盆前倾_深蹲)
        UNIT_TAG="deg"
        HUBER_DELTA="0.05"
        ;;
    *)
        echo "未知体态 '$POSTURE'。退出。"
        echo "支持: 膝超伸 | 膝弯曲 | 骨盆前倾 | 驼背 | 头前伸"
        exit 1
        ;;
esac

# 15 个候选 seed，截取前 N_SEEDS
ALL_SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

# 两个对比 variant — 都用 last_quarter（骨盆前倾 N=15 确认的最佳 schedule）
V2_KWARGS='{"s":40.0,"schedule":"last_quarter","base_weight":1.0}'
V6_KWARGS="{\"Kp\":80,\"Ki\":1,\"Kd\":5,\"s_min\":0.05,\"s_max\":50,\"I_max\":20,\"beta_ema\":0.8,\"lambda_smooth\":0.03,\"manifold_project\":true,\"loss_form\":\"huber\",\"huber_delta\":${HUBER_DELTA},\"normalize_grad\":false,\"band_gate\":false,\"spec_schedule_override\":\"last_quarter\"}"

declare -A VARIANTS=(
    ["v2_dps_s40_last_quarter"]="v2_dps|$V2_KWARGS"
    ["v6_closed_loop_last_quarter"]="v6_closed_loop|$V6_KWARGS"
)

TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

# 输出目录 tag（中文体态名转 ASCII 安全名）
case "$POSTURE" in
    骨盆前倾)     OUT_TAG="apt" ;;
    骨盆前倾_深蹲) OUT_TAG="apt_squat" ;;
    膝超伸)       OUT_TAG="knee" ;;
    膝弯曲)       OUT_TAG="knee_flex" ;;
    膝弯曲_A)     OUT_TAG="knee_flex_a" ;;
    膝弯曲_B)     OUT_TAG="knee_flex_b" ;;
    驼背)         OUT_TAG="kyphosis" ;;
    头前伸)       OUT_TAG="fhp" ;;
    *)            OUT_TAG="$POSTURE" ;;
esac

echo ""
echo "================================================="
echo "  跨体态实验：${POSTURE} (${UNIT_TAG} unit)"
echo "  N_SEEDS = ${N_SEEDS}"
echo "  huber_delta = ${HUBER_DELTA}"
echo "  variants: v2_dps_s40_last_quarter + v6_closed_loop_last_quarter"
echo "================================================="

for CONFIG_NAME in "${!VARIANTS[@]}"; do
    CONFIG="${VARIANTS[$CONFIG_NAME]}"
    VARIANT="${CONFIG%%|*}"
    KWARGS="${CONFIG##*|}"

    OUT_BASE="./output/cross_${OUT_TAG}_${CONFIG_NAME}"
    mkdir -p "${OUT_BASE}"

    echo ""
    echo "-- variant: ${CONFIG_NAME}"
    echo "   kwargs: ${KWARGS}"

    for SEED in "${SEEDS[@]}"; do
        SEED_OUT="${OUT_BASE}/seed${SEED}"
        if [ -f "${SEED_OUT}/comparison.npy" ]; then
            echo "   seed=${SEED} CACHED, skip"
            continue
        fi
        echo "   seed=${SEED}..."
        GUIDANCE_VARIANT="${VARIANT}" \
        GUIDANCE_KWARGS_JSON="${KWARGS}" \
        TEXT_PROMPT="${TEXT_PROMPT}" \
        POSTURE="${POSTURE}" \
        SEED="${SEED}" \
        MOTION_LENGTH="${MOTION_LENGTH}" \
        OUTPUT_DIR="${SEED_OUT}" \
        MAKE_ANIMATION="" \
        ./new/run_posture_pipeline.sh
    done
done

echo ""
echo "================================================="
echo "  ${POSTURE} 跨体态评估完成。聚合统计："
echo "    python -m new.aggregate_seeds ./output/cross_${OUT_TAG}_*"
echo ""
echo "  解读："
echo "    若 V6 CV(corr) < V2 CV(corr) → V6 跨体态稳定性卖点成立"
echo "    若 V2 corr mean > V6 corr mean → V2 仍是 best-mean 基线"
case "$POSTURE" in
    驼背|头前伸)
        echo ""
        echo "  ⚠ 注意：${POSTURE} 是弱可表征体态，hit_band 和 corr 都会比 APT/膝超伸差"
        echo "      这反映表示限制，不代表方法失效。详见 POSTURE_REPRESENTABILITY.md"
        ;;
esac
echo "================================================="
