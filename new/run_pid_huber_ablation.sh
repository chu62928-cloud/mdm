#!/usr/bin/env bash
# new/run_pid_huber_ablation.sh
#
# 躯干前倾上的第二轮受控消融：隔离 PID c_t 时衰增益 vs Huber 损失的贡献。
#
# 背景：第一轮消融（run_trunk_manifold_ablation.sh）排除了 manifold_project 和
#       lambda_smooth 作为 V6 corr 优势（0.477 vs V2 0.335）的来源。
#       v6_noBoth（无 manifold，无 smooth）的 corr 仍为 0.477，说明优势在于
#       PID + Huber 这两个组件。
#
# 本轮聚焦拆解 PID + Huber：
#
#   双向收敛设计（减法 + 加法）：
#   ---- 加法（在 V2 上单独加 Huber）----
#   1. v2_base      : s=40, hinge loss               (baseline, 已知 corr≈0.335)
#   2. v2_huber     : s=40, Huber loss               ← 单独换 Huber，其余同 V2
#
#   ---- 减法（在 V6 上关掉 c_t 时衰）----
#   3. v6_noBoth    : PID+Huber，manifold=F，smooth=0 (baseline, 已知 corr≈0.477)
#   4. v6_noTimeDecay: c_t=1.0（去掉时衰增益），其余同 v6_noBoth
#
# 判读逻辑：
#   - 若 v2_huber corr 升向 0.45+  → Huber 是主因（loss 形式决定 corr）
#   - 若 v6_noTimeDecay corr 掉向 0.35  → c_t 是主因（时衰让 PID 更稳）
#   - 若两者都轻微变化 → PID 的 integral/derivative/EMA 综合作用，无单一因
#
# 用法：
#   bash new/run_pid_huber_ablation.sh [N_SEEDS]   # N_SEEDS 默认 15
# 聚合：
#   python -m new.aggregate_seeds ./output/pid_huber_abl_*

set -e

N_SEEDS="${1:-15}"
POSTURE="${POSTURE:-躯干前倾}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

ALL_SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

# V6 公共参数（与 trunk_manifold_ablation 一致，noBoth 配置）
V6_COMMON='"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"manifold_alpha":1.0,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","manifold_project":false,"lambda_smooth":0.0'

declare -A VARIANTS=(
  # 加法消融（在 V2 上加 Huber）
  ["v2_base"]='v2_dps|{"s":40.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"hinge"}'
  ["v2_huber"]='v2_dps|{"s":40.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"huber","huber_delta":0.05}'
  # 减法消融（在 V6 noBoth 上去掉 c_t）
  ["v6_noBoth"]="v6_closed_loop|{${V6_COMMON},\"use_time_decay\":true}"
  ["v6_noTimeDecay"]="v6_closed_loop|{${V6_COMMON},\"use_time_decay\":false}"
)

echo ""
echo "==================================================="
echo "  躯干前倾 PID c_t vs Huber 受控消融"
echo "  POSTURE=${POSTURE}  N_SEEDS=${N_SEEDS}"
echo "  4 configs × ${N_SEEDS} seeds"
echo "==================================================="

for CONFIG_NAME in "${!VARIANTS[@]}"; do
  CONFIG="${VARIANTS[$CONFIG_NAME]}"
  VARIANT="${CONFIG%%|*}"
  KWARGS="${CONFIG##*|}"

  OUT_BASE="./output/pid_huber_abl_${CONFIG_NAME}"
  mkdir -p "${OUT_BASE}"

  echo ""
  echo "-- config: ${CONFIG_NAME}  (variant=${VARIANT})"
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
echo "==================================================="
echo "  消融完成。分析："
echo ""
echo "  python -m new.aggregate_seeds ./output/pid_huber_abl_*"
echo "==================================================="
