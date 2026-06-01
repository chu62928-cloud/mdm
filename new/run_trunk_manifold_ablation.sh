#!/usr/bin/env bash
# new/run_trunk_manifold_ablation.sh
#
# 躯干前倾上的受控消融：隔离 V6 corr 优势（0.477 vs V2 0.324）的真正来源。
#
# 核心问题：V6 在躯干前倾上 corr 反超 V2，是 manifold_project 的功劳，
#           还是 lambda_smooth（时域平滑）/ Huber 的功劳？
#
# 双向收敛证据设计：
#   A. 减法（在 V6 上逐个关掉组件）—— 无需改代码，全是已有开关
#      1. v6_full          : manifold=T, smooth=0.03, huber       (基准, 已知 corr≈0.477)
#      2. v6_noManifold    : manifold=F                            ← 单独关流形投影
#      3. v6_noSmooth      : lambda_smooth=0                       ← 单独关时域平滑
#      4. v6_noBoth        : manifold=F, smooth=0                  ← 两个都关
#   B. 加法（在 V2 上单独加 manifold）—— 用新加的 V2 manifold_project 开关
#      5. v2_base          : 原始 V2                               (基准, 已知 corr≈0.324)
#      6. v2_manifold      : V2 + manifold_project=true            ← 单独加流形投影
#
# 判读逻辑：
#   - 若 v6_noManifold 的 corr 掉向 0.32  AND  v2_manifold 的 corr 升向 0.45
#       → 流形投影是 corr 优势的因（双向收敛，假设成立）
#   - 若 v6_noSmooth 掉而 v6_noManifold 不掉
#       → 其实是时域平滑在起作用，不是流形投影
#   - 若两个 V6 消融都不怎么掉 → 优势来自 PID/Huber 的综合，非单一组件
#
# 机制验证（跑完后）：
#   python -m new.analyze_trunk_segments ./output/trunk_abl_*
#   看胸椎补偿指数 TCI：若 manifold=on 的配置 TCI 更低 → 流形投影抑制了胸椎补偿捷径
#
# 用法：
#   bash new/run_trunk_manifold_ablation.sh [N_SEEDS]   # N_SEEDS 默认 15
# 聚合：
#   python -m new.aggregate_seeds ./output/trunk_abl_*

set -e

N_SEEDS="${1:-15}"
POSTURE="${POSTURE:-躯干前倾}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

ALL_SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

# V6 公共参数（与 cross_posture / n15 一致，last_quarter）
V6_COMMON='"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"manifold_alpha":1.0,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter"'

declare -A VARIANTS=(
  # A. 减法消融（V6）
  ["v6_full"]="v6_closed_loop|{${V6_COMMON},\"manifold_project\":true,\"lambda_smooth\":0.03}"
  ["v6_noManifold"]="v6_closed_loop|{${V6_COMMON},\"manifold_project\":false,\"lambda_smooth\":0.03}"
  ["v6_noSmooth"]="v6_closed_loop|{${V6_COMMON},\"manifold_project\":true,\"lambda_smooth\":0.0}"
  ["v6_noBoth"]="v6_closed_loop|{${V6_COMMON},\"manifold_project\":false,\"lambda_smooth\":0.0}"
  # B. 加法消融（V2）
  ["v2_base"]='v2_dps|{"s":40.0,"schedule":"last_quarter","base_weight":1.0,"manifold_project":false}'
  ["v2_manifold"]='v2_dps|{"s":40.0,"schedule":"last_quarter","base_weight":1.0,"manifold_project":true,"manifold_alpha":1.0}'
)

echo ""
echo "==================================================="
echo "  躯干前倾 manifold_project 受控消融"
echo "  POSTURE=${POSTURE}  N_SEEDS=${N_SEEDS}"
echo "  6 configs × ${N_SEEDS} seeds"
echo "==================================================="

for CONFIG_NAME in "${!VARIANTS[@]}"; do
  CONFIG="${VARIANTS[$CONFIG_NAME]}"
  VARIANT="${CONFIG%%|*}"
  KWARGS="${CONFIG##*|}"

  OUT_BASE="./output/trunk_abl_${CONFIG_NAME}"
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
echo "  消融完成。两步分析："
echo ""
echo "  1) corr/hit 对比（确定哪个组件驱动 corr 优势）："
echo "     python -m new.aggregate_seeds ./output/trunk_abl_*"
echo ""
echo "  2) 机制诊断（胸椎补偿指数 TCI）："
echo "     python -m new.analyze_trunk_segments ./output/trunk_abl_*"
echo "==================================================="
