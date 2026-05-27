#!/usr/bin/env bash
# new/run_seed_robustness_n15.sh
#
# 骨盆前倾 N=15 seed 稳健性测试（vs run_seed_robustness.sh 的 N=5 升级版）。
# 只跑两个 SOTA 配置：
#   - v2_dps_s40_always         (绝对王者：hit=83, corr=0.46, CV(corr)=21%)
#   - v6_closed_loop_second_half (V6 最稳：hit=63, corr=0.43, CV(corr)=29.5%)
#
# 目的：
#   - 把 V6 的 CV(corr) 29.5% 在 N=15 上验证是否仍 <30%
#   - 把 V2 的 0.461 mean corr 在 N=15 上确认是否稳定
#   - bootstrap 95% CI（在 aggregate_seeds.py 算）需要 N≥10 才有意义
#
# 用法：
#   bash new/run_seed_robustness_n15.sh
#
# 输出：
#   ./output/n15_<variant>/<seed>/comparison.npy
# 聚合：
#   python -m new.aggregate_seeds ./output/n15_*

set -e

declare -A BEST_VARIANTS=(
  # ["v2_dps_s40_always"]='v2_dps|{"s":40.0,"schedule":"always","base_weight":1.0}'
  ["v2_dps_s40_last_quarter"]='v2_dps|{"s":40.0,"schedule":"last_quarter","base_weight":1.0}'
  # ["v6_closed_loop_second_half"]='v6_closed_loop|{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":true,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"second_half"}'
)

# 15 seeds：原 5 + 新 10。前 5 个与 run_seed_robustness.sh 完全一致，方便对比。
# 如果之前的 N=5 结果你想复用，可以 mkdir output/n15_<variant> 并把
# seedtest_*/seed{7,42,99,123,2024} 软链/复制过去，会自动跳过。
SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)

TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

for CONFIG_NAME in "${!BEST_VARIANTS[@]}"; do
  CONFIG="${BEST_VARIANTS[$CONFIG_NAME]}"
  VARIANT="${CONFIG%%|*}"
  KWARGS="${CONFIG##*|}"

  OUT_BASE="./output/n15_${CONFIG_NAME}"
  mkdir -p "${OUT_BASE}"

  echo ""
  echo "================================================="
  echo "  N=15 robustness: ${CONFIG_NAME}"
  echo "  variant=${VARIANT}"
  echo "================================================="

  for SEED in "${SEEDS[@]}"; do
    SEED_OUT="${OUT_BASE}/seed${SEED}"
    # 跳过已经跑过的 seed（cache）
    if [ -f "${SEED_OUT}/comparison.npy" ]; then
      echo "-- seed=${SEED} CACHED, skip"
      continue
    fi
    echo ""
    echo "-- seed=${SEED} --"
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

  echo "--- ${CONFIG_NAME} 完成 ---"
done

echo ""
echo "================================================="
echo "  全部完成。聚合 N=15 统计："
echo "    python -m new.aggregate_seeds ./output/n15_*"
echo ""
echo "  关键判据："
echo "    V6 CV(corr) < 30% 持续保持 → 稳定性卖点确立"
echo "    V2 corr mean ≈ 0.46 持续保持 → best-mean 基线确立"
echo "================================================="
