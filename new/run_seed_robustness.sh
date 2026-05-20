#!/usr/bin/env bash
# new/run_seed_robustness.sh

set -e

# 接收外部传入的参数（体态和目标角度），如果没有传则使用默认值
POSTURE="${1:-骨盆前倾}"
TARGET="${2:-20.0}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

# 在这里配置我们要跑的变体
declare -A BEST_VARIANTS=(
  # 【提醒】这里把之前跑过的 V2/V5 旧配置注释掉了，避免每次跑浪费几个小时的时间。
  # ["v2_dps_s40_always"]='v2_dps|{"s":40.0,"schedule":"always","base_weight":1.0}'
  # ["v2_dps_s80_secondhalf"]='v2_dps|{"s":80.0,"schedule":"secondhalf","base_weight":1.0}'
  # ["v5_lgd_s80_mc1"]='v5_lgd|{"n_mc":1,"s":80.0,"schedule":"always","base_weight":1.0,"mc_noise_scale":0.05}'
  # ["v2b_x0_edit_bw5"]='v2_x0_edit|{"n_inner_steps":15,"lr":0.5,"schedule":"always","base_weight":5.0}'
  
  # ================== 新增的测试方案 ==================
  
  # [测试 1] Step 1 归一化方案
  # 输出路径会自动变成: ./output/seedtest_v2_dps_norm_s2
  # ["v2_dps_norm_s2"]='v2_dps_norm|{"s":2.0,"schedule":"always"}'
  
  # [测试 2] Step 2 闭环控制核心方案
  # 输出路径会自动变成: ./output/seedtest_v6_closed_loop_test
  # ["v6_closed_loop"]='v6_closed_loop|{"Kp":80,"Ki":1,"Kd":5,"s_min":0.5,"s_max":50,
  #                      "I_max":20,"beta_ema":0.8,"lambda_smooth":0.02,
  #                      "manifold_project":true,
  #                      "loss_form":"huber","huber_delta":0.05,
  #                      "normalize_grad":false,"band_gate":false}'
  # ["v6_closed_loop_always_1"]='v6_closed_loop|{"Kp":80,"Ki":1,"Kd":5,"s_min":0.5,"s_max":50,
  #                      "I_max":20,"beta_ema":0.8,"lambda_smooth":0.02,
  #                      "manifold_project":true,
  #                      "loss_form":"huber","huber_delta":0.05,
  #                      "normalize_grad":false,"band_gate":false,
  #                      "spec_schedule_override":"always"}'
  ["v6_closed_loop_second_half"]='v6_closed_loop|{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,
                       "I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,
                       "manifold_project":true,
                       "loss_form":"huber","huber_delta":0.05,
                       "normalize_grad":false,"band_gate":false,
                       "spec_schedule_override":"second_half"}'
)

SEEDS=(7 42 99 123 2024)

for CONFIG_NAME in "${!BEST_VARIANTS[@]}"; do
  CONFIG="${BEST_VARIANTS[$CONFIG_NAME]}"
  VARIANT="${CONFIG%%|*}"
  KWARGS="${CONFIG##*|}"

  # 输出路径是根据上面的名字拼的，所以绝对不会覆盖！
  OUT_BASE="./output/seedtest_${CONFIG_NAME}"
  mkdir -p "${OUT_BASE}"

  echo ""
  echo "================================================="
  echo "  Seed robustness: ${CONFIG_NAME}"
  echo "  variant=${VARIANT}"
  echo "  kwargs=${KWARGS}"
  echo "  posture=${POSTURE}, target=${TARGET}"
  echo "================================================="

  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "-- seed=${SEED} --"
    # 将所有变量传给底层的 pipeline 脚本
    GUIDANCE_VARIANT="${VARIANT}" \
    GUIDANCE_KWARGS_JSON="${KWARGS}" \
    TEXT_PROMPT="${TEXT_PROMPT}" \
    POSTURE="${POSTURE}" \
    TARGET="${TARGET}" \
    SEED="${SEED}" \
    MOTION_LENGTH="${MOTION_LENGTH}" \
    OUTPUT_DIR="${OUT_BASE}/seed${SEED}" \
    MAKE_ANIMATION="" \
    ./new/run_posture_pipeline.sh
  done

  echo ""
  echo "--- ${CONFIG_NAME} 完成 ---"
  echo "评估: python -m new.evaluate_ablation ${OUT_BASE}"
done

echo ""
echo "================================================="
echo "  所有 seed 实验完成。聚合统计："
echo "    python -m new.aggregate_seeds ./output/seedtest_*"
echo "================================================="