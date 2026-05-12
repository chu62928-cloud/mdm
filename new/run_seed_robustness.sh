#!/usr/bin/env bash
# new/run_seed_robustness.sh
#
# 在确定最佳配置后，跑多个 seed 验证稳健性。
# 这是把"单 seed 实验结果"升级到"统计意义实验结果"的关键步骤。
#
# 使用：
#   bash new/run_seed_robustness.sh
#
# 输出：
#   ./output/seedtest_<variant>/<seed>/comparison.npy

set -e

# 等校准 v3 跑完后，把这些值改成实际最佳配置
declare -A BEST_VARIANTS=(
  # 1. 绝对王者 (DPS流派)：全场最高分，hit(band) 高达 69.2%，动作丝滑
  ["v2_dps_s40_always"]='v2_dps|{"s":40.0,"schedule":"always","base_weight":1.0}'
  
  # 2. 动态退火 (DPS流派)：后半程发力，给模型前半程自由度，得分极高且 Jitter 最低
  ["v2_dps_s80_secondhalf"]='v2_dps|{"s":80.0,"schedule":"secondhalf","base_weight":1.0}'
  
  # 3. 蒙特卡洛平滑 (LGD流派)：改成你原始跑通的参数名 n_mc 和 mc_noise_scale
  ["v5_lgd_s80_mc1"]='v5_lgd|{"n_mc":1,"s":80.0,"schedule":"always","base_weight":1.0,"mc_noise_scale":0.05}'
  
  # 4. 直接编辑 (x0_edit流派)：作为 V2b 的对照组，bw=5 是该流派保留结构最好的临界点
  ["v2b_x0_edit_bw5"]='v2_x0_edit|{"n_inner_steps":15,"lr":0.5,"schedule":"always","base_weight":5.0}'
)
SEEDS=(7 42 99 123 2024)
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

for CONFIG_NAME in "${!BEST_VARIANTS[@]}"; do
  CONFIG="${BEST_VARIANTS[$CONFIG_NAME]}"
  VARIANT="${CONFIG%%|*}"
  KWARGS="${CONFIG##*|}"

  OUT_BASE="./output/seedtest_${CONFIG_NAME}"
  mkdir -p "${OUT_BASE}"

  echo ""
  echo "================================================="
  echo "  Seed robustness: ${CONFIG_NAME}"
  echo "  variant=${VARIANT}"
  echo "  kwargs=${KWARGS}"
  echo "================================================="

  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "-- seed=${SEED} --"
    GUIDANCE_VARIANT="${VARIANT}" \
    GUIDANCE_KWARGS_JSON="${KWARGS}" \
    TEXT_PROMPT="${TEXT_PROMPT}" \
    POSTURE="${POSTURE}" \
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