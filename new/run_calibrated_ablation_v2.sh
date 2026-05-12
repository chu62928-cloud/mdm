#!/usr/bin/env bash
# new/run_calibrated_ablation_v2.sh
#
# 基于第一次校准结果的精细化扫描：
#   1. 围绕 V2b bw=10 (corr=0.521) 做 fine-grained 扫描
#   2. 围绕 V2 (DPS) s=80 (corr=0.441) 做 fine-grained 扫描
#   3. V4 (omni) 大幅降低 base_weight (避免塌平)
#   4. V5 修复 sigma_sq bug 后重测
#   5. 把 V1 (corr=-0.11) 改成更弱版本，避免塌平
#
# 已删除的方案：
#   - V3 与 V2b 在 final schedule 下完全一致，只保留 V2b
#   - V2_dps_s30 偏弱、s=120 偏强，已淘汰，集中 s=60~100
#
# 使用：
#   bash new/run_calibrated_ablation_v2.sh

set -e

TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
SEED="${SEED:-42}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TAG="${TAG:-walking_apt_v2}"

OUT_BASE="./output/calibrated_${TAG}_seed${SEED}"
mkdir -p "${OUT_BASE}"

echo "=================================================="
echo "  Calibrated Ablation V2 (refined)"
echo "  Output : ${OUT_BASE}"
echo "=================================================="

run_one() {
  local NAME="$1"
  local VARIANT="$2"
  local KWARGS_JSON="$3"

  echo ""
  echo "-- [${NAME}] variant=${VARIANT} kwargs=${KWARGS_JSON}"

  GUIDANCE_VARIANT="${VARIANT}" \
  GUIDANCE_KWARGS_JSON="${KWARGS_JSON}" \
  TEXT_PROMPT="${TEXT_PROMPT}" \
  POSTURE="${POSTURE}" \
  SEED="${SEED}" \
  MOTION_LENGTH="${MOTION_LENGTH}" \
  OUTPUT_DIR="${OUT_BASE}/${NAME}" \
  MAKE_ANIMATION="" \
  ./new/run_posture_pipeline.sh
}

# ================================================================
# V1 — SGD on mu_t
# 之前 bw=30 塌平 → 降到 bw=[5, 10, 15] 看能否保结构
# ================================================================
for BW in 5 10 15; do
  run_one "v1_bw${BW}" "v1_mu_sgd" \
    "{\"n_inner_steps\":15,\"lr\":0.5,\"schedule\":\"final\",\"base_weight\":${BW}}"
done

# ================================================================
# V2 — DPS (gradient through MDM, update mu_t)
# 上次 s=80 corr=0.441 最佳 → fine-grained [60, 70, 80, 90, 100]
# 同时验证 schedule 影响：用 last_quarter 也跑一次 s=80
# ================================================================
for S in 60 70 80 90 100; do
  run_one "v2_dps_s${S}" "v2_dps" \
    "{\"s\":${S},\"schedule\":\"last_quarter\",\"base_weight\":1.0}"
done

# 验证 schedule 影响（s=80 + final vs last_quarter 对比）
run_one "v2_dps_s80_final" "v2_dps" \
  '{"s":80,"schedule":"final","base_weight":1.0}'

# ================================================================
# V2b — x0_direct edit
# 上次 bw=10 corr=0.521 第一名 → fine-grained [5, 8, 10, 12, 15]
# 同时减小 inner_steps 看是否仍然 work（验证 step 数不关键）
# ================================================================
for BW in 5 8 10 12 15; do
  run_one "v2b_bw${BW}" "v2_x0_edit" \
    "{\"n_inner_steps\":15,\"lr\":0.5,\"schedule\":\"final\",\"base_weight\":${BW}}"
done

# 验证 inner_steps 影响（bw=10 + steps=[5,10,15,25]）
for STEPS in 5 10 25; do
  run_one "v2b_bw10_steps${STEPS}" "v2_x0_edit" \
    "{\"n_inner_steps\":${STEPS},\"lr\":0.5,\"schedule\":\"final\",\"base_weight\":10}"
done

# ================================================================
# V4 — OmniControl
# 上次 bw=30 全部塌平 → 大幅降到 bw=[3, 5, 8]，K_late=3
# ================================================================
for BW in 3 5 8; do
  run_one "v4_omni_bw${BW}" "v4_omni" \
    "{\"K_early\":0,\"K_late\":3,\"T_split\":12,\"lr\":0.5,\"base_weight\":${BW},\"schedule\":\"always\"}"
done

# ================================================================
# V5 — LGD (修复 sigma_sq bug 后重测)
# 现在和 V2 公式一致，s 应该也类似（60-100）
# 看 MC 平均能否让 corr 比 V2 更高
# ================================================================
for S in 60 80 100; do
  run_one "v5_lgd_s${S}_mc2" "v5_lgd" \
    "{\"n_mc\":2,\"s\":${S},\"schedule\":\"last_quarter\",\"base_weight\":1.0,\"mc_noise_scale\":0.05}"
done

# 测 n_mc 影响（s=80 + n_mc=[2,4]）
run_one "v5_lgd_s80_mc4" "v5_lgd" \
  '{"n_mc":4,"s":80,"schedule":"last_quarter","base_weight":1.0,"mc_noise_scale":0.05}'

echo ""
echo "=================================================="
echo "  完成（共约 20 次实验）。下一步："
echo "    python -m new.evaluate_ablation ${OUT_BASE}"
echo "    python -m new.plot_angle_curves ${OUT_BASE}"
echo "=================================================="