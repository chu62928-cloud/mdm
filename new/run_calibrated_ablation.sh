#!/usr/bin/env bash
# new/run_calibrated_ablation.sh
#
# 校准实验：每个 variant 跑多个推力档位，目标是把 Δ angle 控制在 12-15°，
# 然后在等效推力下比较 corr / RMSE / jitter / hit_rate。
#
# 设计原则：
#   - 所有 variant 用同样的 schedule (last_quarter)，避免 schedule 偏差
#   - 推力档位按"等效推力"估算，让每个 variant 都覆盖 Δ ∈ [5, 20]° 范围
#   - 输出按"variant + 推力值"命名，方便 evaluate 脚本批量解析
#
# 使用：
#   bash new/run_calibrated_ablation.sh
#
# 输出：
#   ./output/calibrated_<TAG>_seed<SEED>/<variant>_<strength>/comparison.npy

set -e

TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
SEED="${SEED:-42}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TAG="${TAG:-walking_apt}"

OUT_BASE="./output/calibrated_${TAG}_seed${SEED}"
mkdir -p "${OUT_BASE}"

echo "=================================================="
echo "  Calibrated Ablation"
echo "  Prompt : ${TEXT_PROMPT}"
echo "  Posture: ${POSTURE}"
echo "  Seed   : ${SEED}"
echo "  Output : ${OUT_BASE}"
echo "=================================================="

run_one() {
  local NAME="$1"
  local VARIANT="$2"
  local KWARGS_JSON="$3"

  echo ""
  echo "------------------------------------------------"
  echo "  [${NAME}]  variant=${VARIANT}"
  echo "  kwargs=${KWARGS_JSON}"
  echo "------------------------------------------------"

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
# V1 — SGD on mu_t (baseline reference: bw=30, lr=0.5, 15 steps)
# 已知 work，跑一档作为对照
# ================================================================
run_one "v1_baseline" "v1_mu_sgd" \
  '{"n_inner_steps":15,"lr":0.5,"schedule":"final","base_weight":30.0}'

# ================================================================
# V2 — DPS (gradient through MDM, update mu_t)
# 推力档位：s 在 [30, 50, 80, 120] 扫描
# 上次实验 s=30 → Δ=7°, s=100 → Δ=12°
# ================================================================
for S in 30 50 80 120; do
  run_one "v2_dps_s${S}" "v2_dps" \
    "{\"s\":${S},\"schedule\":\"last_quarter\",\"base_weight\":1.0}"
done

# ================================================================
# V2b — x0 direct edit (SGD on x0_hat, recompose mu_t)
# 参数对齐 V1：bw=[10, 30, 60, 100], lr=0.5, 15 steps
# ================================================================
for BW in 10 30 60 100; do
  run_one "v2b_x0_edit_bw${BW}" "v2_x0_edit" \
    "{\"n_inner_steps\":15,\"lr\":0.5,\"schedule\":\"final\",\"base_weight\":${BW}}"
done

# ================================================================
# V3 — x0_direct (短 inner loop 版本，但参数加大)
# 用 5 步内循环 + 大 bw，验证短 loop 能否达到 V1 效果
# ================================================================
for BW in 10 30 60 100; do
  run_one "v3_x0_direct_bw${BW}" "v3_x0_direct" \
    "{\"n_inner_steps\":5,\"lr\":0.5,\"schedule\":\"final\",\"base_weight\":${BW}}"
done

# ================================================================
# V4 — OmniControl (动态 K_e<<K_l)
# 上次 K_late=10 + base_weight=30 过强 (Δ=55°)
# 改成 K_late=[2, 3, 5, 7], base_weight=30
# ================================================================
for K in 2 3 5 7; do
  run_one "v4_omni_k${K}" "v4_omni" \
    "{\"K_early\":0,\"K_late\":${K},\"T_split\":12,\"lr\":0.5,\"base_weight\":30.0,\"schedule\":\"always\"}"
done

# ================================================================
# V5 — LGD (DPS + MC smoothing)
# 同 V2 需要大 s，n_mc 减少到 2
# ================================================================
for S in 50 80 120 200; do
  run_one "v5_lgd_s${S}" "v5_lgd" \
    "{\"n_mc\":2,\"s\":${S},\"schedule\":\"last_quarter\",\"base_weight\":1.0,\"mc_noise_scale\":0.05}"
done

echo ""
echo "=================================================="
echo "  全部完成。下一步："
echo "    python -m new.evaluate_ablation ${OUT_BASE}"
echo "    python -m new.plot_angle_curves ${OUT_BASE}"
echo "=================================================="