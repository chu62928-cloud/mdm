#!/usr/bin/env bash
# new/run_calibrated_ablation_v3.sh
#
# 围绕 V2 (s≈80) 和 V2b (bw≈8) 的精细化扫描。
# 目标：找到 Δ 最接近 +14°（baseline 6° → target 20°）且 corr 最高的配置。
#
# 设计动机：
#   v2 一次 ablation 结果显示 V2b bw=10-12 corr 最高但 Δ=+27°（过度推力）
#   V2 (DPS) s=80 Δ=+15° corr=+0.45 落带最准
#   需要在这两个邻域做 fine-grained 搜索

set -e

TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
POSTURE="${POSTURE:-骨盆前倾}"
SEED="${SEED:-42}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TAG="${TAG:-walking_apt_v3}"

OUT_BASE="./output/calibrated_${TAG}_seed${SEED}"
mkdir -p "${OUT_BASE}"

echo "============================================================"
echo "  Calibrated Ablation V3 (fine-grained on V2/V2b)"
echo "  Output : ${OUT_BASE}"
echo "============================================================"

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
# V2 (DPS) — 围绕 s=80 fine-grained
# 上次 s=70 corr=0.092 突然崩，s=80 corr=0.45 最佳
# 需要确认 s=75/85 是否更好，找出最稳定窗口
# ================================================================
for S in 75 80 85; do
  run_one "v2_dps_s${S}" "v2_dps" \
    "{\"s\":${S},\"schedule\":\"last_quarter\",\"base_weight\":1.0}"
done

# 也试 schedule=second_half (推力分散到更多步)
run_one "v2_dps_s80_secondhalf" "v2_dps" \
  '{"s":80,"schedule":"second_half","base_weight":1.0}'

# 试 schedule=always (从头推到尾)
run_one "v2_dps_s40_always" "v2_dps" \
  '{"s":40,"schedule":"always","base_weight":1.0}'

run_one "v2_dps_s60_always" "v2_dps" \
  '{"s":60,"schedule":"always","base_weight":1.0}'

# ================================================================
# V2b (x0_edit) — 围绕 bw=8 fine-grained
# 上次 bw=5 Δ=+16° (达标但 corr=0.27 略低)
# 上次 bw=8 Δ=+22° (略过推但 corr=0.49)
# 需要在 bw=[5, 6, 7, 8] 找到 Δ=+14° 且 corr 最高的点
# ================================================================
for BW in 5 6 7 8; do
  run_one "v2b_bw${BW}" "v2_x0_edit" \
    "{\"n_inner_steps\":15,\"lr\":0.5,\"schedule\":\"final\",\"base_weight\":${BW}}"
done

# 同时降 lr 试试（让推力更温和）
for LR in 0.3 0.2 0.1; do
  run_one "v2b_bw10_lr${LR}" "v2_x0_edit" \
    "{\"n_inner_steps\":15,\"lr\":${LR},\"schedule\":\"final\",\"base_weight\":10}"
done

# 测 last_quarter schedule（vs final）
run_one "v2b_bw8_lastquarter" "v2_x0_edit" \
  '{"n_inner_steps":15,"lr":0.5,"schedule":"last_quarter","base_weight":8}'

run_one "v2b_bw10_lastquarter" "v2_x0_edit" \
  '{"n_inner_steps":15,"lr":0.5,"schedule":"last_quarter","base_weight":10}'

# ================================================================
# V4 (omni) — 上次 bw=8 也 work，做精细化
# ================================================================
for BW in 6 7 8 10; do
  run_one "v4_omni_bw${BW}" "v4_omni" \
    "{\"K_early\":0,\"K_late\":3,\"T_split\":12,\"lr\":0.5,\"base_weight\":${BW},\"schedule\":\"always\"}"
done

# ================================================================
# V5 (LGD) — 上次 s=80 mc4 corr=0.39，尝试小 mc 和大 s
# ================================================================
run_one "v5_lgd_s80_mc1" "v5_lgd" \
  '{"n_mc":1,"s":80,"schedule":"last_quarter","base_weight":1.0,"mc_noise_scale":0.05}'

run_one "v5_lgd_s100_mc4" "v5_lgd" \
  '{"n_mc":4,"s":100,"schedule":"last_quarter","base_weight":1.0,"mc_noise_scale":0.05}'

run_one "v5_lgd_s80_noisesmall" "v5_lgd" \
  '{"n_mc":2,"s":80,"schedule":"last_quarter","base_weight":1.0,"mc_noise_scale":0.02}'

echo ""
echo "============================================================"
echo "  完成（约 20 次实验）。下一步："
echo "    python -m new.evaluate_ablation ${OUT_BASE}   # 用 v3 评分"
echo "    python -m new.plot_angle_curves ${OUT_BASE}"
echo "============================================================"