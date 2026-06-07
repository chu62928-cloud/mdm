#!/usr/bin/env bash
# new/run_apt_integrated.sh
# 骨盆前倾（APT）三模式对照：关节角 / 肌肉激活 / 两者组合。
# 统一引导接口走 v2_dps（README 实测 APT 最佳：s=40 + last_quarter）。
#
# 前置：
#   - MODEL_PATH    : MDM checkpoint（如 humanml_trans_dec_512_bert 的 model*.pt）
#   - MUSCLE_CKPT   : motion2muscle 冻结代理权重 net_best_*.pth（muscle/both 需要）
#   需要 GPU。joint 模式不需要 MUSCLE_CKPT。
#
# 用法：
#   MODEL_PATH=./save/.../model.pt MUSCLE_CKPT=./motion2muscle/checkpoints/.../net_best_loss.pth \
#     bash new/run_apt_integrated.sh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?请设置 MODEL_PATH 指向 MDM checkpoint}"
MUSCLE_CKPT="${MUSCLE_CKPT:-}"
PROMPT="${PROMPT:-a person is walking}"
SEED="${SEED:-42}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
OUT_ROOT="${OUT_ROOT:-./output/apt_integrated}"

# 统一 variant：v2_dps，APT 最佳超参
export GUIDANCE_VARIANT="${GUIDANCE_VARIANT:-v2_dps}"
export GUIDANCE_KWARGS_JSON="${GUIDANCE_KWARGS_JSON:-{\"s\":40,\"schedule\":\"last_quarter\",\"base_weight\":20}}"
export GUIDANCE_DIAGNOSTIC="${GUIDANCE_DIAGNOSTIC:-1}"

run_one () {
  local MODE="$1"; shift
  local OUT="${OUT_ROOT}/${MODE}"
  echo "================ MODE=${MODE} -> ${OUT} ================"
  GUIDANCE_MODE="${MODE}" python -m sample.generate \
    --model_path "${MODEL_PATH}" \
    --text_prompt "${PROMPT}" \
    --seed "${SEED}" \
    --motion_length "${MOTION_LENGTH}" \
    --num_samples "${NUM_SAMPLES}" \
    --output_dir "${OUT}" \
    --guidance_mode "${MODE}" \
    --posture_instructions 骨盆前倾 \
    --muscle_posture anterior_pelvic_tilt \
    --muscle_assets_dir motion2muscle \
    --muscle_ckpt "${MUSCLE_CKPT}" \
    --joint_weight 1.0 --muscle_weight 1.0 \
    "$@"
}

# 1) 仅关节角（不需要 MUSCLE_CKPT）
run_one joint

# 2) 仅肌肉激活
if [[ -n "${MUSCLE_CKPT}" ]]; then
  run_one muscle
  # 3) 两者组合
  run_one both
else
  echo "[skip] muscle / both：未设置 MUSCLE_CKPT"
fi

echo "完成。用 new/evaluate_ablation_v3.py 对各 output 目录评关节角指标，"
echo "对照 midterm Table 3/4 看肌肉 posture loss 方向。"
