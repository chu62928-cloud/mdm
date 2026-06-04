#!/usr/bin/env bash
# new/probe_ood_ceiling.sh
#
# OOD 推力上限探测脚本
#
# 背景：delta_max=20 把 CV 从 22% 降到 1.3%（稳定性已解决），
# 但左膝均值卡在 ~182°，距离 190° 还差 8°。
# 每步误差减少约 1.2°，12 步引导不够用。
#
# 本轮测试四个方向：
#   always  — 把引导从 12 步扩展到 15 步（last_quarter → always）
#   kp160   — 把 Kp 从 80 翻到 160，每步推力翻倍
#   comb    — always + Kp=160（最大引导力）
#   t035    — t0=0.35（更高加噪量，MDM 回拉更弱，对照实验）
#
# 用法：
#   bash new/probe_ood_ceiling.sh [N_SEEDS]   默认 N_SEEDS=4
#   bash new/probe_ood_ceiling.sh 8           扩展到 8 seeds
#
# 输出：./output/probe_ceil_<variant>/seed<seed>/comparison.npy
# 聚合：python -m new.aggregate_seeds ./output/probe_ceil_*

set -e

N_SEEDS="${1:-4}"
MODEL_PATH="${MODEL_PATH:-./save/humanml_trans_dec_512_bert/model000200000.pt}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"
TARGET_DEG="${TARGET_DEG:-190}"
MIN_BASE_DEG="${MIN_BASE_DEG:-150}"
DATASET="${DATASET:-humanml}"
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"
T0="${T0:-0.30}"

ALL_SEEDS=(7 42 99 123 2024 17 23 88)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

declare -A KWARGS_MAP

# 当前最优基线（test B）：delta_max=20，last_quarter，Kp=80
KWARGS_MAP["baseline"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":20.0}'

# 修法 A：last_quarter → always（12步→15步引导）
KWARGS_MAP["always"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"always","delta_max":20.0}'

# 修法 B：Kp=80 → 160（每步推力翻倍）
KWARGS_MAP["kp160"]='{"Kp":160,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":20.0}'

# 修法 C：always + Kp=160（组合最强）
KWARGS_MAP["comb"]='{"Kp":160,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"always","delta_max":20.0}'

# 修法 D：t0=0.35（t0_ratio 命令行覆盖，kwargs 同 baseline）
KWARGS_MAP["t035"]='{"Kp":80,"Ki":1,"Kd":5,"s_min":0.05,"s_max":50,"I_max":20,"beta_ema":0.8,"lambda_smooth":0.03,"manifold_project":false,"loss_form":"huber","huber_delta":0.05,"normalize_grad":false,"band_gate":false,"spec_schedule_override":"last_quarter","delta_max":20.0}'

# 实验矩阵：变体名 + 对应的 t0_ratio
declare -A T0_MAP
T0_MAP["baseline"]="${T0}"
T0_MAP["always"]="${T0}"
T0_MAP["kp160"]="${T0}"
T0_MAP["comb"]="${T0}"
T0_MAP["t035"]="0.35"

VARIANTS=(baseline always kp160 comb t035)

echo "================================================================"
echo "  OOD 推力上限探测"
echo "  当前最优：delta_max=20, t0=0.30 → 左膝均值 ~182°"
echo "  目标：找到突破 185°+ 的配置"
echo "  N_SEEDS = ${N_SEEDS}"
echo "  实验数 = ${#VARIANTS[@]} × ${N_SEEDS} = $(( ${#VARIANTS[@]} * N_SEEDS ))"
echo "================================================================"

run_one () {
    local VNAME="$1" SEED="$2"
    local KWARGS="${KWARGS_MAP[$VNAME]}"
    local T0_VAL="${T0_MAP[$VNAME]}"
    local OUT="./output/probe_ceil_${VNAME}/seed${SEED}"

    if [ -f "${OUT}/comparison.npy" ]; then
        echo "   [${VNAME} seed=${SEED}] CACHED, skip"; return
    fi
    mkdir -p "${OUT}"
    echo "   [${VNAME} t0=${T0_VAL} seed=${SEED}] running..."

    GUIDANCE_VARIANT=v6_closed_loop GUIDANCE_KWARGS_JSON="${KWARGS}" \
    ${PYTHON} -m new.sdEdit_ood \
        --model_path "${MODEL_PATH}" \
        --text_prompt "${TEXT_PROMPT}" \
        --posture_instructions 膝超伸 \
        --t0_ratio "${T0_VAL}" \
        --target_signed_deg "${TARGET_DEG}" \
        --min_base_deg "${MIN_BASE_DEG}" \
        --num_samples 1 --motion_length "${MOTION_LENGTH}" \
        --seed "${SEED}" --dataset "${DATASET}" --device "${DEVICE}" \
        --output_dir "${OUT}"
}

for VNAME in "${VARIANTS[@]}"; do
    echo ""
    echo "-- [${VNAME}] t0=${T0_MAP[$VNAME]}  Kp=$(echo ${KWARGS_MAP[$VNAME]} | grep -o '"Kp":[0-9]*' | head -1)"
    for SEED in "${SEEDS[@]}"; do
        run_one "${VNAME}" "${SEED}"
    done
done

# ---- 内联聚合：每个变体打印关键指标 ----
echo ""
echo "================================================================"
echo "  扫描完成，聚合结果："
echo "================================================================"

${PYTHON} - <<'PYEOF'
import os, glob, numpy as np

OUT_BASE = "./output"
variants = ["baseline", "always", "kp160", "comb", "t035"]

print(f"{'变体':<12} {'左膝均值':>8} {'右膝均值':>8} {'双膝均值':>8} {'左CV%':>7} {'seeds':>6}")
print("-" * 58)

for v in variants:
    pattern = os.path.join(OUT_BASE, f"probe_ceil_{v}", "seed*", "comparison.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"{v:<12} {'—':>8} {'—':>8} {'—':>8} {'—':>7} {'0':>6}")
        continue

    lefts, rights = [], []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        # comparison.npy 里有 final_knee_deg 或直接读 motion_xyz
        if "final_knee_deg" in d:
            kd = d["final_knee_deg"]
            lefts.append(kd.get("left", float("nan")))
            rights.append(kd.get("right", float("nan")))

    if not lefts:
        print(f"{v:<12} {'no data':>8}")
        continue

    l, r = np.array(lefts), np.array(rights)
    both = np.concatenate([l, r])
    cv = 100 * l.std() / l.mean() if l.mean() > 0 else 0
    print(f"{v:<12} {l.mean():>8.1f} {r.mean():>8.1f} {both.mean():>8.1f} {cv:>7.1f} {len(files):>6}")

print()
print("期望：comb 或 always 双膝均值突破 185°，CV < 5%")
PYEOF

echo ""
echo "若所有变体仍卡在 ~182°，输出结论：推理时引导触及 OOD 上限，需转 Route E (LoRA)"
