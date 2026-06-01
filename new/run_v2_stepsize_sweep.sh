#!/usr/bin/env bash
# new/run_v2_stepsize_sweep.sh
#
# 决定性验证：corr 是否纯粹是"推动幅度 Δ"的函数，与 V6/V2 架构无关。
#
# 背景（前两轮消融的结论）：
#   - 第一轮排除 manifold_project / lambda_smooth
#   - 第二轮发现 corr 几乎是 Δ 的单调递减函数：
#       v6_noBoth     Δ15.28  hit37.3%  corr0.476
#       v6_noTimeDecay Δ16.49 hit62.3%  corr0.423
#       v2_base       Δ18.52  hit89.0%  corr0.295
#       v2_huber      Δ19.08  hit86.9%  corr0.313
#     → corr↔hit 是同一条 Pareto 权衡曲线，V6 的 corr 优势只是
#       "工作在更低 Δ / 更低 hit 的工作点"的副产物，不是某个神奇组件。
#
# 本实验假设：
#   只要把 V2 的步长 s 调小，让 Δ 降到 ~15，V2 的 corr 也会升到 ~0.47，
#   且 hit 同步掉到 ~37%——证明 corr 是推力幅度的纯函数，与架构无关。
#
#   s 越小 → Δ 越小 → corr 越高 / hit 越低（沿 Pareto 曲线滑动）
#
# 判读逻辑：
#   - 若 s=15~20 的 V2 配置 corr 升到 0.45+ 且 hit 掉到 ~40%
#       → 结论钉死：corr 是 Δ 的纯函数，V6 无架构优势，只是默认工作点不同
#   - 若 V2 怎么调 s 都到不了 0.47（corr 封顶在 ~0.35）
#       → V6 确实有架构上的额外收益（PID 反馈带来的非平凡时序保护）
#
# 用法：
#   bash new/run_v2_stepsize_sweep.sh [N_SEEDS]   # 默认 15
# 聚合：
#   python -m new.aggregate_seeds ./output/v2_sweep_*

set -e

N_SEEDS="${1:-15}"
POSTURE="${POSTURE:-躯干前倾}"
TEXT_PROMPT="${TEXT_PROMPT:-a person is walking forward}"
MOTION_LENGTH="${MOTION_LENGTH:-6.0}"

ALL_SEEDS=(7 42 99 123 2024 17 23 88 251 333 666 777 1337 9999 10000)
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")

# V2 步长扫描：从弱到强覆盖 Δ≈12 → Δ≈19
# s=40 是已知基线 (Δ18.5, corr0.30)；下探 s 看 corr 能否随 Δ 下降而升到 V6 水平
declare -A VARIANTS=(
  ["s10"]='v2_dps|{"s":10.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"hinge"}'
  ["s15"]='v2_dps|{"s":15.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"hinge"}'
  ["s20"]='v2_dps|{"s":20.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"hinge"}'
  ["s30"]='v2_dps|{"s":30.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"hinge"}'
  ["s40"]='v2_dps|{"s":40.0,"schedule":"last_quarter","base_weight":1.0,"loss_form":"hinge"}'
)

echo ""
echo "==================================================="
echo "  V2 步长扫描 — corr 是否为 Δ 的纯函数？"
echo "  POSTURE=${POSTURE}  N_SEEDS=${N_SEEDS}"
echo "  5 个 s 值 × ${N_SEEDS} seeds"
echo "==================================================="

for CONFIG_NAME in "${!VARIANTS[@]}"; do
  CONFIG="${VARIANTS[$CONFIG_NAME]}"
  VARIANT="${CONFIG%%|*}"
  KWARGS="${CONFIG##*|}"

  OUT_BASE="./output/v2_sweep_${CONFIG_NAME}"
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
echo "  扫描完成。分析（按 Δ 排序看 corr 是否单调）："
echo ""
echo "  python -m new.aggregate_seeds ./output/v2_sweep_*"
echo ""
echo "  若 corr 随 s↓ 单调升到 ~0.47 且 hit 掉到 ~37% →"
echo "  corr 是 Δ 的纯函数，V6 无架构优势（只是默认工作点更轻）。"
echo "==================================================="
