#!/bin/bash
set -euo pipefail

source /etc/network_turbo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mdm5090

cd /root/autodl-tmp/motion-diffusion-model

export MODEL_PATH=save/humanml_trans_dec_512_bert/model000600000.pt
export MUSCLE_CKPT=motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth
export GUIDANCE_VARIANT=v2_dps
export GUIDANCE_KWARGS_JSON='{"s":40,"schedule":"last_quarter"}'
export NUM_SAMPLES=3
export SEED=42
export PROMPT='a person is walking'
export OUT_ROOT=./output/apt_integrated_v2

echo "=== Experiment Config ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "MUSCLE_CKPT=$MUSCLE_CKPT"
echo "GUIDANCE_VARIANT=$GUIDANCE_VARIANT"
echo "NUM_SAMPLES=$NUM_SAMPLES SEED=$SEED"
echo "OUT_ROOT=$OUT_ROOT"

bash new/run_apt_integrated.sh 2>&1 | tee /root/autodl-tmp/motion-diffusion-model/run_apt_v2.log

echo "Experiment completed!"
