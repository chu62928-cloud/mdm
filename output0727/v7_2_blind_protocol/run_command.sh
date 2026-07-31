#!/bin/bash
# V7.2 800-run blind test -- exact commands used (backfilled record)
set -e
cd /root/autodl-tmp/motion-diffusion-model
source /root/miniconda3/bin/activate mdm5090
source /etc/network_turbo

# Main run (V2/V6/V7.1/V7.2 x 5 targets x 40 seeds) -- V6 arm's kwargs were
# later found incomplete and rerun separately, see run_v7_2_blind_v6fix.py
MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \
  python scripts/run_v7_2_blind.py 2>&1 | tee output0727/v7_2_blind_test/run.log

# V6-arm fix rerun (corrected kwargs matching validated config)
MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \
  python scripts/run_v7_2_blind_v6fix.py 2>&1 | tee output0727/v7_2_blind_test/run_v6fix.log

# Scorecards (all 20 target x method combinations)
for tau in 05 10 15 20 25; do
  if [ "$tau" = "20" ]; then posture=anterior_pelvic_tilt; else posture=anterior_pelvic_tilt_tau$tau; fi
  tdeg=$((10#$tau))
  for v in v2 v6 v7-1 v7-2; do
    python -m eval.scorecard --run_dir output0727/v7_2_blind_test/tau$tau/$v \
      --posture "$posture" --target "$tdeg" --tolerance 2.0 --no_dist_metrics
  done
done

# Primary analysis (Section 9.4 protocol)
python scripts/analyze_v72_blind.py
