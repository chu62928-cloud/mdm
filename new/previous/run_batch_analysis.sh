#!/bin/bash

python /root/autodl-tmp/motion-diffusion-model/new/batch_analyze_mdm.py \
  --inputs /root/autodl-tmp/motion-diffusion-model/save/batch_single_outputs \
  --analyze_script /root/autodl-tmp/motion-diffusion-model/new/analyze_mdm_case.py \
  --outdir /root/autodl-tmp/motion-diffusion-model/batch_analysis_v2 \
  --recursive \
  --up_axis y \
  --skip_existing