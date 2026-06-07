import os, subprocess, sys

# Set correct env
env = os.environ.copy()
env['MODEL_PATH'] = 'save/humanml_trans_dec_512_bert/model000600000.pt'
env['MUSCLE_CKPT'] = 'motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth'
env['GUIDANCE_VARIANT'] = 'v2_dps'
env['GUIDANCE_KWARGS_JSON'] = '{"s":40,"schedule":"last_quarter"}'
env['NUM_SAMPLES'] = '3'
env['SEED'] = '42'
env['PROMPT'] = 'a person is walking'
env['OUT_ROOT'] = './output/apt_integrated_v3'

print('Running with GUIDANCE_KWARGS_JSON=' + env['GUIDANCE_KWARGS_JSON'])
result = subprocess.run(['bash', 'new/run_apt_integrated.sh'], env=env, cwd='/root/autodl-tmp/motion-diffusion-model')
sys.exit(result.returncode)
