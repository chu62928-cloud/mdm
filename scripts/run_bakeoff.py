#!/usr/bin/env python3
"""scripts/run_bakeoff.py --- V1-V7 bake-off, calls run_seed_batch.py per variant (Task 5)

TMUX:  tmux new -d -s bakeoff "source /root/miniconda3/bin/activate mdm5090 && \
    cd /root/autodl-tmp/motion-diffusion-model && \
    MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \
    python scripts/run_bakeoff.py 2>&1 | tee output0727/bakeoff.log"
"""

import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "output0727" / "bakeoff"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH",
    str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
SEEDS = "7,17,23,42,88,99,123,251,333,666,777,1337,2024,9999,10000"
PROMPT = "a person is walking forward"
POSTURE = "anterior_pelvic_tilt"
PYTHON = sys.executable

VARIANTS = [
    ("v1_mu_sgd", {}),
    ("v2_dps", {"s": 40, "schedule": "last_quarter"}),
    ("v2_dps_norm", {"s": 2.0, "schedule": "last_quarter"}),
    ("v2_x0_edit", {"bw": 5}),
    ("v3_x0_direct", {"n": 5, "lr": 0.05}),
    ("v4_omni", {"K_early": 1, "K_late": 10}),
    ("v5_lgd", {"n_mc": 4, "mc_noise_scale": 0.05}),
    ("v6_closed_loop", {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05,
                         "s_max": 50, "I_max": 20, "beta_ema": 0.8,
                         "lambda_smooth": 0.03, "manifold_project": True,
                         "loss_form": "huber", "huber_delta": 0.05,
                         "normalize_grad": False, "band_gate": False,
                         "spec_schedule_override": "second_half"}),
]

print(f"=== V1-V7 Bake-off (batch mode) ===")
print(f"Output: {OUT}")
print(f"Variants: {len(VARIANTS)}, Seeds: {len(SEEDS.split(','))}")
print(f"Total batches: {len(VARIANTS)}")
print(f"Started: {time.strftime('%H:%M:%S')}")
print()

total_ok = 0
t_total = time.time()

for variant, kwargs in VARIANTS:
    vname = variant.replace("_", "-")
    t0 = time.time()
    kwarg_json = json.dumps(kwargs)

    cmd = [
        PYTHON, "-m", "scripts.run_seed_batch",
        "--model_path", MODEL,
        "--text_prompt", PROMPT,
        "--seeds", SEEDS,
        "--output_dir", str(OUT),
        "--posture_instructions", POSTURE,
        "--variant", variant,
        "--variant_kwargs_json", kwarg_json,
        "--guidance_mode", "joint",
        "--motion_length", "6.0",
    ]

    print(f"\n--- {vname} ---", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True,
                       text=True, timeout=7200)

    print(r.stdout)
    if r.stderr:
        # Filter out non-critical stderr
        for line in r.stderr.strip().split("\n"):
            if any(kw in line.lower() for kw in ["error", "traceback", "fail"]):
                print(f"  [stderr] {line[:200]}")

    dt = time.time() - t0
    if r.returncode == 0:
        print(f"  [DONE] {vname} in {dt/60:.1f} min", flush=True)
    else:
        print(f"  [FAIL] {vname} (rc={r.returncode}) in {dt:.0f}s", flush=True)

print(f"\n=== TOTAL: {time.time()-t_total:.0f}s ===")
print(f"Scorecard: python -m eval.scorecard --run_dir {OUT} "
      f"--target 20.0 --tolerance 2.0 --direction greater_than "
      f"--judge_op pelvis_tilt")