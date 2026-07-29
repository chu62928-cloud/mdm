#!/usr/bin/env python3
"""scripts/run_auto_calibration.py -- V6 PID vs V2 calibration (batch mode, Task 6)

Calls run_seed_batch.py once per (variant, target) pair.
Total: 2 variants × 5 targets = 10 batch calls.

TMUX: tmux new -d -s autocal "source /root/miniconda3/bin/activate mdm5090 && \
    source /etc/network_turbo && cd /root/autodl-tmp/motion-diffusion-model && \
    MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \
    python scripts/run_auto_calibration.py 2>&1 | tee output0727/autocal.log"
"""

import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "output0727" / "autocal"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH",
    str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable
SEEDS = "42,88,123,251,333,666,777,1337,2024,9999"
PROMPT = "a person is walking forward"

TARGETS = {
    5:  "anterior_pelvic_tilt_tau05",
    10: "anterior_pelvic_tilt_tau10",
    15: "anterior_pelvic_tilt_tau15",
    20: "anterior_pelvic_tilt",
    25: "anterior_pelvic_tilt_tau25",
}

V6_KWARGS = {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50,
             "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03,
             "manifold_project": True, "loss_form": "huber", "huber_delta": 0.05,
             "normalize_grad": False, "band_gate": False,
             "spec_schedule_override": "second_half"}

V2_KWARGS = {"s": 40, "schedule": "last_quarter"}

CONFIGS = [
    ("v6_closed_loop", V6_KWARGS),
    ("v2_dps", V2_KWARGS),
]

print(f"=== Auto-Calibration (batch mode) ===")
print(f"Targets: {list(TARGETS.keys())} deg, Seeds: 10, Configs: {len(CONFIGS)}")
print(f"Total batches: {len(CONFIGS) * len(TARGETS)}")
print(f"Output: {OUT}")
print(f"Started: {time.strftime('%H:%M:%S')}")
print()

done, ok = 0, 0
t_total = time.time()
n_batches = len(CONFIGS) * len(TARGETS)

for variant, kwargs in CONFIGS:
    vname = variant.replace("_", "-")
    for target_deg, posture_name in sorted(TARGETS.items()):
        t0 = time.time()
        kwarg_json = json.dumps(kwargs)
        cmd = [
            PYTHON, "-m", "scripts.run_seed_batch",
            "--model_path", MODEL,
            "--text_prompt", PROMPT,
            "--seeds", SEEDS,
            "--output_dir", str(OUT),
            "--posture_instructions", posture_name,
            "--variant", variant,
            "--variant_kwargs_json", kwarg_json,
            "--guidance_mode", "joint",
            "--motion_length", "6.0",
        ]
        print(f"[{done+1}/{n_batches}] {vname} tau={target_deg:02d} ...", flush=True)
        r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True,
                           text=True, timeout=3600)
        if r.stdout:
            for line in r.stdout.strip().split("\n"):
                if "[OK]" in line or "[batch]" in line:
                    print(f"  {line.strip()[:120]}")
        if r.returncode != 0:
            print(f"  [FAIL] rc={r.returncode}", flush=True)
        else:
            ok += 1
        done += 1
        print(f"  [{done}/{n_batches}] {time.time()-t0:.0f}s  total={time.time()-t_total:.0f}s", flush=True)

print(f"\n=== DONE: {ok}/{n_batches} batches OK, total={time.time()-t_total:.0f}s ===")
print(f"Scorecard per target: python -m eval.scorecard --run_dir {OUT}/v6*_tau05* ...")