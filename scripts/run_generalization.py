#!/usr/bin/env python3
"""scripts/run_generalization.py -- Multi-posture x multi-prompt matrix (Task 7)

V2_dps (best hit) across 4 postures x 4 prompts, N=10 seeds.
Uses batch mode (run_seed_batch.py) for joint mode runs.

TMUX: tmux new -d -s gen "source /root/miniconda3/bin/activate mdm5090 && \
    source /etc/network_turbo && cd /root/autodl-tmp/motion-diffusion-model && \
    MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \
    python scripts/run_generalization.py 2>&1 | tee output0727/generalization.log"
"""

import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "output0727" / "generalization"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH",
    str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable
SEEDS = "42,88,123,251,333,666,777,1337,2024,9999"

POSTURES = [
    ("anterior_pelvic_tilt", "APT_20deg"),       # APT 20deg, alias exists
    ("posterior_pelvic_tilt","PPT_minus20deg"),   # PPT -20deg, alias exists
    ("trendelenburg",        "lateral_tilt_5deg"), # Trendelenburg, alias exists
]

PROMPTS = [
    "a person is walking forward",
    "a person walks slowly",
    "a person walks up stairs",
    "a person turns while walking",
]

VARIANT = "v2_dps"
KWARGS = {"s": 40, "schedule": "last_quarter"}

n_batches = len(POSTURES) * len(PROMPTS)
print(f"=== Generalization Matrix ===")
print(f"Postures: {len(POSTURES)}, Prompts: {len(PROMPTS)}")
print(f"Config: {VARIANT}, N=10")
print(f"Total batches: {n_batches}")
print(f"Output: {OUT}")
print(f"Started: {time.strftime('%H:%M:%S')}")
print()

done, ok = 0, 0
t_total = time.time()
kwarg_json = json.dumps(KWARGS)

for posture_name, tag in POSTURES:
    for pi, prompt in enumerate(PROMPTS):
        ptag = prompt.replace(" ", "_")[:30]
        batch_tag = f"{tag}__prompt{pi}"
        sub_out = OUT / batch_tag
        sub_out.mkdir(parents=True, exist_ok=True)

        # Check if already done
        existing = list(sub_out.glob("*__*_joint_seed*/comparison.npy"))
        if len(existing) >= 10:
            print(f"[{done+1}/{n_batches}] {tag} / prompt={pi}  [skip: {len(existing)}]")
            done += 1; ok += 1
            continue

        t0 = time.time()
        cmd = [
            PYTHON, "-m", "scripts.run_seed_batch",
            "--model_path", MODEL,
            "--text_prompt", prompt,
            "--seeds", SEEDS,
            "--output_dir", str(sub_out),
            "--posture_instructions", posture_name,
            "--variant", VARIANT,
            "--variant_kwargs_json", kwarg_json,
            "--guidance_mode", "joint",
            "--motion_length", "6.0",
        ]
        print(f"[{done+1}/{n_batches}] {tag} prompt={pi} ({prompt[:30]}...)", flush=True)
        r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True,
                           text=True, timeout=3600)
        ok_count = r.stdout.count("[OK]")
        if r.returncode == 0 and ok_count > 0:
            ok += 1
            print(f"  [OK] {ok_count} seeds", flush=True)
        else:
            print(f"  [FAIL] rc={r.returncode} ok={ok_count}", flush=True)
            if r.stderr:
                tail = r.stderr.strip().split("\n")[-3:]
                for line in tail:
                    print(f"    {line[:200]}")
        done += 1
        print(f"  [{done}/{n_batches}] {time.time()-t0:.0f}s  total={time.time()-t_total:.0f}s", flush=True)

print(f"\n=== DONE: {ok}/{n_batches} OK, total={time.time()-t_total:.0f}s ===")