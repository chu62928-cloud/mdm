#!/usr/bin/env python3
"""scripts/run_bakeoff_hotfix.py -- Re-run failed variants (V2b, V3) with correct params.

V2b (v2_x0_edit): bw → n_inner_steps=15 (default)
V3 (v3_x0_direct): n → n_inner_steps=5 (default)

Run after main bakeoff completes or in parallel.
"""

import os, sys, json, subprocess
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "output0727" / "bakeoff"
MODEL = os.environ.get("MODEL_PATH",
    str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable
SEEDS = "7,17,23,42,88,99,123,251,333,666,777,1337,2024,9999,10000"

HOTFIX = [
    ("v2_x0_edit", {"n_inner_steps": 15, "lr": 0.5}),
    ("v3_x0_direct", {"n_inner_steps": 5, "lr": 0.05}),
]

print(f"=== Hotfix: V2b, V3 with correct params ===")
print(f"Output: {OUT}")
for variant, kwargs in HOTFIX:
    vname = variant.replace("_", "-")
    kwarg_json = json.dumps(kwargs)
    print(f"\n--- {vname} ---", flush=True)
    cmd = [
        PYTHON, "-m", "scripts.run_seed_batch",
        "--model_path", MODEL,
        "--seeds", SEEDS,
        "--output_dir", str(OUT),
        "--variant", variant,
        "--variant_kwargs_json", kwarg_json,
        "--guidance_mode", "joint",
        "--motion_length", "6.0",
        "--posture_instructions", "anterior_pelvic_tilt",
    ]
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True,
                       text=True, timeout=3600)
    print(r.stdout)
    if r.stderr:
        for line in r.stderr.strip().split("\n"):
            if any(k in line.lower() for k in ["error", "traceback", "fail"]):
                print(f"  [stderr] {line[:200]}")
    print(f"  [DONE] {vname}" if r.returncode == 0 else f"  [FAIL] rc={r.returncode}")

print("\n=== Hotfix complete ===")