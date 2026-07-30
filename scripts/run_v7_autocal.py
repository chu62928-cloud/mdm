#!/usr/bin/env python3
"""V7 Auto-Calibration — locked-config experiment across 5 APT targets.

Compares V2-global, V6-global, V7-global on identical seeds.
Each variant uses ONE fixed config for all targets.
"""

import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(PROJ))

OUT = Path(os.environ.get("V7_AUTOCAL_OUT", str(PROJ / "output0727" / "v7_autocal")))
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH",
    str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable

# ---- CONFIG ----
N_SEEDS = int(os.environ.get("V7_N_SEEDS", "30"))
SEED_BASE = int(os.environ.get("V7_SEED_BASE", "100"))
SEEDS = ",".join(str(SEED_BASE + i) for i in range(N_SEEDS))

PROMPT = "a person is walking forward"

TARGETS = {
    5:  "anterior_pelvic_tilt_tau05",
    10: "anterior_pelvic_tilt_tau10",
    15: "anterior_pelvic_tilt_tau15",
    20: "anterior_pelvic_tilt",
    25: "anterior_pelvic_tilt_tau25",
}

# ---- Locked configs (single config for ALL targets) ----
V7_KWARGS = {
    "schedule": "second_half",
    "radius_scale": 0.05,
    "max_backtracks": 3,
    "trace": True,
}
V2_KWARGS = {"s": 40, "schedule": "last_quarter"}
V6_KWARGS = {
    "Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50,
    "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03,
    "manifold_project": True, "loss_form": "huber", "huber_delta": 0.05,
    "normalize_grad": False, "band_gate": False,
    "spec_schedule_override": "second_half",
}

CONFIGS = [
    ("v7_auto_dps", V7_KWARGS),
    ("v2_dps", V2_KWARGS),
    ("v6_closed_loop", V6_KWARGS),
]

n_batches = len(CONFIGS) * len(TARGETS)
print(f"=== V7 Auto-Calibration ===")
print(f"Targets: {list(TARGETS.keys())} deg")
print(f"Seeds: {N_SEEDS} (base={SEED_BASE})")
print(f"Configs: {len(CONFIGS)}")
print(f"Total batches: {n_batches} (each: {N_SEEDS} seeds)")
print(f"Output: {OUT}")
print(f"Started: {time.strftime('%H:%M:%S')}")
print()

done, ok = 0, 0
t_total = time.time()

for variant, kwargs in CONFIGS:
    vname = variant.replace("_", "-")
    for target_deg, posture_name in sorted(TARGETS.items()):
        t0 = time.time()
        kwarg_json = json.dumps(kwargs)
        out_dir = OUT / f"tau{target_deg:02d}" / vname

        cmd = [
            PYTHON, str(PROJ / "scripts" / "run_seed_batch.py"),
            "--model_path", MODEL,
            "--text_prompt", PROMPT,
            "--seeds", SEEDS,
            "--output_dir", str(out_dir),
            "--posture_instructions", posture_name,
            "--variant", variant,
            "--variant_kwargs_json", kwarg_json,
            "--guidance_mode", "joint",
            "--motion_length", "6.0",
        ]
        print(f"[{done+1}/{n_batches}] {vname} tau={target_deg:02d} ...", flush=True)
        r = subprocess.run(
            cmd, cwd=str(PROJ),
            capture_output=True, text=True, timeout=7200,
        )
        elapsed = time.time() - t0

        # Count successes
        n_ok = r.stdout.count("[OK]")
        n_fail = r.stdout.count("[FAIL]")
        n_skip = r.stdout.count("[skip]")
        ok += n_ok
        done += N_SEEDS

        status = "OK" if r.returncode == 0 and n_fail == 0 else "PARTIAL"
        print(f"  [{status}] {n_ok}ok/{n_fail}fail/{n_skip}skip in {elapsed:.0f}s", flush=True)

        # Save run log
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"tau{target_deg:02d}.log", "w") as f:
            f.write(r.stdout)
            if r.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(r.stderr)

        if r.returncode != 0:
            print(f"  [STDERR] {r.stderr[:500]}", flush=True)

total_elapsed = time.time() - t_total
print(f"\n=== DONE ===")
print(f"Total: {ok}/{done} seeds OK")
print(f"Failed: {done - ok}")
print(f"Time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
print(f"Output: {OUT}")
