"""V7.2 validation: 24 seeds x 5 targets x 4 methods = 480 runs."""
import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path("/root/autodl-tmp/motion-diffusion-model")
OUT = PROJ / "output0727" / "v7_2_validation"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH", str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable

SEEDS = list(range(540, 564))
SEEDS_STR = ",".join(str(s) for s in SEEDS)
PROMPT = "a person is walking forward"

TARGETS = {
    5: "anterior_pelvic_tilt_tau05", 10: "anterior_pelvic_tilt_tau10",
    15: "anterior_pelvic_tilt_tau15", 20: "anterior_pelvic_tilt",
    25: "anterior_pelvic_tilt_tau25",
}

VARIANTS = [
    ("v2_dps", "V2", {"s": 40, "schedule": "last_quarter"}),
    ("v6_closed_loop", "V6", {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50, "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03, "manifold_project": True, "loss_form": "huber", "huber_delta": 0.05, "normalize_grad": False, "band_gate": False, "spec_schedule_override": "second_half"}),
    ("v7_auto_dps", "V7.1", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10}),
    ("v7_auto_dps", "V7.2", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "control_tolerance_deg": 0.5, "evaluation_tolerance_deg": 2.0, "hysteresis_exit_deg": 3.0}),
]

n_batches = len(VARIANTS) * len(TARGETS)
print(f"=== V7.2 24-Seed Validation ===")
print(f"Seeds: 540-563 (N={len(SEEDS)})")
print(f"Targets: {list(TARGETS.keys())}")
print(f"Methods: {[v[1] for v in VARIANTS]}")
print(f"Total: {n_batches * len(SEEDS)} runs")
print()

done, ok = 0, 0
t_total = time.time()

for variant_name, label, kwargs in VARIANTS:
    for target_deg, posture_name in sorted(TARGETS.items()):
        t0 = time.time()
        vtag = label.lower().replace(".", "-")
        out_dir = OUT / f"tau{target_deg:02d}" / vtag
        kwarg_json = json.dumps(kwargs)

        cmd = [
            PYTHON, str(PROJ / "scripts" / "run_seed_batch.py"),
            "--model_path", str(MODEL), "--text_prompt", PROMPT,
            "--seeds", SEEDS_STR, "--output_dir", str(out_dir),
            "--posture_instructions", posture_name,
            "--variant", variant_name,
            "--variant_kwargs_json", kwarg_json,
            "--guidance_mode", "joint", "--motion_length", "6.0",
        ]
        print(f"[{done // len(SEEDS) + 1}/{n_batches}] {label} tau={target_deg:02d} ...", flush=True)
        r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        n_ok = r.stdout.count("[OK]")
        n_fail = r.stdout.count("[FAIL]")
        ok += n_ok
        done += len(SEEDS)
        status = "OK" if r.returncode == 0 and n_fail == 0 else "PARTIAL"
        print(f"  [{status}] {n_ok}ok/{n_fail}fail in {elapsed:.0f}s", flush=True)

total_elapsed = time.time() - t_total
print(f"\n=== DONE: {ok}/{done} OK in {total_elapsed:.0f}s ===")
