"""Phase 2A ablation: 6 V7 internal variants + V2 + V6 on 12 seeds x 3 targets."""
import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path("/root/autodl-tmp/motion-diffusion-model")
OUT = PROJ / "output0727" / "v7_ablation"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH", str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable

SEEDS = list(range(300, 312))
SEEDS_STR = ",".join(str(s) for s in SEEDS)
PROMPT = "a person is walking forward"

TARGETS = {10: "anterior_pelvic_tilt_tau10", 20: "anterior_pelvic_tilt", 25: "anterior_pelvic_tilt_tau25"}

# 6 V7 variants + V2 + V6 = 8 methods
VARIANTS = [
    ("v2_dps", "V2-fixed", {"s": 40, "schedule": "last_quarter"}),
    ("v6_closed_loop", "V6-PID", {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50, "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03, "manifold_project": True, "loss_form": "huber", "huber_delta": 0.05, "normalize_grad": False, "band_gate": False, "spec_schedule_override": "second_half"}),
    ("v7_auto_dps", "V7.0-unfixed", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "band_order_bug": True}),
    ("v7_auto_dps", "V7.1-full", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10}),
    ("v7_auto_dps", "V7-no-trial", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "disable_trial": True}),
    ("v7_auto_dps", "V7-no-band", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "disable_band_stop": True}),
    ("v7_auto_dps", "V7-static-r", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "shrink_factor": 1.0, "grow_factor": 1.0}),
    ("v7_auto_dps", "V7-no-trial-band", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "disable_trial": True, "disable_band_stop": True}),
]

n_batches = len(VARIANTS) * len(TARGETS)
print(f"=== Phase 2A Ablation ===")
print(f"Seeds: {SEEDS_STR} (N={len(SEEDS)})")
print(f"Targets: {list(TARGETS.keys())}")
print(f"Methods: {[v[1] for v in VARIANTS]}")
print(f"Total batches: {n_batches} x {len(SEEDS)} seeds = {n_batches * len(SEEDS)} runs")
print(f"Output: {OUT}")
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
        print(f"[{done // len(SEEDS) + 1}/{n_batches}] {label:20s} tau={target_deg:02d} ...", flush=True)
        r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        n_ok = r.stdout.count("[OK]")
        ok += n_ok
        done += len(SEEDS)
        status = "OK" if r.returncode == 0 and r.stdout.count("[FAIL]") == 0 else "PARTIAL"
        print(f"  [{status}] {n_ok}ok in {elapsed:.0f}s", flush=True)

total_elapsed = time.time() - t_total
print(f"\n=== DONE: {ok}/{done} OK in {total_elapsed:.0f}s ===")
print(f"Output: {OUT}")
