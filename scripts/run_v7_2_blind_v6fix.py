"""V7.2 blind test V6-arm fix-rerun: 40 seeds x 5 targets x 1 method (V6) = 200 runs.

The original run_v7_2_blind.py used an abbreviated V6 kwargs dict that let
lambda_smooth (0.02 vs validated 0.03) and spec_schedule_override ("always"
vs validated "second_half") silently fall back to code defaults instead of
the pre-registered validated V6 config used in run_v7_2_validation.py /
run_v7_validation.py. This reruns ONLY the V6 arm with the correct, full
kwargs so it matches the validated configuration exactly. V2/V7.1/V7.2 arms
from the original 800-run blind test are untouched (already correct).
"""
import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path("/root/autodl-tmp/motion-diffusion-model")
OUT = PROJ / "output0727" / "v7_2_blind_test"
MODEL = os.environ.get("MODEL_PATH", str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
PYTHON = sys.executable

SEEDS = list(range(600, 640))
SEEDS_STR = ",".join(str(s) for s in SEEDS)
PROMPT = "a person is walking forward"

TARGETS = {5: "anterior_pelvic_tilt_tau05", 10: "anterior_pelvic_tilt_tau10",
           15: "anterior_pelvic_tilt_tau15", 20: "anterior_pelvic_tilt",
           25: "anterior_pelvic_tilt_tau25"}

VARIANT = ("v6_closed_loop", "V6", {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50,
           "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03, "manifold_project": True,
           "loss_form": "huber", "huber_delta": 0.05, "normalize_grad": False,
           "band_gate": False, "spec_schedule_override": "second_half"})

variant_name, label, kwargs = VARIANT
n_batches = len(TARGETS)
print(f"=== V7.2 Blind Test V6-ARM FIX RERUN ===")
print(f"Seeds: 600-639 (N={len(SEEDS)})")
print(f"Targets: {list(TARGETS.keys())}")
print(f"Method: {label} (corrected kwargs: lambda_smooth=0.03, spec_schedule_override=second_half)")
print(f"Total: {n_batches * len(SEEDS)} runs")

done, ok = 0, 0
t_total = time.time()

for target_deg, posture_name in sorted(TARGETS.items()):
    t0 = time.time()
    vtag = label.lower().replace(".", "-")
    out_dir = OUT / f"tau{target_deg:02d}" / vtag
    kwarg_json = json.dumps(kwargs)
    cmd = [PYTHON, str(PROJ / "scripts" / "run_seed_batch.py"),
           "--model_path", str(MODEL), "--text_prompt", PROMPT,
           "--seeds", SEEDS_STR, "--output_dir", str(out_dir),
           "--posture_instructions", posture_name,
           "--variant", variant_name, "--variant_kwargs_json", kwarg_json,
           "--guidance_mode", "joint", "--motion_length", "6.0"]
    print(f"[{done // len(SEEDS) + 1}/{n_batches}] {label} tau={target_deg:02d} ...", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True, timeout=3600)
    n_ok = r.stdout.count("[OK]"); n_fail = r.stdout.count("[FAIL]")
    ok += n_ok; done += len(SEEDS)
    print(f"  [{('OK' if r.returncode == 0 and n_fail == 0 else 'PARTIAL')}] {n_ok}ok/{n_fail}fail in {time.time() - t0:.0f}s", flush=True)

print(f"\n=== DONE: {ok}/{done} OK in {time.time() - t_total:.0f}s ===")
