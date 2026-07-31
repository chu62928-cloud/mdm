"""Collect V7.2 (and V7.1 for comparison) diagnostic traces for representative
blind-test seeds, for Section 14 plots #6 (proposal_count vs signed_error) and
#9 (stop/reactivate timeline). This does NOT touch the frozen blind-test
statistical result (comparison.npy) -- it's a supplementary diagnostic-only
rerun of a handful of seeds with V7_TRACE_DIR/V7_TRACE_SEED set, matching the
exact locked V7.1/V7.2 kwargs used in the blind test.
"""
import os, sys, json, subprocess
from pathlib import Path

PROJ = Path("/root/autodl-tmp/motion-diffusion-model")
OUT = PROJ / "output0727" / "v7_2_blind_traces"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt")
PYTHON = sys.executable

# representative seeds identified from pareto_scatter_data.json:
#   hard-tail (most negative signed_error) and near-zero (best) at tau=20 and tau=25
REPS = [
    (20, 628, "hard_tail"), (20, 624, "good"),
    (25, 626, "hard_tail"), (25, 628, "good"),
]

VARIANTS = [
    ("v7_auto_dps", "v7-1", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10, "trace": True}),
    ("v7_auto_dps", "v7-2", {"schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10,
                              "control_tolerance_deg": 0.5, "evaluation_tolerance_deg": 2.0,
                              "hysteresis_exit_deg": 3.0, "trace": True}),
]
TARGET_POSTURE = {20: "anterior_pelvic_tilt", 25: "anterior_pelvic_tilt_tau25"}
PROMPT = "a person is walking forward"

for tdeg, seed, tag in REPS:
    posture = TARGET_POSTURE[tdeg]
    for variant_name, vlabel, kwargs in VARIANTS:
        trace_dir = OUT / f"tau{tdeg}_{tag}_seed{seed}" / vlabel
        trace_dir.mkdir(parents=True, exist_ok=True)
        out_dir = OUT / f"tau{tdeg}_{tag}_seed{seed}" / f"{vlabel}_run"
        env = os.environ.copy()
        env["V7_TRACE_DIR"] = str(trace_dir)
        env["V7_TRACE_SEED"] = str(seed)
        cmd = [PYTHON, str(PROJ / "scripts" / "run_seed_batch.py"),
               "--model_path", MODEL, "--text_prompt", PROMPT,
               "--seeds", str(seed), "--output_dir", str(out_dir),
               "--posture_instructions", posture,
               "--variant", variant_name, "--variant_kwargs_json", json.dumps(kwargs),
               "--guidance_mode", "joint", "--motion_length", "6.0"]
        print(f"[trace] tau={tdeg} seed={seed} ({tag}) {vlabel} ...", flush=True)
        r = subprocess.run(cmd, cwd=str(PROJ), env=env, capture_output=True, text=True, timeout=600)
        ok = "[OK]" in r.stdout
        print(f"  {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])

print("DONE collecting traces ->", OUT)
