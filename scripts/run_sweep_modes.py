#!/usr/bin/env python3
"""scripts/run_sweep_modes.py -- Parameter sweep + muscle/both mode test.

Joint mode: uses run_seed_batch.py (fast, batch)
Muscle/both: uses run_posture_comparison.py per seed (slower but native muscle support)

TMUX: tmux new -d -s sweep "source /root/miniconda3/bin/activate mdm5090 && \
    source /etc/network_turbo && cd /root/autodl-tmp/motion-diffusion-model && \
    MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \
    MUSCLE_CKPT=./motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth \
    python scripts/run_sweep_modes.py 2>&1 | tee output0727/sweep_modes.log"
"""

import os, sys, json, subprocess, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "output0727" / "sweep_modes"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL_PATH",
    str(PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"))
MUSCLE = os.environ.get("MUSCLE_CKPT",
    str(PROJ / "motion2muscle" / "checkpoints" / "transformer_baseline_full" / "net_best_loss.pth"))
PYTHON = sys.executable
BATCH_SEEDS = "42,88,123,251,333,666,777,1337,2024,9999"
SINGLE_SEEDS = [42, 88, 123, 251, 333, 666, 777, 1337, 2024, 9999]
PROMPT = "a person is walking forward"


def run_batch(variant, kwargs, posture_name, mode, tag):
    """Fast batch generation for joint mode via run_seed_batch.py."""
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    kwarg_json = json.dumps(kwargs)
    cmd = [
        PYTHON, "-m", "scripts.run_seed_batch",
        "--model_path", MODEL, "--text_prompt", PROMPT,
        "--seeds", BATCH_SEEDS, "--output_dir", str(out_dir),
        "--posture_instructions", posture_name,
        "--variant", variant, "--variant_kwargs_json", kwarg_json,
        "--guidance_mode", mode, "--motion_length", "6.0",
    ]
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True, timeout=3600)
    ok_count = r.stdout.count("[OK]")
    if r.returncode != 0:
        print(f"  [FAIL] rc={r.returncode}", flush=True); return False
    print(f"  [OK] {ok_count} seeds", flush=True); return True


def run_single(variant, kwargs, posture_name, mode, tag):
    """Per-seed generation for muscle/both via run_posture_comparison.py."""
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    kwarg_json = json.dumps(kwargs)
    ok = 0
    for seed in SINGLE_SEEDS:
        seed_dir = out_dir / f"seed{seed}"
        if (seed_dir / "comparison.npy").exists():
            ok += 1; continue
        env = os.environ.copy()
        env["GUIDANCE_VARIANT"] = variant
        env["GUIDANCE_KWARGS_JSON"] = kwarg_json
        env["GUIDANCE_MODE"] = mode
        cmd = [
            PYTHON, "-m", "scripts.run_posture_comparison",
            "--model_path", MODEL, "--text_prompt", PROMPT,
            "--seed", str(seed), "--motion_length", "6.0",
            "--num_samples", "1", "--output_dir", str(seed_dir),
            "--guidance_mode", mode,
            "--posture_instructions", posture_name,
            "--muscle_ckpt", MUSCLE, "--muscle_posture", "anterior_pelvic_tilt",
            "--muscle_assets_dir", str(PROJ / "motion2muscle"),
            "--joint_weight", "1.0", "--muscle_weight", "22.0",
        ]
        r = subprocess.run(cmd, env=env, cwd=str(PROJ), capture_output=True,
                           text=True, timeout=900)
        if r.returncode == 0:
            ok += 1
        else:
            print(f"  [FAIL] seed={seed}", flush=True); return False
    print(f"  [OK] {ok}/{len(SINGLE_SEEDS)} seeds", flush=True); return True


# ===== Config list =====
SWEEPS = [
    # Part A: Joint parameter sweep (batch mode)
    ("joint", "batch", "v2_x0_edit", {"n_inner_steps": 3, "lr": 0.5}, "anterior_pelvic_tilt", "v2b_n3"),
    ("joint", "batch", "v2_x0_edit", {"n_inner_steps": 5, "lr": 0.5}, "anterior_pelvic_tilt", "v2b_n5"),
    ("joint", "batch", "v2_x0_edit", {"n_inner_steps": 10, "lr": 0.5}, "anterior_pelvic_tilt", "v2b_n10"),
    ("joint", "batch", "v4_omni", {"K_early": 1, "K_late": 3, "lr": 0.5}, "anterior_pelvic_tilt", "v4_Kl3"),
    ("joint", "batch", "v4_omni", {"K_early": 1, "K_late": 5, "lr": 0.5}, "anterior_pelvic_tilt", "v4_Kl5"),
    ("joint", "batch", "v1_mu_sgd", {"base_weight": 60.0, "n_inner_steps": 15, "lr": 0.5}, "anterior_pelvic_tilt", "v1_w60"),
    ("joint", "batch", "v3_x0_direct", {"n_inner_steps": 15, "lr": 0.05}, "anterior_pelvic_tilt", "v3_n15"),
    ("joint", "batch", "v5_lgd", {"n_mc": 4, "s": 5.0, "mc_noise_scale": 0.05}, "anterior_pelvic_tilt", "v5_s5"),
    ("joint", "batch", "v2_dps_norm", {"s": 5.0, "schedule": "last_quarter"}, "anterior_pelvic_tilt", "v2norm_s5"),
    ("joint", "batch", "v2_dps_norm", {"s": 10.0, "schedule": "last_quarter"}, "anterior_pelvic_tilt", "v2norm_s10"),

    # Part B: Muscle/Both modes (per-seed via run_posture_comparison)
    ("muscle", "single", "v2_dps", {"s": 40, "schedule": "last_quarter"}, "anterior_pelvic_tilt", "v2_muscle"),
    ("both",   "single", "v2_dps", {"s": 40, "schedule": "last_quarter"}, "anterior_pelvic_tilt", "v2_both"),
    ("muscle", "single", "v6_closed_loop", {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50,
                                              "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03,
                                              "manifold_project": True, "loss_form": "huber",
                                              "huber_delta": 0.05, "normalize_grad": False,
                                              "band_gate": False, "spec_schedule_override": "second_half"},
     "anterior_pelvic_tilt", "v6_muscle"),
    ("both",   "single", "v6_closed_loop", {"Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50,
                                              "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03,
                                              "manifold_project": True, "loss_form": "huber",
                                              "huber_delta": 0.05, "normalize_grad": False,
                                              "band_gate": False, "spec_schedule_override": "second_half"},
     "anterior_pelvic_tilt", "v6_both"),
]


print(f"=== Parameter Sweep + Muscle/Both ===")
print(f"Configs: {len(SWEEPS)} (10 batch + 4 single)")
print(f"Output: {OUT}")
print(f"Started: {time.strftime('%H:%M:%S')}")
print()

done, ok = 0, 0
t_total = time.time()

for mode, method, variant, kwargs, posture, tag in SWEEPS:
    vname = variant.replace("_", "-")
    t0 = time.time()
    print(f"[{done+1}/{len(SWEEPS)}] mode={mode} {vname} tag={tag} ...", flush=True)

    if method == "batch":
        success = run_batch(variant, kwargs, posture, mode, tag)
    else:
        success = run_single(variant, kwargs, posture, mode, tag)

    if success: ok += 1
    done += 1
    print(f"  [{done}/{len(SWEEPS)}] {time.time()-t0:.0f}s", flush=True)

print(f"\n=== DONE: {ok}/{len(SWEEPS)} OK, total={time.time()-t_total:.0f}s ===")