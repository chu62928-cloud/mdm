"""Backfill V7.2 blind-test protocol freeze artifacts (Section 9.3).

NOTE ON TIMING: per Section 9.3 these artifacts should ideally have been
written BEFORE the blind run executed. In practice the 800-run job was
launched immediately after the 24-seed validation passed, and this freeze
package is being generated after the fact. This script writes an honest
audit trail: it records the git commit and checkpoint checksum that were
ACTUALLY active for the run (verified against the working tree at the time
via `git log`/`git status` clean-check performed manually before this
script ran), plus the discovered-and-fixed V6 kwargs bug and its rerun.
Nothing here is used to justify changing any already-computed result;
it exists purely as the traceability record the plan requires.
"""
import json, hashlib, subprocess
from pathlib import Path

PROJ = Path("/root/autodl-tmp/motion-diffusion-model")
OUT = PROJ / "output0727" / "v7_2_blind_protocol"
OUT.mkdir(parents=True, exist_ok=True)

# ---- git_commit.txt ----
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ, capture_output=True, text=True).stdout.strip()
commit_short = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJ, capture_output=True, text=True).stdout.strip()
(OUT / "git_commit.txt").write_text(
    "commit: %s\nshort: %s\n"
    "note: this is the commit that was HEAD when the 800-run blind test (scripts/run_v7_2_blind.py)\n"
    "was launched, and remained HEAD (working tree clean, verified via git status --short) through\n"
    "completion and through the V6-arm fix rerun (scripts/run_v7_2_blind_v6fix.py).\n" % (commit, commit_short)
)

# ---- checkpoint_checksum.txt ----
ckpt_path = PROJ / "save" / "humanml_trans_dec_512_bert" / "model000600000.pt"
h = hashlib.sha256()
with open(ckpt_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
checksum = h.hexdigest()
(OUT / "checkpoint_checksum.txt").write_text(
    "path: %s\nsha256: %s\n" % (ckpt_path.relative_to(PROJ), checksum)
)

# ---- locked_config.json ----
locked_config = {
    "v2_config": {"s": 40, "schedule": "last_quarter"},
    "v6_config": {
        "Kp": 80, "Ki": 1, "Kd": 5, "s_min": 0.05, "s_max": 50,
        "I_max": 20, "beta_ema": 0.8, "lambda_smooth": 0.03, "manifold_project": True,
        "loss_form": "huber", "huber_delta": 0.05, "normalize_grad": False,
        "band_gate": False, "spec_schedule_override": "second_half",
    },
    "v7_1_config": {
        "schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10,
    },
    "v7_2_config": {
        "schedule": "second_half", "max_backtracks": 3, "max_radius_rms": 0.10,
        "control_tolerance_deg": 0.5, "evaluation_tolerance_deg": 2.0, "hysteresis_exit_deg": 3.0,
    },
    "source_of_truth": "scripts/run_v7_2_validation.py (24-seed validation, commit f484416) -- "
                        "the blind test config must match this exactly for all 4 methods.",
    "known_deviation_fixed": {
        "arm": "v6",
        "description": "scripts/run_v7_2_blind.py originally used abbreviated V6 kwargs "
                        "{Kp,Ki,Kd,s_min,s_max} only, omitting I_max/beta_ema/lambda_smooth/"
                        "manifold_project/loss_form/huber_delta/normalize_grad/band_gate/"
                        "spec_schedule_override. These silently fell back to _guidance_v6_closed_loop() "
                        "code defaults (lambda_smooth=0.02 vs validated 0.03; "
                        "spec_schedule_override='always' vs validated 'second_half'). "
                        "Detected before scorecard generation; original v6 outputs preserved at "
                        "tau*/v6_STALE_wrong_kwargs/ for audit; V6 arm rerun with corrected kwargs "
                        "via scripts/run_v7_2_blind_v6fix.py (200/200 OK). V2/V7.1/V7.2 arms were "
                        "unaffected (kwargs matched validation script from the start).",
    },
}
(OUT / "locked_config.json").write_text(json.dumps(locked_config, indent=2))

# ---- seed_split.json ----
seed_split = {
    "blind_seeds": list(range(600, 640)),
    "excluded_prior_sets": {
        "smoke": [0, 1, 2, 3, 42],
        "autocal": list(range(100, 130)),
        "tuning_v71": [200, 201, 202, 203, 204],
        "diagnostic_v71": [101, 102, 103, 105, 106, 110, 111, 112, 127],
        "validation_v71": list(range(300, 312)),
        "blind_v71": list(range(400, 440)),
        "v72_pilot": None,
        "v72_band_sweep": list(range(500, 520)),
        "v72_validation": list(range(540, 564)),
    },
    "overlap_check": "600-639 has zero overlap with any prior tuning/validation/blind set used "
                     "anywhere in the V7/V7.1/V7.2 project.",
}
(OUT / "seed_split.json").write_text(json.dumps(seed_split, indent=2))

# ---- success_criteria.json (Section 9.4, pre-registered BEFORE seeing blind results) ----
success_criteria = {
    "primary": {
        "cross_target_abs_error": "significantly lower for V7.2 vs V7.1 (paired bootstrap 95% CI excludes 0)",
        "signed_bias_20_25": "pooled median (tau=20 and tau=25 combined) within [-0.75, +0.75] deg -- "
                              "SAME statistic and threshold as Section 8.4 validation gate",
        "frame_hit": "significantly higher for V7.2 vs V7.1 (paired bootstrap 95% CI excludes 0)",
    },
    "non_inferiority": {
        "corr_drop": "<= 0.03 (V7.1 - V7.2, cross-target mean)",
        "foot_skate_increase": "<= 0.01 (V7.2 - V7.1, cross-target mean)",
        "failure_rate": "must not increase vs V7.1",
    },
    "statistics": {
        "method": "paired bootstrap, seed as resampling unit",
        "n_resamples": 10000,
        "reporting": ["mean", "median", "paired_difference", "95%_CI", "effect_size_cohend",
                      "per_seed_scatter", "pareto_plot"],
    },
    "note": "Written into this file AFTER the blind run's raw comparison.npy files were already "
            "generated (see git_commit.txt for the timing caveat), but BEFORE any scorecard or "
            "bootstrap analysis was computed on the blind seeds. These criteria are copied verbatim "
            "from Section 9.4 of V7_2_Three_Band_Execution_Plan.md, which was authored and committed "
            "prior to launching the blind run.",
}
(OUT / "success_criteria.json").write_text(json.dumps(success_criteria, indent=2))

# ---- evaluation_protocol.md ----
(OUT / "evaluation_protocol.md").write_text("""# V7.2 Blind Test Evaluation Protocol (Section 9.3/9.4 backfill)

## Timing note

This freeze package was generated after the 800 raw comparison.npy files
were already produced (Section 9.3 ideally wants it created before the run).
The commit and checkpoint recorded here are verified to be the exact ones
active throughout the run and its V6-arm fix rerun. No parameter was changed,
and no result was viewed, between when the run was launched and when this
package was written -- the only intervening action was discovering and
fixing the V6 kwargs omission bug (see locked_config.json
`known_deviation_fixed`), which was fixed by rerunning V6 only, not by
altering the already-valid V2/V7.1/V7.2 outputs.

## Seeds

600-639 (N=40), zero overlap with any tuning/validation/blind set used
anywhere else in the project (see seed_split.json).

## Matrix

40 seeds x 5 targets (5/10/15/20/25 deg anterior pelvic tilt) x 4 methods
(V2 DPS, V6 PID closed-loop, V7.1 single-band, V7.2 three-band) = 800 runs.

## Scorecard generation

```
python -m eval.scorecard --run_dir output0727/v7_2_blind_test/tau<TT>/<method> \\
  --posture <posture_name> --target <target_deg> --tolerance 2.0 --no_dist_metrics
```

## Primary analysis

`scripts/analyze_v72_blind.py` -- implements Section 9.4 exactly:
paired bootstrap (10,000 resamples, seed as unit), per-target AND
cross-target hierarchical summary, mean/median/CI/effect-size/scatter/Pareto.

## Decision rule

Success criteria fixed in success_criteria.json (copied verbatim from the
plan, authored before the blind run). The 20/25 signed-bias threshold uses
the pooled MEDIAN as the primary statistic (matching Section 8.4's exact
definition) -- the MEAN is also reported for transparency but is NOT the
gating statistic, since the plan requires reporting both without swapping
which one gates the decision after the fact.
""")

# ---- run_command.sh ----
(OUT / "run_command.sh").write_text("""#!/bin/bash
# V7.2 800-run blind test -- exact commands used (backfilled record)
set -e
cd /root/autodl-tmp/motion-diffusion-model
source /root/miniconda3/bin/activate mdm5090
source /etc/network_turbo

# Main run (V2/V6/V7.1/V7.2 x 5 targets x 40 seeds) -- V6 arm's kwargs were
# later found incomplete and rerun separately, see run_v7_2_blind_v6fix.py
MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \\
  python scripts/run_v7_2_blind.py 2>&1 | tee output0727/v7_2_blind_test/run.log

# V6-arm fix rerun (corrected kwargs matching validated config)
MODEL_PATH=./save/humanml_trans_dec_512_bert/model000600000.pt \\
  python scripts/run_v7_2_blind_v6fix.py 2>&1 | tee output0727/v7_2_blind_test/run_v6fix.log

# Scorecards (all 20 target x method combinations)
for tau in 05 10 15 20 25; do
  if [ "$tau" = "20" ]; then posture=anterior_pelvic_tilt; else posture=anterior_pelvic_tilt_tau$tau; fi
  tdeg=$((10#$tau))
  for v in v2 v6 v7-1 v7-2; do
    python -m eval.scorecard --run_dir output0727/v7_2_blind_test/tau$tau/$v \\
      --posture "$posture" --target "$tdeg" --tolerance 2.0 --no_dist_metrics
  done
done

# Primary analysis (Section 9.4 protocol)
python scripts/analyze_v72_blind.py
""")

print("Wrote freeze package to", OUT)
for f in sorted(OUT.iterdir()):
    print(" -", f.name)
