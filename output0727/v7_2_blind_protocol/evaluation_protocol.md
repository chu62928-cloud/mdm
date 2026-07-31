# V7.2 Blind Test Evaluation Protocol (Section 9.3/9.4 backfill)

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
python -m eval.scorecard --run_dir output0727/v7_2_blind_test/tau<TT>/<method> \
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
