# Frozen motion→muscle proxy checkpoint

This directory should contain the frozen proxy weights used by the
muscle-activation guidance module (Module 2):

```
motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth
```

The weight file (`*.pth`) is **not** tracked in git (it is a large binary,
excluded by `.gitignore`). To reproduce the muscle-guidance / muscle-space
results you must place `net_best_loss.pth` here.

- Architecture: 16-layer transformer, width=256, nhead=8, final activation = none.
  See `motion2muscle/models.py` (`MyTransformer`) — its defaults match this checkpoint.
- The loader (`muscle_guidance_mdm/loader.py`) reads from this path by default;
  override with the `MUSCLE_CKPT` environment variable / `--muscle_ckpt` flag.
- Provenance and training details: see `motion2muscle/HANDOFF.md`.
