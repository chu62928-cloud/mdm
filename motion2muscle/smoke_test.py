import numpy as np, torch
from muscle_rollup import ROLLUP_GROUPS
from posture_loss import (build_reference_from_activations,
                          make_synthetic_distortion, compute_posture_loss)
from posture_loss_torch import build_group_index, compute_posture_loss_torch

# ---- fake 402-col header: union of all fascicle names referenced anywhere ----
mint_cols = sorted({f for fasc in ROLLUP_GROUPS.values() for f in fasc})
M = len(mint_cols)
B, T = 4, 28
posture = "anterior_pelvic_tilt"
rng = np.random.default_rng(0)

# a "normal" sample in [0,1]
normal = rng.uniform(0.0, 0.3, size=(B, T, M)).astype(np.float32)
ref = build_reference_from_activations(normal, mint_cols)        # {group: float}
gidx = build_group_index(mint_cols, posture, device="cpu")

def torch_loss(arr, requires_grad=False):
    x = torch.tensor(arr, dtype=torch.float32, requires_grad=requires_grad)
    L = compute_posture_loss_torch(x, gidx, posture, ref)
    return x, L

print("=== column header used:", M, "cols (smoke-test stand-in for the real 402) ===")

# 1) gradients flow
x, L = torch_loss(normal, requires_grad=True)
L.backward()
g = x.grad
print(f"[grad]  loss={float(L):.6f}  grad is None? {g is None}  "
      f"finite? {bool(torch.isfinite(g).all())}  ||grad||={float(g.norm()):.4f}")

# 2) validation property: scoring the reference sample should be ~0
_, L_normal = torch_loss(normal)
print(f"[valid] loss on the reference sample itself = {float(L_normal):.6f}  (should be ~0)")

# 3) directionality: distort TOWARD the posture -> loss should rise
distorted = make_synthetic_distortion(normal, mint_cols, posture, severity=0.8)
_, L_bad = torch_loss(distorted)
print(f"[dir]   loss on synthetic APT distortion   = {float(L_bad):.6f}  (should be >> normal)")

# 4) torch matches numpy on the same input (parity check)
np_out = compute_posture_loss(distorted, mint_cols, posture, ref)
print(f"[parity] numpy total={np_out['total']:.6f}  torch total={float(L_bad):.6f}  "
      f"abs diff={abs(np_out['total']-float(L_bad)):.2e}")
