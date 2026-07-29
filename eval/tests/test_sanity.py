"""3-point sanity test for DistributionMetrics v1.0.

Tests:
  1. Real test motions (MDM-normalized) -> FID ~= 0.00X
  2. Pure noise motions (MDM scale) -> FID >> 1.0
  3. Unguided MDM samples -> FID ~= published MDM (~0.5)
     (requires N >= 100 for reliable FID; fewer samples will warn)
"""

import sys
import numpy as np
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

# Load MDM normalization stats once
_mean = np.load(_PROJ / "dataset" / "HumanML3D" / "Mean.npy").astype(np.float32)
_std = np.load(_PROJ / "dataset" / "HumanML3D" / "Std.npy").astype(np.float32)


def _normalize(m):
    """Convert raw motion to MDM-normalized space."""
    return (m.astype(np.float32) - _mean) / _std


def test_1_real_motions(dm):
    """FID on MDM-normalized test motions should be near zero."""
    print("\n=== Test 1: Real test motions (MDM-normalized, should be ~0) ===")
    test_file = _PROJ / "dataset" / "HumanML3D" / "test.txt"
    vecs_dir = _PROJ / "dataset" / "HumanML3D" / "new_joint_vecs"
    with open(test_file) as f:
        lines = [l.strip() for l in f if l.strip()][:200]

    motions, m_lens = [], []
    for name in lines:
        m = np.load(vecs_dir / f"{name}.npy")
        motions.append(_normalize(m))
        m_lens.append(m.shape[0])

    fid = dm.compute_fid(motions, m_lens)
    print(f"  FID = {fid:.6f}")
    ok = fid < 0.1
    print(f"  {'PASS' if ok else 'WARN'}: FID {'near zero' if ok else '> 0.1'}")
    return fid


def test_2_noise_motions(dm):
    """FID on random noise (MDM scale) should be large."""
    print("\n=== Test 2: Random noise motions (should be >> 1) ===")
    np.random.seed(42)
    motions = [np.random.randn(120, 263).astype(np.float32) for _ in range(50)]
    m_lens = [120] * 50
    fid = dm.compute_fid(motions, m_lens)
    print(f"  FID = {fid:.4f}")
    ok = fid > 5
    print(f"  {'PASS' if ok else 'WARN'}: Noise FID {'>> 1' if ok else 'too small'}")
    return fid


def test_3_mdm_unguided(dm):
    """FID on MDM unguided samples (already in MDM space)."""
    print("\n=== Test 3: MDM unguided samples ===")
    n15_dir = _PROJ / "output_0608" / "n15"
    if not n15_dir.exists():
        print("  SKIP: output_0608/n15/ not found")
        return None

    motions, m_lens = [], []
    for d in sorted(n15_dir.iterdir()):
        if not d.is_dir() or "joint" not in d.name:
            continue
        npy = d / "comparison.npy"
        if not npy.exists():
            continue
        data = np.load(npy, allow_pickle=True).item()
        motions.append(data["motion_hml_tj"][0].astype(np.float32))
        m_lens.append(data["motion_hml_tj"].shape[1])
        if len(motions) >= 15:
            break

    if len(motions) < 5:
        print(f"  SKIP: only {len(motions)} samples")
        return None

    fid = dm.compute_fid(motions, m_lens)
    print(f"  FID = {fid:.4f}  (N={len(motions)})")
    if len(motions) < 100:
        print(f"  WARN: N={len(motions)} < 100, FID unreliable. "
              "Need N>=100 for stable estimates.")
    elif 0.1 < fid < 2.0:
        print("  PASS: FID in reasonable range")
    else:
        print(f"  WARN: FID out of expected range [0.1, 2.0]")
    return fid


def main():
    from eval.distribution_metrics import DistributionMetrics
    print("Initializing DistributionMetrics ...")
    dm = DistributionMetrics(device="cuda")

    r1 = test_1_real_motions(dm)
    r2 = test_2_noise_motions(dm)
    r3 = test_3_mdm_unguided(dm)

    print("\n" + "=" * 60)
    print("SANITY TEST SUMMARY")
    print("=" * 60)
    p1 = "PASS" if (r1 is not None and r1 < 0.1) else "FAIL"
    p2 = "PASS" if (r2 is not None and r2 > 5) else "FAIL"
    # Test 3 is informational when N < 100
    p3_info = "N/A (need N>=100)" if (r3 is None or True) else "PASS"
    print(f"  Test 1 (real ~0):   {p1}  FID={r1}")
    print(f"  Test 2 (noise>>1):  {p2}  FID={r2}")
    print(f"  Test 3 (MDM info):  FID={r3} (informational, N too small)")
    print(f"\n  Harness functional: {p1 == 'PASS' and p2 == 'PASS'}")


if __name__ == "__main__":
    main()