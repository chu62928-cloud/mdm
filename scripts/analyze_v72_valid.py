"""V7.2 validation analysis + Go/No-Go decision."""
import json, numpy as np, sys, glob
sys.path.insert(0, '/root/autodl-tmp/motion-diffusion-model')
from scripts.diagnose_v7_results import compute_per_seed_metrics

base = 'output0727/v7_2_validation'
targets = {5: '05', 10: '10', 15: '15', 20: '20', 25: '25'}

# ---- 1. Scorecard summary ----
print("=== SCORECARD SUMMARY ===")
for tdeg, tstr in sorted(targets.items()):
    for method, mlabel in [('v2', 'V2'), ('v6', 'V6'), ('v7-1', 'V7.1'), ('v7-2', 'V7.2')]:
        sc = json.load(open('%s/tau%s/%s/scorecard.json' % (base, tstr, method)))
        m = sc['control_metrics_aggregated']
        print('tau=%2d %s hit=%.3f corr=%.3f delta=%.1f fs=%.4f' % (
            tdeg, mlabel, m['target_hit_band']['median'],
            m['temporal_corr']['median'], m['delta']['median'],
            m['foot_skate_guided']['median']))
    print()

# ---- 2. Cross-target medians ----
print("=== CROSS-TARGET MEDIANS ===")
for method, mlabel in [('v2', 'V2'), ('v6', 'V6'), ('v7-1', 'V7.1'), ('v7-2', 'V7.2')]:
    vals = {'corr': [], 'hit': [], 'fs': []}
    for tstr in targets.values():
        sc = json.load(open('%s/tau%s/%s/scorecard.json' % (base, tstr, method)))
        m = sc['control_metrics_aggregated']
        vals['corr'].append(m['temporal_corr']['median'])
        vals['hit'].append(m['target_hit_band']['median'])
        vals['fs'].append(m['foot_skate_guided']['median'])
    print('%s: corr=%.3f hit=%.3f fs=%.4f' % (
        mlabel, np.mean(vals['corr']), np.mean(vals['hit']), np.mean(vals['fs'])))

# ---- 3. Extended per-seed diagnostics ----
print()
print("=== EXTENDED DIAGNOSTICS (signed_err = guided_mean - target) ===")
all_data = {}
for method, mlabel in [('v2', 'V2'), ('v6', 'V6'), ('v7-1', 'V7.1'), ('v7-2', 'V7.2')]:
    results = []
    for tdeg, tstr in sorted(targets.items()):
        npy_files = glob.glob('%s/tau%s/%s/**/comparison.npy' % (base, tstr, method), recursive=True)
        for npy_path in npy_files:
            d = np.load(npy_path, allow_pickle=True).item()
            m = compute_per_seed_metrics(d['motion_xyz'], d['motion_xyz_guided'], tdeg)
            results.append({'tau': tdeg, 'signed': m['signed_error_deg'],
                           'abs': m['abs_error_deg'], 'hit': m['frame_hit_band'],
                           'corr': m['temporal_corr'], 'seed': d['seed']})
    all_data[method] = results

    for tdeg in sorted(targets):
        sub = [r for r in results if r['tau'] == tdeg]
        if sub:
            print('%s tau=%2d signed=%+.2f abs=%.2f hit=%.3f corr=%.3f (N=%d)' % (
                mlabel, tdeg, np.median([r['signed'] for r in sub]),
                np.median([r['abs'] for r in sub]),
                np.median([r['hit'] for r in sub]),
                np.median([r['corr'] for r in sub]), len(sub)))
    print()

# ---- 4. V7.2 vs V7.1 comparison ----
print("=== V7.2 vs V7.1 ===")
v71 = all_data['v7-1']
v72 = all_data['v7-2']

v71_ct_abs = np.mean([np.median([r['abs'] for r in v71 if r['tau'] == t]) for t in [5,10,15,20,25]])
v72_ct_abs = np.mean([np.median([r['abs'] for r in v72 if r['tau'] == t]) for t in [5,10,15,20,25]])
v71_ct_corr = np.mean([np.median([r['corr'] for r in v71 if r['tau'] == t]) for t in [5,10,15,20,25]])
v72_ct_corr = np.mean([np.median([r['corr'] for r in v72 if r['tau'] == t]) for t in [5,10,15,20,25]])

v71_2025_signed = np.median([r['signed'] for r in v71 if r['tau'] >= 20])
v72_2025_signed = np.median([r['signed'] for r in v72 if r['tau'] >= 20])
v71_2025_hit = np.mean([np.median([r['hit'] for r in v71 if r['tau'] == t]) for t in [20,25]])
v72_2025_hit = np.mean([np.median([r['hit'] for r in v72 if r['tau'] == t]) for t in [20,25]])

abs_improvement = v71_ct_abs - v72_ct_abs
corr_drop = v71_ct_corr - v72_ct_corr
hit_improvement = v72_2025_hit - v71_2025_hit

print('Cross-target abs_err: V7.1=%.2f V7.2=%.2f improvement=%.2f' % (v71_ct_abs, v72_ct_abs, abs_improvement))
print('Cross-target corr:    V7.1=%.3f V7.2=%.3f drop=%.3f' % (v71_ct_corr, v72_ct_corr, corr_drop))
print('20/25 signed median:  V7.1=%+.2f V7.2=%+.2f' % (v71_2025_signed, v72_2025_signed))
print('20/25 hit mean:       V7.1=%.3f V7.2=%.3f improvement=%.3f' % (v71_2025_hit, v72_2025_hit, hit_improvement))

# ---- 5. Go/No-Go per Section 8.4 ----
print()
print("=== GO/NO-GO (Section 8.4) ===")
checks = [
    ('Cross-target abs >= 0.30 lower', abs_improvement >= 0.30, abs_improvement),
    ('20/25 signed in [-0.75,+0.75]', abs(v72_2025_signed) <= 0.75, v72_2025_signed),
    ('20/25 hit >= 0.08 higher', hit_improvement >= 0.08, hit_improvement),
    ('Cross-target corr drop <= 0.03', corr_drop <= 0.03, corr_drop),
]
all_pass = True
for desc, passed, val in checks:
    status = "PASS" if passed else "FAIL"
    print('  [%s] %s (%.3f)' % (status, desc, val))
    if not passed:
        all_pass = False

# Per-target checks
print()
for tdeg in [20, 25]:
    v72_t = [r for r in v72 if r['tau'] == tdeg]
    v71_t = [r for r in v71 if r['tau'] == tdeg]
    s72 = np.median([r['signed'] for r in v72_t])
    s71 = np.median([r['signed'] for r in v71_t])
    c72 = np.median([r['corr'] for r in v72_t])
    c71 = np.median([r['corr'] for r in v71_t])
    print('tau=%d: V7.1 signed=%+.2f abs=%.2f corr=%.3f | V7.2 signed=%+.2f abs=%.2f corr=%.3f' % (
        tdeg, s71, np.median([r['abs'] for r in v71_t]), c71,
        s72, np.median([r['abs'] for r in v72_t]), c72))

print()
if all_pass:
    print("*** GO: V7.2 passes validation. Proceed to blind test. ***")
else:
    print("*** NO-GO: V7.2 does not pass. V7.1 remains main result. ***")
