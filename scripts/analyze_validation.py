"""Analyze V7.1 12-seed validation results with A/B selection per pre-registered rules."""
import json, sys, os, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/root/autodl-tmp/motion-diffusion-model')
from diagnose_v7_results import compute_per_seed_metrics

base = 'output0727/v7_validation'
targets = {'05': 5, '10': 10, '15': 15, '20': 20, '25': 25}
variants = [
    ('v2', 'V2'), ('v6', 'V6'),
    ('v7-1-a', 'V7.1-A(max=0.10)'), ('v7-1-b', 'V7.1-B(max=0.15)'),
]

# ---- Load scorecard summaries ----
rows = []
for tau_str, tdeg in sorted(targets.items()):
    for vtag, vlabel in variants:
        sc_path = '%s/tau%s/%s/scorecard.json' % (base, tau_str, vtag)
        try:
            sc = json.load(open(sc_path))
        except:
            continue
        for s in sc.get('per_seed_control', []):
            rows.append({
                'target': tdeg, 'variant': vlabel, 'vtag': vtag,
                'seed': s.get('seed'),
                'hit_band': s.get('target_hit_band', 0),
                'delta': s.get('delta', 0), 'overshoot': s.get('overshoot', 0),
                'corr': s.get('temporal_corr', 0),
                'foot_skate_b': s.get('foot_skate_baseline', 0),
                'foot_skate_g': s.get('foot_skate_guided', 0),
            })

print('Total records: %d (expect 240)' % len(rows))

# ---- Scorecard summary ----
print()
print('=== SCORECARD SUMMARY ===')
fmt_hdr = '%-6s %-22s %8s %8s %8s %8s %8s'
fmt_row = '%-6s %-22s %8.3f %8.1f %8.2f %8.3f %8.3f'
print(fmt_hdr % ('Target', 'Method', 'hit_band', 'delta', 'overshoot', 'corr', 'fs_g'))
for tdeg in sorted(targets.values()):
    for vtag, vlabel in variants:
        subset = [r for r in rows if r['target'] == tdeg and r['vtag'] == vtag]
        if not subset:
            continue
        print(fmt_row % (
            str(tdeg), vlabel,
            np.median([r['hit_band'] for r in subset]),
            np.median([r['delta'] for r in subset]),
            np.median([r['overshoot'] for r in subset]),
            np.median([r['corr'] for r in subset]),
            np.median([r['foot_skate_g'] for r in subset]),
        ))
    print()

# ---- Extended per-seed diagnostics from .npy ----
print('=== EXTENDED PER-SEED DIAGNOSTICS ===')
diag_rows = []
for tau_str, tdeg in sorted(targets.items()):
    for vtag, vlabel in variants:
        d = '%s/tau%s/%s' % (base, tau_str, vtag)
        npy_files = sorted(glob.glob(d + '/**/comparison.npy', recursive=True))
        for npy_path in npy_files:
            data = np.load(npy_path, allow_pickle=True).item()
            m = compute_per_seed_metrics(data['motion_xyz'], data['motion_xyz_guided'], tdeg)
            diag_rows.append({
                'target': tdeg, 'variant': vlabel, 'vtag': vtag,
                'seed': data['seed'],
                'baseline_mean': m['baseline_mean_deg'],
                'guided_mean': m['guided_mean_deg'],
                'signed_err': m['signed_error_deg'],
                'abs_err': m['abs_error_deg'],
                'frame_hit': m['frame_hit_band'],
                'pos_overshoot': m['positive_overshoot_deg'],
                'neg_undershoot': m['negative_undershoot_deg'],
                'p90_overshoot': m['overshoot_p90_deg'],
                'corr': m['temporal_corr'],
                'final_hit': m['final_summary_in_band'],
            })

fmt2 = '%-6s %-22s %+10s %10s %10s %10s %10s %10s'
print(fmt2 % ('Target', 'Method', 'signed_err', 'abs_err', 'frame_hit', 'pos_over', 'neg_under', 'corr'))
for tdeg in sorted(targets.values()):
    for vtag, vlabel in variants:
        subset = [r for r in diag_rows if r['target'] == tdeg and r['vtag'] == vtag]
        if not subset:
            continue
        print(fmt2 % (
            str(tdeg), vlabel,
            '%+.2f' % np.median([r['signed_err'] for r in subset]),
            '%.2f' % np.median([r['abs_err'] for r in subset]),
            '%.3f' % np.median([r['frame_hit'] for r in subset]),
            '%.2f' % np.median([r['pos_overshoot'] for r in subset]),
            '%.2f' % np.median([r['neg_undershoot'] for r in subset]),
            '%.3f' % np.median([r['corr'] for r in subset]),
        ))
    print()

# ---- A/B decision per pre-registered rules ----
print('=== A/B DECISION (Pre-registered Section 6) ===')
a_rows = [r for r in diag_rows if r['vtag'] == 'v7-1-a']
b_rows = [r for r in diag_rows if r['vtag'] == 'v7-1-b']
v2_rows = [r for r in diag_rows if r['vtag'] == 'v2']
v6_rows = [r for r in diag_rows if r['vtag'] == 'v6']


def cross_target_median(rows_list, metric):
    by_target = {}
    for r in rows_list:
        by_target.setdefault(r['target'], []).append(r[metric])
    return np.mean([np.median(v) for v in by_target.values()])


mae_a = cross_target_median(a_rows, 'abs_err')
mae_b = cross_target_median(b_rows, 'abs_err')
hit_a = cross_target_median(a_rows, 'frame_hit')
hit_b = cross_target_median(b_rows, 'frame_hit')
corr_a = cross_target_median(a_rows, 'corr')
corr_b = cross_target_median(b_rows, 'corr')
corr_v2 = cross_target_median(v2_rows, 'corr')
corr_v6 = cross_target_median(v6_rows, 'corr')

large_a = [r for r in a_rows if r['target'] >= 20]
large_b = [r for r in b_rows if r['target'] >= 20]
mae_large_a = np.median([r['abs_err'] for r in large_a])
mae_large_b = np.median([r['abs_err'] for r in large_b])

print('Cross-target median MAE:     A=%.2f  B=%.2f  delta=%.2f' % (mae_a, mae_b, mae_a - mae_b))
print('Cross-target median hit:     A=%.3f  B=%.3f  delta=%.3f' % (hit_a, hit_b, hit_b - hit_a))
print('Cross-target median corr:    A=%.3f  B=%.3f  delta=%.3f  (V2=%.3f V6=%.3f)' % (corr_a, corr_b, corr_a - corr_b, corr_v2, corr_v6))
print('Large-target(20/25) MAE:     A=%.2f  B=%.2f  delta=%.2f' % (mae_large_a, mae_large_b, mae_large_a - mae_large_b))
print()

# Apply pre-registered rules
mae_improvement = mae_a - mae_b  # positive = B better
hit_improvement = hit_b - hit_a
corr_loss = corr_a - corr_b  # positive = A better
large_mae_improvement = mae_large_a - mae_large_b

print('Rule check:')
print('  C1: MAE improvement >= 0.25?    %.2f -> %s' % (mae_improvement, 'YES' if mae_improvement >= 0.25 else 'NO'))
print('  C2: hit improvement >= 0.03?    %.3f -> %s' % (hit_improvement, 'YES' if hit_improvement >= 0.03 else 'NO'))
print('  C3: large MAE improv >= 0.35?   %.2f -> %s' % (large_mae_improvement, 'YES' if large_mae_improvement >= 0.35 else 'NO'))
print('  C4: corr loss <= 0.03?          %.3f -> %s' % (corr_loss, 'YES' if corr_loss <= 0.03 else 'NO'))
print()

accuracy_criterion = (mae_improvement >= 0.25) or (hit_improvement >= 0.03) or (large_mae_improvement >= 0.35)
structure_ok = (corr_loss <= 0.03)

if accuracy_criterion and structure_ok:
    print('*** DECISION: SELECT V7.1-B (max_radius_rms=0.15) ***')
    selected = 'B'
elif accuracy_criterion and not structure_ok:
    print('*** DECISION: SELECT V7.1-A (max_radius_rms=0.10) — B exceeds corr budget ***')
    selected = 'A'
else:
    print('*** DECISION: SELECT V7.1-A (max_radius_rms=0.10) — B below accuracy threshold (Section 6.3 default) ***')
    selected = 'A'

# ---- Paired bootstrap ----
print()
print('=== PAIRED BOOTSTRAP (10000 samples) ===')


def paired_bootstrap(list_a, list_b, metric, n_boot=10000):
    seeds_a = {(r['target'], r['seed']): r[metric] for r in list_a}
    seeds_b = {(r['target'], r['seed']): r[metric] for r in list_b}
    common = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))
    if len(common) < 5:
        return None, None, None, len(common)
    diffs = np.array([seeds_a[k] - seeds_b[k] for k in common])
    obs = np.mean(diffs)
    n = len(diffs)
    rng = np.random.RandomState(42)
    boot = np.array([np.mean(diffs[rng.randint(0, n, n)]) for _ in range(n_boot)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    p = float(np.mean(np.abs(boot) >= np.abs(obs)))
    return obs, ci, p, n


for metric, name in [('abs_err', 'abs_error'), ('frame_hit', 'frame_hit'), ('corr', 'temporal_corr')]:
    print('%s:' % name)
    for list_x, list_y, xname, yname in [
        (a_rows, b_rows, 'V7.1-A', 'V7.1-B'),
        (a_rows, v2_rows, 'V7.1-A', 'V2'),
        (b_rows, v2_rows, 'V7.1-B', 'V2'),
    ]:
        obs, ci, p, n = paired_bootstrap(list_x, list_y, metric)
        if obs is None:
            continue
        sig = '**' if p < 0.05 else '  '
        print('  %s - %s: %+.4f  CI=[%.4f, %.4f]  p=%.4f n=%d %s' % (xname, yname, obs, ci[0], ci[1], p, n, sig))

# ---- Accuracy-matched correlation ----
print()
print('=== ACCURACY-MATCHED CORR ===')
for tdeg in sorted(targets.values()):
    print('tau=%d:' % tdeg)
    for vtag, vlabel in variants:
        t_subset = [r for r in diag_rows if r['target'] == tdeg and r['vtag'] == vtag]
        if not t_subset:
            continue
        bins = {'<=1deg': [], '1-2deg': [], '2-4deg': [], '>4deg': []}
        for r in t_subset:
            ae = r['abs_err']
            if ae <= 1:
                bins['<=1deg'].append(r['corr'])
            elif ae <= 2:
                bins['1-2deg'].append(r['corr'])
            elif ae <= 4:
                bins['2-4deg'].append(r['corr'])
            else:
                bins['>4deg'].append(r['corr'])
        parts = []
        for bn, vals in bins.items():
            if vals:
                parts.append('%s:%.3f(n=%d)' % (bn, np.median(vals), len(vals)))
        if parts:
            print('  %-22s %s' % (vlabel, '  '.join(parts)))
    print()

# ---- Per-seed worst cases ----
print('=== WORST CASES (V7.1-B, tau=25) ===')
b25 = sorted([r for r in b_rows if r['target'] == 25], key=lambda r: r['signed_err'])
for r in b25[:3]:
    print('  seed=%d signed_err=%+.1f abs=%.1f hit=%.3f corr=%.3f' % (r['seed'], r['signed_err'], r['abs_err'], r['frame_hit'], r['corr']))

b25_best = sorted([r for r in b_rows if r['target'] == 25], key=lambda r: abs(r['signed_err']))
print('  ... best:')
for r in b25_best[:3]:
    print('  seed=%d signed_err=%+.1f abs=%.1f hit=%.3f corr=%.3f' % (r['seed'], r['signed_err'], r['abs_err'], r['frame_hit'], r['corr']))

print()
print('=== DONE ===')
