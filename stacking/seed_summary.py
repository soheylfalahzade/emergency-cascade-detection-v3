import json, os
import numpy as np

R = os.path.expanduser('~/Portfolio/Smart_Traffic_System/results_v3')
J = json.load(open(R + '/seed_eval.json'))
print(J['note'], '| seeds used:', list(J['seeds']))
if len(J['seeds']) < 3:
    print('WARNING: fewer than three seeds; the spread below is only a coarse indicator, not an interval')
for part in ('dev', 'holdout'):
    D = J[part]; ss = sorted(D)
    print(f"\n=== {part} (light-detector seeds {ss}; thresholds frozen with seed 0; heavy detector fixed)")
    for sy in D[ss[0]]:
        if not any(t in sy for t in ('light_only', 'heavy_only', 'a0.02')):
            continue
        row = {}
        for m in ('map50', 'emerg_map50', 'ap50_firetruck', 'emerg_recall', 'esc', 'loss_rate'):
            v = [D[s][sy].get(m) for s in ss]
            if any(x is None for x in v):
                continue
            row[m] = f"{np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
        print(f"{sy:<14}", row)
    for m in ('map50', 'emerg_map50'):
        den = [D[s]['heavy_only'][m] - D[s]['light_only'][m] for s in ss]
        gap = [(D[s]['cascade_a0.02'][m] - D[s]['light_only'][m]) / d for s, d in zip(ss, den)]
        rnd = [(D[s]['random_a0.02'][m] - D[s]['light_only'][m]) / d for s, d in zip(ss, den)]
        dif = [D[s]['cascade_a0.02'][m] - D[s]['random_a0.02'][m] for s in ss]
        print(f"{m}: gap closed per seed: cascade {np.round(gap, 3).tolist()} | random {np.round(rnd, 3).tolist()} | cascade minus random {np.round(dif, 4).tolist()}")
