import json, os
import numpy as np
ms = ['amb_stopped_s', 'amb_time_to_cross_s', 'norm_stopped_s_per_veh', 'max_wait_s', 'switches_per_h', 'false_preempt_s']
lines = ['HYBRID absolute means (CARLA; n = seeds; amb_time_to_cross_s is survivor-biased)', '%-4s %-13s %-8s %2s ' % ('det', 'weather', 'policy', 'n') + ' '.join('%-22s' % m for m in ms)]
for det in ('ft', 'p1'):
    for w in ('ClearNight', 'ClearNoon', 'Fog', 'HardRainNoon'):
        f = 'v12/results_hyb_%s_%s.jsonl' % (det, w)
        rows = [json.loads(l) for l in open(f, encoding='utf-8-sig') if l.strip()]
        for pol in ('preempt', 'fuzzy'):
            R = [r for r in rows if r['policy'] == pol]
            vals = []
            for m in ms:
                v = [r[m] for r in R if r.get(m) is not None]
                vals.append('%6.2f (sd %5.2f)      ' % (np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else 'n/a')
            lines.append('%-4s %-13s %-8s %2d ' % (det, w, pol, len(R)) + ' '.join(vals))
open('v12/abs_means.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
