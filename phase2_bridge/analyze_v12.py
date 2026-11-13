import glob, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'v12'
REF = sys.argv[2] if len(sys.argv) > 2 else 'v11'
METRICS = ['amb_stopped_s', 'amb_time_to_cross_s', 'norm_stopped_s_per_veh', 'max_wait_s', 'switches_per_h', 'false_preempt_s']

def load(f):
    out = {}
    if os.path.exists(f):
        for l in open(f, encoding='utf-8-sig'):
            if l.strip():
                r = json.loads(l); out.setdefault(r['policy'], {})[r['seed']] = r
    return out

def boot(d, n=4000):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float('nan'), float('nan'))
    m = np.random.default_rng(0).choice(d, (n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def paired(a, b, key):
    seeds = sorted(set(a) & set(b))
    d = [a[s][key] - b[s][key] for s in seeds if a[s].get(key) is not None and b[s].get(key) is not None]
    if not d:
        return None
    lo, hi = boot(d)
    return float(np.mean(d)), lo, hi, len(d)

def fmt(p):
    return 'n/a' if p is None else '%+7.2f [%+7.2f, %+7.2f] n=%d' % p

out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

for det in ('ft', 'p1'):
    for w in ('ClearNight', 'ClearNoon', 'Fog', 'HardRainNoon'):
        R = load(os.path.join(D, 'results_hyb_%s_%s.jsonl' % (det, w)))
        if not R:
            continue
        P('\n=== HYBRID (CARLA, detector=%s, weather=%s)' % (det, w))
        rows = [r for p in R.values() for r in p.values()]
        vis = sum(r['det_amb_vis_frames'] for r in rows)
        P('detector on emergency vehicles visible to the camera (frames, pooled over %d runs): visible=%d  found(IoU>=0.5, any class)=%.2f  recognised as emergency=%.2f  exact class=%.2f' % (
            len(rows), vis, sum(r['det_amb_found_frames'] for r in rows) / max(vis, 1),
            sum(r['det_amb_emerg_frames'] for r in rows) / max(vis, 1), sum(r['det_amb_cls_frames'] for r in rows) / max(vis, 1)))
        never = [r for r in rows if r['det_amb_vis_frames'] > 0 and r['det_amb_emerg_frames'] == 0]
        lat = [r['det_first_emerg_t'] - r['det_first_vis_t'] for r in rows if r['det_first_emerg_t'] is not None and r['det_first_vis_t'] is not None]
        P('runs where the ambulance was visible but NEVER recognised as emergency: %d/%d | latency first-visible -> first-recognised (s): %s' % (
            len(never), len(rows), 'n/a' if not lat else 'median %.1f min %.1f max %.1f (n=%d)' % (np.median(lat), min(lat), max(lat), len(lat))))
        win_min = max(rows[0]['duration_s'] - rows[0]['warmup_s'], 1.0) / 60.0
        P('false emergency alarms (no actor under the box): frames/run=%.1f  detections/min=%.2f | ordinary vehicles reported as emergency: %.1f/run | ambiguous(IoU 0.3-0.5): %.1f/run | detector ms/frame=%.0f | frame lag max=%d' % (
            np.mean([r['det_fp_emerg_frames'] for r in rows]), sum(r['det_fp_emerg_n'] for r in rows) / (len(rows) * win_min),
            np.mean([r['det_car_as_emerg_n'] for r in rows]), np.mean([r['det_ambig_emerg_n'] for r in rows]),
            np.mean([r['det_ms_mean'] for r in rows]), max(r['det_lag_max'] for r in rows)))
        if 'fuzzy' in R and 'preempt' in R:
            P('closed loop, fuzzy - preempt (paired by seed, 95%% bootstrap CI); ambulance not crossed: fuzzy=%d preempt=%d' % (
                sum(1 for r in R['fuzzy'].values() if not r['amb_crossed']), sum(1 for r in R['preempt'].values() if not r['amb_crossed'])))
            for m in METRICS:
                P('  %-24s %s' % (m, fmt(paired(R['fuzzy'], R['preempt'], m))))
        RF = load(os.path.join(REF, 'results_v10_%s_%s.jsonl' % (det, w)))
        for pol in ('preempt', 'fuzzy'):
            if pol in R and pol in RF:
                P('hybrid - injected-noise model (%s, same seeds):' % pol)
                for m in ('amb_stopped_s', 'false_preempt_s', 'norm_stopped_s_per_veh'):
                    P('  %-24s %s' % (m, fmt(paired(R[pol], RF[pol], m))))
open(os.path.join(D, 'analysis_v12.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
