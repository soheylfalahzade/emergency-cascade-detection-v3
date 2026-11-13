import json, glob, os, sys
bad = 0; n = 0
for f in sorted(glob.glob('v11/audit_*.jsonl')):
    _, tag, w, pol, s = os.path.basename(f)[:-6].split('_')
    new = json.loads(open(f).read().strip().splitlines()[-1])
    old = [json.loads(l) for l in open(f'v11/results_v10_{tag}_{w}.jsonl') if l.strip()]
    old = [r for r in old if r['policy'] == pol and str(r['seed']) == s][0]
    diff = [k for k in sorted(set(old) | set(new)) if k != 'wall_s' and old.get(k) != new.get(k)]
    n += 1
    if diff:
        bad += 1
    print(f'{tag:3s} {w:13s} {pol:8s} seed {s}: ' + ('IDENTICAL' if not diff else 'DIFFERS in ' + ', '.join(diff)))
print(f'audit: {n - bad}/{n} rerun rows identical to stored rows (all fields except wall_s)')
