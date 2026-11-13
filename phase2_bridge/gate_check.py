import json, sys
rows = [json.loads(l) for l in open('v11/gate.jsonl') if l.strip()]
by = {}
for r in rows:
    by.setdefault(r['policy'], []).append({k: v for k, v in r.items() if k != 'wall_s'})
ok = len(by) == 2 and all(len(L) == 3 for L in by.values())
for p, L in by.items():
    same = all(x == L[0] for x in L)
    print(p, len(L), 'identical' if same else 'DIFFERENT')
    if not same:
        for k in L[0]:
            vals = [x[k] for x in L]
            if any(v != vals[0] for v in vals):
                print('  differs:', k, vals)
    ok = ok and same
sys.exit(0 if ok else 1)
