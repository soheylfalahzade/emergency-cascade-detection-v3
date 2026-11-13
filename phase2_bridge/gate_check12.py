import json, sys
SKIP = {'wall_s', 'det_ms_mean'}
ok = True
rows = [json.loads(l) for l in open('v12/gate_hyb.jsonl') if l.strip()]
by = {}
for r in rows:
    by.setdefault(r['policy'], []).append({k: v for k, v in r.items() if k not in SKIP})
if not (len(by) == 2 and all(len(L) == 3 for L in by.values())):
    print('FAIL: expected 2 policies x 3 repeats'); sys.exit(1)
for p, L in by.items():
    same = all(x == L[0] for x in L)
    print('determinism hybrid', p, 'identical' if same else 'DIFFERENT')
    if not same:
        for k in L[0]:
            v = [x[k] for x in L]
            if any(a != v[0] for a in v):
                print('  differs:', k, v)
    ok = ok and same
cam = [json.loads(l) for l in open('v12/gate_cam.jsonl') if l.strip()]
same = True
if len(cam) == 2:
    keys = ['norm_n_veh', 'norm_crossed', 'norm_stopped_s_per_veh', 'max_wait_s', 'amb_stopped_s', 'switches_per_h', 'box_jam_s']
    a, b = cam
    same = all(a[k] == b[k] for k in keys)
    print('camera perturbation (fixed policy, with vs without camera): ' + ('NONE detected' if same else 'TRAFFIC DIFFERS'))
    if not same:
        for k in keys:
            print('  ', k, a[k], b[k])
open('v12/camera_perturbs_traffic.txt', 'w').write('0' if same else '1')
sys.exit(0 if ok else 1)
