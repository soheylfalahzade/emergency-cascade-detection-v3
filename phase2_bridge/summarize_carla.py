"""Aggregate carla_runner JSONL: per-policy mean/sd/min/max over seeds and paired (same-seed)
differences against a reference policy with a bootstrap CI over seeds. Exploratory only:
small n, one junction, one map; ambulance not-crossed runs are counted separately."""
import argparse
import json
from collections import defaultdict

import numpy as np

METRICS = ["box_jam_s", "box_stop_max", "norm_crossed_per_min", "amb_stopped_s", "amb_time_to_cross_s", "norm_stopped_s_per_veh", "max_wait_s",
           "switches_per_h", "norm_n_veh", "false_preempt_s"]

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--ref", default="fixed")
ap.add_argument("--boot", type=int, default=5000)
args = ap.parse_args()

rows = [json.loads(l) for f in args.files for l in open(f) if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[r["policy"]][r["seed"]] = r
pols = sorted(by)
print("source=CARLA  rows=%d  policies=%s  seeds/policy=%s" % (len(rows), pols, {p: len(by[p]) for p in pols}))
for p in pols:
    nc = sum(1 for r in by[p].values() if not r.get("amb_crossed"))
    print("  %-12s ambulance not crossed: %d of %d" % (p, nc, len(by[p])))


def val(r, m):
    v = r.get(m)
    return None if v is None else float(v)


print("\nper-policy mean +- sd [min, max] (n)")
for m in METRICS:
    print(m)
    for p in pols:
        v = [x for x in (val(r, m) for r in by[p].values()) if x is not None]
        if v:
            print("  %-12s %8.2f +- %6.2f [%7.2f, %7.2f] (n=%d)" % (p, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0,
                                                                     min(v), max(v), len(v)))

rng = np.random.default_rng(0)
if args.ref in by:
    print("\npaired difference (policy - %s), mean [95%% bootstrap CI over seeds]" % args.ref)
    for m in METRICS:
        print(m)
        for p in pols:
            if p == args.ref:
                continue
            d = []
            for sd, r in by[p].items():
                a, b = val(r, m), val(by[args.ref].get(sd, {}), m)
                if a is not None and b is not None:
                    d.append(a - b)
            if len(d) < 2:
                print("  %-12s n=%d (too few pairs)" % (p, len(d)))
                continue
            d = np.array(d)
            bs = rng.choice(d, (args.boot, len(d)), replace=True).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print("  %-12s %+8.2f [%+7.2f, %+7.2f] (n pairs=%d)" % (p, d.mean(), lo, hi, len(d)))
