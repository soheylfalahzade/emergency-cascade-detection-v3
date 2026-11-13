#!/usr/bin/env python
"""One clean table for the measured-noise sweep: per (condition, weather) the preempt and fuzzy means and the paired difference
fuzzy - preempt with a 95% bootstrap CI over seeds. results_v10b_* files (fuzzy with the visibility proxy) are compared with
the matching results_v10_* fuzzy rows. CARLA simulation only; tag p1 = original Phase-1 detector, ft = CARLA-fine-tuned detector.
Usage: python table_v10.py [dir=.] [--csv out.csv]   NOT RUN ON REAL DATA YET."""
import csv
import glob
import json
import os
import sys

import numpy as np

METRICS = ["amb_stopped_s", "amb_time_to_cross_s", "norm_stopped_s_per_veh", "max_wait_s", "switches_per_h", "false_preempt_s"]
d = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "."
out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None


def load(path):
    by = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            by.setdefault(r["policy"], {})[r["seed"]] = r
    return by


def boot(diff, n=5000):
    diff = np.asarray(diff, float)
    if len(diff) < 2:
        return float(diff.mean()), None, None
    m = np.random.default_rng(0).choice(diff, (n, len(diff)), replace=True).mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pair(a, b, metric):
    seeds = sorted(set(a) & set(b))
    va = [a[s][metric] for s in seeds if a[s].get(metric) is not None and b[s].get(metric) is not None]
    vb = [b[s][metric] for s in seeds if a[s].get(metric) is not None and b[s].get(metric) is not None]
    if not va:
        return None
    mean_d, lo, hi = boot(np.array(va) - np.array(vb))
    return float(np.mean(va)), float(np.mean(vb)), mean_d, lo, hi, len(va)


rows = []
for f in sorted(glob.glob(os.path.join(d, "results_v10*_*.jsonl"))):
    name = os.path.basename(f)[:-6]
    by = load(f)
    if name.startswith("results_v10b_"):
        ref = os.path.join(d, name.replace("results_v10b_", "results_v10_") + ".jsonl")
        if not os.path.exists(ref) or "fuzzy" not in by:
            continue
        a, b, label = by["fuzzy"], load(ref).get("fuzzy", {}), "fuzzy(visibility proxy) - fuzzy(no visibility)"
    else:
        if "fuzzy" not in by or "preempt" not in by:
            continue
        a, b, label = by["fuzzy"], by["preempt"], "fuzzy - preempt"
    nx = sum(1 for r in a.values() if not r.get("amb_crossed")), sum(1 for r in b.values() if not r.get("amb_crossed"))
    print("\n=== %s   [%s]   ambulance not crossed: first=%d second=%d" % (name, label, nx[0], nx[1]))
    print("%-24s %10s %10s %9s %20s %3s" % ("metric", "first", "second", "diff", "95% CI", "n"))
    for m in METRICS:
        r = pair(a, b, m)
        if r is None:
            continue
        ci = "[%+.2f, %+.2f]" % (r[3], r[4]) if r[3] is not None else "n/a"
        print("%-24s %10.2f %10.2f %+9.2f %20s %3d" % (m, r[0], r[1], r[2], ci, r[5]))
        rows.append([name, label, m, r[0], r[1], r[2], r[3], r[4], r[5]])
if out_csv:
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "comparison", "metric", "first_mean", "second_mean", "diff", "ci_lo", "ci_hi", "n_pairs"])
        w.writerows(rows)
    print("\nwrote", out_csv)
