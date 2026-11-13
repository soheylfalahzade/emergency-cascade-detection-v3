"""Abstract queue model, N-junction corridor: 1-hop vs 2-hop coordination, baselines, message delay/loss sweep.
NOT CARLA. Writes hops_ablation.csv and prints paired differences (2-hop minus 1-hop, same seed and demand)."""
import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from toy_sim import POLICIES, Scenario, run_sim

METRICS = ["ord_delay_s", "ord_queue_mean", "em_delay_corridor", "spillback_per_h", "throughput_vph", "false_preempt_s",
           "max_wait_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--lam", type=float, default=0.12)
    ap.add_argument("--out", default="hops_ablation.csv")
    a = ap.parse_args()
    comm = [(0.0, 0.0), (2.0, 0.0), (0.0, 0.2), (0.0, 0.5)]
    rows = []
    for seed in range(a.seeds):
        sc = Scenario(n_junc=a.n, duration_s=a.duration, lam=a.lam, n_emergency=4, corridor_share=1.0)
        for pol in ("Fixed-time", "Fixed+preempt", "Max-pressure", "Fuzzy (no comm)"):
            m, _ = run_sim(POLICIES[pol](seed, n=a.n), sc, seed)
            rows.append(dict(policy=pol, delay_s=0.0, drop_p=0.0, seed=seed, **{k: m[k] for k in METRICS}))
        for pol, (d, p) in itertools.product(("Fuzzy (full)", "Fuzzy (2-hop)"), comm):
            m, _ = run_sim(POLICIES[pol](seed, n=a.n, delay_s=d, drop_p=p), sc, seed)
            rows.append(dict(policy=pol, delay_s=d, drop_p=p, seed=seed, **{k: m[k] for k in METRICS}))
        print("seed", seed, "done", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(pol, d, p, k):
        return np.array([r[k] for r in sorted((r for r in rows if r["policy"] == pol and r["delay_s"] == d and r["drop_p"] == p),
                                              key=lambda r: r["seed"])], float)

    print("\n== means over seeds (n=%d junctions, %d seeds) ==" % (a.n, a.seeds))
    for pol in ("Fixed-time", "Fixed+preempt", "Max-pressure", "Fuzzy (no comm)"):
        print(f"{pol:18s}", "  ".join(f"{k}={np.nanmean(col(pol, 0.0, 0.0, k)):.3f}" for k in METRICS))
    print("\n== paired difference 2-hop minus 1-hop (mean, 95% CI half-width); negative is better except throughput ==")
    for d, p in comm:
        print(f"delay_s={d} drop_p={p}")
        for k in METRICS:
            x = col("Fuzzy (2-hop)", d, p, k) - col("Fuzzy (full)", d, p, k)
            x = x[np.isfinite(x)]
            if len(x) > 1:
                print(f"   {k:18s} {x.mean():+10.4f} +- {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.4f}  (n={len(x)})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
