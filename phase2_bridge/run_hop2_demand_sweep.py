"""Abstract queue model: where does 2-hop help? Demand sweep x which part of 2-hop is on (paired by seed vs 1-hop).
variants: state-only = hop-2 occupancy in pressure (weight 1, no 2-ahead notices); notices-only = weight 0; both."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from toy_sim import POLICIES, Scenario, run_sim

METRICS = ["ord_delay_s", "ord_queue_mean", "em_delay_corridor", "spillback_per_h", "max_wait_s", "throughput_vph"]
VARIANTS = {
    "1-hop": lambda s, n: POLICIES["Fuzzy (full)"](s, n=n),
    "2hop state-only": lambda s, n: POLICIES["Fuzzy (2-hop)"](s, n=n, hop2_weight=1.0, hop2_notices=False),
    "2hop notices-only": lambda s, n: POLICIES["Fuzzy (2-hop)"](s, n=n, hop2_weight=0.0, hop2_notices=True),
    "2hop both": lambda s, n: POLICIES["Fuzzy (2-hop)"](s, n=n, hop2_weight=1.0, hop2_notices=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--lams", default="0.12,0.2,0.28,0.32")
    ap.add_argument("--out", default="hop2_demand_sweep.csv")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]
    rows = []
    for lam in lams:
        for seed in range(a.seeds):
            sc = Scenario(n_junc=a.n, duration_s=a.duration, lam=lam, n_emergency=4, corridor_share=1.0)
            for v, mk in VARIANTS.items():
                m, _ = run_sim(mk(seed, a.n), sc, seed)
                rows.append(dict(lam=lam, variant=v, seed=seed, **{k: m[k] for k in METRICS}))
        print("lam", lam, "done", flush=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def col(lam, v, k):
        return np.array([r[k] for r in sorted((r for r in rows if r["lam"] == lam and r["variant"] == v), key=lambda r: r["seed"])], float)

    for lam in lams:
        print(f"\n== lam={lam}  1-hop means:", "  ".join(f"{k}={np.nanmean(col(lam, '1-hop', k)):.2f}" for k in METRICS))
        for v in list(VARIANTS)[1:]:
            print(f"  {v}: paired diff vs 1-hop (mean +- 95% CI)")
            for k in METRICS:
                x = col(lam, v, k) - col(lam, "1-hop", k)
                x = x[np.isfinite(x)]
                if len(x) > 1:
                    print(f"     {k:18s} {x.mean():+10.3f} +- {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
