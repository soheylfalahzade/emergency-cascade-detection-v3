"""Abstract queue model: sweep hop2_weight of the 2-hop controller against the 1-hop controller (paired by seed)."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from toy_sim import POLICIES, Scenario, run_sim

METRICS = ["ord_delay_s", "ord_queue_mean", "em_delay_corridor", "false_preempt_s", "max_wait_s", "throughput_vph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--lam", type=float, default=0.12)
    ap.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out", default="hop2_weight_sweep.csv")
    a = ap.parse_args()
    ws = [float(x) for x in a.weights.split(",")]
    rows = []
    for seed in range(a.seeds):
        sc = Scenario(n_junc=a.n, duration_s=a.duration, lam=a.lam, n_emergency=4, corridor_share=1.0)
        m, _ = run_sim(POLICIES["Fuzzy (full)"](seed, n=a.n), sc, seed)
        rows.append(dict(policy="1-hop", w=-1.0, seed=seed, **{k: m[k] for k in METRICS}))
        for w in ws:
            m, _ = run_sim(POLICIES["Fuzzy (2-hop)"](seed, n=a.n, hop2_weight=w), sc, seed)
            rows.append(dict(policy="2-hop", w=w, seed=seed, **{k: m[k] for k in METRICS}))
        print("seed", seed, "done", flush=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def col(pol, w, k):
        return np.array([r[k] for r in sorted((r for r in rows if r["policy"] == pol and r["w"] == w), key=lambda r: r["seed"])], float)

    print("\n== 1-hop means ==", "  ".join(f"{k}={np.nanmean(col('1-hop', -1.0, k)):.3f}" for k in METRICS))
    print("== paired difference 2-hop(w) minus 1-hop: mean +- 95% CI half-width ==")
    for w in ws:
        print(f"hop2_weight={w}")
        for k in METRICS:
            x = col("2-hop", w, k) - col("1-hop", -1.0, k)
            x = x[np.isfinite(x)]
            if len(x) > 1:
                print(f"   {k:18s} {x.mean():+10.4f} +- {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.4f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
