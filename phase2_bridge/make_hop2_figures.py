"""Figures and tables from hop2_demand_sweep.csv (and hops_ablation.csv if present).
Every figure carries the 'abstract queue model' watermark: these numbers are NOT evidence about CARLA or real traffic."""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WM = "abstract queue model - not CARLA"
LABEL = {"2hop state-only": "2-hop pressure only", "2hop notices-only": "2-hop emergency notices only",
         "2hop both": "2-hop both"}
METRICS = [("em_delay_corridor", "Emergency delay, corridor (s)"), ("ord_delay_s", "Ordinary delay (s)"),
           ("ord_queue_mean", "Mean ordinary queue (veh-equivalents)"), ("spillback_per_h", "Spillbacks per hour"),
           ("max_wait_s", "Max wait (s)"), ("throughput_vph", "Throughput (veh/h)")]


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in ("variant", "policy"):
                r[k] = float(r[k])
    return rows


def paired(rows, lam, v, k):
    def col(name):
        return np.array([r[k] for r in sorted((r for r in rows if r["lam"] == lam and r["variant"] == name), key=lambda r: r["seed"])])
    x = col(v) - col("1-hop")
    x = x[np.isfinite(x)]
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hop2_demand_sweep.csv")
    ap.add_argument("--outdir", default="figures_hop2")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = read(a.csv)
    lams = sorted({r["lam"] for r in rows})
    variants = [v for v in LABEL if any(r["variant"] == v for r in rows)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (k, title) in zip(axes, METRICS[:2] + METRICS[3:4]):
        w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            st = [paired(rows, lam, v, k) for lam in lams]
            ax.bar(np.arange(len(lams)) + i * w, [s[0] for s in st], w, yerr=[s[1] for s in st], capsize=3, label=LABEL[v])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(lams)) + w * (len(variants) - 1) / 2)
        ax.set_xticklabels([f"{l:g}" for l in lams])
        ax.set_xlabel("arrival rate lam (veh/s per external approach)")
        ax.set_ylabel("difference vs 1-hop (negative = better)")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.text(0.5, 0.01, WM + "   |   paired by seed, mean and 95% CI", ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(a.outdir, "hop2_paired_differences.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(a.outdir, "hop2_paired_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam", "variant", "metric", "mean_diff_vs_1hop", "ci95_half_width", "n_seeds", "significant"])
        for lam in lams:
            for v in variants:
                for k, _ in METRICS:
                    m, h, n = paired(rows, lam, v, k)
                    w.writerow([lam, v, k, f"{m:.4f}", f"{h:.4f}", n, int(abs(m) > h)])
    with open(os.path.join(a.outdir, "hop2_paired_table.tex"), "w") as f:
        f.write("% abstract queue model, NOT CARLA. Paired difference vs 1-hop, mean +- 95% CI half-width.\n")
        f.write("\\begin{tabular}{llrrr}\n\\hline\n$\\lambda$ & variant & em. delay (s) & ord. delay (s) & spillback/h \\\\\n\\hline\n")
        for lam in lams:
            for v in variants:
                c = [paired(rows, lam, v, k) for k in ("em_delay_corridor", "ord_delay_s", "spillback_per_h")]
                f.write(f"{lam:g} & {LABEL[v]} & " + " & ".join(f"{m:+.1f} $\\pm$ {h:.1f}" for m, h, _ in c) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print("wrote", os.listdir(a.outdir))


if __name__ == "__main__":
    main()
