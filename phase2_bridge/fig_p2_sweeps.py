import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Portfolio/Smart_Traffic_System/figures_pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from matplotlib.patches import Patch

from common import *
from toy_sim import POLICIES, NoiseConfig, Scenario, run_sim
import health_policy  # noqa: F401  (registers 'Fuzzy (health)')

setup_style()
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--noise_seeds", type=int, default=5)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
if a.quick:
    a.seeds, a.noise_seeds = 2, 2
WM = "abstract queue model - not CARLA"
SUF = "_quick" if a.quick else ""


def cached(name, fn):
    p = P2_OUT / f"{name}{SUF}.csv"
    if p.exists() and not a.force:
        return pd.read_csv(p)
    df = fn()
    df.to_csv(p, index=False)
    return df


POLS1 = ["Fixed-time", "Fixed+preempt", "Max-pressure", "Fuzzy (no priority)", "Fuzzy (no comm)", "Fuzzy (full)", "Fuzzy (health)"]
COLS1 = [PAL["grey"], PAL["sky"], PAL["orange"], PAL["purple"], PAL["green"], PAL["blue"], PAL["red"]]
LAMS = [0.08, 0.12, 0.16]


def e1():
    rows = []
    for lam in LAMS:
        for pol in POLS1:
            for s in range(a.seeds):
                m, _ = run_sim(POLICIES[pol](s), Scenario(lam=lam, n_emergency=4), s)
                rows.append(dict(policy=pol, lam=lam, seed=s, **m))
        print("E1 lam", lam, flush=True)
    return pd.DataFrame(rows)


def e2():
    rows = []
    for pm in (0.0, 0.1, 0.2, 0.3):
        for pe in (0.0, 0.1, 0.3):
            for pol in ("Fuzzy (full)", "Fixed+preempt"):
                for s in range(a.noise_seeds):
                    nz = NoiseConfig(p_miss=pm, p_emerg_as_car=pe, hold_s=2.0)
                    m, _ = run_sim(POLICIES[pol](s), Scenario(lam=0.12, n_emergency=6, noise=nz), 100 + s)
                    rows.append(dict(kind="grid", policy=pol, p_miss=pm, p_emerg=pe, fp=0.0, seed=s, **m))
        print("E2 grid p_miss", pm, flush=True)
    for fp in (0.0, 0.5, 2.0, 5.0):
        for pol in ("Fuzzy (full)", "Fuzzy (health)", "Fuzzy (no priority)", "Fixed+preempt", "Fixed-time"):
            for s in range(a.noise_seeds):
                nz = NoiseConfig(p_miss=0.1, p_emerg_as_car=0.1, fp_emerg_per_min=fp, hold_s=2.0)
                m, _ = run_sim(POLICIES[pol](s), Scenario(lam=0.12, n_emergency=6, noise=nz), 100 + s)
                rows.append(dict(kind="fp", policy=pol, p_miss=0.1, p_emerg=0.1, fp=fp, seed=s, **m))
        print("E2 fp", fp, flush=True)
    return pd.DataFrame(rows)


def e3():
    rows = []
    good = NoiseConfig(p_miss=0.05, p_emerg_as_car=0.05, fp_emerg_per_min=0.2, hold_s=2.0)
    bad = NoiseConfig(p_miss=0.5, p_emerg_as_car=0.5, fp_emerg_per_min=3.0, hold_s=2.0)
    for pol in ("Fixed-time", "Fuzzy (full)", "Fuzzy (health)", "Fuzzy (ignore visibility)"):
        for s in range(a.seeds):
            sc = Scenario(lam=0.12, n_emergency=6, noise=good, noise_bad=bad, window=(300.0, 600.0), vis_drop=True)
            m, _ = run_sim(POLICIES[pol](s), sc, 200 + s)
            rows.append(dict(policy=pol, seed=s, **m))
        print("E3", pol, flush=True)
    return pd.DataFrame(rows)


def grouped_box(ax, df, metric, groups, hue, order, colors, gcol):
    n = len(order)
    w = 0.8 / n
    for gi, g in enumerate(groups):
        for hi, h in enumerate(order):
            v = df[(df[gcol] == g) & (df[hue] == h)][metric].dropna().values
            if len(v) == 0:
                continue
            bp = ax.boxplot(v, positions=[gi + (hi - (n - 1) / 2) * w], widths=w * 0.9, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[hi])
                patch.set_alpha(0.85)
            for el in ("whiskers", "caps", "medians"):
                for ln in bp[el]:
                    ln.set_color("#333333")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups])


d1 = cached("p2_policy_runs", e1)
metrics = [("em_delay_mean", "emergency delay (s)"), ("ord_delay_s", "ordinary delay per junction passage (s)"),
           ("throughput_vph", "exits per hour"), ("max_wait_s", "max wait (s)"),
           ("switches_per_h", "phase switches per hour (3 junctions)"), ("spillback_per_h", "spillback blocks per hour")]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
for ax, (m, lab) in zip(axes.ravel(), metrics):
    grouped_box(ax, d1, m, LAMS, "policy", POLS1, COLS1, "lam")
    ax.set_xlabel("arrival rate per approach (veh/s)")
    ax.set_ylabel(lab)
fig.legend(handles=[Patch(color=c, label=p) for c, p in zip(COLS1, POLS1)], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(f"Policy comparison, perfect perception, {a.seeds} seeds per cell", y=1.0)
watermark(fig, WM)
save(fig, "p2_policy_comparison", P2_OUT)
summ = d1.groupby(["lam", "policy"])[[m for m, _ in metrics]].agg(["mean", "std"]).round(2)
summ.to_csv(P2_OUT / f"p2_policy_summary{SUF}.csv")

d2 = cached("p2_noise_runs", e2)
g2 = d2[d2["kind"] == "grid"]
vmin, vmax = float(g2["em_delay_mean"].min()), float(g2["em_delay_mean"].max())
fig, axes = plt.subplots(2, 2, figsize=(9.4, 7))
for ax, pol in zip(axes[0], ("Fuzzy (full)", "Fixed+preempt")):
    g = g2[g2["policy"] == pol].groupby(["p_miss", "p_emerg"])["em_delay_mean"].mean().unstack()
    im = ax.imshow(g.values, origin="lower", aspect="auto", cmap="magma_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(g.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in g.columns])
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels([f"{x:.1f}" for x in g.index])
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > (vmin + vmax) / 2 else "black")
    ax.set_xlabel("P(emergency reported as car)")
    ax.set_ylabel("P(miss)")
    ax.set_title(f"mean emergency delay (s): {pol}")
    ax.grid(False)
fig.colorbar(im, ax=list(axes[0]), shrink=0.8)
f2 = d2[d2["kind"] == "fp"]
for ax, (m, lab) in zip(axes[1], (("ord_delay_s", "ordinary delay per junction passage (s)"), ("false_preempt_s", "false preemption seconds per run"))):
    for pol, c in (("Fuzzy (full)", PAL["blue"]), ("Fuzzy (health)", PAL["red"]), ("Fuzzy (no priority)", PAL["purple"]), ("Fixed+preempt", PAL["sky"]), ("Fixed-time", PAL["grey"])):
        g = f2[f2["policy"] == pol].groupby("fp")[m].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o", color=c, label=pol, capsize=2)
    ax.set_xlabel("false emergency reports per minute per approach")
    ax.set_ylabel(lab)
    ax.legend(fontsize=6)
fig.suptitle(f"Sensitivity to perception errors ({a.noise_seeds} seeds)", y=1.0)
watermark(fig, WM)
save(fig, "p2_noise_sensitivity", P2_OUT)

d3 = cached("p2_mode_runs", e3)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
pols3 = ["Fixed-time", "Fuzzy (full)", "Fuzzy (health)", "Fuzzy (ignore visibility)"]
for ax, (m, lab) in zip(axes, (("ord_queue_mean_win", "mean ordinary queue in window (veh)"),
                               ("em_delay_win", "emergency delay, entering in window (s)"),
                               ("false_preempt_s", "false preemption seconds per run"))):
    mu = [d3[d3["policy"] == p][m].mean() for p in pols3]
    sd = [d3[d3["policy"] == p][m].std() for p in pols3]
    ax.bar(range(4), mu, yerr=np.nan_to_num(sd), color=[PAL["grey"], PAL["blue"], PAL["red"], PAL["purple"]], capsize=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["fixed-time", "fuzzy +\nmode manager", "fuzzy +\nhealth", "fuzzy,\nvisibility ignored"], fontsize=6.5)
    ax.set_ylabel(lab, fontsize=8)
fig.suptitle(f"Degraded-perception window (t=300-600 s: visibility 0.1, miss 0.5, class swap 0.5, 3 false alarms/min), {a.seeds} seeds", y=1.03, fontsize=9)
watermark(fig, WM)
save(fig, "p2_visibility_modes", P2_OUT)
