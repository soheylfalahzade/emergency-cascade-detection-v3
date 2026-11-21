#!/usr/bin/env python3
"""Build ABSTRACT_MODEL.md from hops_ablation.csv, hop2_demand_sweep.csv, hop2_weight_sweep.csv.

Every number is computed from the CSV files (nothing typed by hand).
Usage (inside phase2_bridge):  python build_abstract_model_doc.py
Label for everything here: ABSTRACT QUEUE MODEL (toy_sim.py). NOT CARLA, NOT real traffic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
abl = pd.read_csv(HERE / "hops_ablation.csv")
dem = pd.read_csv(HERE / "hop2_demand_sweep.csv")
wsw = pd.read_csv(HERE / "hop2_weight_sweep.csv")


def paired(a, b, col):
    """a, b: frames with a 'seed' column. Mean of a-b over seeds, half-width 1.96*SE (normal approx), n, #seeds a>b."""
    x = (a.sort_values("seed")[col].to_numpy(float) - b.sort_values("seed")[col].to_numpy(float))
    x = x[np.isfinite(x)]
    hw = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return x.mean(), hw, len(x), int((x > 0).sum()), int((x < 0).sum())


def fmt(m, hw, signed=True):
    return f"{m:+.2f} ± {hw:.2f}" if signed else f"{m:.2f} ± {hw:.2f}"


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


seeds = abl["seed"].nunique()
out = []
P = out.append
P("# Abstract queue model: corridor results (NOT CARLA, NOT real traffic)\n")
P("Everything below comes from `toy_sim.py`, a simplified queue model. It says nothing about CARLA or real traffic.\n")
P(f"Seeds: {seeds} (paired by seed). Intervals are mean ± 1.96·SE over seeds (normal approximation, not a bootstrap).")
P("Junction count and duration are NOT stored in the CSV files; the scripts' defaults are 5 junctions and 900 s. "
  "Treat those as the likely settings, not as verified.\n")

# ---- 1. baselines, no message delay/loss
P("## 1. Policies on the corridor (demand lam = 0.12, no message delay or loss)\n")
base = abl[(abl.delay_s == 0) & (abl.drop_p == 0)]
order = ["Fixed-time", "Fixed+preempt", "Max-pressure", "Fuzzy (no comm)", "Fuzzy (full)", "Fuzzy (2-hop)"]
cols = [("em_delay_corridor", "emergency delay, corridor (s)"), ("ord_delay_s", "ordinary delay (s)"),
        ("max_wait_s", "max wait (s)"), ("false_preempt_s", "false-preempt time (s)"), ("throughput_vph", "throughput (veh/h)")]
P("| policy | " + " | ".join(c[1] for c in cols) + " |")
P("|---|" + "---|" * len(cols))
for p in order:
    d = base[base.policy == p]
    P(f"| {p} | " + " | ".join(f"{d[c].mean():.2f}" for c, _ in cols) + " |")
P("\nMeans over seeds. `Fuzzy (full)` = 1-hop messages; `Fuzzy (2-hop)` = messages to 2 junctions each way.\n")

P("Paired differences of `Fuzzy (full)` minus another policy (mean ± 1.96·SE; negative emergency/ordinary delay = Fuzzy better):\n")
P("| versus | emergency delay (s) | ordinary delay (s) | max wait (s) |")
P("|---|---|---|---|")
f = sel(abl, policy="Fuzzy (full)", delay_s=0.0, drop_p=0.0)
for o in ["Fuzzy (no comm)", "Fixed+preempt", "Max-pressure", "Fixed-time"]:
    b = sel(abl, policy=o, delay_s=0.0, drop_p=0.0)
    r = [paired(f, b, c) for c in ("em_delay_corridor", "ord_delay_s", "max_wait_s")]
    P(f"| {o} | " + " | ".join(fmt(m, h) for m, h, *_ in r) + " |")
P("")

# ---- 2. 2-hop vs 1-hop with message delay/loss
P("## 2. 2-hop minus 1-hop under message delay and loss (lam = 0.12)\n")
P("Negative is better for the delay columns. 'seeds worse' = seeds where 2-hop was higher than 1-hop.\n")
P("| message delay (s) | drop prob. | emergency delay (s) | ordinary delay (s) | ordinary: seeds worse | max wait (s) | false-preempt (s) |")
P("|---|---|---|---|---|---|---|")
for d, p in [(0.0, 0.0), (2.0, 0.0), (0.0, 0.2), (0.0, 0.5)]:
    a = sel(abl, policy="Fuzzy (2-hop)", delay_s=d, drop_p=p)
    b = sel(abl, policy="Fuzzy (full)", delay_s=d, drop_p=p)
    em, od, mw, fp = (paired(a, b, c) for c in ("em_delay_corridor", "ord_delay_s", "max_wait_s", "false_preempt_s"))
    P(f"| {d:g} | {p:g} | {fmt(em[0], em[1])} | {fmt(od[0], od[1])} | {od[3]}/{od[2]} | {fmt(mw[0], mw[1])} | {fmt(fp[0], fp[1])} |")
P("")

# ---- 3. demand sweep
P("## 3. Where does 2-hop help? Demand sweep (paired by seed against 1-hop)\n")
P("Variants: `state-only` = hop-2 occupancy enters the pressure term, no 2-ahead notices; "
  "`notices-only` = notices, no hop-2 pressure; `both`.\n")
lam_means = dem[dem.variant == "1-hop"].groupby("lam")[["ord_delay_s", "spillback_per_h", "throughput_vph"]].mean()
P("Regime of the 1-hop controller at each demand (means):\n")
P("| lam | ordinary delay (s) | spillbacks/h | throughput (veh/h) |")
P("|---|---|---|---|")
for lam, r in lam_means.iterrows():
    P(f"| {lam:g} | {r.ord_delay_s:.1f} | {r.spillback_per_h:.1f} | {r.throughput_vph:.0f} |")
P("\nDifference variant − 1-hop (mean ± 1.96·SE):\n")
P("| lam | variant | emergency delay (s) | ordinary delay (s) | spillbacks/h | max wait (s) |")
P("|---|---|---|---|---|---|")
for lam in sorted(dem.lam.unique()):
    b = sel(dem, lam=lam, variant="1-hop")
    for v in ("2hop state-only", "2hop notices-only", "2hop both"):
        a = sel(dem, lam=lam, variant=v)
        r = [paired(a, b, c) for c in ("em_delay_corridor", "ord_delay_s", "spillback_per_h", "max_wait_s")]
        P(f"| {lam:g} | {v} | " + " | ".join(fmt(m, h) for m, h, *_ in r) + " |")
P("")

# ---- 4. weight sweep
P("## 4. Hop-2 weight sweep (lam = 0.12)\n")
g = wsw.groupby(["policy", "w"])[["ord_delay_s", "em_delay_corridor", "false_preempt_s", "max_wait_s"]].mean()
P("| policy | weight | ordinary delay (s) | emergency delay (s) | false-preempt (s) | max wait (s) |")
P("|---|---|---|---|---|---|")
for (pol, w), r in g.iterrows():
    P(f"| {pol} | {'-' if w < 0 else format(w, 'g')} | {r.ord_delay_s:.3f} | {r.em_delay_corridor:.3f} | {r.false_preempt_s:.2f} | {r.max_wait_s:.2f} |")
ident = (g.loc["2-hop"].round(9).nunique() == 1).all()
P("\n" + ("All 2-hop rows are identical across weights: at this demand the weight had no measured effect."
          if ident else "2-hop rows differ across weights."))
P("")

# ---- 0. earlier 3-junction run (10 seeds), only if the file exists
runs_path = HERE / "outputs" / "p2_policy_runs.csv"
if runs_path.exists():
    r3 = pd.read_csv(runs_path)
    P("## 5. Earlier run: 3 junctions (default n_junc=3), 10 seeds, three demands\n")
    P("File: outputs/p2_policy_runs.csv (stored 2026-11-04, before the multi-hop code of 2026-11-13; not re-run with the current code). Different setting from sections 1-4: do not mix them.\n")
    P("| lam | policy | emergency delay (s) | ordinary delay (s) | throughput (veh/h) | max wait (s) |")
    P("|---|---|---|---|---|---|")
    for lam in sorted(r3.lam.unique()):
        for pol in sorted(r3.policy.unique()):
            d = sel(r3, lam=lam, policy=pol)
            P(f"| {lam:g} | {pol} | {d.em_delay_mean.mean():.2f} | {d.ord_delay_s.mean():.2f} | {d.throughput_vph.mean():.0f} | {d.max_wait_s.mean():.2f} |")
    P("\nPaired difference `Fuzzy (full)` minus other (mean ± 1.96·SE over 10 seeds):\n")
    P("| lam | versus | emergency delay (s) | ordinary delay (s) |")
    P("|---|---|---|---|")
    for lam in sorted(r3.lam.unique()):
        f3 = sel(r3, lam=lam, policy="Fuzzy (full)")
        for o in ("Fuzzy (no comm)", "Fixed+preempt", "Max-pressure", "Fixed-time"):
            b3 = sel(r3, lam=lam, policy=o)
            em, od = paired(f3, b3, "em_delay_mean"), paired(f3, b3, "ord_delay_s")
            P(f"| {lam:g} | {o} | {fmt(em[0], em[1])} | {fmt(od[0], od[1])} |")
    P("")

P("## Labels and limits\n")
P("- All numbers: abstract queue model. They are not evidence about CARLA or real traffic.")
P("- The model was designed together with the controller; agreement between them is not independent validation.")
P("- Messaging up to 2 junctions each way exists only in this model; it was not run in CARLA.")
P("- Intervals use a normal approximation over a small number of seeds.")
Path(HERE / "ABSTRACT_MODEL.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("wrote ABSTRACT_MODEL.md", len(out), "lines")
