import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import health_policy  # noqa: F401
from toy_sim import POLICIES, NoiseConfig, Scenario, run_sim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
KEYS = ["em_delay_mean", "em_delay_win", "ord_delay_s", "ord_queue_mean_win", "false_preempt_s",
        "mode_static", "mode_degraded", "max_wait_s"]
S = range(5)


def row(m, **kw):
    return dict(**kw, **{k: m.get(k, float("nan")) for k in KEYS})


good = NoiseConfig(p_miss=0.05, p_emerg_as_car=0.05, fp_emerg_per_min=0.2, hold_s=2.0)
bad = NoiseConfig(p_miss=0.5, p_emerg_as_car=0.5, fp_emerg_per_min=3.0, hold_s=2.0)
conds = {"C0 no window": dict(), "C1 visibility drop only": dict(window=(300.0, 600.0), vis_drop=True),
         "C2 bad noise only": dict(window=(300.0, 600.0), noise_bad=bad),
         "C3 both": dict(window=(300.0, 600.0), noise_bad=bad, vis_drop=True)}
rows = []
for cn, kw in conds.items():
    for pol in ("Fixed-time", "Fuzzy (full)", "Fuzzy (health)", "Fuzzy (ignore visibility)"):
        for s in S:
            m, _ = run_sim(POLICIES[pol](s), Scenario(lam=0.12, n_emergency=6, noise=good, **kw), 200 + s)
            rows.append(row(m, cond=cn, policy=pol))
dA = pd.DataFrame(rows)
dA.to_csv(os.path.join(OUT, "diag_health_windows.csv"), index=False)
print("== A) degraded window (all columns; means over 5 seeds)")
print(dA.groupby(["cond", "policy"]).mean(numeric_only=True).round(2).to_string())

rows = []
for fp in (0.0, 0.5, 2.0, 5.0):
    for pol in ("Fixed-time", "Fixed+preempt", "Fuzzy (full)", "Fuzzy (health)"):
        for s in S:
            nz = NoiseConfig(p_miss=0.1, p_emerg_as_car=0.1, fp_emerg_per_min=fp, hold_s=2.0)
            m, _ = run_sim(POLICIES[pol](s), Scenario(lam=0.12, n_emergency=6, noise=nz), 100 + s)
            rows.append(row(m, fp=fp, policy=pol))
dB = pd.DataFrame(rows)
dB.to_csv(os.path.join(OUT, "diag_health_fp.csv"), index=False)
print("\n== B) false-alarm sweep (corrected false-preemption metric)")
print(dB.groupby(["policy", "fp"]).mean(numeric_only=True).round(2).to_string())
