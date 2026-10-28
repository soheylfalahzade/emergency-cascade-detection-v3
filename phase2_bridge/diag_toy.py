import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from toy_sim import POLICIES, NoiseConfig, Scenario, run_sim

pd.set_option("display.width", 200)
S = range(5)
print("== (1) perfect perception: is 'false preemption' really advance pre-clearing?")
rows = []
for pol in ("Fuzzy (full)", "Fuzzy (no comm)", "Fixed+preempt"):
    for s in S:
        m, _ = run_sim(POLICIES[pol](s), Scenario(lam=0.12, n_emergency=4), s)
        rows.append(dict(policy=pol, false_preempt_s=m["false_preempt_s"], em_delay=m["em_delay_mean"]))
print(pd.DataFrame(rows).groupby("policy").mean().round(2))

print("== (2) degraded window: which ingredient makes emergency delay worse than fixed-time?")
good = NoiseConfig(p_miss=0.05, p_emerg_as_car=0.05, fp_emerg_per_min=0.2, hold_s=2.0)
bad = NoiseConfig(p_miss=0.5, p_emerg_as_car=0.5, fp_emerg_per_min=3.0, hold_s=2.0)
conds = {"C0 no window": dict(),
         "C1 visibility drop only": dict(window=(300.0, 600.0), vis_drop=True),
         "C2 bad noise only": dict(window=(300.0, 600.0), noise_bad=bad),
         "C3 both (as in figure)": dict(window=(300.0, 600.0), noise_bad=bad, vis_drop=True)}
rows = []
for cn, kw in conds.items():
    for pol in ("Fixed-time", "Fuzzy (full)", "Fuzzy (ignore visibility)"):
        for s in S:
            m, _ = run_sim(POLICIES[pol](s), Scenario(lam=0.12, n_emergency=6, noise=good, **kw), 200 + s)
            rows.append(dict(cond=cn, policy=pol, em_delay_all=m["em_delay_mean"], em_delay_win=m["em_delay_win"],
                             ord_queue_win=m["ord_queue_mean_win"], false_preempt_s=m["false_preempt_s"],
                             static_frac=m.get("mode_static", 0.0)))
print(pd.DataFrame(rows).groupby(["cond", "policy"]).mean().round(2))
