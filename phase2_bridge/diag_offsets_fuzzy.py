import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataclasses
import numpy as np, pandas as pd
import toy_sim
from toy_sim import FuzzyPolicy, FixedTimePolicy, Scenario, run_sim, JUNCTIONS


class StaticFuzzy(FuzzyPolicy):
    """Fuzzy controller forced into STATIC from t=0 (visibility=0, mode preset); optional random initial phase/elapsed."""
    def __init__(self, offsets: bool, seed: int = 0) -> None:
        super().__init__(seed=seed)
        ctls = next(v for v in vars(self.net).values()
                    if isinstance(v, dict) and v and all(hasattr(c, "modes") for c in v.values()))
        rng = np.random.default_rng(seed + 55)
        for c in ctls.values():
            c.modes.mode, c.modes.smoothed = "static", 0.0
            if offsets:
                ph = sorted(c.cfg.phases)[int(rng.integers(0, len(c.cfg.phases)))]
                c.phase = ph
                c.t_in_phase = float(rng.uniform(0.0, c.cfg.static_green_s[ph]))

    def step(self, t, dt, obs):
        return super().step(t, dt, {n: dataclasses.replace(o, visibility=0.0) for n, o in obs.items()})


cases = {
    "fuzzy static, synchronised": lambda s: StaticFuzzy(False, s),
    "fuzzy static, random offsets": lambda s: StaticFuzzy(True, s),
    "fixed, synchronised": lambda s: FixedTimePolicy(),
}
rows = []
for name, mk in cases.items():
    for s in range(10):
        m, _ = run_sim(mk(s), Scenario(lam=0.12, n_emergency=6, corridor_share=0.5), 300 + s)
        rows.append(dict(policy=name, em_corridor=m["em_delay_corridor"], em_cross=m["em_delay_cross"],
                         em_all=m["em_delay_mean"], ord_delay=m["ord_delay_s"]))
df = pd.DataFrame(rows)
print(df.groupby("policy").agg(["mean", "std"]).round(2).to_string())
a = df[df.policy == "fuzzy static, random offsets"].em_corridor.mean() - df[df.policy == "fuzzy static, synchronised"].em_corridor.mean()
print(f"corridor delta (random - synchronised) = {a:+.2f}s ; pre-registered criterion: >= +3s")
