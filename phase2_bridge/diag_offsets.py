import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import toy_sim
from toy_sim import FixedTimePolicy, Scenario, run_sim


class FixedOffsets(FixedTimePolicy):
    """Fixed-time with a random initial phase and elapsed time per junction (no coordination)."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed + 55)
        for n in toy_sim.JUNCTIONS:
            s = self.state[n]
            s["phase"] = int(rng.integers(0, 2))
            s["t"] = float(rng.uniform(0.0, self.green))


rows = []
for name, mk in (("fixed, synchronised", lambda s: FixedTimePolicy()), ("fixed, random offsets", lambda s: FixedOffsets(s))):
    for s in range(10):
        m, _ = run_sim(mk(s), Scenario(lam=0.12, n_emergency=6, corridor_share=0.5), 300 + s)
        rows.append(dict(policy=name, em_corridor=m["em_delay_corridor"], em_cross=m["em_delay_cross"],
                         em_all=m["em_delay_mean"], ord_delay=m["ord_delay_s"]))
print(pd.DataFrame(rows).groupby("policy").mean().round(2).to_string())
