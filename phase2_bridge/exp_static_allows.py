import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import toy_sim, health_policy
from toy_sim import POLICIES, Scenario, run_sim
from health_policy import FuzzyHealthPolicy
from perception_noise import NoiseConfig


class HealthAllowEm(FuzzyHealthPolicy):
    """Fuzzy (health) with emergencies still served in STATIC (stricter confirmation applies there)."""
    def __init__(self, **kw):
        super().__init__(**kw)
        ctls = next(v for v in vars(self.net).values()
                    if isinstance(v, dict) and v and all(hasattr(c, "modes") for c in v.values()))
        for c in ctls.values():
            object.__setattr__(c.cfg, "static_allows_emergency", True)


P = dict(POLICIES)
P["Fuzzy (health, static serves emergencies)"] = lambda s=0: HealthAllowEm(seed=s)
for fp in (0.0, 5.0):
    print(f"--- false alarms/min = {fp}")
    for nm in ("Fixed-time", "Fixed+preempt", "Fuzzy (health)", "Fuzzy (health, static serves emergencies)"):
        R, extra = [], {}
        for s in range(10):
            sc = Scenario(lam=0.12, n_emergency=6, corridor_share=0.5, noise=NoiseConfig(fp_emerg_per_min=fp))
            m, _ = run_sim(P[nm](s), sc, 300 + s)
            R.append([m["em_delay_mean"], m["ord_delay_s"], m.get("mode_static", 0.0)])
            for k, v in m.items():
                if isinstance(v, float) and ("false" in k or "switch" in k or "wait_max" in k):
                    extra.setdefault(k, []).append(v)
        R = np.array(R)
        ex = " ".join(f"{k}={np.nanmean(v):.2f}" for k, v in extra.items())
        print(f"{nm:<42} em={R[:,0].mean():.1f}±{R[:,0].std(ddof=1):.1f} ord={R[:,1].mean():.1f} static={R[:,2].mean():.2f} | {ex}")
