"""Registers 'Fuzzy (health)' in toy_sim.POLICIES: visibility = min(raw visibility, alarm-churn health)."""
import dataclasses

from perception_health import AlarmChurnHealth
from toy_sim import JUNCTIONS, POLICIES, FuzzyPolicy


class FuzzyHealthPolicy(FuzzyPolicy):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.health = {n: AlarmChurnHealth() for n in JUNCTIONS}

    def step(self, t, dt, obs):
        o2 = {}
        for n, o in obs.items():
            h = self.health[n].update(t, [x.tid for x in o.emergencies])
            o2[n] = dataclasses.replace(o, visibility=min(o.visibility, h))
        return super().step(t, dt, o2)


POLICIES["Fuzzy (health)"] = lambda s=0: FuzzyHealthPolicy(seed=s)
