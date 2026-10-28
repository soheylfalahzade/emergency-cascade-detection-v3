"""Perception error model for closed-loop controller tests (no GPU, no vision code).

Converts ground-truth vehicles from a simulator API (CARLA or SUMO) into the kind of
observation a camera pipeline could give a controller, with controllable error rates.
All rates are sweep parameters, NOT measured values of the real detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

EMERGENCY: Tuple[str, ...] = ("ambulance", "police", "firetruck")


@dataclass(frozen=True)
class NoiseConfig:
    p_miss: float = 0.0            # chance a vehicle is not reported in a step
    p_miss_far: float = 0.0        # extra miss chance beyond far_m from the stop line
    far_m: float = 60.0
    p_emerg_as_car: float = 0.0    # chance a real emergency vehicle is reported as 'car'
    fp_emerg_per_min: float = 0.0  # false emergency reports per minute per lane (Poisson)
    hold_s: float = 0.0            # tracker coasting: keep a vehicle this long after a miss
    seed: int = 0


@dataclass(frozen=True)
class Observation:
    vid: str
    lane: str
    cls: str
    conf: float
    is_fake: bool = False


class PerceptionNoise:
    """Stateful noisy observer. Call step() once per control step."""

    def __init__(self, cfg: NoiseConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self._last_seen: Dict[str, Tuple[float, Observation]] = {}
        self._fakes: Dict[str, Tuple[float, Observation]] = {}
        self._fake_n = 0

    def step(self, t: float, dt: float, vehicles: Iterable[dict],
             lanes: Iterable[str]) -> List[Observation]:
        """vehicles: dicts with keys id, lane, cls, dist_m (distance to the stop line)."""
        c = self.cfg
        out: Dict[str, Observation] = {}
        for v in vehicles:
            vid, lane, cls = str(v["id"]), str(v["lane"]), str(v["cls"])
            dist = float(v.get("dist_m", 0.0))
            p_miss = min(c.p_miss + (c.p_miss_far if dist > c.far_m else 0.0), 1.0)
            if self.rng.random() < p_miss:
                held = self._last_seen.get(vid)
                if held is not None and t - held[0] <= c.hold_s:
                    out[vid] = held[1]
                continue
            reported = cls
            if cls in EMERGENCY and self.rng.random() < c.p_emerg_as_car:
                reported = "car"
            mean = 0.9 if reported == cls else 0.6
            conf = float(np.clip(self.rng.normal(mean, 0.05), 0.0, 1.0))
            obs = Observation(vid, lane, reported, conf)
            self._last_seen[vid] = (t, obs)
            out[vid] = obs
        self._last_seen = {k: v for k, v in self._last_seen.items() if t - v[0] <= 60.0}

        for lane in lanes:
            for _ in range(int(self.rng.poisson(c.fp_emerg_per_min * dt / 60.0))):
                self._fake_n += 1
                fid = f"fake{self._fake_n}"
                life = float(self.rng.uniform(1.0, 5.0))
                cls = str(self.rng.choice(EMERGENCY))
                conf = float(self.rng.uniform(0.5, 0.9))
                self._fakes[fid] = (t + life, Observation(fid, str(lane), cls, conf, True))
        self._fakes = {k: v for k, v in self._fakes.items() if v[0] >= t}
        return list(out.values()) + [o for _, o in self._fakes.values()]


def summarize(obs: Iterable[Observation], lanes: Iterable[str]) -> Dict[str, dict]:
    """Per-lane view for the controller: vehicle count and emergency flag with confidence."""
    res = {str(l): {"count": 0, "emergency": False, "emergency_conf": 0.0} for l in lanes}
    for o in obs:
        r = res.get(o.lane)
        if r is None:
            continue
        r["count"] += 1
        if o.cls in EMERGENCY:
            r["emergency"] = True
            r["emergency_conf"] = max(r["emergency_conf"], o.conf)
    return res


def sweep_grid() -> List[NoiseConfig]:
    """36 settings: miss rate x emergency-as-car rate x false emergency rate per minute."""
    return [NoiseConfig(p_miss=pm, p_emerg_as_car=pe, fp_emerg_per_min=fp, hold_s=2.0)
            for pm in (0.0, 0.1, 0.2, 0.3)
            for pe in (0.0, 0.1, 0.3)
            for fp in (0.0, 0.5, 2.0)]
