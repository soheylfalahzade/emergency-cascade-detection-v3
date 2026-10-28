"""Abstract queue model of a 3-junction corridor (J1-J2-J3; phase 1 = east-west = the corridor).

NOT CARLA. It exercises the phase-2 controller end to end, for ablations, perception-noise sweeps and
illustrative animations. Every figure made from it must carry the 'abstract queue model' watermark; none of
its numbers is evidence about CARLA or real traffic."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from coordination import MessageBus, NetworkController
from junction_controller import FuzzyJunctionController
from junction_model import (EMERGENCY_CLASSES, ApproachObs, Decision, EmergencyTrack, JunctionConfig,
                            JunctionObs)
from perception_noise import NoiseConfig, PerceptionNoise

APPS = ("A0", "A1", "A2", "A3")
JUNCTIONS = ("J1", "J2", "J3")
PHASES = {0: ("A0", "A2"), 1: ("A1", "A3")}
APP_PHASE = {a: p for p, apps in PHASES.items() for a in apps}
CAP, HEADWAY, TRAVEL, EM_ETA0 = 20, 2.0, 10.0, 30.0


def make_cfg(j: int) -> JunctionConfig:
    in_link, out_link, up, down, tt = {}, {}, {}, {}, {}
    if j == 0:
        in_link["A1"] = "LIN"
    else:
        in_link["A1"] = f"L{j}{j + 1}"
        up[in_link["A1"]] = JUNCTIONS[j - 1]
    if j < 2:
        out_link["A1"] = f"L{j + 1}{j + 2}"
        down[out_link["A1"]] = JUNCTIONS[j + 1]
        tt[out_link["A1"]] = TRAVEL
    return JunctionConfig(name=JUNCTIONS[j], phases=dict(PHASES), in_link=in_link, out_link=out_link,
                          upstream_of=up, downstream_of=down, link_travel_s=tt)


# ----------------------------------------------------------------------------- policies
class _Machine:
    def __init__(self, yellow: float = 3.0) -> None:
        self.yellow = yellow
        self.state = {n: dict(phase=0, t=0.0, yel=0.0, pend=None) for n in JUNCTIONS}

    def _advance(self, n: str, dt: float, want: Optional[int]) -> Tuple[int, bool]:
        s = self.state[n]
        if s["yel"] > 0.0:
            s["yel"] -= dt
            if s["yel"] <= 1e-9:
                s["phase"], s["pend"], s["yel"], s["t"] = s["pend"], None, 0.0, 0.0
                return s["phase"], False
            return s["phase"], True
        s["t"] += dt
        if want is not None and want != s["phase"]:
            s["pend"], s["yel"] = want, self.yellow
            return s["phase"], True
        return s["phase"], False


class FixedTimePolicy(_Machine):
    def __init__(self, green: float = 30.0, yellow: float = 3.0) -> None:
        super().__init__(yellow)
        self.green = green

    def _cycle_want(self, n: str, dt: float) -> Optional[int]:
        s = self.state[n]
        return 1 - s["phase"] if (s["yel"] <= 0.0 and s["t"] + dt >= self.green) else None

    def step(self, t, dt, obs):
        return {n: Decision(t, *self._advance(n, dt, self._cycle_want(n, dt)), "fixed", "", {}, (), ())
                for n in JUNCTIONS}


class FixedPreemptPolicy(FixedTimePolicy):
    """Fixed-time plus naive preemption: the first confirmed emergency track wins, no arbitration."""

    def step(self, t, dt, obs):
        out = {}
        for n in JUNCTIONS:
            s = self.state[n]
            want_em = None
            for tr in obs[n].emergencies:
                if tr.conf >= 0.6 and tr.track_age_s >= 1.0 and tr.cls in EMERGENCY_CLASSES:
                    want_em = APP_PHASE[tr.approach]
                    break
            reason, want = "", None
            if want_em is not None:
                reason = f"emergency[preempt]: phase {want_em}"
                if want_em == s["phase"] and s["yel"] <= 0.0:
                    s["t"] = min(s["t"], self.green - 2.0 * dt)
                elif s["yel"] <= 0.0 and s["t"] >= 4.0:
                    want = want_em
            else:
                want = self._cycle_want(n, dt)
            ph, yel = self._advance(n, dt, want)
            out[n] = Decision(t, ph, yel, "fixed+preempt", reason, {}, (), ())
        return out


class MaxPressurePolicy(_Machine):
    DOWN = {("J1", "A1"): ("J2", "A1"), ("J2", "A1"): ("J3", "A1")}

    def __init__(self, min_green: float = 10.0, max_green: float = 60.0, yellow: float = 3.0) -> None:
        super().__init__(yellow)
        self.min_green, self.max_green = min_green, max_green

    def step(self, t, dt, obs):
        out = {}
        for n in JUNCTIONS:
            s, pr = self.state[n], {}
            for p, apps in PHASES.items():
                v = 0.0
                for a in apps:
                    v += obs[n].approaches[a].queue
                    d = self.DOWN.get((n, a))
                    if d:
                        v -= obs[d[0]].approaches[d[1]].queue
                pr[p] = v
            want = None
            if s["yel"] <= 0.0:
                cur, oth = s["phase"], 1 - s["phase"]
                if s["t"] + dt >= self.max_green or (s["t"] + dt >= self.min_green and pr[oth] > pr[cur]):
                    want = oth
            ph, yel = self._advance(n, dt, want)
            out[n] = Decision(t, ph, yel, "maxpressure", "", pr, (), ())
        return out


class FuzzyPolicy:
    def __init__(self, priority: bool = True, comm: bool = True, ignore_visibility: bool = False, seed: int = 0) -> None:
        self.priority, self.ignore_visibility = priority, ignore_visibility
        ctl = {n: FuzzyJunctionController(make_cfg(j), phase0=0) for j, n in enumerate(JUNCTIONS)}
        self.net = NetworkController(ctl, MessageBus(drop_p=0.0 if comm else 1.0, seed=seed))

    def step(self, t, dt, obs):
        o2 = {}
        for n, o in obs.items():
            kw = {}
            if not self.priority:
                kw["emergencies"] = ()
            if self.ignore_visibility:
                kw["visibility"] = 1.0
            o2[n] = dataclasses.replace(o, **kw) if kw else o
        return self.net.step(t, dt, o2)


POLICIES = {
    "Fixed-time": lambda s=0: FixedTimePolicy(),
    "Fixed+preempt": lambda s=0: FixedPreemptPolicy(),
    "Max-pressure": lambda s=0: MaxPressurePolicy(),
    "Fuzzy (no priority)": lambda s=0: FuzzyPolicy(priority=False, seed=s),
    "Fuzzy (no comm)": lambda s=0: FuzzyPolicy(comm=False, seed=s),
    "Fuzzy (full)": lambda s=0: FuzzyPolicy(seed=s),
    "Fuzzy (ignore visibility)": lambda s=0: FuzzyPolicy(ignore_visibility=True, seed=s),
}


# ----------------------------------------------------------------------------- simulator
@dataclass
class Scenario:
    duration_s: float = 900.0
    lam: float = 0.12                      # ordinary arrivals, veh/s per external approach
    n_emergency: int = 4
    corridor_share: float = 0.5
    emerg_cls: Tuple[str, ...] = EMERGENCY_CLASSES
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    noise_bad: Optional[NoiseConfig] = None
    window: Optional[Tuple[float, float]] = None
    vis_drop: bool = False
    vis_low: float = 0.1


@dataclass
class Em:
    id: str
    cls: str
    j: int
    a: str
    t_enter: float
    corridor: bool = True
    state: str = "pending"      # pending | approach | queued | done
    eta: float = 0.0
    ahead: int = 0
    t_q: float = 0.0
    delays: List[float] = field(default_factory=list)


class CorridorSim:
    def __init__(self, policy, sc: Scenario, seed: int, log: bool = False) -> None:
        self.policy, self.sc, self.log_on, self.log = policy, sc, log, []
        self.rng_arr = np.random.default_rng(seed)
        self.rng_obs = np.random.default_rng(seed + 7919)
        self.ems = self._make_ems(np.random.default_rng(seed + 104729))
        self.em_by_id = {e.id: e for e in self.ems}
        self.queue = {j: {a: 0 for a in APPS} for j in range(3)}
        self.wait = {j: {a: 0.0 for a in APPS} for j in range(3)}
        self.credit = {j: {a: 0.0 for a in APPS} for j in range(3)}
        self.arrivals: List[Tuple[float, int, str]] = []
        self.noise_good = [PerceptionNoise(dataclasses.replace(sc.noise, seed=seed * 10 + j)) for j in range(3)]
        self.noise_bad = ([PerceptionNoise(dataclasses.replace(sc.noise_bad, seed=seed * 10 + j + 5)) for j in range(3)]
                          if sc.noise_bad is not None else None)
        self.first_seen: Dict[tuple, float] = {}
        self.last_seen: Dict[tuple, float] = {}
        self.q_hist: List[float] = []
        self.modes: Dict[str, int] = {}
        self.prev_yellow = {n: False for n in JUNCTIONS}
        self.exits = self.ord_passages = self.arrived = self.spill = self.switches = self.false_preempt = 0
        self.max_wait = 0.0

    def _make_ems(self, rng) -> List[Em]:
        sc, n = self.sc, self.sc.n_emergency
        if n == 0:
            return []
        T = sc.duration_s
        if sc.window is not None:
            assert sc.window[0] >= 150.0
            k = (n + 1) // 2
            lo, hi = sc.window[0] + 20.0, sc.window[1] - 70.0
            times = list(np.sort(rng.uniform(lo, max(hi, lo + 1.0), k)))
            if n - k:
                times += list(np.sort(rng.uniform(60.0, sc.window[0] - 40.0, n - k)))
        else:
            times = list(70.0 + np.arange(n) * (T - 260.0) / n + rng.uniform(0.0, 40.0, n))
        ems = []
        for i, t0 in enumerate(times):
            corridor = bool(rng.random() < sc.corridor_share)
            cls = str(rng.choice(sc.emerg_cls))
            if corridor:
                ems.append(Em(f"em{i}", cls, 0, "A1", float(t0), True))
            else:
                ems.append(Em(f"em{i}", cls, 1, str(rng.choice(["A0", "A2"])), float(t0), False))
        return ems

    # ------------------------------------------------------------------ one step
    def run(self) -> Dict[str, float]:
        dt, T = 1.0, int(self.sc.duration_s)
        for k in range(T):
            t = k * dt
            self._arrivals(t, dt)
            self._update_ems(t, dt)
            active = [e for e in self.ems if e.state in ("approach", "queued")]
            present = {e.j for e in active} | {e.j + 1 for e in active if e.corridor and e.a == "A1" and e.j < 2}
            obs = self._observe(t, dt)
            dec = self.policy.step(t, dt, obs)
            self._apply(t, dt, dec)
            self._metrics_step(dec, present)
            if self.log_on:
                self._snapshot(t, dec)
        return self._summary()

    def _arrivals(self, t, dt):
        for j in range(3):
            for a in APPS:
                if a == "A1" and j > 0:
                    continue
                n = int(self.rng_arr.poisson(self.sc.lam * dt))
                self.queue[j][a] += n
                self.arrived += n
        due = [x for x in self.arrivals if x[0] <= t]
        if due:
            self.arrivals = [x for x in self.arrivals if x[0] > t]
            for _, j, a in due:
                self.queue[j][a] += 1

    def _update_ems(self, t, dt):
        for e in self.ems:
            if e.state == "pending" and t >= e.t_enter:
                e.state, e.eta = "approach", EM_ETA0
            elif e.state == "approach":
                e.eta -= dt
                if e.eta <= 0.0:
                    e.state, e.ahead, e.t_q = "queued", self.queue[e.j][e.a], t
                    self.queue[e.j][e.a] += 1

    def _in_window(self, t):
        w = self.sc.window
        return w is not None and w[0] <= t < w[1]

    def _observe(self, t, dt):
        sc = self.sc
        bad = sc.noise_bad is not None and self._in_window(t)
        noise = self.noise_bad if bad else self.noise_good
        cfg = sc.noise_bad if bad else sc.noise
        vis = sc.vis_low if (sc.vis_drop and self._in_window(t)) else 1.0
        obs = {}
        for j, name in enumerate(JUNCTIONS):
            apps = {}
            for a in APPS:
                q = self.queue[j][a]
                qo = int(self.rng_obs.binomial(q, 1.0 - cfg.p_miss)) if cfg.p_miss > 0 else q
                apps[a] = ApproachObs(a, queue=qo, wait_s=self.wait[j][a], capacity=CAP)
            gt = [e for e in self.ems if e.j == j and e.state in ("approach", "queued")]
            vehicles = [dict(id=e.id, lane=e.a, cls=e.cls,
                             dist_m=(max(e.eta, 0.0) * 10.0 if e.state == "approach" else 0.0)) for e in gt]
            tracks = []
            for o in noise[j].step(t, dt, vehicles, APPS):
                if o.cls not in EMERGENCY_CLASSES:
                    continue
                key = (j, o.vid)
                if t - self.last_seen.get(key, -1e9) > 2.5:
                    self.first_seen[key] = t
                self.last_seen[key] = t
                em = self.em_by_id.get(o.vid)
                if em is not None:
                    eta = max(em.eta, 0.0) if em.state == "approach" else 0.0
                    ahead = self.queue[j][o.lane] if em.state == "approach" else em.ahead
                else:
                    eta, ahead = 15.0, 0
                tracks.append(EmergencyTrack(o.vid, o.cls, o.lane, eta, int(ahead), o.conf, t - self.first_seen[key]))
            obs[name] = JunctionObs(t=t, approaches=apps, emergencies=tuple(tracks), visibility=vis)
        return obs

    def _apply(self, t, dt, dec):
        for j, name in enumerate(JUNCTIONS):
            d, served = dec[name], set()
            if not d.yellow:
                for a in PHASES[d.phase]:
                    self.credit[j][a] += dt / HEADWAY
                    while self.credit[j][a] >= 1.0:
                        if self.queue[j][a] <= 0:
                            self.credit[j][a] = 1.0
                            break
                        if a == "A1" and j < 2 and self.queue[j + 1]["A1"] >= CAP:
                            self.spill += 1
                            self.credit[j][a] = 1.0
                            break
                        self.credit[j][a] -= 1.0
                        self._discharge(j, a, t)
                        served.add(a)
            for a in APPS:
                if d.yellow or a not in PHASES[d.phase]:
                    self.credit[j][a] = 0.0
                if self.queue[j][a] <= 0 or a in served:
                    self.wait[j][a] = 0.0
                else:
                    self.wait[j][a] += dt

    def _discharge(self, j, a, t):
        ems = [e for e in self.ems if e.state == "queued" and e.j == j and e.a == a]
        front = next((e for e in ems if e.ahead == 0), None)
        self.queue[j][a] -= 1
        for e in ems:
            if e is not front:
                e.ahead = max(e.ahead - 1, 0)
        if front is not None:
            front.delays.append(t - front.t_q)
            if front.corridor and a == "A1" and j < 2:
                front.j, front.state, front.eta = j + 1, "approach", TRAVEL
            else:
                front.state = "done"
            return
        self.ord_passages += 1
        if a == "A1" and j < 2:
            self.arrivals.append((t + TRAVEL, j + 1, "A1"))
        else:
            self.exits += 1

    def _metrics_step(self, dec, present):
        emq: Dict[tuple, int] = {}
        for e in self.ems:
            if e.state == "queued":
                emq[(e.j, e.a)] = emq.get((e.j, e.a), 0) + 1
        tot = sum(self.queue[j][a] - emq.get((j, a), 0) for j in range(3) for a in APPS)
        self.q_hist.append(tot / 12.0)
        for j, n in enumerate(JUNCTIONS):
            d = dec[n]
            self.modes[d.mode] = self.modes.get(d.mode, 0) + 1
            if d.yellow and not self.prev_yellow[n]:
                self.switches += 1
            self.prev_yellow[n] = d.yellow
            self.max_wait = max(self.max_wait, max(self.wait[j].values()))
            if d.reason.startswith("emergency[") and j not in present:
                self.false_preempt += 1

    def _snapshot(self, t, dec):
        self.log.append(dict(
            t=t,
            j={n: dict(phase=dec[n].phase, yellow=dec[n].yellow, mode=dec[n].mode, reason=dec[n].reason,
                       q=[self.queue[j][a] for a in APPS]) for j, n in enumerate(JUNCTIONS)},
            ems=[dict(id=e.id, cls=e.cls, j=e.j, a=e.a, state=e.state, eta=e.eta, ahead=e.ahead)
                 for e in self.ems if e.state in ("approach", "queued")]))

    def _summary(self) -> Dict[str, float]:
        sc, T, nan = self.sc, self.sc.duration_s, float("nan")
        qh = np.array(self.q_hist, float)
        em_all, em_c, em_x, em_w, inc = [], [], [], [], 0
        for e in self.ems:
            if e.state == "pending":
                continue
            d = sum(e.delays) + ((T - e.t_q) if e.state == "queued" else 0.0)
            inc += int(e.state != "done")
            em_all.append(d)
            (em_c if e.corridor else em_x).append(d)
            if sc.window is not None and sc.window[0] <= e.t_enter < sc.window[1]:
                em_w.append(d)

        def mean(v):
            return float(np.mean(v)) if len(v) else nan
        out = dict(
            ord_delay_s=float(qh.sum() * 12.0 / max(self.ord_passages, 1)),
            ord_queue_mean=float(qh.mean()),
            ord_queue_mean_win=float(qh[int(sc.window[0]):int(sc.window[1])].mean()) if sc.window is not None else nan,
            max_wait_s=float(self.max_wait), throughput_vph=float(self.exits / T * 3600.0),
            spillback_per_h=float(self.spill / T * 3600.0), switches_per_h=float(self.switches / T * 3600.0),
            em_delay_mean=mean(em_all), em_delay_corridor=mean(em_c), em_delay_cross=mean(em_x), em_delay_win=mean(em_w),
            em_incomplete=float(inc), n_em=float(len(em_all)), false_preempt_s=float(self.false_preempt),
            arrivals=float(self.arrived))
        tot = max(sum(self.modes.values()), 1)
        for md, c in self.modes.items():
            out["mode_" + md.replace("+", "_")] = c / tot
        return out


def run_sim(policy, sc: Scenario, seed: int, log: bool = False):
    sim = CorridorSim(policy, sc, seed, log)
    return sim.run(), sim.log
